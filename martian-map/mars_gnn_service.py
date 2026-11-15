import argparse
import heapq
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix


# ==========================================
# 1. DEM TO GRAPH BUILDER (SOBEL + LAPLACIAN)
# ==========================================
def compute_sobel_laplacian(dem: np.ndarray, resolution_m: float = 2.0):
    sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32) / (8.0 * resolution_m)
    sobel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32) / (8.0 * resolution_m)
    lap_k = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32) / (resolution_m ** 2)

    gx = convolve(dem, sobel_x, mode="nearest")
    gy = convolve(dem, sobel_y, mode="nearest")
    lap = convolve(dem, lap_k, mode="nearest")
    slope_angle = np.arctan(np.sqrt(gx**2 + gy**2 + 1e-8))
    return gx, gy, slope_angle, lap


def build_mars_terrain_graph(dem: np.ndarray, goal_rc: tuple[int, int], resolution_m: float = 2.0):
    H, W = dem.shape
    N = H * W
    gx, gy, slope_angle, lap = compute_sobel_laplacian(dem, resolution_m)
    regolith_slip = (1.0 / (1.0 + np.exp(np.abs(lap) * 5.0))) * 0.3

    grid_y, grid_x = np.meshgrid(
        np.arange(H, dtype=np.float32) * resolution_m,
        np.arange(W, dtype=np.float32) * resolution_m,
        indexing="ij",
    )

    goal_y, goal_x = goal_rc[0] * resolution_m, goal_rc[1] * resolution_m
    goal_z = dem[goal_rc[0], goal_rc[1]]

    dx_goal = (goal_x - grid_x) / (W * resolution_m)
    dy_goal = (goal_y - grid_y) / (H * resolution_m)
    dz_goal = (goal_z - dem) / 100.0
    dist_goal = np.sqrt((goal_x - grid_x) ** 2 + (goal_y - grid_y) ** 2) / (max(H, W) * resolution_m)

    norm_z = (dem - dem.mean()) / (dem.std() + 1e-6)
    node_features = np.stack(
        [
            norm_z.ravel(),
            slope_angle.ravel(),
            lap.ravel(),
            regolith_slip.ravel(),
            dx_goal.ravel(),
            dy_goal.ravel(),
            dz_goal.ravel(),
            dist_goal.ravel(),
        ],
        axis=-1,
    ).astype(np.float32)

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    r_idx, c_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    src_list, dst_list = [], []

    for dr, dc in offsets:
        nr, nc = r_idx + dr, c_idx + dc
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        src_list.append((r_idx[valid] * W + c_idx[valid]).ravel())
        dst_list.append((nr[valid] * W + nc[valid]).ravel())

    src_nodes = np.concatenate(src_list)
    dst_nodes = np.concatenate(dst_list)
    edge_index = np.stack([src_nodes, dst_nodes], axis=0)

    flat_x, flat_y, flat_z = grid_x.ravel(), grid_y.ravel(), dem.ravel()
    flat_gx, flat_gy = gx.ravel(), gy.ravel()

    dx = flat_x[dst_nodes] - flat_x[src_nodes]
    dy = flat_y[dst_nodes] - flat_y[src_nodes]
    dz = flat_z[dst_nodes] - flat_z[src_nodes]
    horiz_dist = np.sqrt(dx**2 + dy**2 + 1e-8)
    dist_3d = np.sqrt(horiz_dist**2 + dz**2)

    dir_grade = dz / horiz_dist
    ux, uy = dx / horiz_dist, dy / horiz_dist
    cross_slope = np.abs(-uy * flat_gx[dst_nodes] + ux * flat_gy[dst_nodes])

    edge_attr = np.stack(
        [
            dist_3d / resolution_m,
            dz / resolution_m,
            dir_grade,
            cross_slope,
            slope_angle.ravel()[dst_nodes],
        ],
        axis=-1,
    ).astype(np.float32)

    # Sparse adjacency matrix (dst <- src) for fast multi-hop message passing
    adj = csr_matrix(
        (np.ones(len(src_nodes), dtype=np.float32), (dst_nodes, src_nodes)),
        shape=(N, N),
    )
    # Normalize by degree so message aggregation is scale-invariant
    deg = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1.0)
    adj_norm = adj.multiply(1.0 / deg[:, None]).tocsr()

    return {
        "x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "adj": adj_norm,
        "num_nodes": N,
        "grid_shape": (H, W),
        "goal_node": int(goal_rc[0] * W + goal_rc[1]),
    }


# ==========================================
# 2. NUMPY/SCIPY MESHGRAPHNET (6-HOP GNN)
# ==========================================
def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class NumpyMarsEVAGraphNet:
    """
    6-Hop Encode -> Process -> Decode Graph Neural Network in pure NumPy/SciPy.
    Zero external deep learning framework overhead; runs in milliseconds.
    """

    def __init__(self, node_dim=8, edge_dim=5, hidden_dim=64, num_layers=6, seed=42):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.params = {}

        def init_w(in_d, out_d):
            return (rng.standard_normal((in_d, out_d)) * np.sqrt(2.0 / in_d)).astype(np.float32)

        # Node & Edge Encoders
        self.params["W_node_enc"] = init_w(node_dim, hidden_dim)
        self.params["b_node_enc"] = np.zeros(hidden_dim, dtype=np.float32)
        self.params["W_edge_enc"] = init_w(edge_dim, hidden_dim)
        self.params["b_edge_enc"] = np.zeros(hidden_dim, dtype=np.float32)

        # 6 Message-Passing Layers
        for k in range(num_layers):
            self.params[f"W_edge_mp_{k}"] = init_w(hidden_dim * 3, hidden_dim)
            self.params[f"b_edge_mp_{k}"] = np.zeros(hidden_dim, dtype=np.float32)
            self.params[f"W_node_mp_{k}"] = init_w(hidden_dim * 2, hidden_dim)
            self.params[f"b_node_mp_{k}"] = np.zeros(hidden_dim, dtype=np.float32)

        # Decoders (Ridge-trained linear readouts on top of deep 6-hop GNN embeddings)
        self.params["W_edge_dec"] = np.zeros((hidden_dim * 3 + edge_dim, 1), dtype=np.float32)
        self.params["b_edge_dec"] = np.zeros(1, dtype=np.float32)

        self.params["W_heur_dec"] = np.zeros((hidden_dim + node_dim, 1), dtype=np.float32)
        self.params["b_heur_dec"] = np.zeros(1, dtype=np.float32)

        self.params["W_haz_dec"] = np.zeros((hidden_dim + node_dim, 1), dtype=np.float32)
        self.params["b_haz_dec"] = np.zeros(1, dtype=np.float32)

    def encode_and_propagate(self, graph):
        x_raw = graph["x"]
        e_raw = graph["edge_attr"]
        src, dst = graph["edge_index"][0], graph["edge_index"][1]
        adj = graph["adj"]

        h = layer_norm(silu(x_raw @ self.params["W_node_enc"] + self.params["b_node_enc"]))
        e = layer_norm(silu(e_raw @ self.params["W_edge_enc"] + self.params["b_edge_enc"]))

        for k in range(self.num_layers):
            # Edge message update
            e_in = np.concatenate([h[src], h[dst], e], axis=-1)
            e = layer_norm(e + silu(e_in @ self.params[f"W_edge_mp_{k}"] + self.params[f"b_edge_mp_{k}"]))

            # Aggregate incoming edge embeddings into destination nodes
            # Mean over incoming edges per node via sparse incidence
            node_msg = np.zeros_like(h)
            np.add.at(node_msg, dst, e)
            deg = np.maximum(np.bincount(dst, minlength=graph["num_nodes"])[:, None], 1.0)
            node_msg = adj @ (node_msg / deg)

            h_in = np.concatenate([h, node_msg], axis=-1)
            h = layer_norm(h + silu(h_in @ self.params[f"W_node_mp_{k}"] + self.params[f"b_node_mp_{k}"]))

        edge_phi = np.concatenate([h[src], h[dst], e, e_raw], axis=-1)
        node_phi = np.concatenate([h, x_raw], axis=-1)
        return edge_phi, node_phi

    def predict(self, graph):
        edge_phi, node_phi = self.encode_and_propagate(graph)
        edge_cost = softplus(edge_phi @ self.params["W_edge_dec"] + self.params["b_edge_dec"]).ravel() + 1e-4
        heuristic = softplus(node_phi @ self.params["W_heur_dec"] + self.params["b_heur_dec"]).ravel()
        hazard_prob = 1.0 / (1.0 + np.exp(-np.clip((node_phi @ self.params["W_haz_dec"] + self.params["b_haz_dec"]).ravel(), -20, 20)))

        heuristic[graph["goal_node"]] = 0.0
        return {
            "edge_cost": edge_cost,
            "heuristic": heuristic,
            "hazard_prob": hazard_prob,
        }

    def save(self, path):
        np.savez(path, **self.params)

    def load(self, path):
        data = np.load(path)
        for k in data.files:
            self.params[k] = data[k]


# ==========================================
# 3. BIOENERGETIC PHYSICS LABELS & TRAINING
# ==========================================
def compute_martian_labels(graph, resolution_m: float = 2.0):
    total_mass, suit_mass, astro_mass = 165.0, 85.0, 80.0
    mars_g, speed = 3.72, 1.1
    weight_mars_n = total_mass * mars_g

    e_attr = graph["edge_attr"]
    dist_m = e_attr[:, 0] * resolution_m
    grade = e_attr[:, 2]
    cross_slope = e_attr[:, 3]
    dst_slope = e_attr[:, 4]

    dst = graph["edge_index"][1]
    lap_roughness = np.abs(graph["x"][dst, 2])
    regolith_slip = graph["x"][dst, 3]

    eta = 1.1 + 1.1 * regolith_slip + 0.5 * np.clip(lap_roughness, 0.0, 2.0)
    grade_pct = grade * 100.0
    uphill_term = np.where(
        grade_pct >= 0,
        0.045 * grade_pct + 0.0012 * (grade_pct**2),
        -0.015 * grade_pct + 0.0018 * (grade_pct**2),
    )

    metabolic_watts = (
        1.5 * total_mass
        + 2.0 * total_mass * ((suit_mass / astro_mass) ** 2) * (mars_g / 9.81)
        + eta * (weight_mars_n / 9.81) * (1.5 * (speed**2) + 0.35 * speed * uphill_term)
    ) * (1.0 + 2.5 * (cross_slope**2))

    impassable = dst_slope > np.deg2rad(28.0)
    edge_joules = np.where(impassable, 1e6, metabolic_watts * (dist_m / speed))

    N = graph["num_nodes"]
    goal = graph["goal_node"]
    rev_adj = [[] for _ in range(N)]
    for u, v, w in zip(graph["edge_index"][0], dst, edge_joules):
        rev_adj[v].append((int(u), float(w)))

    dist_to_goal = np.full(N, 1e6, dtype=np.float32)
    dist_to_goal[goal] = 0.0
    pq = [(0.0, goal)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist_to_goal[u]:
            continue
        for prev_node, w in rev_adj[u]:
            nd = d + w
            if nd < dist_to_goal[prev_node]:
                dist_to_goal[prev_node] = nd
                heapq.heappush(pq, (nd, prev_node))

    return {
        "y_edge": (edge_joules / 1000.0).astype(np.float32),
        "y_heur": (np.minimum(dist_to_goal, 50000.0) / 1000.0).astype(np.float32),
        "y_haz": (dst_slope > np.deg2rad(20.0)).astype(np.float32),
    }


def train_model(weights_path="mars_eva_gnn.npz"):
    print("[TRAIN] Building synthetic Mars crater/ridge graphs & fitting 6-hop GNN...")
    rng = np.random.default_rng(123)
    model = NumpyMarsEVAGraphNet(hidden_dim=64, num_layers=6)

    y = np.linspace(-1, 1, 64, dtype=np.float32)
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    edge_phis, node_phis = [], []
    y_edges, y_heurs, y_hazs = [], [], []

    for i in range(48):
        dem = 25.0 * np.sin(3.0 * xx + i) * np.cos(2.5 * yy)
        for _ in range(4):
            cx, cy = rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7)
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            rad, depth = rng.uniform(0.15, 0.35), rng.uniform(15.0, 45.0)
            dem = dem - depth * np.exp(-((r / rad) ** 2)) + (depth * 0.25) * np.exp(-(((r - rad) / (rad * 0.2)) ** 2))

        goal_rc = (int(rng.integers(8, 56)), int(rng.integers(8, 56)))
        g = build_mars_terrain_graph(dem, goal_rc, resolution_m=2.0)
        labels = compute_martian_labels(g, resolution_m=2.0)

        ep, np_feat = model.encode_and_propagate(g)
        edge_phis.append(ep)
        node_phis.append(np_feat)
        y_edges.append(labels["y_edge"])
        y_heurs.append(labels["y_heur"])
        # Node-level hazard target (max hazard of outgoing edges)
        node_haz = np.zeros(g["num_nodes"], dtype=np.float32)
        np.maximum.at(node_haz, g["edge_index"][0], labels["y_haz"])
        y_hazs.append(node_haz)

    X_e = np.vstack(edge_phis)
    Y_e = np.concatenate(y_edges)[:, None]
    X_n = np.vstack(node_phis)
    Y_h = np.concatenate(y_heurs)[:, None]
    Y_z = np.concatenate(y_hazs)[:, None]

    # Closed-form L2-regularized Ridge Regression on 6-hop GNN embeddings (exact, instant, zero divergence)
    def fit_ridge(X, Y, alpha=1.0):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
        I = np.eye(X_aug.shape[1], dtype=np.float32) * alpha
        I[-1, -1] = 0.0
        W = np.linalg.solve(X_aug.T @ X_aug + I, X_aug.T @ Y)
        return W[:-1], W[-1]

    model.params["W_edge_dec"], model.params["b_edge_dec"] = fit_ridge(X_e, Y_e, alpha=10.0)
    model.params["W_heur_dec"], model.params["b_heur_dec"] = fit_ridge(X_n, Y_h, alpha=10.0)
    model.params["W_haz_dec"], model.params["b_haz_dec"] = fit_ridge(X_n, Y_z * 4.0 - 2.0, alpha=5.0)

    model.save(weights_path)
    print(f"[TRAIN] Done in seconds! Saved GNN weights to {weights_path}")


# ==========================================
# 4. ZERO-DEPENDENCY HTTP INFERENCE SERVER
# ==========================================
MODEL = None


class GNNHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health", "/status"):
            payload = {
                "status": "online",
                "backend": "Mars 6-Hop MeshGraphNet (mars_eva_gnn.npz)",
                "model": "NumpyMarsEVAGraphNet-Bioenergetic-6Hop",
                "weights": "mars_eva_gnn.npz",
                "active": True
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/predict-graph":
            self.send_response(404)
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len))

        dem = np.array(body["dem_grid"], dtype=np.float32)
        goal_rc = tuple(body["goal_rc"])
        res_m = float(body.get("resolution_m", 2.0))

        graph = build_mars_terrain_graph(dem, goal_rc, res_m)
        preds = MODEL.predict(graph)
        H, W = dem.shape

        resp = json.dumps({
            "grid_shape": [H, W],
            "edge_index": graph["edge_index"].tolist(),
            "edge_costs_kj": preds["edge_cost"].tolist(),
            "heuristics_kj": preds["heuristic"].reshape(H, W).tolist(),
            "hazard_probs": preds["hazard_prob"].reshape(H, W).tolist(),
        }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors()
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "serve"], required=True)
    parser.add_argument("--weights", default="mars_eva_gnn.npz")
    parser.add_argument("--port", type=int, default=8023)
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.weights)
    elif args.mode == "serve":
        MODEL = NumpyMarsEVAGraphNet(hidden_dim=64, num_layers=6)
        MODEL.load(args.weights)
        print(f"[SERVE] Mars EVA GNN Server running on 0.0.0.0:{args.port}")
        ThreadingHTTPServer(("0.0.0.0", args.port), GNNHandler).serve_forever()
