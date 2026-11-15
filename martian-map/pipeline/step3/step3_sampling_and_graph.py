#!/usr/bin/env python3
"""
step3_sampling_and_graph.py
===========================
Step 3: Passable-Only 1D Stratified CDF Node Sampling, Waypoint Pinning,
Vectorized KDTree Line-of-Sight Edge Validation, and Inter-Chunk Seam Stitching.
Reads all constants from .env and outputs step3_output.npz for Step 4.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial import KDTree


# =========================================================================
# 0. ZERO-DEPENDENCY .env LOADER & CONSTANT BINDINGS
# =========================================================================

def load_env_file(env_path: str = ".env") -> None:
    """Parses a .env file into os.environ without external dependencies."""
    p = Path(env_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing environment configuration file: {p.resolve()}")
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split("#", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


load_env_file(".env")

# --- Shared Grid & Math Constants from .env ---
GRID_CHUNKS_SIDE              = int(os.environ["GRID_CHUNKS_SIDE"])
CHUNK_SIZE_PX                 = int(os.environ["CHUNK_SIZE_PX"])
GLOBAL_HALO_PX                = int(os.environ["GLOBAL_HALO_PX"])
CENTER_LON_DEG                = float(os.environ["CENTER_LON_DEG"])
CENTER_LAT_DEG                = float(os.environ["CENTER_LAT_DEG"])
MAP_PIXEL_SIZE_X_M            = float(os.environ["MAP_PIXEL_SIZE_X_M"])
MAP_PIXEL_SIZE_Y_M            = float(os.environ["MAP_PIXEL_SIZE_Y_M"])
DUMMY_BASE_ALT_M              = float(os.environ["DUMMY_BASE_ALT_M"])
DUMMY_REGIONAL_SLOPE_M        = float(os.environ["DUMMY_REGIONAL_SLOPE_M"])
STEP2_DUMMY_CLIFF_COL_MIN     = float(os.environ["STEP2_DUMMY_CLIFF_COL_MIN"])
STEP2_DUMMY_CLIFF_COL_MAX     = float(os.environ["STEP2_DUMMY_CLIFF_COL_MAX"])
STEP2_DUMMY_GAP_ROW_MIN       = float(os.environ["STEP2_DUMMY_GAP_ROW_MIN"])
STEP2_DUMMY_GAP_ROW_MAX       = float(os.environ["STEP2_DUMMY_GAP_ROW_MAX"])
ZERO_FLOAT                    = float(os.environ["ZERO_FLOAT"])
ONE_FLOAT                     = float(os.environ["ONE_FLOAT"])
TWO_FLOAT                     = float(os.environ["TWO_FLOAT"])

# --- Step 3 File Paths ---
STEP3_INPUT_NPZ_PATH          = os.environ["STEP3_INPUT_NPZ_PATH"]
STEP3_OUTPUT_NPZ_PATH         = os.environ["STEP3_OUTPUT_NPZ_PATH"]

# --- Step 3 Active Chunks & Stratified CDF Sampling Constants ---
STEP3_ACTIVE_CHUNKS_RAW       = os.environ["STEP3_ACTIVE_CHUNKS"]
STEP3_SAMPLES_PER_CHUNK       = int(os.environ["STEP3_SAMPLES_PER_CHUNK"])
STEP3_CDF_BASELINE_EPS        = float(os.environ["STEP3_CDF_BASELINE_EPS"])
STEP3_HASH_PRIME_I            = int(os.environ["STEP3_HASH_PRIME_I"])
STEP3_HASH_PRIME_J            = int(os.environ["STEP3_HASH_PRIME_J"])
STEP3_HASH_SEED               = int(os.environ["STEP3_HASH_SEED"])
STEP3_HASH_BITMASK            = int(os.environ["STEP3_HASH_BITMASK"])
STEP3_INDEX_ZERO              = int(os.environ["STEP3_INDEX_ZERO"])
STEP3_INDEX_ONE               = int(os.environ["STEP3_INDEX_ONE"])

# --- Step 3 KDTree & Line-of-Sight Edge Validation Constants ---
STEP3_EDGE_RADIUS_M           = float(os.environ["STEP3_EDGE_RADIUS_M"])
STEP3_LOS_INTERP_POINTS       = int(os.environ["STEP3_LOS_INTERP_POINTS"])
STEP3_LAMBDA_MIN              = float(os.environ["STEP3_LAMBDA_MIN"])
STEP3_LAMBDA_MAX              = float(os.environ["STEP3_LAMBDA_MAX"])
STEP3_SEAM_STRIP_HALFWIDTH_PX = int(os.environ["STEP3_SEAM_STRIP_HALFWIDTH_PX"])

# --- Step 3 Pinned Waypoints List ---
PINNED_WAYPOINT_SPECS: List[Dict[str, Any]] = [
    {"id": os.environ["WP_1_ID"], "rel_r": float(os.environ["WP_1_REL_R"]), "rel_c": float(os.environ["WP_1_REL_C"])},
    {"id": os.environ["WP_2_ID"], "rel_r": float(os.environ["WP_2_REL_R"]), "rel_c": float(os.environ["WP_2_REL_C"])},
    {"id": os.environ["WP_3_ID"], "rel_r": float(os.environ["WP_3_REL_R"]), "rel_c": float(os.environ["WP_3_REL_C"])},
    {"id": os.environ["WP_4_ID"], "rel_r": float(os.environ["WP_4_REL_R"]), "rel_c": float(os.environ["WP_4_REL_C"])},
    {"id": os.environ["WP_5_ID"], "rel_r": float(os.environ["WP_5_REL_R"]), "rel_c": float(os.environ["WP_5_REL_C"])},
    {"id": os.environ["WP_6_ID"], "rel_r": float(os.environ["WP_6_REL_R"]), "rel_c": float(os.environ["WP_6_REL_C"])},
]

# --- Step 3 Dummy Input Generator Constants ---
STEP3_DUMMY_RNG_SEED          = int(os.environ["STEP3_DUMMY_RNG_SEED"])
STEP3_DUMMY_D_BASE            = float(os.environ["STEP3_DUMMY_D_BASE"])
STEP3_DUMMY_D_AMP             = float(os.environ["STEP3_DUMMY_D_AMP"])
STEP3_DUMMY_ETA_BASE          = float(os.environ["STEP3_DUMMY_ETA_BASE"])
STEP3_DUMMY_ETA_AMP           = float(os.environ["STEP3_DUMMY_ETA_AMP"])


def parse_active_chunks(raw_str: str) -> List[Tuple[int, int]]:
    """Parses semicolon-separated chunk pairs like '1,1;1,2' into [(1, 1), (1, 2)]."""
    chunks: List[Tuple[int, int]] = []
    for item in raw_str.split(";"):
        parts = item.strip().split(",")
        chunks.append((int(parts[STEP3_INDEX_ZERO]), int(parts[STEP3_INDEX_ONE])))
    return chunks


# =========================================================================
# 1. INPUT LOADERS: (A) REAL INPUT [ACTIVE] vs. (B) DUMMY INPUT [COMMENTED]
# =========================================================================

# -------------------------------------------------------------------------
# (A) REAL INPUT LOADER (Active: reads step2_output.npz)
# -------------------------------------------------------------------------
def load_step3_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads the precomputed Step 2 bundle from step2_output.npz."""
    bundle = np.load(STEP3_INPUT_NPZ_PATH, allow_pickle=True)
    meta = json.loads(str(bundle["metadata_json"]))
    arrays = {
        k: bundle[k]
        for k in bundle.files
        if k != "metadata_json"
    }
    meta["step3_input_mode"] = f"REAL_STEP2_NPZ ({STEP3_INPUT_NPZ_PATH})"
    return arrays, meta


# -------------------------------------------------------------------------
# (B) DUMMY INPUT GENERATOR (Commented out; use only for standalone testing)
# -------------------------------------------------------------------------
# def load_step3_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
#     """Generates synthetic Step 2 deviation scores, passability, friction, and elevation."""
#     G_core = GRID_CHUNKS_SIDE * CHUNK_SIZE_PX
#     G_pad = G_core + int(TWO_FLOAT) * GLOBAL_HALO_PX
#     rng = np.random.default_rng(STEP3_DUMMY_RNG_SEED)
#
#     r_idx, c_idx = np.indices((G_pad, G_pad), dtype=np.float64)
#     r_norm = (r_idx - GLOBAL_HALO_PX) / G_core
#     c_norm = (c_idx - GLOBAL_HALO_PX) / G_core
#
#     cliff_band = (c_norm >= STEP2_DUMMY_CLIFF_COL_MIN) & (c_norm <= STEP2_DUMMY_CLIFF_COL_MAX)
#     saddle_gap = (r_norm >= STEP2_DUMMY_GAP_ROW_MIN) & (r_norm <= STEP2_DUMMY_GAP_ROW_MAX)
#     pass_pad = ~(cliff_band & (~saddle_gap))
#
#     z_pad = (DUMMY_BASE_ALT_M + DUMMY_REGIONAL_SLOPE_M * r_norm).astype(np.float32)
#     eta_pad = (
#         STEP3_DUMMY_ETA_BASE + STEP3_DUMMY_ETA_AMP * (np.sin(math.pi * r_norm) ** TWO_FLOAT)
#     ).astype(np.float32)
#
#     r_core, c_core = np.indices((G_core, G_core), dtype=np.float64) / G_core
#     pass_core = pass_pad[GLOBAL_HALO_PX:-GLOBAL_HALO_PX, GLOBAL_HALO_PX:-GLOBAL_HALO_PX]
#     D_raw = (
#         STEP3_DUMMY_D_BASE
#         + STEP3_DUMMY_D_AMP * np.exp(-((c_core - STEP2_DUMMY_CLIFF_COL_MIN) ** TWO_FLOAT) / 0.005)
#         + rng.uniform(ZERO_FLOAT, STEP3_CDF_BASELINE_EPS, size=(G_core, G_core))
#     )
#     D_core_global = np.where(pass_core, np.clip(D_raw, ZERO_FLOAT, ONE_FLOAT), ZERO_FLOAT).astype(np.float32)
#
#     dx_m = MAP_PIXEL_SIZE_X_M * math.cos(math.radians(CENTER_LAT_DEG))
#     dy_m = MAP_PIXEL_SIZE_Y_M
#
#     arrays = {
#         "z_pad": z_pad,
#         "eta_pad": eta_pad,
#         "pass_pad": pass_pad,
#         "D_core_global": D_core_global,
#     }
#     meta = {
#         "step": 2,
#         "step3_input_mode": "DUMMY_SYNTHETIC_STEP2_INPUT",
#         "grid_chunks_side": GRID_CHUNKS_SIDE,
#         "chunk_size_L": CHUNK_SIZE_PX,
#         "halo_h": GLOBAL_HALO_PX,
#         "global_core_size": G_core,
#         "center_lon": CENTER_LON_DEG,
#         "center_lat": CENTER_LAT_DEG,
#         "dx_ground_m": round(dx_m, 6),
#         "dy_ground_m": round(dy_m, 6),
#     }
#     return arrays, meta


# =========================================================================
# 2. STEP 3 CORE FUNCTIONS: STRATIFIED CDF SAMPLING & KDTREE GRAPH BUILDER
# =========================================================================

def build_waypoint_catalog(G_core: int, L: int) -> List[Dict[str, Any]]:
    """Converts normalized waypoint specs from .env into chunk & pixel coordinates."""
    catalog: List[Dict[str, Any]] = []
    for wp in PINNED_WAYPOINT_SPECS:
        r_g = int(round(wp["rel_r"] * G_core))
        c_g = int(round(wp["rel_c"] * G_core))
        catalog.append({
            "id": wp["id"],
            "global_rc": [r_g, c_g],
            "chunk_ij": [r_g // L, c_g // L],
            "local_rc": [r_g % L, c_g % L],
        })
    return catalog


def sample_single_chunk_graph(
    I: int,
    J: int,
    L: int,
    D_chunk: np.ndarray,
    pass_chunk: np.ndarray,
    eta_chunk: np.ndarray,
    dx_m: float,
    dy_m: float,
    waypoint_catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    3.1-3.3: Performs passable-only 1D stratified inverse-CDF node sampling,
    pins mission waypoints, and validates intra-chunk KDTree edges via K-point LOS.
    """
    seed_ij = ((I * STEP3_HASH_PRIME_I) ^ (J * STEP3_HASH_PRIME_J) ^ STEP3_HASH_SEED) & STEP3_HASH_BITMASK
    rng = np.random.default_rng(seed_ij)

    pass_rc = np.argwhere(pass_chunk)
    weights = D_chunk[pass_chunk].astype(np.float64) + STEP3_CDF_BASELINE_EPS
    cdf = np.cumsum(weights)
    cdf /= cdf[-STEP3_INDEX_ONE]

    n_samples = STEP3_SAMPLES_PER_CHUNK
    strata = (np.arange(n_samples, dtype=np.float64) + rng.random(n_samples)) / n_samples
    sampled_idx = np.unique(np.searchsorted(cdf, strata))
    local_rc = pass_rc[sampled_idx]

    # Pin any mission waypoints falling inside chunk (I, J)
    pinned_global: Dict[str, List[int]] = {}
    for wp in waypoint_catalog:
        if tuple(wp["chunk_ij"]) == (I, J):
            wp_rc = np.array(wp["local_rc"], dtype=int)
            if not pass_chunk[wp_rc[STEP3_INDEX_ZERO], wp_rc[STEP3_INDEX_ONE]]:
                dist_sq = np.sum((pass_rc - wp_rc) ** int(TWO_FLOAT), axis=STEP3_INDEX_ONE)
                wp_rc = pass_rc[int(np.argmin(dist_sq))]
            local_rc = np.vstack([wp_rc, local_rc])
            pinned_global[wp["id"]] = [
                int(wp_rc[STEP3_INDEX_ZERO] + I * L),
                int(wp_rc[STEP3_INDEX_ONE] + J * L),
            ]

    _, uniq_idx = np.unique(local_rc, axis=STEP3_INDEX_ZERO, return_index=True)
    local_rc = local_rc[np.sort(uniq_idx)]
    global_rc = local_rc + np.array([I * L, J * L], dtype=int)

    # Query KDTree in true ground meters (dy_m, dx_m)
    coords_m = np.column_stack([
        global_rc[:, STEP3_INDEX_ZERO] * dy_m,
        global_rc[:, STEP3_INDEX_ONE] * dx_m,
    ])
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(r=STEP3_EDGE_RADIUS_M, output_type="ndarray")

    if pairs.size == STEP3_INDEX_ZERO:
        return {
            "chunk_ij": (I, J),
            "global_rc": global_rc,
            "valid_pairs": np.empty((STEP3_INDEX_ZERO, int(TWO_FLOAT)), dtype=int),
            "eta_bar": np.empty((STEP3_INDEX_ZERO,), dtype=np.float32),
            "pinned_waypoints": pinned_global,
        }

    # Vectorized K-point Line-of-Sight cliff-intersection check
    u_pts = local_rc[pairs[:, STEP3_INDEX_ZERO]]
    v_pts = local_rc[pairs[:, STEP3_INDEX_ONE]]
    lambdas = np.linspace(STEP3_LAMBDA_MIN, STEP3_LAMBDA_MAX, STEP3_LOS_INTERP_POINTS)[None, :, None]
    seg_rc = np.clip(
        np.rint((ONE_FLOAT - lambdas) * u_pts[:, None, :] + lambdas * v_pts[:, None, :]).astype(int),
        STEP3_INDEX_ZERO,
        L - STEP3_INDEX_ONE,
    )
    valid_mask = np.all(
        pass_chunk[seg_rc[:, :, STEP3_INDEX_ZERO], seg_rc[:, :, STEP3_INDEX_ONE]],
        axis=STEP3_INDEX_ONE,
    )
    valid_pairs = pairs[valid_mask]
    valid_seg = seg_rc[valid_mask]
    eta_bar = np.mean(
        eta_chunk[valid_seg[:, :, STEP3_INDEX_ZERO], valid_seg[:, :, STEP3_INDEX_ONE]],
        axis=STEP3_INDEX_ONE,
    ).astype(np.float32)

    return {
        "chunk_ij": (I, J),
        "global_rc": global_rc,
        "valid_pairs": valid_pairs,
        "eta_bar": eta_bar,
        "pinned_waypoints": pinned_global,
    }


def run_step3_pipeline() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Executes Step 3 chunk sampling, edge validation, and seam stitching."""
    arrays, meta = load_step3_inputs()

    L = int(meta["chunk_size_L"])
    h_global = int(meta["halo_h"])
    G_core = int(meta["global_core_size"])
    dx_m = float(meta["dx_ground_m"])
    dy_m = float(meta["dy_ground_m"])

    pass_core_global = arrays["pass_pad"][h_global:-h_global, h_global:-h_global].astype(bool)
    eta_core_global = arrays["eta_pad"][h_global:-h_global, h_global:-h_global].astype(np.float32)
    z_core_global = arrays["z_pad"][h_global:-h_global, h_global:-h_global].astype(np.float32)
    D_core_global = arrays["D_core_global"].astype(np.float32)

    waypoint_catalog = build_waypoint_catalog(G_core, L)
    active_chunks = parse_active_chunks(STEP3_ACTIVE_CHUNKS_RAW)

    chunk_graphs: List[Dict[str, Any]] = []
    pinned_waypoints: Dict[str, List[int]] = {}

    for I, J in active_chunks:
        r0, r1 = I * L, (I + STEP3_INDEX_ONE) * L
        c0, c1 = J * L, (J + STEP3_INDEX_ONE) * L
        cg = sample_single_chunk_graph(
            I=I,
            J=J,
            L=L,
            D_chunk=D_core_global[r0:r1, c0:c1],
            pass_chunk=pass_core_global[r0:r1, c0:c1],
            eta_chunk=eta_core_global[r0:r1, c0:c1],
            dx_m=dx_m,
            dy_m=dy_m,
            waypoint_catalog=waypoint_catalog,
        )
        chunk_graphs.append(cg)
        pinned_waypoints.update(cg["pinned_waypoints"])

    # Combine nodes across all active chunks into a unified global node table
    all_nodes_list: List[np.ndarray] = []
    all_edges_list: List[np.ndarray] = []
    all_eta_list: List[np.ndarray] = []
    node_offset = STEP3_INDEX_ZERO
    chunk_node_ranges: List[Tuple[Tuple[int, int], int, int]] = []

    for cg in chunk_graphs:
        n_c = int(cg["global_rc"].shape[STEP3_INDEX_ZERO])
        all_nodes_list.append(cg["global_rc"])
        all_edges_list.append(cg["valid_pairs"] + node_offset)
        all_eta_list.append(cg["eta_bar"])
        chunk_node_ranges.append((cg["chunk_ij"], node_offset, node_offset + n_c))
        node_offset += n_c

    nodes_global_rc = np.vstack(all_nodes_list).astype(np.int32)
    intra_edges_uv = np.vstack(all_edges_list).astype(np.int32)
    intra_eta_bar = np.concatenate(all_eta_list).astype(np.float32)

    # --- 3.4: Inter-Chunk Seam Stitching Across Adjacent Active Chunks ---
    bridge_edges: List[Tuple[int, int]] = []
    bridge_etas: List[float] = []

    for idx_a in range(len(chunk_node_ranges)):
        (Ia, Ja), start_a, end_a = chunk_node_ranges[idx_a]
        rc_a = nodes_global_rc[start_a:end_a]
        for idx_b in range(idx_a + STEP3_INDEX_ONE, len(chunk_node_ranges)):
            (Ib, Jb), start_b, end_b = chunk_node_ranges[idx_b]
            if abs(Ia - Ib) + abs(Ja - Jb) != STEP3_INDEX_ONE:
                continue

            rc_b = nodes_global_rc[start_b:end_b]
            if Ia == Ib:
                seam_c = max(Ja, Jb) * L
                mask_a = np.abs(rc_a[:, STEP3_INDEX_ONE] - seam_c) <= STEP3_SEAM_STRIP_HALFWIDTH_PX
                mask_b = np.abs(rc_b[:, STEP3_INDEX_ONE] - seam_c) <= STEP3_SEAM_STRIP_HALFWIDTH_PX
            else:
                seam_r = max(Ia, Ib) * L
                mask_a = np.abs(rc_a[:, STEP3_INDEX_ZERO] - seam_r) <= STEP3_SEAM_STRIP_HALFWIDTH_PX
                mask_b = np.abs(rc_b[:, STEP3_INDEX_ZERO] - seam_r) <= STEP3_SEAM_STRIP_HALFWIDTH_PX

            loc_a = np.where(mask_a)[STEP3_INDEX_ZERO]
            loc_b = np.where(mask_b)[STEP3_INDEX_ZERO]
            if loc_a.size == STEP3_INDEX_ZERO or loc_b.size == STEP3_INDEX_ZERO:
                continue

            tree_a = KDTree(np.column_stack([rc_a[loc_a, STEP3_INDEX_ZERO] * dy_m, rc_a[loc_a, STEP3_INDEX_ONE] * dx_m]))
            tree_b = KDTree(np.column_stack([rc_b[loc_b, STEP3_INDEX_ZERO] * dy_m, rc_b[loc_b, STEP3_INDEX_ONE] * dx_m]))

            for i_sub, nbrs in enumerate(tree_a.query_ball_tree(tree_b, r=STEP3_EDGE_RADIUS_M)):
                u_idx = int(start_a + loc_a[i_sub])
                u_rc = nodes_global_rc[u_idx]
                for j_sub in nbrs:
                    v_idx = int(start_b + loc_b[j_sub])
                    v_rc = nodes_global_rc[v_idx]
                    pts = np.rint(np.linspace(u_rc, v_rc, STEP3_LOS_INTERP_POINTS)).astype(int)
                    if np.all(pass_core_global[pts[:, STEP3_INDEX_ZERO], pts[:, STEP3_INDEX_ONE]]):
                        bridge_edges.append((u_idx, v_idx))
                        bridge_etas.append(float(np.mean(eta_core_global[pts[:, STEP3_INDEX_ZERO], pts[:, STEP3_INDEX_ONE]])))

    if bridge_edges:
        edges_uv_idx = np.vstack([intra_edges_uv, np.array(bridge_edges, dtype=np.int32)])
        edges_eta_bar = np.concatenate([intra_eta_bar, np.array(bridge_etas, dtype=np.float32)])
    else:
        edges_uv_idx = intra_edges_uv
        edges_eta_bar = intra_eta_bar

    nodes_z_m = z_core_global[
        nodes_global_rc[:, STEP3_INDEX_ZERO],
        nodes_global_rc[:, STEP3_INDEX_ONE],
    ].astype(np.float32)

    arrays_out = {
        **arrays,
        "nodes_global_rc": nodes_global_rc,
        "nodes_z_m": nodes_z_m,
        "edges_uv_idx": edges_uv_idx,
        "edges_eta_bar": edges_eta_bar,
    }

    meta_out = {
        **meta,
        "step": 3,
        "active_chunks": [list(c) for c in active_chunks],
        "total_nodes": int(nodes_global_rc.shape[STEP3_INDEX_ZERO]),
        "intra_chunk_edges": int(intra_edges_uv.shape[STEP3_INDEX_ZERO]),
        "seam_bridge_edges": len(bridge_edges),
        "total_undirected_edges": int(edges_uv_idx.shape[STEP3_INDEX_ZERO]),
        "pinned_waypoints": pinned_waypoints,
    }
    return arrays_out, meta_out


if __name__ == "__main__":
    arrays_res, meta_res = run_step3_pipeline()
    np.savez_compressed(
        STEP3_OUTPUT_NPZ_PATH,
        **arrays_res,
        metadata_json=np.array(json.dumps(meta_res)),
    )
    print("=" * 70)
    print(f"[STEP 3 COMPLETE] Mode: {meta_res['step3_input_mode']}")
    print(f"Active Chunks Stitched : {meta_res['active_chunks']}")
    print(f"Total Graph Nodes      : {meta_res['total_nodes']} (Pinned WPs: {list(meta_res['pinned_waypoints'].keys())})")
    print(f"Validated Edges (LOS)  : {meta_res['total_undirected_edges']} ({meta_res['intra_chunk_edges']} intra + {meta_res['seam_bridge_edges']} seam bridges)")
    print(f"Saved Output for Step4 : {STEP3_OUTPUT_NPZ_PATH}")
    print("=" * 70)