#!/usr/bin/env python3
"""
step5_astar_and_pnr.py
======================
Step 5: Multi-Leg Payload-Aware A* Pathfinding with Admissible 3D Euclidean O2
Heuristic and Single-Pass Reversed-Graph Dijkstra Point-of-No-Return (PNR) Envelope.
Reads all constants from .env and outputs step5_output.npz for Step 6.
"""

import heapq
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


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

# --- Shared Grid, Math & Leg Constants from .env ---
GRID_CHUNKS_SIDE                = int(os.environ["GRID_CHUNKS_SIDE"])
CHUNK_SIZE_PX                   = int(os.environ["CHUNK_SIZE_PX"])
GLOBAL_HALO_PX                  = int(os.environ["GLOBAL_HALO_PX"])
CENTER_LON_DEG                  = float(os.environ["CENTER_LON_DEG"])
CENTER_LAT_DEG                  = float(os.environ["CENTER_LAT_DEG"])
MAP_PIXEL_SIZE_X_M              = float(os.environ["MAP_PIXEL_SIZE_X_M"])
MAP_PIXEL_SIZE_Y_M              = float(os.environ["MAP_PIXEL_SIZE_Y_M"])
DUMMY_BASE_ALT_M                = float(os.environ["DUMMY_BASE_ALT_M"])
DUMMY_REGIONAL_SLOPE_M          = float(os.environ["DUMMY_REGIONAL_SLOPE_M"])
ZERO_FLOAT                      = float(os.environ["ZERO_FLOAT"])
ONE_FLOAT                       = float(os.environ["ONE_FLOAT"])
TWO_FLOAT                       = float(os.environ["TWO_FLOAT"])
STEP3_INDEX_ZERO                = int(os.environ["STEP3_INDEX_ZERO"])
STEP3_INDEX_ONE                 = int(os.environ["STEP3_INDEX_ONE"])
STEP4_NUM_MISSION_LEGS          = int(os.environ["STEP4_NUM_MISSION_LEGS"])
STEP4_DUMMY_GRID_ROWS           = int(os.environ["STEP4_DUMMY_GRID_ROWS"])
STEP4_DUMMY_GRID_COLS           = int(os.environ["STEP4_DUMMY_GRID_COLS"])
STEP4_DUMMY_ROW_STEP_PX         = int(os.environ["STEP4_DUMMY_ROW_STEP_PX"])
STEP4_DUMMY_COL_STEP_PX         = int(os.environ["STEP4_DUMMY_COL_STEP_PX"])
STEP4_DUMMY_ROW_OFFSET_PX       = int(os.environ["STEP4_DUMMY_ROW_OFFSET_PX"])
STEP4_DUMMY_COL_OFFSET_PX       = int(os.environ["STEP4_DUMMY_COL_OFFSET_PX"])
STEP4_DUMMY_Z_WAVE_AMP_M        = float(os.environ["STEP4_DUMMY_Z_WAVE_AMP_M"])

# --- Step 5 File Paths ---
STEP5_INPUT_NPZ_PATH            = os.environ["STEP5_INPUT_NPZ_PATH"]
STEP5_OUTPUT_NPZ_PATH           = os.environ["STEP5_OUTPUT_NPZ_PATH"]

# --- Step 5 Life-Support & PNR Constants ---
STEP5_O2_TANK_CAPACITY_L        = float(os.environ["STEP5_O2_TANK_CAPACITY_L"])
STEP5_PNR_SAFETY_FACTOR         = float(os.environ["STEP5_PNR_SAFETY_FACTOR"])
STEP5_PNR_REFERENCE_LEG_INDEX   = int(os.environ["STEP5_PNR_REFERENCE_LEG_INDEX"])
STEP5_HAB_WAYPOINT_ID           = os.environ["STEP5_HAB_WAYPOINT_ID"]
STEP5_INF_COST                  = float(os.environ["STEP5_INF_COST"])
STEP5_PARENT_NONE               = int(os.environ["STEP5_PARENT_NONE"])
STEP5_ROUND_DECIMALS            = int(os.environ["STEP5_ROUND_DECIMALS"])

# --- Step 5 Dummy Input Generator Constants ---
STEP5_DUMMY_RNG_SEED            = int(os.environ["STEP5_DUMMY_RNG_SEED"])
STEP5_DUMMY_O2_BASE_L           = float(os.environ["STEP5_DUMMY_O2_BASE_L"])
STEP5_DUMMY_O2_LOAD_STEP_L      = float(os.environ["STEP5_DUMMY_O2_LOAD_STEP_L"])
STEP5_DUMMY_O2_NOISE_L          = float(os.environ["STEP5_DUMMY_O2_NOISE_L"])
STEP5_DUMMY_GRADE_PENALTY_SCALE = float(os.environ["STEP5_DUMMY_GRADE_PENALTY_SCALE"])
STEP5_DUMMY_P_FLOOR_BASE_W      = float(os.environ["STEP5_DUMMY_P_FLOOR_BASE_W"])
STEP5_DUMMY_P_FLOOR_STEP_W      = float(os.environ["STEP5_DUMMY_P_FLOOR_STEP_W"])
STEP5_DUMMY_GAMMA_O2            = float(os.environ["STEP5_DUMMY_GAMMA_O2"])


# =========================================================================
# 1. INPUT LOADERS: (A) REAL INPUT [ACTIVE] vs. (B) DUMMY INPUT [COMMENTED]
# =========================================================================

# -------------------------------------------------------------------------
# (A) REAL INPUT LOADER (Active: reads step4_output.npz)
# -------------------------------------------------------------------------
def load_step5_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads the precomputed Step 4 bioenergetic directed graph from step4_output.npz."""
    bundle = np.load(STEP5_INPUT_NPZ_PATH, allow_pickle=True)
    meta = json.loads(str(bundle["metadata_json"]))
    arrays = {
        k: bundle[k]
        for k in bundle.files
        if k != "metadata_json"
    }
    meta["step5_input_mode"] = f"REAL_STEP4_NPZ ({STEP5_INPUT_NPZ_PATH})"
    return arrays, meta


# -------------------------------------------------------------------------
# (B) DUMMY INPUT GENERATOR (Commented out; use only for standalone testing)
# -------------------------------------------------------------------------
# def load_step5_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
#     """Generates a connected synthetic directed 3D graph and O2 cost matrix."""
#     G_core = GRID_CHUNKS_SIDE * CHUNK_SIZE_PX
#     rng = np.random.default_rng(STEP5_DUMMY_RNG_SEED)
#
#     n_rows = STEP4_DUMMY_GRID_ROWS
#     n_cols = STEP4_DUMMY_GRID_COLS
#     grid_r, grid_c = np.meshgrid(
#         STEP4_DUMMY_ROW_OFFSET_PX + np.arange(n_rows) * STEP4_DUMMY_ROW_STEP_PX,
#         STEP4_DUMMY_COL_OFFSET_PX + np.arange(n_cols) * STEP4_DUMMY_COL_STEP_PX,
#         indexing="ij",
#     )
#     nodes_global_rc = np.column_stack([grid_r.ravel(), grid_c.ravel()]).astype(np.int32)
#
#     r_norm = nodes_global_rc[:, STEP3_INDEX_ZERO] / G_core
#     c_norm = nodes_global_rc[:, STEP3_INDEX_ONE] / G_core
#     nodes_z_m = (
#         DUMMY_BASE_ALT_M
#         + DUMMY_REGIONAL_SLOPE_M * r_norm
#         + STEP4_DUMMY_Z_WAVE_AMP_M * np.sin(TWO_FLOAT * math.pi * c_norm)
#     ).astype(np.float32)
#
#     undir_list: List[Tuple[int, int]] = []
#     for r_i in range(n_rows):
#         for c_i in range(n_cols):
#             u_idx = r_i * n_cols + c_i
#             for dr, dc in (
#                 (STEP3_INDEX_ZERO, STEP3_INDEX_ONE),
#                 (STEP3_INDEX_ONE, STEP3_INDEX_ZERO),
#                 (STEP3_INDEX_ONE, STEP3_INDEX_ONE),
#                 (STEP3_INDEX_ONE, -STEP3_INDEX_ONE),
#             ):
#                 nr, nc = r_i + dr, c_i + dc
#                 if STEP3_INDEX_ZERO <= nr < n_rows and STEP3_INDEX_ZERO <= nc < n_cols:
#                     undir_list.append((u_idx, nr * n_cols + nc))
#
#     undir_uv = np.array(undir_list, dtype=np.int32)
#     u_idx = undir_uv[:, STEP3_INDEX_ZERO]
#     v_idx = undir_uv[:, STEP3_INDEX_ONE]
#     dir_edges_uv = np.column_stack([
#         np.concatenate([u_idx, v_idx]),
#         np.concatenate([v_idx, u_idx]),
#     ]).astype(np.int32)
#
#     dx_m = MAP_PIXEL_SIZE_X_M * math.cos(math.radians(CENTER_LAT_DEG))
#     dy_m = MAP_PIXEL_SIZE_Y_M
#
#     du = dir_edges_uv[:, STEP3_INDEX_ZERO]
#     dv = dir_edges_uv[:, STEP3_INDEX_ONE]
#     dr_m = (nodes_global_rc[dv, STEP3_INDEX_ZERO] - nodes_global_rc[du, STEP3_INDEX_ZERO]) * dy_m
#     dc_m = (nodes_global_rc[dv, STEP3_INDEX_ONE] - nodes_global_rc[du, STEP3_INDEX_ONE]) * dx_m
#     dir_dz_m = (nodes_z_m[dv] - nodes_z_m[du]).astype(np.float64)
#     dir_d_3d_m = np.hypot(np.hypot(dr_m, dc_m), dir_dz_m).astype(np.float32)
#
#     num_dir = int(dir_edges_uv.shape[STEP3_INDEX_ZERO])
#     dir_o2_cost_matrix = np.zeros((STEP4_NUM_MISSION_LEGS, num_dir), dtype=np.float32)
#     mission_legs_bioenergetics: List[Dict[str, Any]] = []
#
#     noise_vec = rng.uniform(ZERO_FLOAT, STEP5_DUMMY_O2_NOISE_L, size=num_dir)
#     uphill_penalty = np.maximum(ZERO_FLOAT, dir_dz_m) * STEP5_DUMMY_GRADE_PENALTY_SCALE
#
#     for idx in range(STEP3_INDEX_ONE, STEP4_NUM_MISSION_LEGS + STEP3_INDEX_ONE):
#         leg_i = idx - STEP3_INDEX_ONE
#         base_o2 = STEP5_DUMMY_O2_BASE_L + leg_i * STEP5_DUMMY_O2_LOAD_STEP_L
#         dir_o2_cost_matrix[leg_i, :] = (base_o2 + uphill_penalty + noise_vec).astype(np.float32)
#         mission_legs_bioenergetics.append({
#             "leg_index": leg_i,
#             "leg_id": os.environ[f"STEP4_LEG_{idx}_ID"],
#             "start_wp": os.environ[f"STEP4_LEG_{idx}_START_WP"],
#             "goal_wp": os.environ[f"STEP4_LEG_{idx}_GOAL_WP"],
#             "sample_mass_kg": float(os.environ[f"STEP4_LEG_{idx}_SAMPLE_KG"]),
#             "label": os.environ[f"STEP4_LEG_{idx}_LABEL"],
#             "P_floor_W": STEP5_DUMMY_P_FLOOR_BASE_W + leg_i * STEP5_DUMMY_P_FLOOR_STEP_W,
#             "gamma_O2_L_per_Jm": STEP5_DUMMY_GAMMA_O2,
#         })
#
#     last_row = n_rows - STEP3_INDEX_ONE
#     last_col = n_cols - STEP3_INDEX_ONE
#     mid_row = n_rows // int(TWO_FLOAT)
#     mid_col = n_cols // int(TWO_FLOAT)
#
#     pinned_waypoints = {
#         os.environ["WP_1_ID"]: nodes_global_rc[STEP3_INDEX_ZERO].tolist(),
#         os.environ["WP_2_ID"]: nodes_global_rc[mid_row * n_cols + mid_col // int(TWO_FLOAT)].tolist(),
#         os.environ["WP_3_ID"]: nodes_global_rc[last_row * n_cols + mid_col].tolist(),
#         os.environ["WP_5_ID"]: nodes_global_rc[mid_row * n_cols + last_col].tolist(),
#     }
#
#     arrays = {
#         "nodes_global_rc": nodes_global_rc,
#         "nodes_z_m": nodes_z_m,
#         "dir_edges_uv": dir_edges_uv,
#         "dir_d_3d_m": dir_d_3d_m,
#         "dir_o2_cost_matrix": dir_o2_cost_matrix,
#     }
#     meta = {
#         "step": 4,
#         "step5_input_mode": "DUMMY_SYNTHETIC_STEP4_INPUT",
#         "grid_chunks_side": GRID_CHUNKS_SIDE,
#         "chunk_size_L": CHUNK_SIZE_PX,
#         "halo_h": GLOBAL_HALO_PX,
#         "global_core_size": G_core,
#         "center_lon": CENTER_LON_DEG,
#         "center_lat": CENTER_LAT_DEG,
#         "dx_ground_m": round(dx_m, STEP5_ROUND_DECIMALS),
#         "dy_ground_m": round(dy_m, STEP5_ROUND_DECIMALS),
#         "total_nodes": int(nodes_global_rc.shape[STEP3_INDEX_ZERO]),
#         "total_directed_edges": num_dir,
#         "pinned_waypoints": pinned_waypoints,
#         "mission_legs_bioenergetics": mission_legs_bioenergetics,
#     }
#     return arrays, meta


# =========================================================================
# 2. STEP 5 CORE FUNCTIONS: ADMISSIBLE A* & REVERSED DIJKSTRA PNR ENVELOPE
# =========================================================================

def resolve_waypoint_node_index(nodes_rc: np.ndarray, target_rc: List[int]) -> int:
    """Finds the exact or nearest graph node index matching a waypoint's [r, c]."""
    dr = nodes_rc[:, STEP3_INDEX_ZERO] - target_rc[STEP3_INDEX_ZERO]
    dc = nodes_rc[:, STEP3_INDEX_ONE] - target_rc[STEP3_INDEX_ONE]
    return int(np.argmin(dr * dr + dc * dc))


def run_astar_single_leg(
    start_idx: int,
    goal_idx: int,
    adj_list: List[List[Tuple[int, float, float]]],
    nodes_rc: np.ndarray,
    nodes_z: np.ndarray,
    dx_m: float,
    dy_m: float,
    gamma_o2: float,
    p_floor_w: float,
) -> Tuple[List[int], float, float, int]:
    """
    5.1: Finds the minimum-O2 path from start_idx to goal_idx using A* with an
    admissible 3D Euclidean lower-bound O2 heuristic.
    """
    goal_r = float(nodes_rc[goal_idx, STEP3_INDEX_ZERO])
    goal_c = float(nodes_rc[goal_idx, STEP3_INDEX_ONE])
    goal_z = float(nodes_z[goal_idx])
    h_scale = gamma_o2 * p_floor_w

    def heuristic(u_idx: int) -> float:
        dr_m = (float(nodes_rc[u_idx, STEP3_INDEX_ZERO]) - goal_r) * dy_m
        dc_m = (float(nodes_rc[u_idx, STEP3_INDEX_ONE]) - goal_c) * dx_m
        dz_m = float(nodes_z[u_idx]) - goal_z
        return h_scale * math.hypot(math.hypot(dr_m, dc_m), dz_m)

    num_nodes = int(nodes_rc.shape[STEP3_INDEX_ZERO])
    g_cost = np.full(num_nodes, STEP5_INF_COST, dtype=np.float64)
    parent = np.full(num_nodes, STEP5_PARENT_NONE, dtype=np.int32)

    g_cost[start_idx] = ZERO_FLOAT
    pq: List[Tuple[float, float, int]] = [(heuristic(start_idx), ZERO_FLOAT, start_idx)]

    while pq:
        _, g_u, u = heapq.heappop(pq)
        if u == goal_idx:
            break
        if g_u > g_cost[u]:
            continue
        for v, o2_uv, _ in adj_list[u]:
            new_g = g_u + o2_uv
            if new_g < g_cost[v]:
                g_cost[v] = new_g
                parent[v] = u
                heapq.heappush(pq, (new_g + heuristic(v), new_g, v))

    # Fallback to nearest reachable node if goal_idx lies in a disconnected pocket
    reached_idx = goal_idx
    if g_cost[reached_idx] >= STEP5_INF_COST and reached_idx != start_idx:
        reachable = np.where(g_cost < STEP5_INF_COST)[STEP3_INDEX_ZERO]
        dr = nodes_rc[reachable, STEP3_INDEX_ZERO] - goal_r
        dc = nodes_rc[reachable, STEP3_INDEX_ONE] - goal_c
        reached_idx = int(reachable[int(np.argmin(dr * dr + dc * dc))])

    path: List[int] = []
    curr = reached_idx
    while curr != STEP5_PARENT_NONE:
        path.append(int(curr))
        curr = int(parent[curr])
    path.reverse()

    # Sum 3D distance along the recovered path
    dist_3d_m = ZERO_FLOAT
    for i in range(len(path) - STEP3_INDEX_ONE):
        u_node, v_node = path[i], path[i + STEP3_INDEX_ONE]
        for nbr, _, d3d in adj_list[u_node]:
            if nbr == v_node:
                dist_3d_m += d3d
                break

    return path, float(g_cost[reached_idx]), dist_3d_m, reached_idx


def compute_reversed_dijkstra_pnr(
    hab_idx: int,
    rev_adj_list: List[List[Tuple[int, float]]],
    num_nodes: int,
) -> np.ndarray:
    """
    5.2: Runs single-pass Dijkstra on the reversed directed graph from hab_idx
    to obtain exact minimum return O2 cost R_O2(u -> Hab) for every node u.
    """
    ret_o2 = np.full(num_nodes, STEP5_INF_COST, dtype=np.float64)
    ret_o2[hab_idx] = ZERO_FLOAT
    pq: List[Tuple[float, int]] = [(ZERO_FLOAT, hab_idx)]

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > ret_o2[u]:
            continue
        for prev_node, o2_prev_to_u in rev_adj_list[u]:
            new_d = d_u + o2_prev_to_u
            if new_d < ret_o2[prev_node]:
                ret_o2[prev_node] = new_d
                heapq.heappush(pq, (new_d, prev_node))

    return ret_o2


def run_step5_pipeline() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Executes Step 5 multi-leg A* routing and reversed-Dijkstra PNR analysis."""
    arrays, meta = load_step5_inputs()

    nodes_rc = arrays["nodes_global_rc"].astype(np.int32)
    nodes_z = arrays["nodes_z_m"].astype(np.float64)
    dir_uv = arrays["dir_edges_uv"].astype(np.int32)
    dir_d3d = arrays["dir_d_3d_m"].astype(np.float64)
    o2_matrix = arrays["dir_o2_cost_matrix"].astype(np.float64)

    dx_m = float(meta["dx_ground_m"])
    dy_m = float(meta["dy_ground_m"])
    num_nodes = int(nodes_rc.shape[STEP3_INDEX_ZERO])
    num_dir_edges = int(dir_uv.shape[STEP3_INDEX_ZERO])

    pinned_wps: Dict[str, List[int]] = meta["pinned_waypoints"]
    wp_node_idx = {
        wp_id: resolve_waypoint_node_index(nodes_rc, rc)
        for wp_id, rc in pinned_wps.items()
    }
    hab_idx = wp_node_idx[STEP5_HAB_WAYPOINT_ID]

    # --- 5.1: Multi-Leg Payload-Aware A* Traverse ---
    o2_rem_L = STEP5_O2_TANK_CAPACITY_L
    full_route_indices: List[int] = []
    leg_ids_per_step: List[int] = []
    eva_mission_legs: List[Dict[str, Any]] = []
    curr_start_idx = hab_idx

    for leg_spec in meta["mission_legs_bioenergetics"]:
        leg_i = int(leg_spec["leg_index"])
        goal_wp_id = str(leg_spec["goal_wp"])
        target_goal_idx = wp_node_idx[goal_wp_id]

        # Build forward adjacency list for this leg's payload O2 weights
        adj_leg: List[List[Tuple[int, float, float]]] = [[] for _ in range(num_nodes)]
        leg_o2_vec = o2_matrix[leg_i]
        for e_i in range(num_dir_edges):
            u_e = int(dir_uv[e_i, STEP3_INDEX_ZERO])
            v_e = int(dir_uv[e_i, STEP3_INDEX_ONE])
            adj_leg[u_e].append((v_e, float(leg_o2_vec[e_i]), float(dir_d3d[e_i])))

        path, o2_used, d3d_m, reached_idx = run_astar_single_leg(
            start_idx=curr_start_idx,
            goal_idx=target_goal_idx,
            adj_list=adj_leg,
            nodes_rc=nodes_rc,
            nodes_z=nodes_z,
            dx_m=dx_m,
            dy_m=dy_m,
            gamma_o2=float(leg_spec["gamma_O2_L_per_Jm"]),
            p_floor_w=float(leg_spec["P_floor_W"]),
        )
        curr_start_idx = reached_idx
        o2_rem_L -= o2_used

        if not full_route_indices:
            full_route_indices.extend(path)
            leg_ids_per_step.extend([leg_i] * len(path))
        else:
            full_route_indices.extend(path[STEP3_INDEX_ONE:])
            leg_ids_per_step.extend([leg_i] * (len(path) - STEP3_INDEX_ONE))

        eva_mission_legs.append({
            "leg_index": leg_i,
            "leg_id": leg_spec["leg_id"],
            "label": leg_spec["label"],
            "start_id": leg_spec["start_wp"],
            "goal_id": goal_wp_id,
            "sample_mass_kg": float(leg_spec["sample_mass_kg"]),
            "vertex_count": len(path),
            "distance_3d_m": round(d3d_m, STEP5_ROUND_DECIMALS),
            "o2_consumed_L": round(o2_used, STEP5_ROUND_DECIMALS),
            "o2_tank_remaining_L": round(o2_rem_L, STEP5_ROUND_DECIMALS),
        })

    # --- 5.2: Reversed-Graph Dijkstra Point-of-No-Return (PNR) Safety Envelope ---
    rev_adj: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    ref_o2_vec = o2_matrix[STEP5_PNR_REFERENCE_LEG_INDEX]
    for e_i in range(num_dir_edges):
        u_e = int(dir_uv[e_i, STEP3_INDEX_ZERO])
        v_e = int(dir_uv[e_i, STEP3_INDEX_ONE])
        # Reverse directed edge v_e <- u_e stores forward traversal cost u_e -> v_e
        rev_adj[v_e].append((u_e, float(ref_o2_vec[e_i])))

    node_return_o2_L = compute_reversed_dijkstra_pnr(hab_idx, rev_adj, num_nodes)
    pnr_hazard_mask = (STEP5_PNR_SAFETY_FACTOR * node_return_o2_L) > o2_rem_L
    pnr_hazard_indices = np.where(pnr_hazard_mask)[STEP3_INDEX_ZERO].astype(np.int32)

    route_idx_arr = np.array(full_route_indices, dtype=np.int32)
    route_rc_arr = nodes_rc[route_idx_arr]
    route_z_arr = nodes_z[route_idx_arr].astype(np.float32)

    arrays_out = {
        **arrays,
        "full_route_node_indices": route_idx_arr,
        "full_route_global_rc": route_rc_arr,
        "full_route_z_m": route_z_arr,
        "full_route_leg_ids": np.array(leg_ids_per_step, dtype=np.int32),
        "node_return_o2_L": node_return_o2_L.astype(np.float32),
        "pnr_hazard_node_indices": pnr_hazard_indices,
    }

    total_dist_m = sum(leg["distance_3d_m"] for leg in eva_mission_legs)
    total_o2_used = STEP5_O2_TANK_CAPACITY_L - o2_rem_L

    meta_out = {
        **meta,
        "step": 5,
        "eva_mission_legs": eva_mission_legs,
        "total_route_vertices": len(full_route_indices),
        "total_traverse_distance_3d_m": round(total_dist_m, STEP5_ROUND_DECIMALS),
        "o2_tank_capacity_L": STEP5_O2_TANK_CAPACITY_L,
        "o2_total_consumed_L": round(total_o2_used, STEP5_ROUND_DECIMALS),
        "o2_end_of_mission_L": round(o2_rem_L, STEP5_ROUND_DECIMALS),
        "pnr_safety_factor": STEP5_PNR_SAFETY_FACTOR,
        "pnr_flagged_node_count": int(pnr_hazard_indices.size),
    }
    return arrays_out, meta_out


if __name__ == "__main__":
    arrays_res, meta_res = run_step5_pipeline()
    np.savez_compressed(
        STEP5_OUTPUT_NPZ_PATH,
        **arrays_res,
        metadata_json=np.array(json.dumps(meta_res)),
    )
    print("=" * 74)
    print(f"[STEP 5 COMPLETE] Mode: {meta_res['step5_input_mode']}")
    for leg in meta_res["eva_mission_legs"]:
        print(
            f"  [A* OK] {leg['label']} | Dist: {leg['distance_3d_m']} m | "
            f"O2: {leg['o2_consumed_L']} L | Rem: {leg['o2_tank_remaining_L']} L"
        )
    print(f"Total 3D Traverse     : {meta_res['total_traverse_distance_3d_m']} m ({meta_res['total_route_vertices']} vertices)")
    print(f"End-of-Mission O2 Rem : {meta_res['o2_end_of_mission_L']} L / {meta_res['o2_tank_capacity_L']} L")
    print(f"PNR Hazard Envelope   : {meta_res['pnr_flagged_node_count']} nodes flagged (Safety Factor = {meta_res['pnr_safety_factor']}x)")
    print(f"Saved Output for Step6: {STEP5_OUTPUT_NPZ_PATH}")
    print("=" * 74)