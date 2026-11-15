#!/usr/bin/env python3
"""
step4_bioenergetic_graph.py
===========================
Step 4: Directed 3D Spatial Graph Construction, Mars-Adapted Pandolf-Minetti
Metabolic Power Model, and Payload-Dependent O2 Edge Weighting.
Reads all constants from .env and outputs step4_output.npz for Step 5.
"""

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

# --- Shared Grid & Math Constants from .env ---
GRID_CHUNKS_SIDE             = int(os.environ["GRID_CHUNKS_SIDE"])
CHUNK_SIZE_PX                = int(os.environ["CHUNK_SIZE_PX"])
GLOBAL_HALO_PX               = int(os.environ["GLOBAL_HALO_PX"])
CENTER_LON_DEG               = float(os.environ["CENTER_LON_DEG"])
CENTER_LAT_DEG               = float(os.environ["CENTER_LAT_DEG"])
MAP_PIXEL_SIZE_X_M           = float(os.environ["MAP_PIXEL_SIZE_X_M"])
MAP_PIXEL_SIZE_Y_M           = float(os.environ["MAP_PIXEL_SIZE_Y_M"])
DUMMY_BASE_ALT_M             = float(os.environ["DUMMY_BASE_ALT_M"])
DUMMY_REGIONAL_SLOPE_M       = float(os.environ["DUMMY_REGIONAL_SLOPE_M"])
PERCENT_SCALE                = float(os.environ["PERCENT_SCALE"])
ZERO_FLOAT                   = float(os.environ["ZERO_FLOAT"])
ONE_FLOAT                    = float(os.environ["ONE_FLOAT"])
TWO_FLOAT                    = float(os.environ["TWO_FLOAT"])
STEP3_INDEX_ZERO             = int(os.environ["STEP3_INDEX_ZERO"])
STEP3_INDEX_ONE              = int(os.environ["STEP3_INDEX_ONE"])

# --- Step 4 File Paths ---
STEP4_INPUT_NPZ_PATH         = os.environ["STEP4_INPUT_NPZ_PATH"]
STEP4_OUTPUT_NPZ_PATH        = os.environ["STEP4_OUTPUT_NPZ_PATH"]

# --- Step 4 Astronaut & Martian Bioenergetic Constants ---
STEP4_ASTRONAUT_MASS_KG      = float(os.environ["STEP4_ASTRONAUT_MASS_KG"])
STEP4_SUIT_BASE_MASS_KG      = float(os.environ["STEP4_SUIT_BASE_MASS_KG"])
STEP4_WALK_SPEED_MS          = float(os.environ["STEP4_WALK_SPEED_MS"])
STEP4_MARS_GRAVITY_RATIO     = float(os.environ["STEP4_MARS_GRAVITY_RATIO"])
STEP4_PANDOLF_STAND_W_COEFF  = float(os.environ["STEP4_PANDOLF_STAND_W_COEFF"])
STEP4_PANDOLF_STAND_L_COEFF  = float(os.environ["STEP4_PANDOLF_STAND_L_COEFF"])
STEP4_PANDOLF_WALK_COEFF     = float(os.environ["STEP4_PANDOLF_WALK_COEFF"])
STEP4_PANDOLF_GRADE_COEFF    = float(os.environ["STEP4_PANDOLF_GRADE_COEFF"])
STEP4_DOWNHILL_BRAKING_FLOOR = float(os.environ["STEP4_DOWNHILL_BRAKING_FLOOR"])
STEP4_O2_JOULES_PER_LITER    = float(os.environ["STEP4_O2_JOULES_PER_LITER"])
STEP4_HEURISTIC_ETA_MIN      = float(os.environ["STEP4_HEURISTIC_ETA_MIN"])
STEP4_DIST_EPSILON_M         = float(os.environ["STEP4_DIST_EPSILON_M"])
STEP4_ROUND_DECIMALS         = int(os.environ["STEP4_ROUND_DECIMALS"])

# --- Step 4 Mission Leg Payload Schedule ---
STEP4_NUM_MISSION_LEGS       = int(os.environ["STEP4_NUM_MISSION_LEGS"])
MISSION_LEGS_SPEC: List[Dict[str, Any]] = [
    {
        "leg_index": idx - STEP3_INDEX_ONE,
        "leg_id": os.environ[f"STEP4_LEG_{idx}_ID"],
        "start_wp": os.environ[f"STEP4_LEG_{idx}_START_WP"],
        "goal_wp": os.environ[f"STEP4_LEG_{idx}_GOAL_WP"],
        "sample_mass_kg": float(os.environ[f"STEP4_LEG_{idx}_SAMPLE_KG"]),
        "label": os.environ[f"STEP4_LEG_{idx}_LABEL"],
    }
    for idx in range(STEP3_INDEX_ONE, STEP4_NUM_MISSION_LEGS + STEP3_INDEX_ONE)
]

# --- Step 4 Dummy Input Generator Constants ---
STEP4_DUMMY_RNG_SEED         = int(os.environ["STEP4_DUMMY_RNG_SEED"])
STEP4_DUMMY_GRID_ROWS        = int(os.environ["STEP4_DUMMY_GRID_ROWS"])
STEP4_DUMMY_GRID_COLS        = int(os.environ["STEP4_DUMMY_GRID_COLS"])
STEP4_DUMMY_ROW_STEP_PX      = int(os.environ["STEP4_DUMMY_ROW_STEP_PX"])
STEP4_DUMMY_COL_STEP_PX      = int(os.environ["STEP4_DUMMY_COL_STEP_PX"])
STEP4_DUMMY_ROW_OFFSET_PX    = int(os.environ["STEP4_DUMMY_ROW_OFFSET_PX"])
STEP4_DUMMY_COL_OFFSET_PX    = int(os.environ["STEP4_DUMMY_COL_OFFSET_PX"])
STEP4_DUMMY_Z_WAVE_AMP_M     = float(os.environ["STEP4_DUMMY_Z_WAVE_AMP_M"])
STEP4_DUMMY_ETA_MIN          = float(os.environ["STEP4_DUMMY_ETA_MIN"])
STEP4_DUMMY_ETA_MAX          = float(os.environ["STEP4_DUMMY_ETA_MAX"])


# =========================================================================
# 1. INPUT LOADERS: (A) REAL INPUT [ACTIVE] vs. (B) DUMMY INPUT [COMMENTED]
# =========================================================================

# -------------------------------------------------------------------------
# (A) REAL INPUT LOADER (Active: reads step3_output.npz)
# -------------------------------------------------------------------------
def load_step4_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads the precomputed Step 3 graph bundle from step3_output.npz."""
    bundle = np.load(STEP4_INPUT_NPZ_PATH, allow_pickle=True)
    meta = json.loads(str(bundle["metadata_json"]))
    arrays = {
        k: bundle[k]
        for k in bundle.files
        if k != "metadata_json"
    }
    meta["step4_input_mode"] = f"REAL_STEP3_NPZ ({STEP4_INPUT_NPZ_PATH})"
    return arrays, meta


# -------------------------------------------------------------------------
# (B) DUMMY INPUT GENERATOR (Commented out; use only for standalone testing)
# -------------------------------------------------------------------------
# def load_step4_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
#     """Generates a connected synthetic 2D lattice graph across Chunks (1,1) and (1,2)."""
#     G_core = GRID_CHUNKS_SIDE * CHUNK_SIZE_PX
#     rng = np.random.default_rng(STEP4_DUMMY_RNG_SEED)
#
#     n_rows = STEP4_DUMMY_GRID_ROWS
#     n_cols = STEP4_DUMMY_GRID_COLS
#
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
#     # Connect 8-neighborhood lattice edges so every waypoint pair is mutually reachable
#     edges_list: List[Tuple[int, int]] = []
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
#                     edges_list.append((u_idx, nr * n_cols + nc))
#
#     edges_uv_idx = np.array(edges_list, dtype=np.int32)
#     edges_eta_bar = rng.uniform(
#         STEP4_DUMMY_ETA_MIN,
#         STEP4_DUMMY_ETA_MAX,
#         size=(edges_uv_idx.shape[STEP3_INDEX_ZERO],),
#     ).astype(np.float32)
#
#     # Map the 4 mission waypoints onto deterministic lattice nodes
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
#     dx_m = MAP_PIXEL_SIZE_X_M * math.cos(math.radians(CENTER_LAT_DEG))
#     dy_m = MAP_PIXEL_SIZE_Y_M
#
#     arrays = {
#         "nodes_global_rc": nodes_global_rc,
#         "nodes_z_m": nodes_z_m,
#         "edges_uv_idx": edges_uv_idx,
#         "edges_eta_bar": edges_eta_bar,
#     }
#     meta = {
#         "step": 3,
#         "step4_input_mode": "DUMMY_SYNTHETIC_STEP3_INPUT",
#         "grid_chunks_side": GRID_CHUNKS_SIDE,
#         "chunk_size_L": CHUNK_SIZE_PX,
#         "halo_h": GLOBAL_HALO_PX,
#         "global_core_size": G_core,
#         "center_lon": CENTER_LON_DEG,
#         "center_lat": CENTER_LAT_DEG,
#         "dx_ground_m": round(dx_m, STEP4_ROUND_DECIMALS),
#         "dy_ground_m": round(dy_m, STEP4_ROUND_DECIMALS),
#         "total_nodes": int(nodes_global_rc.shape[STEP3_INDEX_ZERO]),
#         "total_undirected_edges": int(edges_uv_idx.shape[STEP3_INDEX_ZERO]),
#         "pinned_waypoints": pinned_waypoints,
#     }
#     return arrays, meta


# =========================================================================
# 2. STEP 4 CORE FUNCTIONS: DIRECTED 3D GRAPH & BIOENERGETIC O2 WEIGHTING
# =========================================================================

def compute_payload_bioenergetic_constants(sample_mass_kg: float) -> Dict[str, float]:
    """
    Computes the Mars-adapted Pandolf-Minetti standing power P_stand, walking
    coefficient alpha_V, grade coefficient beta_V, O2 conversion gamma_O2, and
    admissible A* lower-bound power floor P_floor for a given sample mass.
    """
    W_kg = STEP4_ASTRONAUT_MASS_KG
    L_kg = STEP4_SUIT_BASE_MASS_KG + sample_mass_kg
    V_ms = STEP4_WALK_SPEED_MS
    g_ratio = STEP4_MARS_GRAVITY_RATIO

    load_ratio_sq = (L_kg / W_kg) ** TWO_FLOAT
    P_stand = g_ratio * (
        STEP4_PANDOLF_STAND_W_COEFF * W_kg
        + STEP4_PANDOLF_STAND_L_COEFF * (W_kg + L_kg) * load_ratio_sq
    )
    alpha_V = g_ratio * STEP4_PANDOLF_WALK_COEFF * (W_kg + L_kg) * (V_ms ** TWO_FLOAT)
    beta_V = g_ratio * STEP4_PANDOLF_GRADE_COEFF * (W_kg + L_kg) * V_ms
    gamma_O2 = ONE_FLOAT / (STEP4_O2_JOULES_PER_LITER * V_ms)
    P_floor = P_stand + STEP4_HEURISTIC_ETA_MIN * STEP4_DOWNHILL_BRAKING_FLOOR * alpha_V

    return {
        "total_carried_load_kg": round(L_kg, STEP4_ROUND_DECIMALS),
        "P_stand_W": round(P_stand, STEP4_ROUND_DECIMALS),
        "alpha_V_W": round(alpha_V, STEP4_ROUND_DECIMALS),
        "beta_V_W": round(beta_V, STEP4_ROUND_DECIMALS),
        "gamma_O2_L_per_Jm": round(gamma_O2, int(TWO_FLOAT) * STEP4_ROUND_DECIMALS),
        "P_floor_W": round(P_floor, STEP4_ROUND_DECIMALS),
    }


def run_step4_pipeline() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Builds the directed 3D graph and computes O2 edge costs across all payload tiers."""
    arrays, meta = load_step4_inputs()

    nodes_rc = arrays["nodes_global_rc"].astype(np.int32)
    nodes_z = arrays["nodes_z_m"].astype(np.float64)
    undir_uv = arrays["edges_uv_idx"].astype(np.int32)
    undir_eta = arrays["edges_eta_bar"].astype(np.float64)

    dx_m = float(meta["dx_ground_m"])
    dy_m = float(meta["dy_ground_m"])

    # --- 4.1: Expand Undirected Edges into Forward (u->v) and Reverse (v->u) Directed Edges ---
    u_idx = undir_uv[:, STEP3_INDEX_ZERO]
    v_idx = undir_uv[:, STEP3_INDEX_ONE]

    dir_u = np.concatenate([u_idx, v_idx])
    dir_v = np.concatenate([v_idx, u_idx])
    dir_edges_uv = np.column_stack([dir_u, dir_v]).astype(np.int32)
    dir_eta_bar = np.concatenate([undir_eta, undir_eta]).astype(np.float64)

    # --- 4.2: Compute Directed 3D Geometry (d_2d, dz, d_3d, and asymmetric grade %) ---
    dr_m = (nodes_rc[dir_v, STEP3_INDEX_ZERO] - nodes_rc[dir_u, STEP3_INDEX_ZERO]) * dy_m
    dc_m = (nodes_rc[dir_v, STEP3_INDEX_ONE] - nodes_rc[dir_u, STEP3_INDEX_ONE]) * dx_m
    dir_d_2d_m = np.hypot(dr_m, dc_m)
    dir_dz_m = nodes_z[dir_v] - nodes_z[dir_u]
    dir_d_3d_m = np.hypot(dir_d_2d_m, dir_dz_m)
    dir_grade_pct = PERCENT_SCALE * (dir_dz_m / np.maximum(dir_d_2d_m, STEP4_DIST_EPSILON_M))

    # --- 4.3: Evaluate Pandolf-Minetti Power & O2 Cost (Liters) per Mission Leg Payload ---
    num_dir_edges = int(dir_edges_uv.shape[STEP3_INDEX_ZERO])
    dir_o2_cost_matrix = np.zeros((STEP4_NUM_MISSION_LEGS, num_dir_edges), dtype=np.float32)
    dir_power_matrix = np.zeros((STEP4_NUM_MISSION_LEGS, num_dir_edges), dtype=np.float32)
    leg_bioenergetics_meta: List[Dict[str, Any]] = []

    for leg in MISSION_LEGS_SPEC:
        leg_idx = int(leg["leg_index"])
        b_consts = compute_payload_bioenergetic_constants(float(leg["sample_mass_kg"]))

        P_stand = b_consts["P_stand_W"]
        alpha_V = b_consts["alpha_V_W"]
        beta_V = b_consts["beta_V_W"]
        gamma_O2 = b_consts["gamma_O2_L_per_Jm"]

        # Asymmetric metabolic power P(u->v) != P(v->u) when grade != 0
        locomotion_term = np.maximum(
            STEP4_DOWNHILL_BRAKING_FLOOR * alpha_V,
            alpha_V + beta_V * dir_grade_pct,
        )
        P_edge_W = P_stand + dir_eta_bar * locomotion_term
        o2_edge_L = gamma_O2 * P_edge_W * dir_d_3d_m

        dir_power_matrix[leg_idx, :] = P_edge_W.astype(np.float32)
        dir_o2_cost_matrix[leg_idx, :] = o2_edge_L.astype(np.float32)

        leg_bioenergetics_meta.append({
            **leg,
            **b_consts,
            "mean_edge_power_W": round(float(np.mean(P_edge_W)), STEP4_ROUND_DECIMALS),
            "mean_edge_o2_L": round(float(np.mean(o2_edge_L)), STEP4_ROUND_DECIMALS),
        })

    arrays_out = {
        **arrays,
        "dir_edges_uv": dir_edges_uv,
        "dir_d_2d_m": dir_d_2d_m.astype(np.float32),
        "dir_dz_m": dir_dz_m.astype(np.float32),
        "dir_d_3d_m": dir_d_3d_m.astype(np.float32),
        "dir_grade_pct": dir_grade_pct.astype(np.float32),
        "dir_eta_bar": dir_eta_bar.astype(np.float32),
        "dir_power_matrix": dir_power_matrix,
        "dir_o2_cost_matrix": dir_o2_cost_matrix,
    }

    meta_out = {
        **meta,
        "step": 4,
        "total_directed_edges": num_dir_edges,
        "astronaut_mass_kg": STEP4_ASTRONAUT_MASS_KG,
        "suit_base_mass_kg": STEP4_SUIT_BASE_MASS_KG,
        "walk_speed_ms": STEP4_WALK_SPEED_MS,
        "mars_gravity_ratio": STEP4_MARS_GRAVITY_RATIO,
        "mission_legs_bioenergetics": leg_bioenergetics_meta,
    }
    return arrays_out, meta_out


if __name__ == "__main__":
    arrays_res, meta_res = run_step4_pipeline()
    np.savez_compressed(
        STEP4_OUTPUT_NPZ_PATH,
        **arrays_res,
        metadata_json=np.array(json.dumps(meta_res)),
    )
    print("=" * 74)
    print(f"[STEP 4 COMPLETE] Mode: {meta_res['step4_input_mode']}")
    print(f"Directed 3D Graph     : {meta_res['total_nodes']} nodes | {meta_res['total_directed_edges']} directed edges")
    for leg_info in meta_res["mission_legs_bioenergetics"]:
        print(
            f"  - {leg_info['label']} | Carried Load = {leg_info['total_carried_load_kg']} kg | "
            f"P_stand = {leg_info['P_stand_W']:.1f} W | Mean Edge P = {leg_info['mean_edge_power_W']:.1f} W"
        )
    print(f"Saved Output for Step5: {STEP4_OUTPUT_NPZ_PATH}")
    print("=" * 74)