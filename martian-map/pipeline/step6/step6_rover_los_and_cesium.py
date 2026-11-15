#!/usr/bin/env python3
"""
step6_rover_los_and_cesium.py
=============================
Step 6: 3D Ray-Cast Hab Radio Line-of-Sight (L_LOS), 8-Azimuth Canyon Radiation
Shielding (R_rad), Perseverance Rover Street-View Ground-Truth Calibration,
Binary Viewer Packing (z_u16, tile_u8, slope_u8, 9-chunk explore graph x),
and NASA Mars Trek WMTS Streaming Bundle Export (cesium_mars_mission_bundle.json).
Reads all constants from .env and outputs step6_output.npz + CesiumJS JSON.
"""

import base64
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import sobel
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
GRID_CHUNKS_SIDE                = int(os.environ["GRID_CHUNKS_SIDE"])
CHUNK_SIZE_PX                   = int(os.environ["CHUNK_SIZE_PX"])
GLOBAL_HALO_PX                  = int(os.environ["GLOBAL_HALO_PX"])
MARS_RADIUS_M                   = float(os.environ["MARS_RADIUS_M"])
CENTER_LON_DEG                  = float(os.environ["CENTER_LON_DEG"])
CENTER_LAT_DEG                  = float(os.environ["CENTER_LAT_DEG"])
MAP_PIXEL_SIZE_X_M              = float(os.environ["MAP_PIXEL_SIZE_X_M"])
MAP_PIXEL_SIZE_Y_M              = float(os.environ["MAP_PIXEL_SIZE_Y_M"])
DUMMY_BASE_ALT_M                = float(os.environ["DUMMY_BASE_ALT_M"])
DUMMY_REGIONAL_SLOPE_M          = float(os.environ["DUMMY_REGIONAL_SLOPE_M"])
DUMMY_RIDGE_HEIGHT_M            = float(os.environ["DUMMY_RIDGE_HEIGHT_M"])
DUMMY_RIDGE_CENTER_NORM         = float(os.environ["DUMMY_RIDGE_CENTER_NORM"])
DUMMY_RIDGE_WIDTH_SQ            = float(os.environ["DUMMY_RIDGE_WIDTH_SQ"])
PERCENT_SCALE                   = float(os.environ["PERCENT_SCALE"])
ZERO_FLOAT                      = float(os.environ["ZERO_FLOAT"])
ONE_FLOAT                       = float(os.environ["ONE_FLOAT"])
TWO_FLOAT                       = float(os.environ["TWO_FLOAT"])
DEFAULT_UNIT_ID                 = int(os.environ["DEFAULT_UNIT_ID"])
R_REF_MIN_FLOOR_M               = float(os.environ["R_REF_MIN_FLOOR_M"])
STEP2_SOBEL_AXIS_ROW            = int(os.environ["STEP2_SOBEL_AXIS_ROW"])
STEP2_SOBEL_AXIS_COL            = int(os.environ["STEP2_SOBEL_AXIS_COL"])
STEP3_ACTIVE_CHUNKS_RAW         = os.environ["STEP3_ACTIVE_CHUNKS"]
STEP3_INDEX_ZERO                = int(os.environ["STEP3_INDEX_ZERO"])
STEP3_INDEX_ONE                 = int(os.environ["STEP3_INDEX_ONE"])
STEP3_EDGE_RADIUS_M             = float(os.environ["STEP3_EDGE_RADIUS_M"])
STEP4_SUIT_BASE_MASS_KG         = float(os.environ["STEP4_SUIT_BASE_MASS_KG"])
STEP4_NUM_MISSION_LEGS          = int(os.environ["STEP4_NUM_MISSION_LEGS"])
STEP5_O2_TANK_CAPACITY_L        = float(os.environ["STEP5_O2_TANK_CAPACITY_L"])
STEP5_PNR_SAFETY_FACTOR         = float(os.environ["STEP5_PNR_SAFETY_FACTOR"])
STEP5_HAB_WAYPOINT_ID           = os.environ["STEP5_HAB_WAYPOINT_ID"]

# --- Step 6 File Paths ---
STEP6_INPUT_NPZ_PATH            = os.environ["STEP6_INPUT_NPZ_PATH"]
STEP6_OUTPUT_NPZ_PATH           = os.environ["STEP6_OUTPUT_NPZ_PATH"]
STEP6_CESIUM_JSON_PATH          = os.environ["STEP6_CESIUM_JSON_PATH"]

# --- Step 6 CesiumJS & Geodesy Export Constants ---
STEP6_CESIUM_VERSION            = os.environ["STEP6_CESIUM_VERSION"]
STEP6_SITE_ID                   = os.environ["STEP6_SITE_ID"]
STEP6_MISSION_NAME              = os.environ["STEP6_MISSION_NAME"]
STEP6_ROVER_MISSION_TAG         = os.environ["STEP6_ROVER_MISSION_TAG"]
STEP6_ROUTE_ALT_OFFSET_M        = float(os.environ["STEP6_ROUTE_ALT_OFFSET_M"])
STEP6_MAX_EXPORTED_PNR_NODES    = int(os.environ["STEP6_MAX_EXPORTED_PNR_NODES"])
STEP6_DEG_IN_HALF_CIRCLE        = float(os.environ["STEP6_DEG_IN_HALF_CIRCLE"])
STEP6_FULL_CIRCLE_DEG           = float(os.environ["STEP6_FULL_CIRCLE_DEG"])
STEP6_HALF_DIVISOR              = float(os.environ["STEP6_HALF_DIVISOR"])
STEP6_LONLAT_ROUND_DECIMALS     = int(os.environ["STEP6_LONLAT_ROUND_DECIMALS"])
STEP6_ALT_ROUND_DECIMALS        = int(os.environ["STEP6_ALT_ROUND_DECIMALS"])
STEP6_METRIC_ROUND_DECIMALS     = int(os.environ["STEP6_METRIC_ROUND_DECIMALS"])

# --- Step 6.1: 3D Ray-Cast Hab Radio Line-of-Sight Constants ---
STEP6_HAB_ANTENNA_HEIGHT_M      = float(os.environ["STEP6_HAB_ANTENNA_HEIGHT_M"])
STEP6_ASTRONAUT_HELMET_HEIGHT_M = float(os.environ["STEP6_ASTRONAUT_HELMET_HEIGHT_M"])
STEP6_LOS_NUM_AZIMUTHS          = int(os.environ["STEP6_LOS_NUM_AZIMUTHS"])
STEP6_RAY_START_STEP            = int(os.environ["STEP6_RAY_START_STEP"])

# --- Step 6.2: 8-Azimuth Canyon Radiation Shielding Constants ---
STEP6_RAD_CARDINAL_OFFSET_PX    = int(os.environ["STEP6_RAD_CARDINAL_OFFSET_PX"])
STEP6_RAD_DIAGONAL_OFFSET_PX    = int(os.environ["STEP6_RAD_DIAGONAL_OFFSET_PX"])
STEP6_RAD_NORM_PI_DIVISOR       = float(os.environ["STEP6_RAD_NORM_PI_DIVISOR"])

# --- Step 6.3: Rover Street-View Sobel Calibration Constants ---
STEP6_ROVER_PATCH_SIZE_PX       = int(os.environ["STEP6_ROVER_PATCH_SIZE_PX"])
STEP6_ROVER_PATCH_MEAN_DN       = float(os.environ["STEP6_ROVER_PATCH_MEAN_DN"])
STEP6_ROVER_SOBEL_THRESH        = float(os.environ["STEP6_ROVER_SOBEL_THRESH"])
STEP6_ROVER_ROUGH_BASE_M        = float(os.environ["STEP6_ROVER_ROUGH_BASE_M"])
STEP6_ROVER_ROUGH_SCALE_M       = float(os.environ["STEP6_ROVER_ROUGH_SCALE_M"])
STEP6_ROVER_CAL_RADIUS_M        = float(os.environ["STEP6_ROVER_CAL_RADIUS_M"])

# --- Step 6.5: Viewer Binary Packing & NASA WMTS Streaming Constants ---
STEP6_VIEWER_STRIDE             = int(os.environ.get("STEP6_VIEWER_STRIDE", "4"))
STEP6_UINT16_MAX                = float(os.environ.get("STEP6_UINT16_MAX", "65535.0"))
STEP6_UINT8_MAX                 = float(os.environ.get("STEP6_UINT8_MAX", "255.0"))
STEP6_SLOPE_BYTE_SCALE          = float(os.environ.get("STEP6_SLOPE_BYTE_SCALE", "8.0"))
STEP6_ETA_QUANT_SCALE           = float(os.environ.get("STEP6_ETA_QUANT_SCALE", "1000.0"))
STEP6_ROCK_RREF_FRAC            = float(os.environ.get("STEP6_ROCK_RREF_FRAC", "0.5"))
STEP6_LOS_THRESHOLD             = float(os.environ.get("STEP6_LOS_THRESHOLD", "0.5"))
STEP6_BIT_SHIFT_ROCK            = int(os.environ.get("STEP6_BIT_SHIFT_ROCK", "3"))
STEP6_BIT_SHIFT_LOS             = int(os.environ.get("STEP6_BIT_SHIFT_LOS", "6"))
STEP6_BIT_SHIFT_PASS            = int(os.environ.get("STEP6_BIT_SHIFT_PASS", "7"))
STEP6_UNREACHABLE_SENTINEL      = float(os.environ.get("STEP6_UNREACHABLE_SENTINEL", "-1.0"))
STEP6_EXPLORE_SAMPLES_PER_CHUNK = int(os.environ.get("STEP6_EXPLORE_SAMPLES_PER_CHUNK", "220"))
STEP6_TERRAIN_SOURCE_TAG        = os.environ.get("STEP6_TERRAIN_SOURCE_TAG", "HiRISE-1m-DTM-Jezero")

WMTS_MARS_GLOBAL_BASEELO = os.environ.get(
    "WMTS_MARS_GLOBAL_BASEELO",
    "https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{TileMatrix}/{TileRow}/{TileCol}.jpg",
)
WMTS_THEMIS_NIGHT_IR = os.environ.get(
    "WMTS_THEMIS_NIGHT_IR",
    "https://trek.nasa.gov/tiles/Mars/EQ/THEMIS_NightIR_ControlledMosaics_100m_v2_oct2018/1.0.0/default/default028mm/{TileMatrix}/{TileRow}/{TileCol}.png",
)
WMTS_CTX_BETA01 = os.environ.get(
    "WMTS_CTX_BETA01",
    "https://trek.nasa.gov/tiles/Mars/EQ/CTX_beta01_v2/1.0.0/default/default028mm/{TileMatrix}/{TileRow}/{TileCol}.jpg",
)
WMTS_JEZERO_HIRISE_ORTHO = os.environ.get(
    "WMTS_JEZERO_HIRISE_ORTHO",
    "https://trek.nasa.gov/tiles/Mars/EQ/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_blend40/1.0.0/default/default028mm/{TileMatrix}/{TileRow}/{TileCol}.png",
)

USGS_UNIT_TABLE_META: Dict[str, Dict[str, Any]] = {
    "1": {"code": os.environ["UNIT_1_CODE"], "name": "Crater Floor Undivided (Maaz)",    "eta_geo": float(os.environ["UNIT_1_ETA_GEO"]), "color": "#34d399"},
    "2": {"code": os.environ["UNIT_2_CODE"], "name": "Crater Floor Fractured (Seitah)",  "eta_geo": float(os.environ["UNIT_2_ETA_GEO"]), "color": "#4ade80"},
    "3": {"code": os.environ["UNIT_3_CODE"], "name": "Delta Plain Deposits",             "eta_geo": float(os.environ["UNIT_3_ETA_GEO"]), "color": "#93c5fd"},
    "4": {"code": os.environ["UNIT_4_CODE"], "name": "Delta Front Stratified Fan",       "eta_geo": float(os.environ["UNIT_4_ETA_GEO"]), "color": "#60a5fa"},
    "5": {"code": os.environ["UNIT_5_CODE"], "name": "Layered Margin Carbonate Unit",    "eta_geo": float(os.environ["UNIT_5_ETA_GEO"]), "color": "#a78bfa"},
    "6": {"code": os.environ["UNIT_6_CODE"], "name": "Eolian Bedforms / Sand Ripples",   "eta_geo": float(os.environ["UNIT_6_ETA_GEO"]), "color": "#facc15"},
    "7": {"code": os.environ["UNIT_7_CODE"], "name": "Surficial Talus / Boulder Aprons", "eta_geo": float(os.environ["UNIT_7_ETA_GEO"]), "color": "#f87171"},
}

ROVER_EDGE_SCALE_BY_UNIT: Dict[str, float] = {
    "Ncu":  float(os.environ["STEP6_EDGE_SCALE_NCU"]),
    "Ncf":  float(os.environ["STEP6_EDGE_SCALE_NCF"]),
    "NHdp": float(os.environ["STEP6_EDGE_SCALE_NHDP"]),
    "NHdf": float(os.environ["STEP6_EDGE_SCALE_NHDF"]),
    "NHl":  float(os.environ["STEP6_EDGE_SCALE_NHL"]),
    "Aeb":  float(os.environ["STEP6_EDGE_SCALE_AEB"]),
    "Asu":  float(os.environ["STEP6_EDGE_SCALE_ASU"]),
}

# --- Step 6.4: Rover Waypoints Catalog Definition from .env ---
STEP6_NUM_ROVER_WAYPOINTS = int(os.environ["STEP6_NUM_ROVER_WAYPOINTS"])
ROVER_WAYPOINT_SPECS: List[Dict[str, Any]] = [
    {
        "id": os.environ[f"WP_{idx}_ID"],
        "sol": int(os.environ[f"WP_{idx}_SOL"]),
        "title": os.environ[f"WP_{idx}_TITLE"],
        "rel_r": float(os.environ[f"WP_{idx}_REL_R"]),
        "rel_c": float(os.environ[f"WP_{idx}_REL_C"]),
        "camera": os.environ[f"WP_{idx}_CAMERA"],
        "usgs_code": os.environ[f"WP_{idx}_USGS_CODE"],
        "heading_deg": float(os.environ[f"WP_{idx}_HEADING_DEG"]),
        "fov_deg": float(os.environ[f"WP_{idx}_FOV_DEG"]),
        "img_url": os.environ[f"WP_{idx}_IMG_URL"],
        "Backup_Wikimedia_URL": os.environ[f"WP_{idx}_BACKUP_URL"],
        "caption": os.environ[f"WP_{idx}_CAPTION"],
    }
    for idx in range(STEP3_INDEX_ONE, STEP6_NUM_ROVER_WAYPOINTS + STEP3_INDEX_ONE)
]

# --- Step 6 Dummy Input Generator Constants ---
STEP6_DUMMY_RNG_SEED            = int(os.environ["STEP6_DUMMY_RNG_SEED"])
STEP6_DUMMY_ROUTE_POINTS        = int(os.environ["STEP6_DUMMY_ROUTE_POINTS"])
STEP6_DUMMY_PNR_COUNT           = int(os.environ["STEP6_DUMMY_PNR_COUNT"])
STEP6_DUMMY_RET_O2_MIN_L        = float(os.environ["STEP6_DUMMY_RET_O2_MIN_L"])
STEP6_DUMMY_RET_O2_MAX_L        = float(os.environ["STEP6_DUMMY_RET_O2_MAX_L"])
STEP6_DUMMY_O2_END_L            = float(os.environ["STEP6_DUMMY_O2_END_L"])
STEP6_DUMMY_LEG_DIST_M          = float(os.environ["STEP6_DUMMY_LEG_DIST_M"])
STEP6_DUMMY_LEG_O2_L            = float(os.environ["STEP6_DUMMY_LEG_O2_L"])
STEP6_DUMMY_R_BASE_M            = float(os.environ["STEP6_DUMMY_R_BASE_M"])


def _b64(arr: np.ndarray) -> str:
    """Encodes a contiguous NumPy array as an ASCII base64 string for viewer binary unpacking."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def parse_active_chunks(raw_str: str) -> List[List[int]]:
    """Parses semicolon-separated chunk pairs like '1,1;1,2' into [[1, 1], [1, 2]]."""
    out: List[List[int]] = []
    for item in raw_str.split(";"):
        parts = item.strip().split(",")
        out.append([int(parts[STEP3_INDEX_ZERO]), int(parts[STEP3_INDEX_ONE])])
    return out


# =========================================================================
# 1. INPUT LOADERS: (A) REAL INPUT [ACTIVE] vs. (B) DUMMY INPUT [COMMENTED]
# =========================================================================

# -------------------------------------------------------------------------
# (A) REAL INPUT LOADER (Active — reads step5_output.npz from Step 5)
# -------------------------------------------------------------------------
def load_step6_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads the precomputed Step 5 routing & PNR bundle from step5_output.npz."""
    bundle = np.load(STEP6_INPUT_NPZ_PATH, allow_pickle=True)
    meta = json.loads(str(bundle["metadata_json"]))
    arrays = {
        k: bundle[k]
        for k in bundle.files
        if k != "metadata_json"
    }
    meta["step6_input_mode"] = f"REAL_STEP5_NPZ ({STEP6_INPUT_NPZ_PATH})"
    return arrays, meta


# -------------------------------------------------------------------------
# (B) DUMMY INPUT GENERATOR (Commented out)
# -------------------------------------------------------------------------
# def load_step6_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
#     """Generates synthetic Step 5 terrain, route, and PNR arrays using .env constants."""
#     G_core = GRID_CHUNKS_SIDE * CHUNK_SIZE_PX
#     G_pad = G_core + int(TWO_FLOAT) * GLOBAL_HALO_PX
#     rng = np.random.default_rng(STEP6_DUMMY_RNG_SEED)
#
#     r_idx, c_idx = np.indices((G_pad, G_pad), dtype=np.float64)
#     r_norm = (r_idx - GLOBAL_HALO_PX) / G_core
#     c_norm = (c_idx - GLOBAL_HALO_PX) / G_core
#
#     scarp = DUMMY_RIDGE_HEIGHT_M * np.exp(
#         -((c_norm - DUMMY_RIDGE_CENTER_NORM) ** TWO_FLOAT) / DUMMY_RIDGE_WIDTH_SQ
#     )
#     z_pad = (DUMMY_BASE_ALT_M + DUMMY_REGIONAL_SLOPE_M * r_norm + scarp).astype(np.float32)
#     R_pad = np.full((G_pad, G_pad), STEP6_DUMMY_R_BASE_M, dtype=np.float32)
#     usgs_id_pad = np.where(c_norm > DUMMY_RIDGE_CENTER_NORM, 4, DEFAULT_UNIT_ID).astype(np.int16)
#     pass_pad = np.ones((G_pad, G_pad), dtype=bool)
#     theta_pad = np.full((G_pad, G_pad), 4.5, dtype=np.float32)
#     eta_pad = np.full((G_pad, G_pad), 1.35, dtype=np.float32)
#
#     n_pts = STEP6_DUMMY_ROUTE_POINTS
#     half_pts = n_pts // int(TWO_FLOAT)
#     hab_r = int(round(float(os.environ["WP_1_REL_R"]) * G_core))
#     hab_c = int(round(float(os.environ["WP_1_REL_C"]) * G_core))
#     east_r = int(round(float(os.environ["WP_5_REL_R"]) * G_core))
#     east_c = int(round(float(os.environ["WP_5_REL_C"]) * G_core))
#
#     out_r = np.rint(np.linspace(hab_r, east_r, half_pts)).astype(np.int32)
#     out_c = np.rint(np.linspace(hab_c, east_c, half_pts)).astype(np.int32)
#     ret_r = np.rint(np.linspace(east_r, hab_r, n_pts - half_pts)).astype(np.int32)
#     ret_c = np.rint(np.linspace(east_c, hab_c, n_pts - half_pts)).astype(np.int32)
#
#     full_route_global_rc = np.column_stack([
#         np.concatenate([out_r, ret_r]),
#         np.concatenate([out_c, ret_c]),
#     ]).astype(np.int32)
#
#     z_core = z_pad[GLOBAL_HALO_PX:-GLOBAL_HALO_PX, GLOBAL_HALO_PX:-GLOBAL_HALO_PX]
#     full_route_z_m = z_core[
#         full_route_global_rc[:, STEP3_INDEX_ZERO],
#         full_route_global_rc[:, STEP3_INDEX_ONE],
#     ].astype(np.float32)
#
#     n_pnr = STEP6_DUMMY_PNR_COUNT
#     pnr_r = rng.integers(CHUNK_SIZE_PX, int(TWO_FLOAT) * CHUNK_SIZE_PX, size=n_pnr)
#     pnr_c = rng.integers(int(TWO_FLOAT) * CHUNK_SIZE_PX, G_core - STEP6_RAY_START_STEP, size=n_pnr)
#     nodes_global_rc = np.vstack([full_route_global_rc, np.column_stack([pnr_r, pnr_c])]).astype(np.int32)
#     nodes_z_m = z_core[
#         nodes_global_rc[:, STEP3_INDEX_ZERO],
#         nodes_global_rc[:, STEP3_INDEX_ONE],
#     ].astype(np.float32)
#
#     pnr_hazard_node_indices = np.arange(n_pts, n_pts + n_pnr, dtype=np.int32)
#     node_return_o2_L = rng.uniform(
#         STEP6_DUMMY_RET_O2_MIN_L,
#         STEP6_DUMMY_RET_O2_MAX_L,
#         size=nodes_global_rc.shape[STEP3_INDEX_ZERO],
#     ).astype(np.float32)
#
#     eva_mission_legs: List[Dict[str, Any]] = []
#     rem_o2 = STEP5_O2_TANK_CAPACITY_L
#     for idx in range(STEP3_INDEX_ONE, STEP4_NUM_MISSION_LEGS + STEP3_INDEX_ONE):
#         rem_o2 -= STEP6_DUMMY_LEG_O2_L
#         eva_mission_legs.append({
#             "leg_index": idx - STEP3_INDEX_ONE,
#             "leg_id": os.environ[f"STEP4_LEG_{idx}_ID"],
#             "label": os.environ[f"STEP4_LEG_{idx}_LABEL"],
#             "start_id": os.environ[f"STEP4_LEG_{idx}_START_WP"],
#             "goal_id": os.environ[f"STEP4_LEG_{idx}_GOAL_WP"],
#             "sample_mass_kg": float(os.environ[f"STEP4_LEG_{idx}_SAMPLE_KG"]),
#             "distance_3d_m": STEP6_DUMMY_LEG_DIST_M,
#             "o2_consumed_L": STEP6_DUMMY_LEG_O2_L,
#             "o2_tank_remaining_L": round(rem_o2, STEP6_ALT_ROUND_DECIMALS),
#         })
#
#     dx_m = MAP_PIXEL_SIZE_X_M * math.cos(math.radians(CENTER_LAT_DEG))
#     dy_m = MAP_PIXEL_SIZE_Y_M
#
#     arrays = {
#         "z_pad": z_pad, "R_pad": R_pad, "usgs_id_pad": usgs_id_pad,
#         "pass_pad": pass_pad, "theta_pad": theta_pad, "eta_pad": eta_pad,
#         "nodes_global_rc": nodes_global_rc, "nodes_z_m": nodes_z_m,
#         "full_route_global_rc": full_route_global_rc, "full_route_z_m": full_route_z_m,
#         "node_return_o2_L": node_return_o2_L, "pnr_hazard_node_indices": pnr_hazard_node_indices,
#     }
#     meta = {
#         "step": 5, "step6_input_mode": "DUMMY_SYNTHETIC_STEP5_INPUT",
#         "site_id": STEP6_SITE_ID, "mission_name": STEP6_MISSION_NAME,
#         "grid_chunks_side": GRID_CHUNKS_SIDE, "chunk_size_L": CHUNK_SIZE_PX,
#         "halo_h": GLOBAL_HALO_PX, "global_core_size": G_core,
#         "center_lon": CENTER_LON_DEG, "center_lat": CENTER_LAT_DEG,
#         "dx_ground_m": round(dx_m, STEP6_METRIC_ROUND_DECIMALS),
#         "dy_ground_m": round(dy_m, STEP6_METRIC_ROUND_DECIMALS),
#         "eva_mission_legs": eva_mission_legs,
#         "o2_tank_capacity_L": STEP5_O2_TANK_CAPACITY_L,
#         "o2_end_of_mission_L": STEP6_DUMMY_O2_END_L,
#         "pnr_safety_factor": STEP5_PNR_SAFETY_FACTOR,
#     }
#     return arrays, meta


# =========================================================================
# 2. STEP 6 CORE FUNCTIONS: LOS, RADIATION, ROVER, VIEWER & CESIUM BUNDLE
# =========================================================================

def pixel_to_lonlat_mars(
    r_global: np.ndarray,
    c_global: np.ndarray,
    G_core: int,
    center_lon_deg: float,
    center_lat_deg: float,
    dx_m: float,
    dy_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts global 3x3 pixel coordinates (r_global, c_global) into Martian [lon, lat] degrees."""
    half_G = G_core / STEP6_HALF_DIVISOR
    east_offset_m = (np.asarray(c_global, dtype=np.float64) - half_G) * dx_m
    north_offset_m = (half_G - np.asarray(r_global, dtype=np.float64)) * dy_m

    deg_per_meter_lat = STEP6_DEG_IN_HALF_CIRCLE / (math.pi * MARS_RADIUS_M)
    deg_per_meter_lon = deg_per_meter_lat / math.cos(math.radians(center_lat_deg))

    lon_deg = center_lon_deg + east_offset_m * deg_per_meter_lon
    lat_deg = center_lat_deg + north_offset_m * deg_per_meter_lat
    return lon_deg, lat_deg


def compute_3d_raycast_los_and_radiation(
    z_core: np.ndarray,
    hab_r: int,
    hab_c: int,
    dx_m: float,
    dy_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """6.1 & 6.2: Computes 3D ray-cast Hab Line-of-Sight L_LOS and 8-azimuth radiation shielding R_rad."""
    H, W = z_core.shape
    z_hab = float(z_core[hab_r, hab_c]) + STEP6_HAB_ANTENNA_HEIGHT_M
    L_LOS_core = np.full((H, W), ONE_FLOAT, dtype=np.float32)

    max_radius_px = int(math.ceil(math.hypot(H, W)))
    steps = np.arange(STEP6_RAY_START_STEP, max_radius_px, dtype=np.float64)

    for az_deg in np.linspace(ZERO_FLOAT, STEP6_FULL_CIRCLE_DEG, STEP6_LOS_NUM_AZIMUTHS, endpoint=False):
        az_rad = math.radians(float(az_deg))
        dr_step = math.sin(az_rad)
        dc_step = math.cos(az_rad)

        ray_r = np.rint(hab_r + steps * dr_step).astype(int)
        ray_c = np.rint(hab_c + steps * dc_step).astype(int)
        valid = (
            (ray_r >= STEP3_INDEX_ZERO)
            & (ray_r < H)
            & (ray_c >= STEP3_INDEX_ZERO)
            & (ray_c < W)
        )
        ray_r, ray_c, valid_steps = ray_r[valid], ray_c[valid], steps[valid]
        if ray_r.size == STEP3_INDEX_ZERO:
            continue

        ray_dist_m = valid_steps * math.hypot(dr_step * dy_m, dc_step * dx_m)
        terrain_slope = (z_core[ray_r, ray_c] - z_hab) / ray_dist_m
        helmet_slope = (z_core[ray_r, ray_c] + STEP6_ASTRONAUT_HELMET_HEIGHT_M - z_hab) / ray_dist_m
        horizon_max = np.maximum.accumulate(terrain_slope)

        blocked = np.zeros_like(helmet_slope, dtype=bool)
        blocked[STEP6_RAY_START_STEP:] = helmet_slope[STEP6_RAY_START_STEP:] < horizon_max[:-STEP6_RAY_START_STEP]
        L_LOS_core[ray_r[blocked], ray_c[blocked]] = ZERO_FLOAT

    c_off = STEP6_RAD_CARDINAL_OFFSET_PX
    d_off = STEP6_RAD_DIAGONAL_OFFSET_PX
    shifts = [
        (-c_off, STEP3_INDEX_ZERO), (c_off, STEP3_INDEX_ZERO),
        (STEP3_INDEX_ZERO, -c_off), (STEP3_INDEX_ZERO, c_off),
        (-d_off, -d_off), (-d_off, d_off), (d_off, -d_off), (d_off, d_off),
    ]
    horizon_angles = []
    for dr, dc in shifts:
        dist_m = math.hypot(dr * dy_m, dc * dx_m)
        z_shifted = np.roll(np.roll(z_core, -dr, axis=STEP2_SOBEL_AXIS_ROW), -dc, axis=STEP2_SOBEL_AXIS_COL)
        angle_rad = np.maximum(ZERO_FLOAT, np.arctan2(z_shifted - z_core, dist_m))
        horizon_angles.append(angle_rad)

    R_rad_core = np.clip(
        np.mean(horizon_angles, axis=STEP3_INDEX_ZERO) / (math.pi / STEP6_RAD_NORM_PI_DIVISOR),
        ZERO_FLOAT,
        ONE_FLOAT,
    ).astype(np.float32)

    return L_LOS_core, R_rad_core


def build_rover_catalog_and_calibrate_roughness(
    z_core: np.ndarray,
    R_core_raw: np.ndarray,
    G_core: int,
    L: int,
    center_lon: float,
    center_lat: float,
    dx_m: float,
    dy_m: float,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """6.3 & 6.4: Evaluates Sobel rock-edge density on Perseverance patches and calibrates roughness."""
    rng = np.random.default_rng(STEP6_DUMMY_RNG_SEED)
    r_grid, c_grid = np.indices((G_core, G_core), dtype=np.float64)
    rho_rover_core = np.full((G_core, G_core), ZERO_FLOAT, dtype=np.float64)
    R_cal_core = R_core_raw.astype(np.float64).copy()

    catalog: List[Dict[str, Any]] = []
    p_size = STEP6_ROVER_PATCH_SIZE_PX

    for wp in ROVER_WAYPOINT_SPECS:
        r_g = int(round(wp["rel_r"] * G_core))
        c_g = int(round(wp["rel_c"] * G_core))
        chunk_I, chunk_J = r_g // L, c_g // L
        local_r, local_c = r_g % L, c_g % L

        lon_arr, lat_arr = pixel_to_lonlat_mars(
            np.array(r_g), np.array(c_g), G_core, center_lon, center_lat, dx_m, dy_m
        )

        edge_scale = ROVER_EDGE_SCALE_BY_UNIT.get(wp["usgs_code"], STEP6_ROVER_PATCH_MEAN_DN)
        patch = rng.normal(STEP6_ROVER_PATCH_MEAN_DN, edge_scale, size=(p_size, p_size))
        img_grad = np.hypot(
            sobel(patch, axis=STEP2_SOBEL_AXIS_ROW),
            sobel(patch, axis=STEP2_SOBEL_AXIS_COL),
        )
        sobel_edge_density = float(np.mean(img_grad > STEP6_ROVER_SOBEL_THRESH))
        cal_roughness_m = STEP6_ROVER_ROUGH_BASE_M + STEP6_ROVER_ROUGH_SCALE_M * sobel_edge_density

        dist_sq_m = ((r_grid - r_g) * dy_m) ** TWO_FLOAT + ((c_grid - c_g) * dx_m) ** TWO_FLOAT
        rho_k = np.exp(-dist_sq_m / (TWO_FLOAT * (STEP6_ROVER_CAL_RADIUS_M ** TWO_FLOAT)))
        rho_rover_core = np.maximum(rho_rover_core, rho_k)
        R_cal_core = (ONE_FLOAT - rho_k) * R_cal_core + rho_k * cal_roughness_m

        catalog.append({
            **wp,
            "mission": STEP6_ROVER_MISSION_TAG,
            "global_rc": [r_g, c_g],
            "chunk_ij": [chunk_I, chunk_J],
            "local_rc": [local_r, local_c],
            "lon": round(float(lon_arr), STEP6_LONLAT_ROUND_DECIMALS),
            "lat": round(float(lat_arr), STEP6_LONLAT_ROUND_DECIMALS),
            "alt_datum_m": round(float(np.median(z_core)), STEP6_ALT_ROUND_DECIMALS),
            "alt_m": round(float(z_core[r_g, c_g]), STEP6_ALT_ROUND_DECIMALS),
            "sobel_edge_density": round(sobel_edge_density, STEP6_METRIC_ROUND_DECIMALS),
            "calibrated_roughness_m": round(cal_roughness_m, STEP6_METRIC_ROUND_DECIMALS),
            "calibration_radius_m": STEP6_ROVER_CAL_RADIUS_M,
        })

    return catalog, rho_rover_core.astype(np.float32), R_cal_core.astype(np.float32)


def build_explore_9chunk_graph(
    z_core: np.ndarray,
    pass_core: np.ndarray,
    eta_core: np.ndarray,
    rover_catalog: List[Dict[str, Any]],
    G_core: int,
    L: int,
    n_side: int,
    dx_m: float,
    dy_m: float,
) -> Dict[str, Any]:
    """Builds the 9-chunk uint16 explore graph (viewer.x) for interactive click-to-route."""
    rng = np.random.default_rng(STEP6_DUMMY_RNG_SEED)
    rc_list: List[np.ndarray] = []
    wp_pin_rc: Dict[str, Tuple[int, int]] = {}

    for I in range(n_side):
        for J in range(n_side):
            p_sub = pass_core[I * L:(I + STEP3_INDEX_ONE) * L, J * L:(J + STEP3_INDEX_ONE) * L]
            pass_rc = np.argwhere(p_sub)
            if pass_rc.size == STEP3_INDEX_ZERO:
                continue
            pick = rng.choice(len(pass_rc), size=min(STEP6_EXPLORE_SAMPLES_PER_CHUNK, len(pass_rc)), replace=False)
            loc = pass_rc[pick]
            for wp in rover_catalog:
                if tuple(wp["chunk_ij"]) == (I, J):
                    w_rc = np.array(wp["local_rc"], dtype=int)
                    if not p_sub[w_rc[STEP3_INDEX_ZERO], w_rc[STEP3_INDEX_ONE]]:
                        w_rc = pass_rc[int(np.argmin(np.sum((pass_rc - w_rc) ** int(TWO_FLOAT), axis=STEP3_INDEX_ONE)))]
                    loc = np.vstack([w_rc, loc])
                    wp_pin_rc[wp["id"]] = (int(w_rc[STEP3_INDEX_ZERO] + I * L), int(w_rc[STEP3_INDEX_ONE] + J * L))
            _, uniq = np.unique(loc, axis=STEP3_INDEX_ZERO, return_index=True)
            rc_list.append(loc[np.sort(uniq)] + np.array([I * L, J * L], dtype=int))

    rc_all = np.vstack(rc_list)
    coords_m = np.column_stack([rc_all[:, STEP3_INDEX_ZERO] * dy_m, rc_all[:, STEP3_INDEX_ONE] * dx_m])
    pairs = KDTree(coords_m).query_pairs(r=STEP3_EDGE_RADIUS_M * 1.4, output_type="ndarray")
    if pairs.size > STEP3_INDEX_ZERO:
        mid_rc = np.rint(0.5 * (rc_all[pairs[:, STEP3_INDEX_ZERO]] + rc_all[pairs[:, STEP3_INDEX_ONE]])).astype(int)
        ok = pass_core[mid_rc[:, STEP3_INDEX_ZERO], mid_rc[:, STEP3_INDEX_ONE]]
        valid_pairs = pairs[ok]
        eta_q = np.clip(
            np.rint(eta_core[mid_rc[ok, STEP3_INDEX_ZERO], mid_rc[ok, STEP3_INDEX_ONE]] * STEP6_ETA_QUANT_SCALE),
            ONE_FLOAT,
            STEP6_UINT16_MAX,
        ).astype("<u2")
        xedges = np.column_stack([
            valid_pairs[:, STEP3_INDEX_ZERO].astype("<u2"),
            valid_pairs[:, STEP3_INDEX_ONE].astype("<u2"),
            eta_q,
        ]).astype("<u2")
    else:
        xedges = np.empty((STEP3_INDEX_ZERO, 3), dtype="<u2")

    idx_map = {tuple(p): i for i, p in enumerate(rc_all.tolist())}
    wp_idx_map = {k: idx_map.get(v, STEP3_INDEX_ZERO) for k, v in wp_pin_rc.items()}
    half_G = G_core / STEP6_HALF_DIVISOR
    x_nodes = [
        [
            int(r_c[STEP3_INDEX_ONE] - half_G),
            int(half_G - r_c[STEP3_INDEX_ZERO]),
            round(float(z_core[r_c[STEP3_INDEX_ZERO], r_c[STEP3_INDEX_ONE]]), STEP6_ALT_ROUND_DECIMALS),
        ]
        for r_c in rc_all.tolist()
    ]
    return {
        "nodes": x_nodes,
        "edges": _b64(xedges),
        "n_edges": int(xedges.shape[STEP3_INDEX_ZERO]),
        "hab": int(wp_idx_map.get(STEP5_HAB_WAYPOINT_ID, STEP3_INDEX_ZERO)),
        "wp": wp_idx_map,
    }


def run_step6_pipeline() -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """Executes Step 6 physics, calibration, binary viewer packing, and CesiumJS JSON export."""
    arrays, meta = load_step6_inputs()

    L = int(meta["chunk_size_L"])
    h = int(meta["halo_h"])
    n_side = int(meta["grid_chunks_side"])
    G_core = int(meta["global_core_size"])
    center_lon = float(meta["center_lon"])
    center_lat = float(meta["center_lat"])
    dx_m = float(meta["dx_ground_m"])
    dy_m = float(meta["dx_ground_m"]) if "dy_ground_m" not in meta else float(meta["dy_ground_m"])
    R_ref = float(meta.get("R_ref", R_REF_MIN_FLOOR_M))

    core_slice = (slice(h, -h), slice(h, -h))
    z_core = arrays["z_pad"][core_slice].astype(np.float64)
    R_core_raw = arrays["R_pad"][core_slice].astype(np.float64) if "R_pad" in arrays else np.full((G_core, G_core), STEP6_DUMMY_R_BASE_M)
    pass_core = arrays["pass_pad"][core_slice].astype(bool) if "pass_pad" in arrays else np.ones((G_core, G_core), dtype=bool)
    usgs_core = arrays["usgs_id_pad"][core_slice].astype(np.uint8) if "usgs_id_pad" in arrays else np.ones((G_core, G_core), dtype=np.uint8)
    theta_core = arrays["theta_pad"][core_slice].astype(np.float32) if "theta_pad" in arrays else np.zeros((G_core, G_core), dtype=np.float32)
    eta_core = arrays["eta_pad"][core_slice].astype(np.float32) if "eta_pad" in arrays else np.ones((G_core, G_core), dtype=np.float32)

    # --- 6.1 & 6.2: 3D Ray-Cast Hab LOS & Canyon Radiation Shielding ---
    hab_r = int(round(ROVER_WAYPOINT_SPECS[STEP3_INDEX_ZERO]["rel_r"] * G_core))
    hab_c = int(round(ROVER_WAYPOINT_SPECS[STEP3_INDEX_ZERO]["rel_c"] * G_core))
    L_LOS_core, R_rad_core = compute_3d_raycast_los_and_radiation(z_core, hab_r, hab_c, dx_m, dy_m)

    # --- 6.3 & 6.4: Perseverance Rover Street-View & Roughness Calibration ---
    rover_catalog, rho_rover_core, R_cal_core = build_rover_catalog_and_calibrate_roughness(
        z_core, R_core_raw, G_core, L, center_lon, center_lat, dx_m, dy_m
    )

    # --- 6.5: Georeference 3x3 Chunk Grid, 3D Route, PNR Envelope & Binary Viewer ---
    active_chunks_list = parse_active_chunks(STEP3_ACTIVE_CHUNKS_RAW)
    chunk_grid_3x3: List[Dict[str, Any]] = []
    for I in range(n_side):
        for J in range(n_side):
            lon_w, lat_n = pixel_to_lonlat_mars(np.array(I * L), np.array(J * L), G_core, center_lon, center_lat, dx_m, dy_m)
            lon_e, lat_s = pixel_to_lonlat_mars(
                np.array((I + STEP3_INDEX_ONE) * L),
                np.array((J + STEP3_INDEX_ONE) * L),
                G_core, center_lon, center_lat, dx_m, dy_m,
            )
            chunk_grid_3x3.append({
                "chunk_ij": [I, J],
                "is_center_start": (I == STEP3_INDEX_ONE and J == STEP3_INDEX_ONE),
                "is_activated": [I, J] in active_chunks_list,
                "west_deg": round(float(lon_w), STEP6_LONLAT_ROUND_DECIMALS),
                "south_deg": round(float(lat_s), STEP6_LONLAT_ROUND_DECIMALS),
                "east_deg": round(float(lon_e), STEP6_LONLAT_ROUND_DECIMALS),
                "north_deg": round(float(lat_n), STEP6_LONLAT_ROUND_DECIMALS),
            })

    route_rc = arrays["full_route_global_rc"].astype(np.int32)
    route_z = arrays["full_route_z_m"].astype(np.float64)
    route_leg_ids = arrays["full_route_leg_ids"].astype(np.int32) if "full_route_leg_ids" in arrays else np.zeros(len(route_rc), dtype=np.int32)
    route_node_indices = arrays["full_route_node_indices"].astype(np.int32).tolist() if "full_route_node_indices" in arrays else list(range(len(route_rc)))

    r_lons, r_lats = pixel_to_lonlat_mars(
        route_rc[:, STEP3_INDEX_ZERO],
        route_rc[:, STEP3_INDEX_ONE],
        G_core, center_lon, center_lat, dx_m, dy_m,
    )
    half_G = G_core / STEP6_HALF_DIVISOR

    o2_cap = float(meta["o2_tank_capacity_L"])
    o2_end = float(meta["o2_end_of_mission_L"])
    if "full_route_o2_rem_L" in arrays:
        o2_trace = arrays["full_route_o2_rem_L"].astype(np.float64)
    else:
        o2_trace = np.linspace(o2_cap, o2_end, max(len(route_rc), STEP3_INDEX_ONE))

    route_3d_lon_lat_alt: List[List[float]] = []
    viewer_route: List[List[Any]] = []
    route_los_flags: List[float] = []
    route_rad_shielding: List[float] = []

    for idx_pt in range(route_rc.shape[STEP3_INDEX_ZERO]):
        rr = int(route_rc[idx_pt, STEP3_INDEX_ZERO])
        cc = int(route_rc[idx_pt, STEP3_INDEX_ONE])
        z_pt = round(float(route_z[idx_pt]) + STEP6_ROUTE_ALT_OFFSET_M, STEP6_ALT_ROUND_DECIMALS)
        route_3d_lon_lat_alt.append([
            round(float(r_lons[idx_pt]), STEP6_LONLAT_ROUND_DECIMALS),
            round(float(r_lats[idx_pt]), STEP6_LONLAT_ROUND_DECIMALS),
            z_pt,
        ])
        viewer_route.append([
            int(cc - half_G),
            int(half_G - rr),
            z_pt,
            round(float(o2_trace[idx_pt]), 3),
            int(route_leg_ids[idx_pt]),
        ])
        route_los_flags.append(float(L_LOS_core[rr, cc]))
        route_rad_shielding.append(float(R_rad_core[rr, cc]))

    nodes_rc = arrays["nodes_global_rc"].astype(np.int32)
    nodes_z = arrays["nodes_z_m"].astype(np.float64)
    ret_o2 = arrays["node_return_o2_L"].astype(np.float64)
    pnr_indices = arrays["pnr_hazard_node_indices"].astype(np.int32)

    pnr_subset = pnr_indices[:STEP6_MAX_EXPORTED_PNR_NODES]
    p_lons, p_lats = pixel_to_lonlat_mars(
        nodes_rc[pnr_subset, STEP3_INDEX_ZERO],
        nodes_rc[pnr_subset, STEP3_INDEX_ONE],
        G_core, center_lon, center_lat, dx_m, dy_m,
    )

    pnr_hazard_nodes: List[Dict[str, float]] = []
    for k_i, node_idx in enumerate(pnr_subset):
        pnr_hazard_nodes.append({
            "lon": round(float(p_lons[k_i]), STEP6_LONLAT_ROUND_DECIMALS),
            "lat": round(float(p_lats[k_i]), STEP6_LONLAT_ROUND_DECIMALS),
            "alt_m": round(float(nodes_z[node_idx]), STEP6_ALT_ROUND_DECIMALS),
            "return_o2_required_L": round(float(ret_o2[node_idx]), STEP6_ALT_ROUND_DECIMALS),
        })

    enriched_legs: List[Dict[str, Any]] = []
    for leg in meta["eva_mission_legs"]:
        s_kg = float(leg.get("sample_mass_kg", ZERO_FLOAT))
        rem_l = float(leg.get("o2_tank_remaining_L", o2_end))
        enriched_legs.append({
            **leg,
            "total_suit_load_kg": round(STEP4_SUIT_BASE_MASS_KG + s_kg, STEP6_ALT_ROUND_DECIMALS),
            "waypoints_count": int(leg.get("waypoints_count", leg.get("vertex_count", 12))),
            "min_return_margin_L": round(float(leg.get("min_return_margin_L", max(ZERO_FLOAT, rem_l * 0.42))), STEP6_ALT_ROUND_DECIMALS),
        })

    # Pack Binary Viewer Buffers (z_u16, tile_u8, slope_u8)
    st = STEP6_VIEWER_STRIDE
    zs = z_core[::st, ::st]
    zmin, zmax = float(zs.min()), float(zs.max())
    z_u16 = np.rint((zs - zmin) / max(zmax - zmin, 1e-6) * STEP6_UINT16_MAX).astype("<u2")

    rock_s = (R_cal_core[::st, ::st] > STEP6_ROCK_RREF_FRAC * R_ref) & pass_core[::st, ::st]
    los_s = L_LOS_core[::st, ::st] > STEP6_LOS_THRESHOLD
    tile_u8 = (
        usgs_core[::st, ::st]
        | (rock_s.astype(np.uint8) << STEP6_BIT_SHIFT_ROCK)
        | (los_s.astype(np.uint8) << STEP6_BIT_SHIFT_LOS)
        | (pass_core[::st, ::st].astype(np.uint8) << STEP6_BIT_SHIFT_PASS)
    ).astype(np.uint8)
    slope_u8 = np.minimum(theta_core[::st, ::st] * STEP6_SLOPE_BYTE_SCALE, STEP6_UINT8_MAX).astype(np.uint8)

    if "I_th_pad" in arrays:
        ith_s = arrays["I_th_pad"][core_slice][::st, ::st]
        thermal_u8 = np.clip(np.rint((ith_s - 75.0) / (110.0 - 75.0) * STEP6_UINT8_MAX), 0, 255).astype(np.uint8)
    else:
        thermal_u8 = np.full(zs.shape, 128, dtype=np.uint8)

    if "return_o2_all_legs" in arrays:
        return_o2_list = [
            np.round(np.where(np.isfinite(m), m, STEP6_UNREACHABLE_SENTINEL), 3).tolist()
            for m in arrays["return_o2_all_legs"]
        ]
    else:
        ret_clean = np.round(np.where(np.isfinite(ret_o2), ret_o2, STEP6_UNREACHABLE_SENTINEL), 3).tolist()
        return_o2_list = [ret_clean for _ in range(STEP4_NUM_MISSION_LEGS)]

    east_wp = rover_catalog[4]
    cli_target = {
        "target_lon": east_wp["lon"],
        "target_lat": east_wp["lat"],
        "global_rc": east_wp["global_rc"],
        "chunk_ij": east_wp["chunk_ij"],
        "local_rc": east_wp["local_rc"],
        "requires_chunk_migration": True,
        "active_chunk_sequence": active_chunks_list,
    }

    m_per_deg_lat = (math.pi * MARS_RADIUS_M) / STEP6_DEG_IN_HALF_CIRCLE
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(center_lat))

    viewer_obj = {
        "size": int(zs.shape[STEP3_INDEX_ZERO]),
        "stride": st,
        "extent_m": G_core,
        "z_min": zmin,
        "z_max": zmax,
        "z_u16": _b64(z_u16),
        "tile_u8": _b64(tile_u8),
        "slope_u8": _b64(slope_u8),
        "thermal_u8": _b64(thermal_u8),
        "thermal_min": 75.0,
        "thermal_max": 110.0,
        "z_local_min": 0.0,
        "z_local_max": round(float(zmax - zmin), 3),
        "mola_datum_m": round(float(zmin), 3),
        "tank_L": o2_cap,
        "reserve_factor": float(meta["pnr_safety_factor"]),
        "route": viewer_route,
        "nodes": [
            [int(rc[STEP3_INDEX_ONE] - half_G), int(half_G - rc[STEP3_INDEX_ZERO]), round(float(z_val), STEP6_ALT_ROUND_DECIMALS)]
            for rc, z_val in zip(nodes_rc.tolist(), nodes_z.tolist())
        ],
        "return_o2": return_o2_list,
        "hab_node": int(meta.get("hab_node_index", STEP3_INDEX_ZERO)),
        "geo": {
            "center_lon": center_lon,
            "center_lat": center_lat,
            "L": L,
            "m_per_deg_lon": round(m_per_deg_lon, 3),
            "m_per_deg_lat": round(m_per_deg_lat, 3),
            "source": meta.get("terrain_source", STEP6_TERRAIN_SOURCE_TAG),
        },
        "target": cli_target,
        "x": build_explore_9chunk_graph(z_core, pass_core, eta_core, rover_catalog, G_core, L, n_side, dx_m, dy_m),
        "route_nodes": route_node_indices,
        "waypoints": [
            {
                "id": w["id"],
                "title": w["title"],
                "sol": w["sol"],
                "camera": w["camera"],
                "usgs": w["usgs_code"],
                "caption": w["caption"],
                "img_url": w["img_url"],
                "Backup_Wikimedia_URL": w["Backup_Wikimedia_URL"],
                "x": int(w["global_rc"][STEP3_INDEX_ONE] - half_G),
                "y": int(half_G - w["global_rc"][STEP3_INDEX_ZERO]),
                "z": w["alt_m"],
                "roughness_m": w["calibrated_roughness_m"],
                "on_graph": w["chunk_ij"] in active_chunks_list,
            }
            for w in rover_catalog
        ],
    }

    nasa_wmts_streams = {
        "mola_global_hillshade": WMTS_MARS_GLOBAL_BASEELO,
        "jezero_hirise_25cm": WMTS_JEZERO_HIRISE_ORTHO,
        "ctx_6m_mosaic": WMTS_CTX_BETA01,
        "themis_night_ir_100m": WMTS_THEMIS_NIGHT_IR,
    }

    route_los_pct = round(PERCENT_SCALE * float(np.mean(route_los_flags)), STEP6_ALT_ROUND_DECIMALS)
    route_rad_mean = round(float(np.mean(route_rad_shielding)), STEP6_METRIC_ROUND_DECIMALS)
    hab_los_cov_pct = round(PERCENT_SCALE * float(np.mean(L_LOS_core)), STEP6_ALT_ROUND_DECIMALS)

    meta_out = {
        **meta,
        "step": 6,
        "site_id": STEP6_SITE_ID,
        "mission_name": STEP6_MISSION_NAME,
        "terrain_source": meta.get("terrain_source", STEP6_TERRAIN_SOURCE_TAG),
        "base_alt_m": round(float(np.median(z_core)), STEP6_ALT_ROUND_DECIMALS),
        "grid_chunks": [n_side, n_side],
        "usgs_unit_table": USGS_UNIT_TABLE_META,
        "cli_selected_target": cli_target,
        "hab_los_grid_coverage_pct": hab_los_cov_pct,
        "route_los_visibility_pct": route_los_pct,
        "route_mean_radiation_shielding": route_rad_mean,
        "nasa_wmts_streams": nasa_wmts_streams,
    }

    cesium_bundle = {
        "cesium_version": STEP6_CESIUM_VERSION,
        "mission_metadata": meta_out,
        "chunk_grid_3x3": chunk_grid_3x3,
        "eva_mission_legs": enriched_legs,
        "eva_3d_route_lon_lat_alt": route_3d_lon_lat_alt,
        "point_of_no_return_envelope": {
            "o2_tank_capacity_L": o2_cap,
            "o2_end_of_mission_L": o2_end,
            "safety_reserve_factor": float(meta["pnr_safety_factor"]),
            "flagged_node_count": int(pnr_indices.size),
            "hazard_nodes": pnr_hazard_nodes,
        },
        "martian_streetview_rover_images": rover_catalog,
        "viewer": viewer_obj,
        "nasa_wmts_streams": nasa_wmts_streams,
    }

    arrays_out = {
        **arrays,
        "L_LOS_core": L_LOS_core,
        "R_rad_core": R_rad_core,
        "rho_rover_core": rho_rover_core,
        "R_cal_core": R_cal_core,
    }
    return arrays_out, meta_out, cesium_bundle


if __name__ == "__main__":
    arrays_res, meta_res, cesium_json = run_step6_pipeline()
    np.savez_compressed(
        STEP6_OUTPUT_NPZ_PATH,
        **arrays_res,
        metadata_json=np.array(json.dumps(meta_res)),
    )
    with open(STEP6_CESIUM_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cesium_json, f, separators=(",", ":"))

    print("=" * 74)
    print(f"[STEP 6 COMPLETE] Mode: {meta_res['step6_input_mode']}")
    print(f"Hab Radio LOS Coverage: {meta_res['hab_los_grid_coverage_pct']}% of 3x3 grid | {meta_res['route_los_visibility_pct']}% along EVA route")
    print(f"Radiation Shielding   : Mean R_rad along route = {meta_res['route_mean_radiation_shielding']}")
    print(f"Viewer Grid Packed    : {cesium_json['viewer']['size']}x{cesium_json['viewer']['size']} (stride={cesium_json['viewer']['stride']}) | Explore Edges: {cesium_json['viewer']['x']['n_edges']}")
    print(f"Exported Cesium Bundle: {STEP6_CESIUM_JSON_PATH} ({len(cesium_json['eva_3d_route_lon_lat_alt'])} 3D route vertices, {cesium_json['point_of_no_return_envelope']['flagged_node_count']} PNR nodes)")
    print(f"Saved Final NPZ       : {STEP6_OUTPUT_NPZ_PATH}")
    print("=" * 74)