#!/usr/bin/env python3
"""
step1_terrain_and_cost.py
=========================
Step 1: Terrain Geometry (Slope & Denoised Roughness), Material Friction (eta),
Orbital Uncertainty (U), and Continuous Traversal Cost Field C(r,c) on a 3x3 Chunk Grid.
Reads all constants from .env and outputs step1_output.npz for Step 2.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import uniform_filter


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

# --- File Paths ---
REAL_DEM_TIF_PATH            = os.environ["REAL_DEM_TIF_PATH"]
REAL_TES_TIF_PATH            = os.environ["REAL_TES_TIF_PATH"]
REAL_THEMIS_TIF_PATH         = os.environ["REAL_THEMIS_TIF_PATH"]
REAL_SIM3464_TIF_PATH        = os.environ["REAL_SIM3464_TIF_PATH"]
STEP1_OUTPUT_NPZ_PATH        = os.environ["STEP1_OUTPUT_NPZ_PATH"]

# --- 3x3 Grid & Geodesy Constants ---
GRID_CHUNKS_SIDE             = int(os.environ["GRID_CHUNKS_SIDE"])
CHUNK_SIZE_PX                = int(os.environ["CHUNK_SIZE_PX"])
GLOBAL_HALO_PX               = int(os.environ["GLOBAL_HALO_PX"])
MARS_RADIUS_M                = float(os.environ["MARS_RADIUS_M"])
CENTER_LON_DEG               = float(os.environ["CENTER_LON_DEG"])
CENTER_LAT_DEG               = float(os.environ["CENTER_LAT_DEG"])
MAP_PIXEL_SIZE_X_M           = float(os.environ["MAP_PIXEL_SIZE_X_M"])
MAP_PIXEL_SIZE_Y_M           = float(os.environ["MAP_PIXEL_SIZE_Y_M"])
NODATA_SENTINEL_LOW          = float(os.environ["NODATA_SENTINEL_LOW"])
NODATA_SENTINEL_HIGH         = float(os.environ["NODATA_SENTINEL_HIGH"])
PERCENT_SCALE                = float(os.environ["PERCENT_SCALE"])

# --- Dummy Input Generator Constants ---
DUMMY_RNG_SEED               = int(os.environ["DUMMY_RNG_SEED"])
DUMMY_BASE_ALT_M             = float(os.environ["DUMMY_BASE_ALT_M"])
DUMMY_REGIONAL_SLOPE_M       = float(os.environ["DUMMY_REGIONAL_SLOPE_M"])
DUMMY_RIDGE_HEIGHT_M         = float(os.environ["DUMMY_RIDGE_HEIGHT_M"])
DUMMY_RIDGE_CENTER_NORM      = float(os.environ["DUMMY_RIDGE_CENTER_NORM"])
DUMMY_RIDGE_WIDTH_SQ         = float(os.environ["DUMMY_RIDGE_WIDTH_SQ"])
DUMMY_SADDLE_GAP_DEPTH       = float(os.environ["DUMMY_SADDLE_GAP_DEPTH"])
DUMMY_SADDLE_CENTER_NORM     = float(os.environ["DUMMY_SADDLE_CENTER_NORM"])
DUMMY_SADDLE_WIDTH_SQ        = float(os.environ["DUMMY_SADDLE_WIDTH_SQ"])
DUMMY_DEM_NOISE_STD_M        = float(os.environ["DUMMY_DEM_NOISE_STD_M"])
DUMMY_BOULDER_NOISE_STD_M    = float(os.environ["DUMMY_BOULDER_NOISE_STD_M"])
DUMMY_BOULDER_R_MIN          = float(os.environ["DUMMY_BOULDER_R_MIN"])
DUMMY_BOULDER_R_MAX          = float(os.environ["DUMMY_BOULDER_R_MAX"])
DUMMY_BOULDER_C_MIN          = float(os.environ["DUMMY_BOULDER_C_MIN"])
DUMMY_BOULDER_C_MAX          = float(os.environ["DUMMY_BOULDER_C_MAX"])
DUMMY_TES_BASELINE_TIU       = float(os.environ["DUMMY_TES_BASELINE_TIU"])
DUMMY_THEMIS_MEAN_DN         = float(os.environ["DUMMY_THEMIS_MEAN_DN"])
DUMMY_THEMIS_STD_DN          = float(os.environ["DUMMY_THEMIS_STD_DN"])
DUMMY_DUST_TRAP_DN           = float(os.environ["DUMMY_DUST_TRAP_DN"])
DUMMY_SAND_R_MIN             = float(os.environ["DUMMY_SAND_R_MIN"])
DUMMY_SAND_R_MAX             = float(os.environ["DUMMY_SAND_R_MAX"])
DUMMY_SAND_C_MIN             = float(os.environ["DUMMY_SAND_C_MIN"])
DUMMY_SAND_C_MAX             = float(os.environ["DUMMY_SAND_C_MAX"])
DUMMY_DELTA_FRONT_C_THRESH   = float(os.environ["DUMMY_DELTA_FRONT_C_THRESH"])
DUMMY_DELTA_PLAIN_C_THRESH   = float(os.environ["DUMMY_DELTA_PLAIN_C_THRESH"])
DUMMY_LOS_SHADOW_C_THRESH    = float(os.environ["DUMMY_LOS_SHADOW_C_THRESH"])
DUMMY_UNIT_ID_DELTA_PLAIN    = int(os.environ["DUMMY_UNIT_ID_DELTA_PLAIN"])
DUMMY_UNIT_ID_DELTA_FRONT    = int(os.environ["DUMMY_UNIT_ID_DELTA_FRONT"])
DUMMY_UNIT_ID_SAND           = int(os.environ["DUMMY_UNIT_ID_SAND"])
DUMMY_UNIT_ID_BOULDER        = int(os.environ["DUMMY_UNIT_ID_BOULDER"])

# --- USGS SIM 3464 Geologic Unit Friction Priors (eta_geo) ---
DEFAULT_UNIT_ID              = int(os.environ["DEFAULT_UNIT_ID"])
DEFAULT_ETA_GEO              = float(os.environ["DEFAULT_ETA_GEO"])
USGS_UNIT_PRIORS: Dict[int, Dict[str, Any]] = {
    1: {"code": os.environ["UNIT_1_CODE"], "eta_geo": float(os.environ["UNIT_1_ETA_GEO"])},
    2: {"code": os.environ["UNIT_2_CODE"], "eta_geo": float(os.environ["UNIT_2_ETA_GEO"])},
    3: {"code": os.environ["UNIT_3_CODE"], "eta_geo": float(os.environ["UNIT_3_ETA_GEO"])},
    4: {"code": os.environ["UNIT_4_CODE"], "eta_geo": float(os.environ["UNIT_4_ETA_GEO"])},
    5: {"code": os.environ["UNIT_5_CODE"], "eta_geo": float(os.environ["UNIT_5_ETA_GEO"])},
    6: {"code": os.environ["UNIT_6_CODE"], "eta_geo": float(os.environ["UNIT_6_ETA_GEO"])},
    7: {"code": os.environ["UNIT_7_CODE"], "eta_geo": float(os.environ["UNIT_7_ETA_GEO"])},
}

# --- Thermal Inertia Fusion Constants ---
TI_ZSCORE_CLIP_MIN           = float(os.environ["TI_ZSCORE_CLIP_MIN"])
TI_ZSCORE_CLIP_MAX           = float(os.environ["TI_ZSCORE_CLIP_MAX"])
TI_CONTRAST_SCALE_TIU        = float(os.environ["TI_CONTRAST_SCALE_TIU"])
TI_PHYSICAL_MIN_TIU          = float(os.environ["TI_PHYSICAL_MIN_TIU"])
TI_PHYSICAL_MAX_TIU          = float(os.environ["TI_PHYSICAL_MAX_TIU"])
EPSILON_STD                  = float(os.environ["EPSILON_STD"])

# --- Step 1.1: Slope & Cliff Passability Constants ---
MAX_PASSABLE_SLOPE_DEG       = float(os.environ["MAX_PASSABLE_SLOPE_DEG"])
ZERO_FLOAT                   = float(os.environ["ZERO_FLOAT"])
ONE_FLOAT                    = float(os.environ["ONE_FLOAT"])
TWO_FLOAT                    = float(os.environ["TWO_FLOAT"])

# --- Step 1.2: Stereo Noise & Denoised Roughness Constants ---
STEEP_SLOPE_NOISE_THRESH_DEG = float(os.environ["STEEP_SLOPE_NOISE_THRESH_DEG"])
SIGMA_Z_FLAT_M               = float(os.environ["SIGMA_Z_FLAT_M"])
SIGMA_Z_STEEP_M              = float(os.environ["SIGMA_Z_STEEP_M"])
ROUGHNESS_WINDOW_PX          = int(os.environ["ROUGHNESS_WINDOW_PX"])
ROUGHNESS_PERCENTILE_Q       = float(os.environ["ROUGHNESS_PERCENTILE_Q"])
R_REF_MIN_FLOOR_M            = float(os.environ["R_REF_MIN_FLOOR_M"])

# --- Step 1.3: Material Friction Multiplier eta(r,c) Constants ---
ETA_BASE_TERM                = float(os.environ["ETA_BASE_TERM"])
DUST_PENALTY_INTERCEPT       = float(os.environ["DUST_PENALTY_INTERCEPT"])
DUST_PENALTY_SLOPE           = float(os.environ["DUST_PENALTY_SLOPE"])
ROUGHNESS_WEIGHT_ALPHA       = float(os.environ["ROUGHNESS_WEIGHT_ALPHA"])

# --- Step 1.4: Orbital Uncertainty U(r,c) Constants ---
UNC_SIGMA_Z_WEIGHT           = float(os.environ["UNC_SIGMA_Z_WEIGHT"])
UNC_SIGMA_Z_REF_M            = float(os.environ["UNC_SIGMA_Z_REF_M"])
UNC_THERMAL_WEIGHT           = float(os.environ["UNC_THERMAL_WEIGHT"])
UNC_THERMAL_CENTER_TIU       = float(os.environ["UNC_THERMAL_CENTER_TIU"])
UNC_THERMAL_SIGMA_TIU        = float(os.environ["UNC_THERMAL_SIGMA_TIU"])
UNC_GAUSS_DENOM_FACTOR       = float(os.environ["UNC_GAUSS_DENOM_FACTOR"])
UNC_ROVER_REDUCTION_WEIGHT   = float(os.environ["UNC_ROVER_REDUCTION_WEIGHT"])
UNC_CLIP_MIN                 = float(os.environ["UNC_CLIP_MIN"])
UNC_CLIP_MAX                 = float(os.environ["UNC_CLIP_MAX"])

# --- Step 1.5: Continuous Traversal Cost Field C(r,c) Constants ---
COST_FLAT_BASE               = float(os.environ["COST_FLAT_BASE"])
COST_SLOPE_FACTOR            = float(os.environ["COST_SLOPE_FACTOR"])
COST_UNCERTAINTY_WEIGHT      = float(os.environ["COST_UNCERTAINTY_WEIGHT"])
COST_LOS_SHADOW_WEIGHT       = float(os.environ["COST_LOS_SHADOW_WEIGHT"])
DEFAULT_LOS_VISIBLE          = float(os.environ["DEFAULT_LOS_VISIBLE"])


# =========================================================================
# 1. INPUT LOADERS: (A) REAL INPUT [COMMENTED] vs. (B) DUMMY INPUT [ACTIVE]
# =========================================================================

# -------------------------------------------------------------------------
# (A) REAL INPUT LOADER (Uncomment this block to read your real .tif files)
# -------------------------------------------------------------------------
import rasterio
from scipy.ndimage import distance_transform_edt
def load_step1_inputs() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Loads the 4 aligned Jezero 3x3 rasters and pads them by GLOBAL_HALO_PX."""
    with rasterio.open(REAL_DEM_TIF_PATH) as src_z:
        z_core = src_z.read(1).astype(np.float64)
        invalid = np.isnan(z_core) | (z_core <= NODATA_SENTINEL_LOW) | (z_core >= NODATA_SENTINEL_HIGH)
        if np.any(invalid):
            _, nearest_idx = distance_transform_edt(invalid, return_indices=True)
            z_core = z_core[tuple(nearest_idx)]
    with rasterio.open(REAL_TES_TIF_PATH) as src_tes:
        tes_core = src_tes.read(1).astype(np.float64)
    with rasterio.open(REAL_THEMIS_TIF_PATH) as src_th:
        themis_core = src_th.read(1).astype(np.float64)
    with rasterio.open(REAL_SIM3464_TIF_PATH) as src_geo:
        usgs_core = src_geo.read(1).astype(np.int32)
    h = GLOBAL_HALO_PX
    z_pad = np.pad(z_core, h, mode="edge")
    tes_pad = np.pad(tes_core, h, mode="edge")
    themis_pad = np.pad(themis_core, h, mode="edge")
    usgs_id_pad = np.pad(usgs_core, h, mode="edge")
    rho_rover_pad = np.full_like(z_pad, ZERO_FLOAT, dtype=np.float64)
    L_LOS_pad = np.full_like(z_pad, DEFAULT_LOS_VISIBLE, dtype=np.float64)
    return z_pad, tes_pad, themis_pad, usgs_id_pad, rho_rover_pad, L_LOS_pad, "REAL_JEZERO_TIFS"


# -------------------------------------------------------------------------
# (B) DUMMY INPUT GENERATOR (Active out-of-the-box for standalone testing)
# -------------------------------------------------------------------------
#def load_step1_inputs() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
#    """Generates synthetic 3x3 Jezero input arrays using only .env constants."""
#    G_core = GRID_CHUNKS_SIDE * CHUNK_SIZE_PX
#    G_pad = G_core + int(TWO_FLOAT) * GLOBAL_HALO_PX
#    rng = np.random.default_rng(DUMMY_RNG_SEED)
#
#    r_idx, c_idx = np.indices((G_pad, G_pad), dtype=np.float64)
#    r_norm = (r_idx - GLOBAL_HALO_PX) / G_core
#    c_norm = (c_idx - GLOBAL_HALO_PX) / G_core
#
#    # 1. Synthetic HiRISE Elevation z_pad [m]
#    scarp = (
#        DUMMY_RIDGE_HEIGHT_M
#        * np.exp(-((c_norm - DUMMY_RIDGE_CENTER_NORM) ** TWO_FLOAT) / DUMMY_RIDGE_WIDTH_SQ)
#        * (ONE_FLOAT - DUMMY_SADDLE_GAP_DEPTH * np.exp(-((r_norm - DUMMY_SADDLE_CENTER_NORM) ** TWO_FLOAT) / DUMMY_SADDLE_WIDTH_SQ))
#    )
#    z_pad = (
#        DUMMY_BASE_ALT_M
#        + DUMMY_REGIONAL_SLOPE_M * r_norm
#        + scarp
#        + rng.normal(ZERO_FLOAT, DUMMY_DEM_NOISE_STD_M, size=(G_pad, G_pad))
#    )
#
#    boulder_mask = (
#        (r_norm > DUMMY_BOULDER_R_MIN)
#        & (r_norm < DUMMY_BOULDER_R_MAX)
#        & (c_norm > DUMMY_BOULDER_C_MIN)
#        & (c_norm < DUMMY_BOULDER_C_MAX)
#    )
#    z_pad[boulder_mask] += rng.normal(ZERO_FLOAT, DUMMY_BOULDER_NOISE_STD_M, size=int(boulder_mask.sum()))
#
#    # 2. Synthetic TES Thermal Inertia [TIU] & THEMIS Night IR [DN]
#    tes_pad = np.full((G_pad, G_pad), DUMMY_TES_BASELINE_TIU, dtype=np.float64)
#    themis_pad = rng.normal(DUMMY_THEMIS_MEAN_DN, DUMMY_THEMIS_STD_DN, size=(G_pad, G_pad))
#    sand_trap_mask = (
#        (r_norm > DUMMY_SAND_R_MIN)
#        & (r_norm < DUMMY_SAND_R_MAX)
#        & (c_norm > DUMMY_SAND_C_MIN)
#        & (c_norm < DUMMY_SAND_C_MAX)
#    )
#    themis_pad[sand_trap_mask] = DUMMY_DUST_TRAP_DN
#
#    # 3. Synthetic USGS SIM 3464 Unit IDs
#    usgs_id_pad = np.full((G_pad, G_pad), DEFAULT_UNIT_ID, dtype=np.int32)
#    usgs_id_pad[c_norm > DUMMY_DELTA_FRONT_C_THRESH] = DUMMY_UNIT_ID_DELTA_FRONT
#    usgs_id_pad[c_norm > DUMMY_DELTA_PLAIN_C_THRESH] = DUMMY_UNIT_ID_DELTA_PLAIN
#    usgs_id_pad[sand_trap_mask] = DUMMY_UNIT_ID_SAND
#    usgs_id_pad[boulder_mask] = DUMMY_UNIT_ID_BOULDER
#
#    # 4. Baseline Rover Confidence & Hab Line-of-Sight
#    rho_rover_pad = np.full((G_pad, G_pad), ZERO_FLOAT, dtype=np.float64)
#    L_LOS_pad = np.full((G_pad, G_pad), DEFAULT_LOS_VISIBLE, dtype=np.float64)
#    L_LOS_pad[c_norm > DUMMY_LOS_SHADOW_C_THRESH] = ZERO_FLOAT
#
#    return z_pad, tes_pad, themis_pad, usgs_id_pad, rho_rover_pad, L_LOS_pad, "DUMMY_SYNTHETIC_INPUT"


# =========================================================================
# 2. STEP 1 CORE COMPUTATION PIPELINE
# =========================================================================

def run_step1_pipeline() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Executes Step 1 equations and returns all output arrays and metadata."""
    z_pad, tes_pad, themis_pad, usgs_id_pad, rho_rover_pad, L_LOS_pad, input_mode = load_step1_inputs()

    h = GLOBAL_HALO_PX
    G_core = GRID_CHUNKS_SIDE * CHUNK_SIZE_PX

    dx_m = MAP_PIXEL_SIZE_X_M * math.cos(math.radians(CENTER_LAT_DEG))
    dy_m = MAP_PIXEL_SIZE_Y_M

    # --- 1.1: Squared Central-Difference Slope G^2(r,c) & Cliff Passability Mask ---
    dz_dy, dz_dx = np.gradient(z_pad, dy_m, dx_m)
    G2_pad = dz_dx ** TWO_FLOAT + dz_dy ** TWO_FLOAT
    max_tan2 = math.tan(math.radians(MAX_PASSABLE_SLOPE_DEG)) ** TWO_FLOAT

    pass_pad = G2_pad <= max_tan2
    theta_pad = np.full_like(z_pad, ZERO_FLOAT)
    theta_pad[pass_pad] = np.degrees(np.arctan(np.sqrt(G2_pad[pass_pad])))

    # --- 1.2: Stereo Noise Floor sigma_z(r,c) & Denoised WxW Roughness R(r,c) ---
    sigma_z_pad = np.where(
        theta_pad > STEEP_SLOPE_NOISE_THRESH_DEG,
        SIGMA_Z_STEEP_M,
        SIGMA_Z_FLAT_M,
    )
    mean_z = uniform_filter(z_pad, size=ROUGHNESS_WINDOW_PX, mode="reflect")
    mean_z2 = uniform_filter(z_pad ** TWO_FLOAT, size=ROUGHNESS_WINDOW_PX, mode="reflect")
    R_pad = np.sqrt(np.maximum(ZERO_FLOAT, mean_z2 - mean_z ** TWO_FLOAT - sigma_z_pad ** TWO_FLOAT))

    pass_core = pass_pad[h:-h, h:-h]
    R_core_passable = R_pad[h:-h, h:-h][pass_core]
    R_ref = max(float(np.percentile(R_core_passable, ROUGHNESS_PERCENTILE_Q)), R_REF_MIN_FLOOR_M)

    # --- 1.3: Thermal Inertia Fusion I_th(r,c) & Material Friction eta(r,c) ---
    tes_baseline_tiu = float(np.median(tes_pad))
    th_mean = float(np.mean(themis_pad))
    th_std = max(float(np.std(themis_pad)), EPSILON_STD)
    z_night = np.clip((themis_pad - th_mean) / th_std, TI_ZSCORE_CLIP_MIN, TI_ZSCORE_CLIP_MAX)
    I_th_pad = np.clip(
        tes_baseline_tiu + TI_CONTRAST_SCALE_TIU * z_night,
        TI_PHYSICAL_MIN_TIU,
        TI_PHYSICAL_MAX_TIU,
    )

    eta_geo_pad = np.full_like(z_pad, DEFAULT_ETA_GEO)
    for uid, umeta in USGS_UNIT_PRIORS.items():
        eta_geo_pad[usgs_id_pad == uid] = float(umeta["eta_geo"])

    dust_penalty = np.maximum(ZERO_FLOAT, DUST_PENALTY_INTERCEPT - DUST_PENALTY_SLOPE * I_th_pad)
    roughness_penalty = (ROUGHNESS_WEIGHT_ALPHA / R_ref) * R_pad
    eta_pad = eta_geo_pad * (ETA_BASE_TERM + dust_penalty + roughness_penalty)

    # --- 1.4: Orbital Data Uncertainty Field U(r,c) ---
    dem_unc_term = (UNC_SIGMA_Z_WEIGHT / UNC_SIGMA_Z_REF_M) * sigma_z_pad
    thermal_amb_term = UNC_THERMAL_WEIGHT * np.exp(
        -((I_th_pad - UNC_THERMAL_CENTER_TIU) ** TWO_FLOAT)
        / (UNC_GAUSS_DENOM_FACTOR * (UNC_THERMAL_SIGMA_TIU ** TWO_FLOAT))
    )
    rover_cal_term = UNC_ROVER_REDUCTION_WEIGHT * rho_rover_pad
    U_pad = np.clip(dem_unc_term + thermal_amb_term - rover_cal_term, UNC_CLIP_MIN, UNC_CLIP_MAX)

    # --- 1.5: Continuous Traversal Cost Field C(r,c) ---
    C_pad = np.full_like(z_pad, np.inf)
    C_pad[pass_pad] = (
        eta_pad[pass_pad] * (COST_FLAT_BASE + COST_SLOPE_FACTOR * theta_pad[pass_pad])
        + COST_UNCERTAINTY_WEIGHT * U_pad[pass_pad]
        + COST_LOS_SHADOW_WEIGHT * (ONE_FLOAT - L_LOS_pad[pass_pad])
    )

    arrays = {
        "z_pad": z_pad.astype(np.float32),
        "sigma_z_pad": sigma_z_pad.astype(np.float32),
        "usgs_id_pad": usgs_id_pad.astype(np.int16),
        "eta_geo_pad": eta_geo_pad.astype(np.float32),
        "I_th_pad": I_th_pad.astype(np.float32),
        "theta_pad": theta_pad.astype(np.float32),
        "R_pad": R_pad.astype(np.float32),
        "eta_pad": eta_pad.astype(np.float32),
        "U_pad": U_pad.astype(np.float32),
        "L_LOS_pad": L_LOS_pad.astype(np.float32),
        "pass_pad": pass_pad,
        "C_pad": C_pad.astype(np.float32),
    }

    metadata = {
        "step": 1,
        "input_mode": input_mode,
        "grid_chunks_side": GRID_CHUNKS_SIDE,
        "chunk_size_L": CHUNK_SIZE_PX,
        "halo_h": GLOBAL_HALO_PX,
        "global_core_size": G_core,
        "center_lon": CENTER_LON_DEG,
        "center_lat": CENTER_LAT_DEG,
        "dx_ground_m": round(dx_m, 6),
        "dy_ground_m": round(dy_m, 6),
        "R_ref": round(R_ref, 5),
        "impassable_cliff_pct": round(PERCENT_SCALE * float(np.mean(~pass_core)), 2),
        "finite_cost_min": round(float(C_pad[pass_pad].min()), 2),
        "finite_cost_max": round(float(C_pad[pass_pad].max()), 2),
    }
    return arrays, metadata


if __name__ == "__main__":
    arrays_out, meta_out = run_step1_pipeline()
    np.savez_compressed(
        STEP1_OUTPUT_NPZ_PATH,
        **arrays_out,
        metadata_json=np.array(json.dumps(meta_out)),
    )
    print("=" * 70)
    print(f"[STEP 1 COMPLETE] Mode: {meta_out['input_mode']}")
    print(f"Grid Shape (with halo): {arrays_out['C_pad'].shape} | Core: {meta_out['global_core_size']}x{meta_out['global_core_size']} px")
    print(f"Ground Spacing        : dx = {meta_out['dx_ground_m']} m, dy = {meta_out['dy_ground_m']} m")
    print(f"Reference Roughness   : R_ref = {meta_out['R_ref']} m")
    print(f"Impassable Cliffs     : {meta_out['impassable_cliff_pct']}% (> {MAX_PASSABLE_SLOPE_DEG} deg)")
    print(f"Finite Cost C(r,c)    : [{meta_out['finite_cost_min']}, {meta_out['finite_cost_max']}]")
    print(f"Saved Output for Step2: {STEP1_OUTPUT_NPZ_PATH}")
    print("=" * 70)