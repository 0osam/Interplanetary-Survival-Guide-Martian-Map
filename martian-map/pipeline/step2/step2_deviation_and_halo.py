#!/usr/bin/env python3
"""
step2_deviation_and_halo.py
===========================
Step 2: Exact Nearest-Passable EDT Sanitization (C_safe), Global Shift-Invariant
Normalization (S_grad, S_lap), and Halo-Protected 3x3 Chunk Deviation Score D^(I,J)(r,c).
Reads all constants from .env and outputs step2_output.npz for Step 3.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, laplace, sobel


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
GRID_CHUNKS_SIDE        = int(os.environ["GRID_CHUNKS_SIDE"])
CHUNK_SIZE_PX           = int(os.environ["CHUNK_SIZE_PX"])
GLOBAL_HALO_PX          = int(os.environ["GLOBAL_HALO_PX"])
CENTER_LON_DEG          = float(os.environ["CENTER_LON_DEG"])
CENTER_LAT_DEG          = float(os.environ["CENTER_LAT_DEG"])
MAP_PIXEL_SIZE_X_M      = float(os.environ["MAP_PIXEL_SIZE_X_M"])
MAP_PIXEL_SIZE_Y_M      = float(os.environ["MAP_PIXEL_SIZE_Y_M"])
DUMMY_BASE_ALT_M        = float(os.environ["DUMMY_BASE_ALT_M"])
DUMMY_REGIONAL_SLOPE_M  = float(os.environ["DUMMY_REGIONAL_SLOPE_M"])
ZERO_FLOAT              = float(os.environ["ZERO_FLOAT"])
TWO_FLOAT               = float(os.environ["TWO_FLOAT"])

# --- Step 2 File Paths ---
STEP2_INPUT_NPZ_PATH    = os.environ["STEP2_INPUT_NPZ_PATH"]
STEP2_OUTPUT_NPZ_PATH   = os.environ["STEP2_OUTPUT_NPZ_PATH"]

# --- Step 2 Stencil & Normalization Constants ---
STEP2_STENCIL_HALO_PX   = int(os.environ["STEP2_STENCIL_HALO_PX"])
STEP2_SOBEL_AXIS_ROW    = int(os.environ["STEP2_SOBEL_AXIS_ROW"])
STEP2_SOBEL_AXIS_COL    = int(os.environ["STEP2_SOBEL_AXIS_COL"])
STEP2_GRAD_PERCENTILE_Q = float(os.environ["STEP2_GRAD_PERCENTILE_Q"])
STEP2_LAP_PERCENTILE_Q  = float(os.environ["STEP2_LAP_PERCENTILE_Q"])
STEP2_NORM_MIN_FLOOR    = float(os.environ["STEP2_NORM_MIN_FLOOR"])
STEP2_CLIP_MIN          = float(os.environ["STEP2_CLIP_MIN"])
STEP2_CLIP_MAX          = float(os.environ["STEP2_CLIP_MAX"])

# --- Step 2 Deviation Score Weights ---
STEP2_WEIGHT_GRAD       = float(os.environ["STEP2_WEIGHT_GRAD"])
STEP2_WEIGHT_LAP        = float(os.environ["STEP2_WEIGHT_LAP"])
STEP2_WEIGHT_UNC        = float(os.environ["STEP2_WEIGHT_UNC"])

# --- Step 2 Dummy Input Generator Constants ---
STEP2_DUMMY_RNG_SEED       = int(os.environ["STEP2_DUMMY_RNG_SEED"])
STEP2_DUMMY_CLIFF_COL_MIN  = float(os.environ["STEP2_DUMMY_CLIFF_COL_MIN"])
STEP2_DUMMY_CLIFF_COL_MAX  = float(os.environ["STEP2_DUMMY_CLIFF_COL_MAX"])
STEP2_DUMMY_GAP_ROW_MIN    = float(os.environ["STEP2_DUMMY_GAP_ROW_MIN"])
STEP2_DUMMY_GAP_ROW_MAX    = float(os.environ["STEP2_DUMMY_GAP_ROW_MAX"])
STEP2_DUMMY_BASE_COST      = float(os.environ["STEP2_DUMMY_BASE_COST"])
STEP2_DUMMY_SLOPE_COST_AMP = float(os.environ["STEP2_DUMMY_SLOPE_COST_AMP"])
STEP2_DUMMY_NOISE_COST_STD = float(os.environ["STEP2_DUMMY_NOISE_COST_STD"])
STEP2_DUMMY_ETA_BASE       = float(os.environ["STEP2_DUMMY_ETA_BASE"])
STEP2_DUMMY_UNC_BASE       = float(os.environ["STEP2_DUMMY_UNC_BASE"])
STEP2_DUMMY_UNC_BAND_AMP   = float(os.environ["STEP2_DUMMY_UNC_BAND_AMP"])


# =========================================================================
# 1. INPUT LOADERS: (A) REAL INPUT [ACTIVE] vs. (B) DUMMY INPUT [COMMENTED]
# =========================================================================

# -------------------------------------------------------------------------
# (A) REAL INPUT LOADER (Active: reads step1_output.npz)
# -------------------------------------------------------------------------
def load_step2_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads the precomputed Step 1 bundle from step1_output.npz."""
    bundle = np.load(STEP2_INPUT_NPZ_PATH, allow_pickle=True)
    meta = json.loads(str(bundle["metadata_json"]))
    arrays = {
        k: bundle[k]
        for k in bundle.files
        if k != "metadata_json"
    }
    meta["step2_input_mode"] = f"REAL_STEP1_NPZ ({STEP2_INPUT_NPZ_PATH})"
    return arrays, meta


# -------------------------------------------------------------------------
# (B) DUMMY INPUT GENERATOR (Commented out; use only for standalone testing)
# -------------------------------------------------------------------------
# def load_step2_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
#     """Generates synthetic Step 1 cost, passability, uncertainty, and elevation arrays."""
#     G_core = GRID_CHUNKS_SIDE * CHUNK_SIZE_PX
#     G_pad = G_core + int(TWO_FLOAT) * GLOBAL_HALO_PX
#     rng = np.random.default_rng(STEP2_DUMMY_RNG_SEED)
#
#     r_idx, c_idx = np.indices((G_pad, G_pad), dtype=np.float64)
#     r_norm = (r_idx - GLOBAL_HALO_PX) / G_core
#     c_norm = (c_idx - GLOBAL_HALO_PX) / G_core
#
#     # Passability mask with a vertical cliff band pierced by a central saddle gap
#     cliff_band = (c_norm >= STEP2_DUMMY_CLIFF_COL_MIN) & (c_norm <= STEP2_DUMMY_CLIFF_COL_MAX)
#     saddle_gap = (r_norm >= STEP2_DUMMY_GAP_ROW_MIN) & (r_norm <= STEP2_DUMMY_GAP_ROW_MAX)
#     pass_pad = ~(cliff_band & (~saddle_gap))
#
#     # Synthetic elevation z_pad, friction eta_pad, uncertainty U_pad, and raw cost C_pad
#     z_pad = DUMMY_BASE_ALT_M + DUMMY_REGIONAL_SLOPE_M * r_norm
#     eta_pad = np.full((G_pad, G_pad), STEP2_DUMMY_ETA_BASE, dtype=np.float32)
#     U_pad = np.clip(
#         STEP2_DUMMY_UNC_BASE + STEP2_DUMMY_UNC_BAND_AMP * np.sin(math.pi * c_norm) ** TWO_FLOAT,
#         STEP2_CLIP_MIN,
#         STEP2_CLIP_MAX,
#     ).astype(np.float32)
#
#     C_pad = np.full((G_pad, G_pad), np.inf, dtype=np.float64)
#     C_finite = (
#         STEP2_DUMMY_BASE_COST
#         + STEP2_DUMMY_SLOPE_COST_AMP * np.sin( TWO_FLOAT * math.pi * r_norm ) ** TWO_FLOAT
#         + rng.normal(ZERO_FLOAT, STEP2_DUMMY_NOISE_COST_STD, size=(G_pad, G_pad))
#     )
#     C_pad[pass_pad] = np.maximum(STEP2_DUMMY_BASE_COST, C_finite[pass_pad])
#
#     dx_m = MAP_PIXEL_SIZE_X_M * math.cos(math.radians(CENTER_LAT_DEG))
#     dy_m = MAP_PIXEL_SIZE_Y_M
#
#     arrays = {
#         "z_pad": z_pad.astype(np.float32),
#         "eta_pad": eta_pad,
#         "U_pad": U_pad,
#         "pass_pad": pass_pad,
#         "C_pad": C_pad.astype(np.float32),
#     }
#     meta = {
#         "step": 1,
#         "step2_input_mode": "DUMMY_SYNTHETIC_STEP1_INPUT",
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
# 2. STEP 2 CORE FUNCTIONS: EDT SANITIZATION & HALO-SLICED DEVIATION SCORE
# =========================================================================

def slice_chunk_with_stencil_halo(
    arr_pad: np.ndarray,
    I: int,
    J: int,
    L: int,
    h_global: int,
    h_stencil: int,
) -> np.ndarray:
    """Extracts an (L + 2*h_stencil) x (L + 2*h_stencil) window for chunk (I, J)."""
    r0 = h_global + I * L - h_stencil
    r1 = h_global + (I + 1) * L + h_stencil
    c0 = h_global + J * L - h_stencil
    c1 = h_global + (J + 1) * L + h_stencil
    return arr_pad[r0:r1, c0:c1]


def compute_chunk_deviation_score(
    C_safe_pad: np.ndarray,
    pass_pad: np.ndarray,
    U_pad: np.ndarray,
    I: int,
    J: int,
    L: int,
    h_global: int,
    S_grad: float,
    S_lap: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes shift-invariant Sobel gradient, Laplacian, and Deviation Score D^(I,J)(r,c)
    on Chunk (I,J) using a STEP2_STENCIL_HALO_PX border halo.
    """
    sh = STEP2_STENCIL_HALO_PX
    C_win = slice_chunk_with_stencil_halo(C_safe_pad, I, J, L, h_global, sh)
    pass_core = slice_chunk_with_stencil_halo(pass_pad, I, J, L, h_global, sh)[sh:-sh, sh:-sh]
    U_core = slice_chunk_with_stencil_halo(U_pad, I, J, L, h_global, sh)[sh:-sh, sh:-sh]

    grad_r = sobel(C_win, axis=STEP2_SOBEL_AXIS_ROW)
    grad_c = sobel(C_win, axis=STEP2_SOBEL_AXIS_COL)
    grad_core = np.hypot(grad_r, grad_c)[sh:-sh, sh:-sh]
    lap_core = np.abs(laplace(C_win))[sh:-sh, sh:-sh]

    norm_grad = np.clip(grad_core / S_grad, STEP2_CLIP_MIN, STEP2_CLIP_MAX)
    norm_lap = np.clip(lap_core / S_lap, STEP2_CLIP_MIN, STEP2_CLIP_MAX)

    D_core = np.where(
        pass_core,
        STEP2_WEIGHT_GRAD * norm_grad + STEP2_WEIGHT_LAP * norm_lap + STEP2_WEIGHT_UNC * U_core,
        ZERO_FLOAT,
    )
    return D_core.astype(np.float32), grad_core.astype(np.float32), lap_core.astype(np.float32)


def run_step2_pipeline() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Executes Step 2 sanitization, global scaling, and 3x3 chunk deviation scoring."""
    arrays, meta = load_step2_inputs()

    L = int(meta["chunk_size_L"])
    h_global = int(meta["halo_h"])
    n_side = int(meta["grid_chunks_side"])
    G_core = int(meta["global_core_size"])

    C_pad = arrays["C_pad"].astype(np.float64)
    pass_pad = arrays["pass_pad"].astype(bool)
    U_pad = arrays["U_pad"].astype(np.float64)

    # --- 2.1: Exact Nearest-Passable EDT Sanitization C_safe_pad ---
    if np.any(~pass_pad):
        _, nearest_idx = distance_transform_edt(~pass_pad, return_indices=True)
        C_safe_pad = C_pad[tuple(nearest_idx)]
    else:
        C_safe_pad = C_pad.copy()

    # --- 2.2: Global Shift-Invariant Normalizers (S_grad, S_lap) ---
    grad_full = np.hypot(
        sobel(C_safe_pad, axis=STEP2_SOBEL_AXIS_ROW),
        sobel(C_safe_pad, axis=STEP2_SOBEL_AXIS_COL),
    )
    lap_full = np.abs(laplace(C_safe_pad))

    pass_core_global = pass_pad[h_global:-h_global, h_global:-h_global]
    grad_core_ref = grad_full[h_global:-h_global, h_global:-h_global]
    lap_core_ref = lap_full[h_global:-h_global, h_global:-h_global]

    S_grad = max(float(np.percentile(grad_core_ref[pass_core_global], STEP2_GRAD_PERCENTILE_Q)), STEP2_NORM_MIN_FLOOR)
    S_lap = max(float(np.percentile(lap_core_ref[pass_core_global], STEP2_LAP_PERCENTILE_Q)), STEP2_NORM_MIN_FLOOR)

    # --- 2.3: Compute D^(I,J)(r,c) Chunk-by-Chunk Across the 3x3 Grid ---
    D_core_global = np.zeros((G_core, G_core), dtype=np.float32)
    grad_core_stitched = np.zeros((G_core, G_core), dtype=np.float32)
    lap_core_stitched = np.zeros((G_core, G_core), dtype=np.float32)

    for I in range(n_side):
        for J in range(n_side):
            D_ij, g_ij, l_ij = compute_chunk_deviation_score(
                C_safe_pad, pass_pad, U_pad, I, J, L, h_global, S_grad, S_lap
            )
            r0, r1 = I * L, (I + 1) * L
            c0, c1 = J * L, (J + 1) * L
            D_core_global[r0:r1, c0:c1] = D_ij
            grad_core_stitched[r0:r1, c0:c1] = g_ij
            lap_core_stitched[r0:r1, c0:c1] = l_ij

    # Verify exact shift-invariance between chunk-wise stencil and global stencil
    max_shift_err = float(np.max(np.abs(grad_core_stitched - grad_core_ref.astype(np.float32))))

    arrays_out = {
        **arrays,
        "C_safe_pad": C_safe_pad.astype(np.float64),
        "D_core_global": D_core_global,
        "grad_core_global": grad_core_stitched,
        "lap_core_global": lap_core_stitched,
    }

    meta_out = {
        **meta,
        "step": 2,
        "S_grad": round(S_grad, 4),
        "S_lap": round(S_lap, 4),
        "D_passable_min": round(float(D_core_global[pass_core_global].min()), 4),
        "D_passable_max": round(float(D_core_global[pass_core_global].max()), 4),
        "D_passable_mean": round(float(D_core_global[pass_core_global].mean()), 4),
        "shift_invariance_max_err": max_shift_err,
    }
    return arrays_out, meta_out


if __name__ == "__main__":
    arrays_res, meta_res = run_step2_pipeline()
    np.savez_compressed(
        STEP2_OUTPUT_NPZ_PATH,
        **arrays_res,
        metadata_json=np.array(json.dumps(meta_res)),
    )
    print("=" * 70)
    print(f"[STEP 2 COMPLETE] Mode: {meta_res['step2_input_mode']}")
    print(f"Sanitized C_safe Shape : {arrays_res['C_safe_pad'].shape} (NaN count = {int(np.isnan(arrays_res['C_safe_pad']).sum())})")
    print(f"Global Normalizers     : S_grad = {meta_res['S_grad']} | S_lap = {meta_res['S_lap']}")
    print(f"Deviation Score D(r,c) : min = {meta_res['D_passable_min']}, mean = {meta_res['D_passable_mean']}, max = {meta_res['D_passable_max']}")
    print(f"Shift-Invariance Error : {meta_res['shift_invariance_max_err']:.2e} (0.00e+00 = exact)")
    print(f"Saved Output for Step3 : {STEP2_OUTPUT_NPZ_PATH}")
    print("=" * 70)