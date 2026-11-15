#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
NASA Space Apps Challenge — Autonomous Martian EVA Path Planning Pipeline.

End-to-end execution of Steps 1 through 6:
  Step 1: Terrain Geometry, Roughness & Traversal Cost Field C(r,c)
  Step 2: Dual-Map Separation & Sobel/Laplacian Deviation Score D(r,c)
  Step 3: Adaptive Vertex Sampling (N=1,200) & KDTree Topological Graph
  Step 4: Mars Gravity-Adjusted Pandolf Bioenergetics & O2 Edge Weights
  Step 5: Directed A* Routing & Real-Time Point of No Return (PNR)
  Step 6: 3D Raycasting Viewshed, SPE Radiation & Cesium Mission Bundle Export

Usage:
  python run_full_pipeline.py
  python run_full_pipeline.py --clean-intermediate
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent.resolve()
STEPS = [
    ("Step 1: Terrain Cost Field", PIPELINE_DIR / "step1" / "step1_terrain_and_cost.py", PIPELINE_DIR / "step1"),
    ("Step 2: Deviation & Dual-Map", PIPELINE_DIR / "step2" / "step2_deviation_and_halo.py", PIPELINE_DIR / "step2"),
    ("Step 3: Sampling & Adaptive Graph", PIPELINE_DIR / "step3" / "step3_sampling_and_graph.py", PIPELINE_DIR / "step3"),
    ("Step 4: Bioenergetic Graph", PIPELINE_DIR / "step4" / "step4_bioenergetic_graph.py", PIPELINE_DIR / "step4"),
    ("Step 5: A* & Point of No Return", PIPELINE_DIR / "step5" / "step5_astar_and_pnr.py", PIPELINE_DIR / "step5"),
    ("Step 6: 3D Raycasting & Cesium Export", PIPELINE_DIR / "step6" / "step6_rover_los_and_cesium.py", PIPELINE_DIR / "step6"),
]


def run_pipeline(clean_intermediate: bool = True) -> bool:
    print("=" * 72)
    print("NASA Space Apps Challenge — Martian Map End-to-End Pipeline")
    print("=" * 72)

    total_start = time.time()
    python_exe = sys.executable

    for step_num, (name, script_path, work_dir) in enumerate(STEPS, start=1):
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            return False

        print(f"\n[{step_num}/6] Executing {name}...")
        t0 = time.time()
        res = subprocess.run([python_exe, str(script_path.name)], cwd=str(work_dir))
        dt = time.time() - t0

        if res.returncode != 0:
            print(f"[FAIL] {name} exited with error code {res.returncode}")
            return False
        print(f"[OK] {name} completed in {dt:.2f}s")

    # Run settlement setup & bundle validation
    print("\n[VALIDATION] Validating and Patching Settlements & Mission Bundle...")
    patch_script = PIPELINE_DIR / "validate_and_patch_bundle.py"
    if patch_script.exists():
        subprocess.run([python_exe, str(patch_script.name)], cwd=str(PIPELINE_DIR))

    # Clean intermediate .npz outputs to keep repo lightweight
    if clean_intermediate:
        print("\n[CLEANUP] Removing intermediate .npz pipeline files...")
        cleaned = 0
        for npz in PIPELINE_DIR.rglob("step*_output.npz"):
            try:
                npz.unlink()
                cleaned += 1
            except Exception as e:
                print(f"  Warning: could not delete {npz}: {e}")
        print(f"[CLEANUP] Removed {cleaned} intermediate .npz file(s).")

    total_time = time.time() - total_start
    print(f"\n[SUCCESS] Complete 6-Step Pipeline executed successfully in {total_time:.2f}s!")
    print(f"[READY] Deliverable available: frontend/public/data/cesium_mars_mission_bundle.json")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete 6-step Mars EVA pipeline")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate stepX_output.npz files")
    args = parser.parse_args()

    success = run_pipeline(clean_intermediate=not args.keep_intermediate)
    sys.exit(0 if success else 1)
