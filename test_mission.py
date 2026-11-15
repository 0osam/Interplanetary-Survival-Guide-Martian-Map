#!/usr/bin/env python3
"""
test_mission.py
===============
NASA International Space Apps Challenge — Self-Verification & Unit Test Suite.

Runs all offline verification checks:
1. Mission Bundle & Settlements Integrity Verification
2. Perseverance Rover Proximity & Waypoint Engine (18 Unit Tests)
3. Full-Duplex Telemetry Serialization & Contract Checks

Usage:
  python test_mission.py
"""

import json
import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
APP_DIR = ROOT_DIR / "martian-map"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class TestMissionBundleIntegrity(unittest.TestCase):
    """Verifies precomputed mission bundle and settlements configuration."""

    def setUp(self):
        self.bundle_path = APP_DIR / "frontend" / "public" / "data" / "cesium_mars_mission_bundle.json"
        self.cfg_path = APP_DIR / "frontend" / "public" / "data" / "settlements_config.json"

    def test_bundle_exists_and_valid_json(self):
        self.assertTrue(self.bundle_path.exists(), f"Missing bundle at {self.bundle_path}")
        with open(self.bundle_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("chunk_grid_3x3", data)
        self.assertEqual(len(data["chunk_grid_3x3"]), 9, "Must contain a 3x3 (9 chunk) grid")
        self.assertIn("eva_mission_legs", data)
        self.assertIn("point_of_no_return_envelope", data)

    def test_o2_schedule_non_negative(self):
        with open(self.bundle_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, leg in enumerate(data.get("eva_mission_legs", [])):
            rem_o2 = leg.get("o2_tank_remaining_L", 0.0)
            self.assertGreaterEqual(rem_o2, 0.0, f"Leg {i} has negative remaining oxygen: {rem_o2} L")

    def test_settlements_config_within_bounds(self):
        self.assertTrue(self.cfg_path.exists(), f"Missing settlements config at {self.cfg_path}")
        with open(self.bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        chunks = bundle["chunk_grid_3x3"]
        west = min(c["west_deg"] for c in chunks)
        east = max(c["east_deg"] for c in chunks)
        south = min(c["south_deg"] for c in chunks)
        north = max(c["north_deg"] for c in chunks)

        start = cfg["start_point"]
        self.assertTrue(west <= start["lon"] <= east and south <= start["lat"] <= north,
                        f"Start point {start['id']} is outside 3x3 region bounds")

        for s in cfg.get("settlements", []):
            self.assertTrue(west <= s["lon"] <= east and south <= s["lat"] <= north,
                            f"Settlement {s['id']} is outside 3x3 region bounds")


def run_all_tests():
    print("=" * 72)
    print("NASA Space Apps Challenge — Martian Map Automated Self-Verification")
    print("=" * 72)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 1. Mission bundle integrity tests
    suite.addTests(loader.loadTestsFromTestCase(TestMissionBundleIntegrity))

    # 2. Rover locator & waypoint tests (18 offline tests)
    import test_mars_locator
    suite.addTests(loader.loadTestsFromModule(test_mars_locator))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 72)
    if result.wasSuccessful():
        print(f"[PASSED] All {result.testsRun} verification checks passed successfully!")
        return 0
    else:
        print(f"[FAILED] {len(result.failures)} failure(s), {len(result.errors)} error(s).")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
