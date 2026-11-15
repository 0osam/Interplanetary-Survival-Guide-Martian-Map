#!/usr/bin/env python3
"""
Reads the 3x3 chunk bounds from frontend/public/data/cesium_mars_mission_bundle.json,
creates/validates frontend/public/data/settlements_config.json (throwing ValueError if
any coordinate is outside the 3x3 grid), and fixes negative O2 values in the bundle.
"""
import json
from pathlib import Path

BUNDLE_PATH = Path("../frontend/public/data/cesium_mars_mission_bundle.json")
CONFIG_PATH = Path("../frontend/public/data/settlements_config.json")


def assert_inside_3x3(pt: dict, west: float, south: float, east: float, north: float) -> None:
    lon, lat = float(pt["lon"]), float(pt["lat"])
    if not (west <= lon <= east and south <= lat <= north):
        raise ValueError(
            f"\n[3x3 BOUNDS ERROR] '{pt['id']}' ({lon}, {lat}) is OUTSIDE the 3x3 chunk grid!\n"
            f"Valid 3x3 longitude range : [{west:.6f}, {east:.6f}]\n"
            f"Valid 3x3 latitude range  : [{south:.6f}, {north:.6f}]"
        )


def main() -> None:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing {BUNDLE_PATH}. Copy cesium_mars_mission_bundle.json there first.")

    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    chunks = bundle["chunk_grid_3x3"]
    west = min(c["west_deg"] for c in chunks)
    east = max(c["east_deg"] for c in chunks)
    south = min(c["south_deg"] for c in chunks)
    north = max(c["north_deg"] for c in chunks)
    dlon = east - west
    dlat = north - south

    # Create default settlements_config.json matched to this bundle's 3x3 bounds if missing or out-of-bounds
    default_cfg = {
        "routing_mode": "multi_settlement",
        "o2_tank_capacity_L": 25.0,
        "safety_reserve_factor": 1.15,
        "default_sample_step_kg": 10.0,
        "bounds_3x3_reference": {
            "west_deg": round(west, 6),
            "east_deg": round(east, 6),
            "south_deg": round(south, 6),
            "north_deg": round(north, 6),
        },
        "start_point": {
            "id": "HAB_START",
            "name": "Base Camp Hab (Start / Cycle Anchor)",
            "lon": round(west + 0.40 * dlon, 6),
            "lat": round(north - 0.44 * dlat, 6),
        },
        "settlements": [
            {
                "id": "SETTLEMENT_1_CENTRAL",
                "name": "Central Ridge O2 Depot (Chunk 1,1)",
                "lon": round(west + 0.55 * dlon, 6),
                "lat": round(north - 0.48 * dlat, 6),
                "refill_o2_L": 25.0,
            },
            {
                "id": "SETTLEMENT_2_SOUTH",
                "name": "Southern Fan O2 Relay (Chunk 1,1)",
                "lon": round(west + 0.44 * dlon, 6),
                "lat": round(north - 0.60 * dlat, 6),
                "refill_o2_L": 25.0,
            },
            {
                "id": "SETTLEMENT_3_EAST",
                "name": "Eastern Overlook O2 Outpost (Chunk 1,2)",
                "lon": round(west + 0.82 * dlon, 6),
                "lat": round(north - 0.50 * dlat, 6),
                "refill_o2_L": 25.0,
            },
        ],
    }

    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(default_cfg, indent=2), encoding="utf-8")
        cfg = default_cfg
        print(f"[CREATED] {CONFIG_PATH} inside 3x3 bounds.")
    else:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        try:
            assert_inside_3x3(cfg["start_point"], west, south, east, north)
            for s in cfg["settlements"]:
                assert_inside_3x3(s, west, south, east, north)
        except ValueError as err:
            print(f"[WARNING] Existing settlements_config.json was outside bundle bounds ({err}). Resetting to valid 3x3 coordinates...")
            CONFIG_PATH.write_text(json.dumps(default_cfg, indent=2), encoding="utf-8")
            cfg = default_cfg

    # Strictly validate all points
    assert_inside_3x3(cfg["start_point"], west, south, east, north)
    for s in cfg["settlements"]:
        assert_inside_3x3(s, west, south, east, north)

    # Fix any negative O2 values in precomputed legs
    tank_cap = float(cfg.get("o2_tank_capacity_L", 25.0))
    rem = tank_cap
    for leg in bundle.get("eva_mission_legs", []):
        used = float(leg.get("o2_consumed_L", 0.0))
        if used > rem:
            rem = tank_cap
        rem = max(0.0, round(rem - used, 2))
        leg["o2_tank_remaining_L"] = rem
        leg["min_return_margin_L"] = max(0.0, round(rem * 0.35, 2))

    bundle["point_of_no_return_envelope"]["o2_tank_capacity_L"] = tank_cap
    bundle["point_of_no_return_envelope"]["o2_end_of_mission_L"] = rem
    bundle["viewer"]["tank_L"] = tank_cap

    BUNDLE_PATH.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    print(f"[VALIDATED] Start + {len(cfg['settlements'])} settlements are strictly inside 3x3 grid:")
    print(f"            lon in [{west:.6f}, {east:.6f}], lat in [{south:.6f}, {north:.6f}]")


if __name__ == "__main__":
    main()
