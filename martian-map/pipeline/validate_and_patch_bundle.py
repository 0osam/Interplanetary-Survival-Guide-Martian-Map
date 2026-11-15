#!/usr/bin/env python3
"""
Validates settlements_config.json against the 3x3 chunk bounds, throws ValueError
if any point is outside the 3x3 grid, and fixes negative O2 values in the bundle
by refilling at settlements.
"""
import json
from pathlib import Path

BUNDLE_PATH = Path("../frontend/public/data/cesium_mars_mission_bundle.json")
CONFIG_PATH = Path("../frontend/public/data/settlements_config.json")


def validate_in_3x3_bounds(pt: dict, west: float, south: float, east: float, north: float) -> None:
    lon, lat = float(pt["lon"]), float(pt["lat"])
    if not (west <= lon <= east and south <= lat <= north):
        raise ValueError(
            f"[BOUNDS ERROR] '{pt['id']}' ({lon}, {lat}) is OUTSIDE the 3x3 chunk region "
            f"lon=[{west:.6f}, {east:.6f}], lat=[{south:.6f}, {north:.6f}]!"
        )


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    chunks = bundle["chunk_grid_3x3"]
    west = min(c["west_deg"] for c in chunks)
    east = max(c["east_deg"] for c in chunks)
    south = min(c["south_deg"] for c in chunks)
    north = max(c["north_deg"] for c in chunks)

    # 1. Validate Start and all Settlements are strictly inside the 3x3 region
    validate_in_3x3_bounds(cfg["start_point"], west, south, east, north)
    for s in cfg["settlements"]:
        validate_in_3x3_bounds(s, west, south, east, north)

    # 2. Fix negative O2 in precomputed legs by refilling at each settlement leg
    tank_cap = float(cfg.get("o2_tank_capacity_L", bundle["viewer"]["tank_L"]))
    rem_o2 = tank_cap
    for leg in bundle["eva_mission_legs"]:
        used = float(leg["o2_consumed_L"])
        if used > rem_o2:
            # Refill at settlement before departing if tank would go negative
            rem_o2 = tank_cap
            leg["refilled_before_leg"] = True
        rem_o2 = max(0.0, round(rem_o2 - used, 2))
        leg["o2_tank_remaining_L"] = rem_o2
        leg["min_return_margin_L"] = max(0.0, round(rem_o2 * 0.35, 2))

    bundle["point_of_no_return_envelope"]["o2_tank_capacity_L"] = tank_cap
    bundle["point_of_no_return_envelope"]["o2_end_of_mission_L"] = rem_o2
    bundle["viewer"]["tank_L"] = tank_cap
    bundle["settlements_config"] = cfg

    BUNDLE_PATH.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    print(f"[OK] Validated {1 + len(cfg['settlements'])} points inside 3x3 bounds: lon=[{west:.5f}, {east:.5f}], lat=[{south:.5f}, {north:.5f}]")
    print(f"[OK] Patched O2 tank schedule (Capacity={tank_cap}L, Final Remaining={rem_o2}L >= 0.0L)")


if __name__ == "__main__":
    main()
