#!/usr/bin/env python3
"""
mars_locator.py - find Perseverance (Mars 2020) raw images taken near a given
longitude/latitude in the Jezero Crater region, and return image URLs and metadata.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

log = logging.getLogger("mars_locator")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MARS_RADIUS_M = 3_389_500.0  # IAU mean radius

WAYPOINTS_URL = "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json"
RAW_IMAGES_API = "https://mars.nasa.gov/rss/api/"
RAW_IMAGES_PAGE = "https://mars.nasa.gov/mars2020/multimedia/raw-images/"

JEZERO_BBOX = (77.0, 17.9, 78.1, 18.9)

DEFAULT_CAMERAS = (
    "MCZ_LEFT", "MCZ_RIGHT",
    "NAVCAM_LEFT", "NAVCAM_RIGHT",
    "FRONT_HAZCAM_LEFT_A", "FRONT_HAZCAM_RIGHT_A",
    "REAR_HAZCAM_LEFT", "REAR_HAZCAM_RIGHT",
)

CAMERA_PREFERENCE = {
    "MCZ_LEFT": 0, "MCZ_RIGHT": 0,
    "NAVCAM_LEFT": 1, "NAVCAM_RIGHT": 1,
    "FRONT_HAZCAM_LEFT_A": 2, "FRONT_HAZCAM_RIGHT_A": 2,
    "FRONT_HAZCAM_LEFT_B": 2, "FRONT_HAZCAM_RIGHT_B": 2,
    "REAR_HAZCAM_LEFT": 2, "REAR_HAZCAM_RIGHT": 2,
}
SAMPLE_TYPE_PREFERENCE = {"Full": 0, "Downsampled": 1, "Subframe": 2, "Thumbnail": 3}

PAGE_SIZE = 100
LAST_WAYPOINT_LOOKAHEAD_SOLS = 30
MAX_FALLBACK_SOLS = 40
RETRY_STATUS = {429, 500, 502, 503, 504}
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache_mars_locator")

IMAGE_SIZES = ("full_res", "large", "medium", "small")


class NasaDataError(RuntimeError):
    """NASA endpoint unreachable, or its response is not in the expected shape."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres on a spherical Mars."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * MARS_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def validate_coordinates(lon: float, lat: float, enforce_region: bool = True) -> None:
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError("longitude and latitude must be finite numbers")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude {lat} is outside [-90, 90]")
    if not -180.0 <= lon <= 360.0:
        raise ValueError(f"longitude {lon} is outside [-180, 360]")
    if enforce_region:
        lon_e = lon % 360.0
        lon_min, lat_min, lon_max, lat_max = JEZERO_BBOX
        if not (lon_min <= lon_e <= lon_max and lat_min <= lat <= lat_max):
            raise ValueError(
                f"({lon}, {lat}) is outside the Jezero region guard box "
                f"lon {lon_min}..{lon_max}E, lat {lat_min}..{lat_max}N"
            )


# --------------------------------------------------------------------------- #
# HTTP Session (Supports requests with standard library urllib fallback)
# --------------------------------------------------------------------------- #

class StandardUrllibSession:
    """Lightweight requests-compatible session using urllib."""
    def __init__(self):
        self.headers = {
            "User-Agent": "mars-locator/1.0 (+NASA Space Apps Challenge; Python)",
            "Accept": "application/json",
        }

    def get(self, url: str, params: Optional[dict] = None, timeout: float = 30.0):
        import urllib.request
        import urllib.parse
        full_url = url
        if params:
            query_str = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_str}"
        req = urllib.request.Request(full_url, headers=self.headers)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            class RespWrapper:
                def __init__(self, r):
                    self.status_code = r.status
                    self._data = r.read()
                def json(self):
                    return json.loads(self._data.decode("utf-8"))
            return RespWrapper(resp)
        except urllib.error.HTTPError as e:
            class ErrWrapper:
                def __init__(self, err):
                    self.status_code = err.code
                def json(self):
                    return {}
            return ErrWrapper(e)
        except Exception as e:
            raise NasaDataError(f"Network error accessing {url}: {e}")

def make_session():
    try:
        import requests
        s = requests.Session()
        s.headers.update({
            "User-Agent": "mars-locator/1.0 (+NASA Space Apps; python-requests)",
            "Accept": "application/json",
        })
        return s
    except ImportError:
        return StandardUrllibSession()


def _get_json(session, url: str, params: Optional[dict] = None, *,
              timeout: float = 20.0, retries: int = 3) -> Any:
    last_err: Any = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except Exception as exc:
            last_err = exc
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise NasaDataError(f"{url} did not return valid JSON") from exc
            if resp.status_code in RETRY_STATUS:
                last_err = f"HTTP {resp.status_code}"
            else:
                raise NasaDataError(f"HTTP {resp.status_code} from {url}")
        if attempt < retries - 1:
            time.sleep(1.0 * 2 ** attempt)
    raise NasaDataError(f"Could not fetch {url}: {last_err}")


# --------------------------------------------------------------------------- #
# Rover waypoints
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Waypoint:
    site: int
    drive: int
    sol: int
    lon: float
    lat: float
    note: str = ""

    @property
    def key(self) -> tuple[int, int]:
        return (self.site, self.drive)


def parse_waypoints(data: Any) -> list[Waypoint]:
    feats = data.get("features") if isinstance(data, dict) else None
    if not feats:
        raise NasaDataError("Waypoint file has no 'features'.")
    out: list[Waypoint] = []
    for f in feats:
        props = f.get("properties") or {}
        lon, lat = props.get("lon"), props.get("lat")
        if lon is None or lat is None:
            coords = (f.get("geometry") or {}).get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
        try:
            out.append(Waypoint(int(props["site"]), int(props["drive"]),
                                int(props["sol"]), float(lon), float(lat),
                                str(props.get("Note") or "")))
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise NasaDataError("No usable waypoints found.")
    return out


def load_waypoints(session, cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
                   ttl_s: float = 86400.0) -> list[Waypoint]:
    cache_file = Path(cache_dir).expanduser() / "M20_waypoints.json" if cache_dir else None
    data = None
    if cache_file and cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl_s:
        try:
            data = json.loads(cache_file.read_text("utf-8"))
        except (OSError, ValueError):
            pass
    if data is None:
        try:
            data = _get_json(session, WAYPOINTS_URL)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data), "utf-8")
        except Exception as e:
            if cache_file and cache_file.exists():
                data = json.loads(cache_file.read_text("utf-8"))
            else:
                raise e
    return parse_waypoints(data)


def _note_rank(w: Waypoint) -> int:
    n = w.note.lower()
    return 2 if n.startswith("interpolated") else 1 if n.startswith("mid-drive") else 0


class WaypointIndex:
    def __init__(self, waypoints: Sequence[Waypoint]):
        self.sorted = sorted(waypoints, key=lambda w: (w.site, w.drive, w.sol))
        self.keys = [w.key for w in self.sorted]
        self.exact: dict[tuple[int, int], Waypoint] = {}
        for w in self.sorted:
            cur = self.exact.get(w.key)
            if cur is None or _note_rank(w) <= _note_rank(cur):
                self.exact[w.key] = w
        self.pos = {id(w): i for i, w in enumerate(self.sorted)}

    def next_after(self, wp: Waypoint) -> Optional[Waypoint]:
        i = self.pos[id(wp)] + 1
        return self.sorted[i] if i < len(self.sorted) else None

    def for_image(self, site_drive: tuple[int, int]) -> Optional[tuple[Waypoint, str]]:
        wp = self.exact.get(site_drive)
        if wp is not None:
            return wp, "exact_site_drive"
        i = bisect.bisect_right(self.keys, site_drive) - 1
        if i >= 0:
            return self.sorted[i], "preceding_waypoint"
        return None


# --------------------------------------------------------------------------- #
# Raw image feed
# --------------------------------------------------------------------------- #

class _FilterIgnored(Exception):
    pass


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _query_params(extra: dict, cameras: Optional[Sequence[str]]) -> dict:
    p = {"feed": "raw_images", "category": "mars2020", "feedtype": "json",
         "ver": "1.2", "order": "sol asc"}
    p.update(extra)
    if cameras:
        p["search"] = "|".join(cameras)
    return p


def _paged(session, base: dict, max_pages: int,
           bounds: Optional[tuple[int, int]] = None) -> tuple[list[dict], bool]:
    images: list[dict] = []
    for page in range(max_pages):
        data = _get_json(session, RAW_IMAGES_API, dict(base, num=PAGE_SIZE, page=page))
        batch = data.get("images") if isinstance(data, dict) else None
        if batch is None:
            raise NasaDataError("Raw-image response has no 'images' list.")
        if bounds and page == 0:
            for img in batch:
                s = _as_int(img.get("sol"))
                if s is not None and not bounds[0] <= s <= bounds[1]:
                    raise _FilterIgnored()
        images.extend(batch)
        total = _as_int(data.get("total_results")) or 0
        if not batch or len(images) >= total:
            return images, False
    return images, True


def fetch_images(session, sol_min: int, sol_max: int,
                 cameras: Optional[Sequence[str]], max_pages: int) -> tuple[list[dict], bool]:
    base = _query_params({"condition_2": f"{sol_min}:sol:gte",
                          "condition_3": f"{sol_max}:sol:lte"}, cameras)
    try:
        return _paged(session, base, max_pages, bounds=(sol_min, sol_max))
    except _FilterIgnored:
        pass
    images: list[dict] = []
    truncated = sol_max - sol_min + 1 > MAX_FALLBACK_SOLS
    for sol in range(sol_min, min(sol_max, sol_min + MAX_FALLBACK_SOLS - 1) + 1):
        try:
            batch, trunc = _paged(session, _query_params({"sol": sol}, cameras), max_pages)
            images.extend(batch)
            truncated = truncated or trunc
        except Exception:
            continue
    return images, truncated


def _pixels(img: dict) -> int:
    m = re.findall(r"\d+", str((img.get("extended") or {}).get("dimension", "")))
    return int(m[0]) * int(m[1]) if len(m) >= 2 else 0


def _quality_key(img: dict, match_type: str) -> tuple:
    inst = (img.get("camera") or {}).get("instrument", "")
    return (
        0 if match_type == "exact_site_drive" else 1,
        CAMERA_PREFERENCE.get(inst, 9),
        SAMPLE_TYPE_PREFERENCE.get(img.get("sample_type"), 9),
        -_pixels(img),
        str(img.get("imageid", "")),
    )


def _pick_url(files: dict, size: str = "large") -> Optional[str]:
    order = [size] + [s for s in IMAGE_SIZES if s != size]
    return next((files[s] for s in order if files.get(s)), None)


def _build_match(img: dict, match_type: str, wp: Waypoint, dist_m: float,
                 query: dict, size: str, diagnostics: dict) -> dict:
    files = img.get("image_files") or {}
    cam = img.get("camera") or {}
    ext = img.get("extended") or {}
    image_id = img.get("imageid", "")
    return {
        "status": "match",
        "query": query,
        "image_url": _pick_url(files, size),
        "image_urls": {s: files.get(s) for s in IMAGE_SIZES},
        "image_id": image_id,
        "title": img.get("title") or f"Perseverance Sol {img.get('sol')}: {cam.get('instrument', 'Rover Camera')}",
        "source_url": img.get("link") or f"{RAW_IMAGES_PAGE}{image_id}",
        "capture": {
            "sol": img.get("sol"),
            "date_taken_utc": img.get("date_taken_utc"),
            "date_taken_mars": img.get("date_taken_mars"),
        },
        "camera": {
            "instrument": cam.get("instrument"),
            "title": img.get("title"),
            "filter_name": cam.get("filter_name"),
            "dimension": ext.get("dimension"),
        },
        "rover_position": {
            "lon": wp.lon, "lat": wp.lat, "site": wp.site, "drive": wp.drive,
            "waypoint_sol": wp.sol,
            "position_match": match_type,
            "image_site": _as_int(img.get("site")),
            "image_drive": _as_int(img.get("drive")),
            "waypoint_note": wp.note or None,
        },
        "distance_m": round(dist_m, 1),
        "diagnostics": diagnostics,
    }


def find_image(lon: float, lat: float, *,
               max_distance_m: float = 1500.0,
               cameras: Optional[Sequence[str]] = DEFAULT_CAMERAS,
               size: str = "large",
               max_candidates: int = 6,
               max_pages: int = 5,
               enforce_region: bool = True,
               session: Optional[Any] = None,
               cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
               cache_ttl_s: float = 86400.0) -> dict:
    validate_coordinates(lon, lat, enforce_region)
    session = session or make_session()
    query = {"lon": lon, "lat": lat, "max_distance_m": max_distance_m}

    index = WaypointIndex(load_waypoints(session, cache_dir, cache_ttl_s))
    ranked = sorted(((haversine_m(lon, lat, w.lon, w.lat), w) for w in index.sorted),
                    key=lambda t: t[0])
    nearest_d, nearest_w = ranked[0]
    in_range = [(d, w) for d, w in ranked if d <= max_distance_m]
    candidates = in_range[:max_candidates]

    if not candidates:
        return {"status": "no_match", "query": query,
                "reason": f"No rover waypoint within {max_distance_m:g} m.",
                "nearest_waypoint": {"distance_m": round(nearest_d, 1),
                                     "lon": nearest_w.lon, "lat": nearest_w.lat,
                                     "sol": nearest_w.sol}}

    diag = {"waypoints_in_range": len(in_range), "candidates_checked": 0, "images_examined": 0}
    wanted = set(cameras) if cameras else None

    for dist, wp in candidates:
        diag["candidates_checked"] += 1
        nxt = index.next_after(wp)
        sol_max = nxt.sol if nxt and nxt.sol >= wp.sol else wp.sol + 10
        images, _ = fetch_images(session, wp.sol, sol_max, cameras, max_pages)
        diag["images_examined"] += len(images)

        matches = []
        for img in images:
            inst = (img.get("camera") or {}).get("instrument", "")
            if inst.startswith("HELI_") or (wanted and inst not in wanted):
                continue
            site, drive = _as_int(img.get("site")), _as_int(img.get("drive"))
            if site is None or drive is None:
                continue
            assigned = index.for_image((site, drive))
            if assigned and assigned[0] is wp and (img.get("image_files") or {}):
                matches.append((img, assigned[1]))
        if matches:
            img, mtype = min(matches, key=lambda m: _quality_key(*m))
            return _build_match(img, mtype, wp, dist, query, size, diag)

    return {"status": "no_match", "query": query,
            "reason": "Rover waypoints exist within range, but no matching images found."}


# --------------------------------------------------------------------------- #
# Enhanced Multi-Image Search for Interactive Carousel Gallery
# --------------------------------------------------------------------------- #

# In-memory LRU-style cache for fast repeated queries
_NEARBY_CACHE = {}

def find_nearby_images(lon: float, lat: float, count: int = 8, max_distance_m: float = 6000.0) -> list[dict]:
    """
    Returns up to `count` nearby Mars images dynamically ranked by true proximity to (lon, lat).
    Uses the full M20 waypoints traverse dataset and the verified Mars 2020 raw image database
    to provide location-specific, working photos anywhere in Jezero Crater.
    """
    cache_key = (round(lon, 4), round(lat, 4), count)
    if cache_key in _NEARBY_CACHE:
        return _NEARBY_CACHE[cache_key]

    results: list[dict] = []
    seen_ids = set()
    seen_urls = set()

    session = make_session()

    # Load local images_by_sol database if present
    sol_images_db = {}
    cache_db_path = os.path.join(DEFAULT_CACHE_DIR, "images_by_sol.json")
    if os.path.exists(cache_db_path):
        try:
            with open(cache_db_path, "r", encoding="utf-8") as f:
                sol_images_db = json.load(f)
        except Exception as e:
            log.warning("Could not read images_by_sol.json: %s", e)

    # 1. Rank all rover waypoints by geographic distance to the requested (lon, lat)
    try:
        waypoints = load_waypoints(session, DEFAULT_CACHE_DIR)
        ranked_wps = sorted(((haversine_m(lon, lat, w.lon, w.lat), w) for w in waypoints),
                            key=lambda t: t[0])
    except Exception as e:
        log.warning("Could not rank waypoints: %s", e)
        ranked_wps = []

    # 2. Collect photos from the closest waypoints first
    for dist_m, wp in ranked_wps:
        if dist_m > max_distance_m and len(results) >= count:
            break
        sol_key = str(wp.sol)
        wp_imgs = sol_images_db.get(sol_key)

        # If not cached locally, query online NASA raw images for this sol if we still need results
        if not wp_imgs and len(results) < count:
            try:
                fetched, _ = fetch_images(session, wp.sol, wp.sol, DEFAULT_CAMERAS, max_pages=2)
                if fetched:
                    wp_imgs = []
                    for f_img in fetched:
                        f_files = f_img.get("image_files") or {}
                        f_url = _pick_url(f_files, "large")
                        if not f_url: continue
                        f_cam = (f_img.get("camera") or {}).get("instrument", "Rover Camera")
                        wp_imgs.append({
                            "id": f_img.get("imageid", ""),
                            "sol": wp.sol,
                            "camera": f_cam,
                            "title": f_img.get("title") or f"Perseverance Sol {wp.sol}: {f_cam}",
                            "img_url": f_url,
                            "caption": f"Mars 2020 Raw Image on Sol {wp.sol} ({f_cam})",
                            "site": wp.site,
                            "drive": wp.drive,
                        })
                        if len(wp_imgs) >= 4:
                            break
                    sol_images_db[sol_key] = wp_imgs
            except Exception:
                pass

        if wp_imgs:
            for item in wp_imgs:
                img_id = item.get("id") or item.get("img_url")
                img_url = item.get("img_url")
                if not img_url or img_id in seen_ids or img_url in seen_urls:
                    continue
                seen_ids.add(img_id)
                seen_urls.add(img_url)

                cam_name = item.get("camera", "Rover Camera")
                results.append({
                    "id": img_id,
                    "title": item.get("title", f"Perseverance Sol {wp.sol}: {cam_name}"),
                    "camera": cam_name,
                    "sol": wp.sol,
                    "distance_m": round(dist_m, 1),
                    "img_url": img_url,
                    "proxy_url": f"/api/image-proxy?url={urllib.parse.quote(img_url, safe='')}",
                    "lon": wp.lon,
                    "lat": wp.lat,
                    "caption": f"Perseverance Sol {wp.sol} raw image by {cam_name} ({round(dist_m, 1)}m away)",
                    "usgs_code": "Jezero Crater Floor",
                    "source_url": f"{RAW_IMAGES_PAGE}{img_id}",
                })
                # Add at most 2 photos per waypoint so the gallery offers spatial diversity
                if len([r for r in results if r["sol"] == wp.sol]) >= 2:
                    break
        if len(results) >= count * 2:
            break

    # 3. Check mission bundle stations for any nearby verified stations
    possible_bundle_paths = [
        os.path.join(os.path.dirname(__file__), "frontend", "public", "data", "cesium_mars_mission_bundle.json"),
        os.path.join(os.path.dirname(__file__), "pipeline", "cesium_mars_mission_bundle.json"),
        os.path.join(os.path.dirname(__file__), "data", "cesium_mars_mission_bundle.json"),
    ]
    for b_path in possible_bundle_paths:
        if os.path.exists(b_path):
            try:
                with open(b_path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                stations = bundle.get("martian_streetview_rover_images", [])
                for st in stations:
                    st_lon = st.get("lon")
                    st_lat = st.get("lat")
                    if st_lon is not None and st_lat is not None:
                        d = haversine_m(lon, lat, st_lon, st_lat)
                        # Only add bundle station if within reasonable range
                        if d <= max_distance_m:
                            url = st.get("img_url") or st.get("Backup_Wikimedia_URL")
                            sid = st.get("id")
                            if url and sid not in seen_ids and url not in seen_urls:
                                seen_ids.add(sid)
                                seen_urls.add(url)
                                results.append({
                                    "id": sid,
                                    "title": st.get("title", f"Perseverance Station (Sol {st.get('sol')})"),
                                    "camera": st.get("camera", "Mastcam-Z"),
                                    "sol": st.get("sol", 0),
                                    "distance_m": round(d, 1),
                                    "img_url": url,
                                    "proxy_url": f"/api/image-proxy?url={urllib.parse.quote(url, safe='')}",
                                    "backup_url": st.get("Backup_Wikimedia_URL"),
                                    "lon": st_lon,
                                    "lat": st_lat,
                                    "alt_m": st.get("alt_m", -2540.0),
                                    "caption": st.get("caption", "Martian outcrop topography in Jezero Crater"),
                                    "usgs_code": st.get("usgs_code", "N/A"),
                                    "source_url": f"{RAW_IMAGES_PAGE}?id={sid}",
                                })
                break
            except Exception as e:
                log.warning("Could not load local mission bundle rover stations: %s", e)

    # Sort strictly by distance to the user's chosen location
    results.sort(key=lambda x: x["distance_m"])
    final_res = results[:count]
    _NEARBY_CACHE[cache_key] = final_res
    return final_res


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ln, lt = float(sys.argv[1]), float(sys.argv[2])
    else:
        ln, lt = 77.4526, 18.4449
    print(f"Searching nearby Mars images for ({ln}, {lt})...")
    res = find_nearby_images(ln, lt, count=6)
    print(json.dumps(res, indent=2))
