"""
Offline tests for mars_locator.py.
"""
import unittest
from unittest import mock

import mars_locator as ml


def feature(site, drive, sol, lon, lat):
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"site": site, "drive": drive, "sol": sol, "lon": lon, "lat": lat}}


WAYPOINTS = {"type": "FeatureCollection", "features": [
    feature(3, 0, 13, 77.4509, 18.4446),
    feature(3, 100, 20, 77.4520, 18.4450),
    feature(4, 0, 30, 77.4600, 18.4500),
]}


def image(imageid, sol, site, drive, instrument, sample="Full", dim="(1280,960)"):
    base = f"https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/{sol:05d}/ids/edr/browse/x/{imageid}"
    return {
        "imageid": imageid, "sol": sol, "site": site, "drive": str(drive),  # drive is a STRING in the feed
        "sample_type": sample, "title": f"Mars Perseverance Sol {sol}: {instrument}",
        "date_taken_utc": "2021-03-01T01:02:03.000", "date_taken_mars": f"Sol-{sol:05d}M12:00:00.000",
        "date_received": "2021-03-01T08:00:00Z",
        "link": f"https://mars.nasa.gov/mars2020/multimedia/raw-images/?id={imageid}",
        "json_link": f"https://mars.nasa.gov/rss/api/?id={imageid}",
        "camera": {"instrument": instrument, "filter_name": "UNK"},
        "extended": {"dimension": dim, "mastAz": "10.0", "mastEl": "5.0"},
        "image_files": {"full_res": base + ".png", "large": base + "_1200.jpg",
                        "medium": base + "_800.jpg", "small": base + "_320.jpg"},
    }


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class FakeNasa:
    """Serves waypoints and a raw-image list; honours condition_2/3, sol, search, num, page."""

    def __init__(self, images, honor_range=True):
        self.images, self.honor_range, self.calls = images, honor_range, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url == ml.WAYPOINTS_URL:
            return FakeResponse(WAYPOINTS)
        p = params
        imgs = sorted(self.images, key=lambda i: i["sol"])
        if "sol" in p:
            imgs = [i for i in imgs if i["sol"] == int(p["sol"])]
        if self.honor_range and "condition_2" in p:
            lo = int(p["condition_2"].split(":")[0])
            hi = int(p["condition_3"].split(":")[0])
            imgs = [i for i in imgs if lo <= i["sol"] <= hi]
        if p.get("search"):
            cams = p["search"].split("|")
            imgs = [i for i in imgs if i["camera"]["instrument"] in cams]
        n, pg = int(p["num"]), int(p["page"])
        return FakeResponse({"images": imgs[pg * n:(pg + 1) * n], "total_results": len(imgs),
                             "page": pg, "per_page": str(n)})


def run(images, lon, lat, **kw):
    kw.setdefault("cache_dir", None)
    return ml.find_image(lon, lat, session=FakeNasa(images, kw.pop("honor_range", True)), **kw)


class Geometry(unittest.TestCase):
    def test_one_degree_latitude(self):
        self.assertAlmostEqual(ml.haversine_m(77, 18, 77, 19), 59158, delta=5)

    def test_zero_distance(self):
        self.assertEqual(ml.haversine_m(77.4, 18.4, 77.4, 18.4), 0.0)


class Validation(unittest.TestCase):
    def test_outside_jezero_rejected(self):
        with self.assertRaises(ValueError):
            run([], 10.0, 40.0)

    def test_bad_latitude_rejected(self):
        with self.assertRaises(ValueError):
            run([], 77.4, 95.0, enforce_region=False)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            run([], float("nan"), 18.4)

    def test_region_override(self):
        r = run([], 10.0, 40.0, enforce_region=False)
        self.assertEqual(r["status"], "no_match")


class Matching(unittest.TestCase):
    def test_exact_match_prefers_mastcam_full_largest(self):
        imgs = [
            image("HAZ_FULL", 21, 3, 100, "FRONT_HAZCAM_LEFT_A"),
            image("MCZ_THUMB", 21, 3, 100, "MCZ_LEFT", sample="Thumbnail"),
            image("MCZ_SMALL", 21, 3, 100, "MCZ_LEFT", dim="(640,480)"),
            image("MCZ_BIG", 22, 3, 100, "MCZ_RIGHT", dim="(1648,1200)"),
        ]
        r = run(imgs, 77.4520, 18.4450)
        self.assertEqual(r["status"], "match")
        self.assertEqual(r["image_id"], "MCZ_BIG")
        self.assertEqual(r["rover_position"]["position_match"], "exact_site_drive")
        self.assertEqual(r["distance_m"], 0.0)
        self.assertTrue(r["image_url"].endswith("MCZ_BIG.png") or r["image_url"].endswith("MCZ_BIG_1200.jpg"))
        self.assertEqual(r["camera"]["instrument"], "MCZ_RIGHT")
        self.assertEqual(r["capture"]["sol"], 22)
        self.assertIn("raw-images/?id=MCZ_BIG", r["source_url"])

    def test_nearby_point_reports_distance(self):
        imgs = [image("A", 21, 3, 100, "NAVCAM_LEFT")]
        r = run(imgs, 77.4521, 18.4451)  # ~10-20 m from waypoint
        self.assertEqual(r["status"], "match")
        self.assertTrue(0 < r["distance_m"] < 30)

    def test_inferred_from_preceding_waypoint(self):
        imgs = [image("MID", 15, 3, 50, "NAVCAM_LEFT")]
        r = run(imgs, 77.4509, 18.4446)
        self.assertEqual(r["status"], "match")
        self.assertEqual(r["rover_position"]["position_match"], "preceding_waypoint")
        self.assertEqual((r["rover_position"]["site"], r["rover_position"]["drive"]), (3, 0))

    def test_exact_beats_inferred_at_same_waypoint(self):
        imgs = [image("INFER", 15, 3, 50, "MCZ_LEFT"), image("EXACT", 14, 3, 0, "NAVCAM_LEFT")]
        r = run(imgs, 77.4509, 18.4446)
        self.assertEqual(r["image_id"], "EXACT")

    def test_images_after_drive_belong_to_new_position(self):
        imgs = [image("LATE", 25, 3, 100, "NAVCAM_LEFT")]
        r = run(imgs, 77.4509, 18.4446, max_distance_m=50)
        self.assertEqual(r["status"], "no_match")

    def test_out_of_range(self):
        r = run([], 77.40, 18.40, max_distance_m=100)
        self.assertEqual(r["status"], "no_match")
        self.assertIn("nearest_waypoint", r)
        self.assertGreater(r["nearest_waypoint"]["distance_m"], 100)

    def test_helicopter_images_excluded(self):
        imgs = [image("HELI", 14, 3, 0, "HELI_NAV")]
        r = run(imgs, 77.4509, 18.4446, cameras=None)
        self.assertEqual(r["status"], "no_match")

    def test_size_selection(self):
        imgs = [image("A", 14, 3, 0, "NAVCAM_LEFT")]
        r = run(imgs, 77.4509, 18.4446, size="medium")
        self.assertTrue(r["image_url"].endswith("_800.jpg"))


class WaypointLabels(unittest.TestCase):
    def test_note_exposed_and_duplicate_key_prefers_end_of_drive(self):
        feats = [feature(3, 100, 20, 77.4520, 18.4450), feature(3, 100, 20, 77.4521, 18.4451)]
        feats[0]["properties"]["Note"] = "Interpolated"
        feats[1]["properties"]["Note"] = "End-of-drive localization"
        idx = ml.WaypointIndex(ml.parse_waypoints({"features": feats}))
        self.assertEqual(idx.exact[(3, 100)].note, "End-of-drive localization")


class SchemaErrors(unittest.TestCase):
    def test_bad_waypoint_file(self):
        with self.assertRaises(ml.NasaDataError):
            ml.parse_waypoints({"foo": "bar"})

    def test_geometry_fallback(self):
        wp = ml.parse_waypoints({"features": [{"geometry": {"coordinates": [77.4, 18.4]},
                                               "properties": {"site": 1, "drive": 2, "sol": 3}}]})
        self.assertEqual((wp[0].lon, wp[0].lat), (77.4, 18.4))

    def test_http_error_message(self):
        class S:
            def get(self, *a, **k):
                return FakeResponse({}, status=403)
        with self.assertRaises(ml.NasaDataError) as cm:
            ml.find_image(77.45, 18.44, session=S(), cache_dir=None)
        self.assertIn("403", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
