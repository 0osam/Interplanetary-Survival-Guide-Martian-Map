#!/usr/bin/env python3
"""
crop_jezero_3x3.py
==================
Crops a single 3x3 chunk GeoTIFF window out of the full Jezero HiRISE DTM
with zero terrain/routing calculations.
"""

import argparse
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

MARS_LONLAT = "+proj=longlat +R=3396190 +no_defs"
MARS_EQC = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs"


def crop_3x3_tif(
    input_tif: str,
    output_tif: str = "jezero_3x3.tif",
    center_lon: float = 77.4508,
    center_lat: float = 18.4447,
    chunk_size: int = 512,
    halo: int = 0,
):
    total_px = 3 * chunk_size + 2 * halo
    half_px = total_px // 2

    with rasterio.open(input_tif) as src:
        try:
            transformer = Transformer.from_crs(MARS_LONLAT, src.crs, always_xy=True)
        except Exception:
            transformer = Transformer.from_crs(MARS_LONLAT, MARS_EQC, always_xy=True)

        cx_m, cy_m = transformer.transform(center_lon, center_lat)
        row_c, col_c = src.index(cx_m, cy_m)

        win = Window(
            col_off=int(col_c - half_px),
            row_off=int(row_c - half_px),
            width=total_px,
            height=total_px,
        )

        data = src.read(1, window=win, boundless=True, fill_value=src.nodata)
        profile = src.profile.copy()
        profile.update(
            height=total_px,
            width=total_px,
            transform=src.window_transform(win),
            compress="deflate",
        )

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(data, 1)

    print(f"Saved {output_tif} ({total_px}x{total_px} px = 3x3 chunks of {chunk_size}px, halo={halo}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop 3x3 Jezero DTM GeoTIFF")
    parser.add_argument(
        "--input",
        default="JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif",
        help="Path to full Jezero HiRISE DTM .tif",
    )
    parser.add_argument("--output", default="jezero_3x3.tif", help="Output 3x3 .tif path")
    parser.add_argument("--lon", type=float, default=77.4508, help="Center Longitude (deg East)")
    parser.add_argument("--lat", type=float, default=18.4447, help="Center Latitude (deg North)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Pixels per chunk L (512 -> 1536x1536)")
    parser.add_argument("--halo", type=int, default=0, help="Optional global border halo in pixels (e.g. 4)")
    args = parser.parse_args()

    crop_3x3_tif(args.input, args.output, args.lon, args.lat, args.chunk_size, args.halo)
