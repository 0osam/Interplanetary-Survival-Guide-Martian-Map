#!/usr/bin/env python3
"""
crop_tes_jezero.py
==================
Extracts the Jezero Crater 3x3 region from TES_Thermal_Inertia.tif and aligns it
1:1 to the exact spatial extent, CRS, and pixel grid of jezero_3x3.tif.
"""

import argparse
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def crop_tes_to_jezero_grid(
    tes_tif: str,
    ref_dem_tif: str = "jezero_3x3.tif",
    output_tif: str = "tes_jezero_3x3.tif",
):
    # 1. Read exact 3x3 target grid (1536x1536, CRS, transform) from jezero_3x3.tif
    with rasterio.open(ref_dem_tif) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height = ref.height
        dst_width = ref.width

    # 2. Reproject & crop TES Thermal Inertia onto the exact Jezero 3x3 grid
    with rasterio.open(tes_tif) as src:
        src_data = src.read(1).astype(np.float32)

        # Clean NoData sentinels so they don't bleed during bilinear interpolation
        if src.nodata is not None and not np.isnan(src.nodata):
            src_data[np.isclose(src_data, src.nodata)] = np.nan
        src_data[(src_data <= 0.0) | (src_data > 2000.0)] = np.nan

        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = dst_data[np.isfinite(dst_data)]
    if valid.size == 0:
        raise ValueError(f"No valid TES pixels overlapped {ref_dem_tif}. Check CRS/bounds of {tes_tif}.")

    # Fill any remaining edge NaNs with the Jezero median TIU
    dst_data = np.where(np.isfinite(dst_data), dst_data, float(np.median(valid)))

    # 3. Write aligned 3x3 Thermal Inertia GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": -9999.0,
        "compress": "deflate",
    }
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(dst_data, 1)

    print(
        f"Saved {output_tif} ({dst_width}x{dst_height} px, matched to {ref_dem_tif}) | "
        f"Jezero TIU range: {valid.min():.1f} to {valid.max():.1f} TIU"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align TES Thermal Inertia to Jezero 3x3 grid")
    parser.add_argument("--tes", default="TES_Thermal_Inertia.tif", help="Path to MGS TES Thermal Inertia .tif")
    parser.add_argument("--ref-dem", default="jezero_3x3.tif", help="Path to cropped jezero_3x3.tif")
    parser.add_argument("--output", default="tes_jezero_3x3.tif", help="Output aligned TES .tif")
    args = parser.parse_args()

    crop_tes_to_jezero_grid(args.tes, args.ref_dem, args.output)
