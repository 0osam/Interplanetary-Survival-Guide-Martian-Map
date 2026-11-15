import io
import math
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

# Single Mars Trek WMTS tile covering Jezero Crater at ~100m/px (Zoom 9)
Z, ROW, COL = 9, 203, 732
TILE_URL = (
    f"https://trek.nasa.gov/tiles/Mars/EQ/"
    f"THEMIS_NightIR_ControlledMosaics_100m_v2_oct2018/1.0.0/default/default028mm/{Z}/{ROW}/{COL}.png"
)

resp = requests.get(TILE_URL, timeout=15)
resp.raise_for_status()
print(f"Downloaded {len(resp.content) / 1024:.1f} KB from NASA Mars Trek")

tile_gray = np.array(Image.open(io.BytesIO(resp.content)).convert("L"), dtype=np.float32)

# Exact geographic bounds of tile (Z=9, ROW=203, COL=732)
n_cols, n_rows = 2 ** (Z + 1), 2 ** Z
west = -180.0 + (COL / n_cols) * 360.0
east = -180.0 + ((COL + 1) / n_cols) * 360.0
north = 90.0 - (ROW / n_rows) * 180.0
south = 90.0 - ((ROW + 1) / n_rows) * 180.0
src_transform = from_bounds(west, south, east, north, tile_gray.shape[1], tile_gray.shape[0])

with rasterio.open("jezero_3x3.tif") as ref:
    dst = np.zeros((ref.height, ref.width), dtype=np.float32)
    reproject(
        source=tile_gray,
        destination=dst,
        src_transform=src_transform,
        src_crs="+proj=longlat +R=3396190 +no_defs",
        dst_transform=ref.transform,
        dst_crs=ref.crs,
        resampling=Resampling.bilinear,
    )
    prof = ref.profile.copy()
    prof.update(dtype="float32", nodata=None, compress="deflate")
    with rasterio.open("themis_night_jezero_3x3.tif", "w", **prof) as out:
        out.write(dst, 1)

print("Saved aligned 3x3 raster -> themis_night_jezero_3x3.tif")
