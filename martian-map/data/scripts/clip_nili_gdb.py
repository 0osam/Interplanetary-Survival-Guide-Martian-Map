#!/usr/bin/env python3
import argparse
import geopandas as gpd
import pyogrio
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

MARS_LONLAT = "+proj=longlat +R=3396190 +no_defs"

SIM3464_TO_ID = {
    "Ncu": 1, "Nbu": 1,
    "Ncf": 2, "Nnp": 2,
    "NHdp": 3,
    "NHdf": 4,
    "NHl": 5, "Nm": 5,
    "Aeb": 6,
    "Asu": 7, "Ac": 7,
}

def clip_nili_geology(
    gdb_path: str = "sim3464_database/sim3464_JezeroNili_FINAL/Geodatabase/Nili.gdb",
    ref_dem_tif: str = "jezero_3x3.tif",
    out_geojson: str = "jezero_sim3464_units.geojson",
    out_tif: str = "sim3464_jezero_3x3.tif",
):
    layers = pyogrio.list_layers(gdb_path)
    unit_layer = next(
        (name for name, gtype in layers if "Polygon" in str(gtype) and "unit" in name.lower() and "anno" not in name.lower()),
        "GeoUnits_v1",
    )
    gdf = gpd.read_file(gdb_path, layer=unit_layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(MARS_LONLAT)

    sym_col = next((c for c in ("symbol_noSub", "symbol", "Unit", "UNIT") if c in gdf.columns), "unit")
    gdf["Unit"] = gdf[sym_col].astype(str).str.strip()
    if "name" not in gdf.columns:
        gdf["name"] = gdf["Unit"]

    with rasterio.open(ref_dem_tif) as ref:
        dst_crs, dst_transform = ref.crs, ref.transform
        dst_h, dst_w = ref.height, ref.width
        b = ref.bounds
        clip_box = box(b.left - 500.0, b.bottom - 500.0, b.right + 500.0, b.top + 500.0)

    gdf_clip = gpd.clip(gdf.to_crs(dst_crs), clip_box)[["Unit", "name", "geometry"]].copy()
    gdf_clip = gdf_clip[~gdf_clip.geometry.is_empty].reset_index(drop=True)

    gdf_clip.to_crs(MARS_LONLAT).to_file(out_geojson, driver="GeoJSON")

    shapes = [(geom, SIM3464_TO_ID.get(code, 1)) for geom, code in zip(gdf_clip.geometry, gdf_clip["Unit"]) if geom is not None]
    unit_raster = rasterize(shapes=shapes, out_shape=(dst_h, dst_w), transform=dst_transform, fill=1, dtype="uint8")

    with rasterio.open(
        out_tif, "w", driver="GTiff", height=dst_h, width=dst_w, count=1,
        dtype="uint8", crs=dst_crs, transform=dst_transform, nodata=0, compress="deflate"
    ) as dst:
        dst.write(unit_raster, 1)

    print(f"[OK] Saved {out_geojson} & {out_tif} | True SIM 3464 Units: {sorted(gdf_clip['Unit'].unique().tolist())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdb", default="sim3464_database/sim3464_JezeroNili_FINAL/Geodatabase/Nili.gdb")
    parser.add_argument("--ref-dem", default="jezero_3x3.tif")
    args = parser.parse_args()
    clip_nili_geology(args.gdb, args.ref_dem)
