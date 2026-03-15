"""
Vector Enrichment Pipeline

Enrich vector feature layers with spatial statistics derived from raster data.
Computes zonal statistics, spatial joins, and proximity metrics.

Usage:
    python vector_enrichment.py --vectors parcels.geojson \
        --raster elevation.tif --output enriched_parcels.geojson
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import shape, mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def zonal_statistics(vectors_path: str, raster_path: str,
                     stats: list = None, band: int = 1,
                     prefix: str = "zonal") -> gpd.GeoDataFrame:
    """Compute zonal statistics for each polygon feature.

    Args:
        vectors_path: Path to vector file (GeoJSON, Shapefile, etc).
        raster_path: Path to raster file.
        stats: List of statistics to compute. Default: mean, std, min, max, median.
        band: Raster band number (1-based).
        prefix: Column name prefix for statistics.

    Returns:
        GeoDataFrame with statistics columns added.
    """
    if stats is None:
        stats = ["mean", "std", "min", "max", "median", "count"]

    gdf = gpd.read_file(vectors_path)
    logger.info(f"Computing zonal stats for {len(gdf)} features")

    stat_columns = {f"{prefix}_{s}": [] for s in stats}

    with rasterio.open(raster_path) as src:
        # Reproject vectors to raster CRS if needed
        if gdf.crs and str(gdf.crs) != str(src.crs):
            gdf = gdf.to_crs(src.crs)
            logger.info(f"Reprojected vectors to {src.crs}")

        for _, row in gdf.iterrows():
            try:
                geom = [mapping(row.geometry)]
                clipped, _ = rasterio_mask(src, geom, crop=True, band=band,
                                           nodata=src.nodata or 0, filled=True)
                values = clipped.flatten()

                # Remove nodata
                if src.nodata is not None:
                    values = values[values != src.nodata]
                values = values[np.isfinite(values)]

                computed = _compute_stats(values, stats)
                for s in stats:
                    stat_columns[f"{prefix}_{s}"].append(computed.get(s))

            except Exception:
                for s in stats:
                    stat_columns[f"{prefix}_{s}"].append(None)

    for col_name, values in stat_columns.items():
        gdf[col_name] = values

    logger.info(f"Added {len(stats)} statistic columns")
    return gdf


def _compute_stats(values: np.ndarray, stats: list) -> dict:
    """Compute requested statistics on an array of values."""
    if len(values) == 0:
        return {s: None for s in stats}

    result = {}
    for s in stats:
        if s == "mean":
            result[s] = float(np.mean(values))
        elif s == "std":
            result[s] = float(np.std(values))
        elif s == "min":
            result[s] = float(np.min(values))
        elif s == "max":
            result[s] = float(np.max(values))
        elif s == "median":
            result[s] = float(np.median(values))
        elif s == "count":
            result[s] = int(len(values))
        elif s == "sum":
            result[s] = float(np.sum(values))
    return result


def spatial_join_nearest(target_path: str, join_path: str,
                         columns: list = None,
                         max_distance: float = None) -> gpd.GeoDataFrame:
    """Spatial join: attach attributes from nearest features.

    Args:
        target_path: Path to target vector file.
        join_path: Path to join vector file.
        columns: Columns to transfer from join layer. None = all.
        max_distance: Maximum join distance in CRS units. None = unlimited.

    Returns:
        Target GeoDataFrame with joined columns.
    """
    target = gpd.read_file(target_path)
    join_layer = gpd.read_file(join_path)

    if target.crs != join_layer.crs:
        join_layer = join_layer.to_crs(target.crs)

    result = gpd.sjoin_nearest(
        target, join_layer,
        how="left",
        max_distance=max_distance,
        distance_col="join_distance",
    )

    if columns:
        keep_cols = list(target.columns) + columns + ["join_distance"]
        result = result[[c for c in keep_cols if c in result.columns]]

    # Drop duplicates (keep nearest)
    result = result.drop_duplicates(subset=[target.index.name or "index_right"])

    logger.info(f"Spatial join: {len(target)} features joined with {join_path}")
    return result


def compute_proximity(vectors_path: str, target_path: str,
                      col_name: str = "distance_to_target") -> gpd.GeoDataFrame:
    """Compute minimum distance from each feature to a target layer.

    Args:
        vectors_path: Path to source vector file.
        target_path: Path to target features (e.g., roads, rivers).
        col_name: Name for the distance column.

    Returns:
        GeoDataFrame with distance column added.
    """
    gdf = gpd.read_file(vectors_path)
    targets = gpd.read_file(target_path)

    if gdf.crs != targets.crs:
        targets = targets.to_crs(gdf.crs)

    # Union all target geometries for efficient distance computation
    target_union = targets.union_all()

    gdf[col_name] = gdf.geometry.distance(target_union)
    logger.info(f"Computed proximity to {target_path}: "
                f"min={gdf[col_name].min():.2f}, max={gdf[col_name].max():.2f}")
    return gdf


def compute_area_perimeter(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add area and perimeter columns for polygon features.

    If the CRS is geographic (degrees), reprojects to a suitable
    UTM zone for accurate metric calculations.
    """
    work_gdf = gdf.copy()

    if work_gdf.crs and work_gdf.crs.is_geographic:
        centroid = work_gdf.geometry.union_all().centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        work_gdf = work_gdf.to_crs(epsg=epsg)
        logger.info(f"Reprojected to EPSG:{epsg} for area/perimeter calculation")

    gdf["area_m2"] = work_gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000
    gdf["perimeter_m"] = work_gdf.geometry.length
    gdf["compactness"] = (4 * np.pi * gdf["area_m2"]) / (gdf["perimeter_m"] ** 2)

    return gdf


def main():
    parser = argparse.ArgumentParser(description="Vector enrichment pipeline")
    parser.add_argument("--vectors", required=True, help="Input vector file")
    parser.add_argument("--raster", help="Raster for zonal statistics")
    parser.add_argument("--output", required=True, help="Output vector file")
    parser.add_argument("--proximity_target", help="Target layer for proximity calc")
    parser.add_argument("--join_layer", help="Layer for spatial join")
    args = parser.parse_args()

    gdf = gpd.read_file(args.vectors)
    gdf = compute_area_perimeter(gdf)

    if args.raster:
        gdf = zonal_statistics(args.vectors, args.raster)
        gdf = compute_area_perimeter(gdf)

    if args.proximity_target:
        gdf = compute_proximity(args.vectors, args.proximity_target)

    gdf.to_file(args.output, driver="GeoJSON")
    logger.info(f"Enriched output: {args.output} ({len(gdf)} features, {len(gdf.columns)} columns)")


if __name__ == "__main__":
    main()
