"""
Vector Enrichment Pipeline
===========================

Enrich vector feature layers (polygons, points) with statistics derived
from raster data and other vector layers. This is a common GIS workflow
for preparing data for analysis or ML model input.

Capabilities:
    - Zonal Statistics: compute mean/std/min/max/median of raster values
      within each polygon (e.g., average elevation per parcel)
    - Spatial Joins: attach attributes from the nearest feature in another
      layer (e.g., nearest road name for each building)
    - Proximity Analysis: distance from each feature to target features
      (e.g., distance to nearest river for each parcel)
    - Geometric Metrics: area, perimeter, compactness ratio with automatic
      UTM reprojection for accurate metric calculations

Usage:
    python vector_enrichment.py --vectors parcels.geojson \\
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


# =============================================================================
# Zonal Statistics
# =============================================================================

def zonal_statistics(vectors_path: str, raster_path: str,
                     stats: list = None, band: int = 1,
                     prefix: str = "zonal") -> gpd.GeoDataFrame:
    """Compute zonal statistics: summarize raster values within each polygon.

    For each polygon feature, clips the raster to the polygon's extent,
    extracts pixel values, and computes the requested statistics.

    This is equivalent to QGIS's "Zonal Statistics" processing tool but
    runs as a standalone script with more control over output format.

    Args:
        vectors_path: Path to vector file (GeoJSON, Shapefile, GeoPackage).
        raster_path: Path to raster file (GeoTIFF, etc.).
        stats: Statistics to compute. Default: ["mean", "std", "min", "max", "median", "count"].
        band: Raster band number to analyze (1-based).
        prefix: Column name prefix (e.g., "elevation" → "elevation_mean").

    Returns:
        GeoDataFrame with new columns for each statistic.

    Example:
        >>> gdf = zonal_statistics("parcels.geojson", "dem.tif", prefix="elev")
        >>> print(gdf[["name", "elev_mean", "elev_std"]].head())
        #    name           elev_mean   elev_std
        # 0  Parcel A       342.5       12.3
        # 1  Parcel B       289.1       8.7
    """
    if stats is None:
        stats = ["mean", "std", "min", "max", "median", "count"]

    gdf = gpd.read_file(vectors_path)
    logger.info(f"Computing zonal stats for {len(gdf)} features")

    # Pre-allocate columns for each statistic
    stat_columns = {f"{prefix}_{s}": [] for s in stats}

    with rasterio.open(raster_path) as src:
        # Ensure vectors and raster share the same CRS.
        # Mismatched CRS is a common source of bugs in spatial analysis.
        if gdf.crs and str(gdf.crs) != str(src.crs):
            gdf = gdf.to_crs(src.crs)
            logger.info(f"Reprojected vectors to {src.crs}")

        # Process each feature individually
        for _, row in gdf.iterrows():
            try:
                # Clip raster to this polygon's geometry
                geom = [mapping(row.geometry)]
                clipped, _ = rasterio_mask(src, geom, crop=True, band=band,
                                           nodata=src.nodata or 0, filled=True)
                values = clipped.flatten()

                # Remove nodata pixels before computing stats
                if src.nodata is not None:
                    values = values[values != src.nodata]
                # Also remove any NaN/Inf values
                values = values[np.isfinite(values)]

                # Compute requested statistics
                computed = _compute_stats(values, stats)
                for s in stats:
                    stat_columns[f"{prefix}_{s}"].append(computed.get(s))

            except Exception:
                # If clipping fails (e.g., polygon outside raster extent),
                # fill with None for this feature
                for s in stats:
                    stat_columns[f"{prefix}_{s}"].append(None)

    # Add computed columns to the GeoDataFrame
    for col_name, values in stat_columns.items():
        gdf[col_name] = values

    logger.info(f"Added {len(stats)} statistic columns")
    return gdf


def _compute_stats(values: np.ndarray, stats: list) -> dict:
    """Compute requested statistics on a 1D array of pixel values.

    Args:
        values: Array of raster values (nodata already removed).
        stats: List of statistic names to compute.

    Returns:
        Dictionary mapping stat name → computed value.
    """
    # Empty zone (polygon didn't overlap any valid pixels)
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


# =============================================================================
# Spatial Join
# =============================================================================

def spatial_join_nearest(target_path: str, join_path: str,
                         columns: list = None,
                         max_distance: float = None) -> gpd.GeoDataFrame:
    """Spatial join: attach attributes from the nearest feature in another layer.

    For each feature in the target layer, finds the nearest feature in the
    join layer and copies its attributes. Useful for questions like:
    - "What's the nearest school to each residential parcel?"
    - "Which road segment is closest to each accident point?"

    Args:
        target_path: Path to the target vector file (will receive new columns).
        join_path: Path to the join vector file (source of attributes).
        columns: Specific columns to transfer. None = transfer all columns.
        max_distance: Maximum join distance in CRS units. Features farther
                      than this won't be joined. None = no limit.

    Returns:
        Target GeoDataFrame with joined columns + "join_distance" column.
    """
    target = gpd.read_file(target_path)
    join_layer = gpd.read_file(join_path)

    # CRS alignment — both layers must be in the same coordinate system
    if target.crs != join_layer.crs:
        join_layer = join_layer.to_crs(target.crs)

    # GeoPandas nearest join — finds the single closest feature per target row
    result = gpd.sjoin_nearest(
        target, join_layer,
        how="left",
        max_distance=max_distance,
        distance_col="join_distance",  # Distance to the matched feature
    )

    # Optionally filter to only keep specific columns from the join layer
    if columns:
        keep_cols = list(target.columns) + columns + ["join_distance"]
        result = result[[c for c in keep_cols if c in result.columns]]

    # If a target feature matched multiple equidistant join features,
    # keep only the first match to avoid duplicate rows
    result = result.drop_duplicates(subset=[target.index.name or "index_right"])

    logger.info(f"Spatial join: {len(target)} features joined with {join_path}")
    return result


# =============================================================================
# Proximity Analysis
# =============================================================================

def compute_proximity(vectors_path: str, target_path: str,
                      col_name: str = "distance_to_target") -> gpd.GeoDataFrame:
    """Compute minimum distance from each feature to a target layer.

    Measures the shortest distance from each polygon/point in the source
    layer to the nearest geometry in the target layer. Useful for:
    - Distance to nearest road
    - Distance to nearest water body
    - Distance to nearest protected area boundary

    Args:
        vectors_path: Path to source features (e.g., buildings).
        target_path: Path to target features (e.g., rivers, roads).
        col_name: Name for the new distance column.

    Returns:
        GeoDataFrame with the distance column added.
        Distance is in CRS units (meters for projected CRS, degrees for geographic).
    """
    gdf = gpd.read_file(vectors_path)
    targets = gpd.read_file(target_path)

    # Ensure same CRS
    if gdf.crs != targets.crs:
        targets = targets.to_crs(gdf.crs)

    # Merge all target geometries into one for efficient distance computation.
    # This avoids an O(n*m) loop — instead we compute distance to one unified geometry.
    target_union = targets.union_all()

    # Compute distance from each source feature to the unified target
    gdf[col_name] = gdf.geometry.distance(target_union)

    logger.info(f"Computed proximity to {target_path}: "
                f"min={gdf[col_name].min():.2f}, max={gdf[col_name].max():.2f}")
    return gdf


# =============================================================================
# Geometric Metrics
# =============================================================================

def compute_area_perimeter(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add area, perimeter, and compactness columns to polygon features.

    If the CRS is geographic (lat/lon in degrees), automatically reprojects
    to the appropriate UTM zone for accurate metric calculations. This is
    critical — computing area in degrees gives meaningless numbers.

    Added columns:
        area_m2:     Area in square meters
        area_ha:     Area in hectares (1 ha = 10,000 m2)
        perimeter_m: Perimeter in meters
        compactness: Polsby-Popper compactness ratio (4*pi*area / perimeter^2).
                     1.0 = perfect circle, lower = more irregular shape.

    Args:
        gdf: GeoDataFrame with polygon geometries.

    Returns:
        Same GeoDataFrame with four new columns added.
    """
    work_gdf = gdf.copy()

    # Auto-detect and reproject geographic CRS to UTM for accurate measurements.
    # UTM zones are determined by the centroid longitude of the dataset.
    if work_gdf.crs and work_gdf.crs.is_geographic:
        centroid = work_gdf.geometry.union_all().centroid
        # UTM zone formula: zone = floor((longitude + 180) / 6) + 1
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        # EPSG codes: 326xx = UTM North, 327xx = UTM South
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        work_gdf = work_gdf.to_crs(epsg=epsg)
        logger.info(f"Reprojected to EPSG:{epsg} for area/perimeter calculation")

    # Compute metrics using the projected geometry
    gdf["area_m2"] = work_gdf.geometry.area
    gdf["area_ha"] = gdf["area_m2"] / 10000  # Convert to hectares
    gdf["perimeter_m"] = work_gdf.geometry.length
    # Polsby-Popper compactness: how close to a circle is this polygon?
    gdf["compactness"] = (4 * np.pi * gdf["area_m2"]) / (gdf["perimeter_m"] ** 2)

    return gdf


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for the vector enrichment pipeline."""
    parser = argparse.ArgumentParser(description="Vector enrichment pipeline")
    parser.add_argument("--vectors", required=True, help="Input vector file")
    parser.add_argument("--raster", help="Raster for zonal statistics")
    parser.add_argument("--output", required=True, help="Output vector file")
    parser.add_argument("--proximity_target", help="Target layer for proximity calc")
    parser.add_argument("--join_layer", help="Layer for spatial join")
    args = parser.parse_args()

    # Start with geometric metrics (always computed)
    gdf = gpd.read_file(args.vectors)
    gdf = compute_area_perimeter(gdf)

    # Add raster-derived statistics if a raster is provided
    if args.raster:
        gdf = zonal_statistics(args.vectors, args.raster)
        gdf = compute_area_perimeter(gdf)

    # Add proximity metrics if a target layer is provided
    if args.proximity_target:
        gdf = compute_proximity(args.vectors, args.proximity_target)

    # Write enriched output
    gdf.to_file(args.output, driver="GeoJSON")
    logger.info(f"Enriched output: {args.output} ({len(gdf)} features, {len(gdf.columns)} columns)")


if __name__ == "__main__":
    main()
