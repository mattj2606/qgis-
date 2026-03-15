"""
Multi-temporal Change Detection
=================================

Detect land use / land cover changes between two dates using satellite imagery.
This is one of the most valuable remote sensing analyses — it answers the question
"What changed on the ground between Date A and Date B?"

Three detection methods are supported:

1. Image Differencing:  Simple subtraction (after - before), normalized.
   Best for: quick visual assessment with single-band data.

2. Log Ratio:  log(after / before). Symmetric around zero, robust to
   multiplicative noise common in radar/SAR imagery.

3. NDVI Differencing:  Computes NDVI for each date, then subtracts.
   Best for: vegetation gain/loss detection (deforestation, agriculture,
   urban expansion into green areas).

Outputs:
    - Change magnitude raster (continuous values)
    - Binary change map (gain=1, no change=0, loss=-1)
    - Statistics JSON report
    - Transition matrix (for post-classification comparison)

Usage:
    python change_detection.py --before 2020_composite.tif \\
        --after 2023_composite.tif --output changes.tif --method difference
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Main Change Detection Function
# =============================================================================

def detect_changes(before: str, after: str, output: str = None,
                   method: str = "difference", threshold: float = 0.3,
                   band: int = 1) -> dict:
    """Detect changes between two raster datasets from different dates.

    The workflow:
    1. Read both rasters (reprojecting 'after' if needed to match 'before')
    2. Build a valid pixel mask (excluding nodata from both dates)
    3. Compute change magnitude using the chosen method
    4. Apply threshold to create binary gain/loss/nochange map
    5. Calculate statistics and optionally write outputs

    Args:
        before: Path to the earlier date raster.
        after: Path to the later date raster.
        output: Optional path to save the change magnitude GeoTIFF.
                Also creates _binary.tif and _stats.json alongside it.
        method: Detection method:
                - 'difference': simple image subtraction
                - 'ratio': log ratio (better for radar)
                - 'ndvi_change': vegetation index differencing
        threshold: Magnitude threshold for classifying a pixel as "changed".
                   Values above threshold = gain, below -threshold = loss.
        band: Which band to analyze (1-based). Ignored for ndvi_change.

    Returns:
        Dictionary with change statistics:
        {
            "method": "difference",
            "change_percentage": 12.5,
            "gain_pixels": 15000,
            "loss_pixels": 8000,
            "magnitude_mean": 0.05,
            ...
        }
    """
    # --- Read the two date rasters ---
    with rasterio.open(before) as src_before:
        data_before = src_before.read(band).astype(np.float64)
        profile = src_before.profile.copy()
        nodata_before = src_before.nodata

    with rasterio.open(after) as src_after:
        data_after = src_after.read(band).astype(np.float64)
        nodata_after = src_after.nodata

        # If the rasters have different extents or resolutions, reproject
        # 'after' to match 'before'. This ensures pixel-to-pixel comparison.
        if data_after.shape != data_before.shape:
            logger.info("Reprojecting 'after' raster to match 'before'")
            with rasterio.open(before) as src_before_ref:
                data_after_aligned = np.zeros_like(data_before)
                reproject(
                    source=rasterio.band(src_after, band),
                    destination=data_after_aligned,
                    src_transform=src_after.transform,
                    src_crs=src_after.crs,
                    dst_transform=src_before_ref.transform,
                    dst_crs=src_before_ref.crs,
                    resampling=Resampling.bilinear,
                )
                data_after = data_after_aligned

    # --- Build valid data mask ---
    # Only analyze pixels that have valid data in BOTH dates
    valid_mask = np.ones(data_before.shape, dtype=bool)
    if nodata_before is not None:
        valid_mask &= data_before != nodata_before
    if nodata_after is not None:
        valid_mask &= data_after != nodata_after

    # --- Compute change magnitude ---
    if method == "difference":
        change_magnitude = _difference_change(data_before, data_after, valid_mask)
    elif method == "ratio":
        change_magnitude = _ratio_change(data_before, data_after, valid_mask)
    elif method == "ndvi_change":
        change_magnitude = _ndvi_change(before, after)
        valid_mask = np.isfinite(change_magnitude)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'difference', 'ratio', or 'ndvi_change'.")

    # --- Create binary change map ---
    # Classify each pixel as gain (+1), loss (-1), or no change (0)
    change_binary = np.zeros(change_magnitude.shape, dtype=np.int32)
    change_binary[valid_mask & (change_magnitude > threshold)] = 1     # Gain
    change_binary[valid_mask & (change_magnitude < -threshold)] = -1   # Loss
    change_binary[~valid_mask] = -9999                                 # Nodata

    # --- Compute summary statistics ---
    valid_total = valid_mask.sum()
    stats = {
        "method": method,
        "threshold": threshold,
        "total_pixels": int(valid_total),
        "changed_pixels": int(np.abs(change_binary[valid_mask]).sum()),
        "change_percentage": round(
            float(np.abs(change_binary[valid_mask]).sum()) / max(valid_total, 1) * 100, 2
        ),
        "gain_pixels": int((change_binary == 1).sum()),
        "loss_pixels": int((change_binary == -1).sum()),
        "magnitude_mean": float(np.nanmean(change_magnitude[valid_mask])),
        "magnitude_std": float(np.nanstd(change_magnitude[valid_mask])),
    }

    logger.info(
        f"Change detection complete: {stats['change_percentage']}% changed "
        f"(gain={stats['gain_pixels']}, loss={stats['loss_pixels']})"
    )

    # --- Write outputs if path provided ---
    if output:
        _write_change_outputs(
            change_magnitude, change_binary, profile, output, stats
        )

    return stats


# =============================================================================
# Change Detection Methods
# =============================================================================

def _difference_change(before: np.ndarray, after: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """Simple image differencing: (after - before), normalized to [-1, 1].

    The simplest change detection approach. Positive values indicate an
    increase in pixel value (e.g., more reflectance), negative = decrease.

    Normalization ensures the threshold parameter works consistently
    regardless of the original data range.
    """
    result = np.full(before.shape, np.nan, dtype=np.float64)
    result[mask] = after[mask] - before[mask]

    # Normalize to [-1, 1] for consistent threshold behavior
    valid_values = result[mask]
    if len(valid_values) > 0:
        max_abs = max(abs(valid_values.min()), abs(valid_values.max()), 1)
        result[mask] = result[mask] / max_abs

    return result


def _ratio_change(before: np.ndarray, after: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """Log ratio change detection: log(after / before).

    The log ratio is preferred over simple ratio because:
    - It's symmetric: a 2x increase and 2x decrease have equal magnitudes
    - It's additive: changes can be summed meaningfully
    - It handles multiplicative noise well (common in radar/SAR data)

    Values near 0 = no change, positive = increase, negative = decrease.
    """
    result = np.full(before.shape, np.nan, dtype=np.float64)
    # Only compute where both values are positive (log is undefined for <= 0)
    safe_mask = mask & (before > 0) & (after > 0)
    result[safe_mask] = np.log(after[safe_mask] / before[safe_mask])
    return result


def _ndvi_change(before_path: str, after_path: str,
                 red_band: int = 3, nir_band: int = 4) -> np.ndarray:
    """NDVI differencing: NDVI(after) - NDVI(before).

    The most widely used method for vegetation change detection.
    Positive values = vegetation gain (greening, regrowth).
    Negative values = vegetation loss (deforestation, drought, urbanization).

    This method requires multispectral imagery with at least Red and NIR bands.

    Args:
        before_path: Path to earlier date multispectral raster.
        after_path: Path to later date multispectral raster.
        red_band: 1-based band number for red (default 3 = Sentinel-2 Band 4).
        nir_band: 1-based band number for NIR (default 4 = Sentinel-2 Band 8).
    """
    def _read_ndvi(path):
        """Read a raster and compute its NDVI."""
        with rasterio.open(path) as src:
            if src.count < max(red_band, nir_band):
                raise ValueError(f"{path} has only {src.count} bands, need {max(red_band, nir_band)}")
            red = src.read(red_band).astype(np.float64)
            nir = src.read(nir_band).astype(np.float64)
        denom = nir + red
        ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
        return ndvi

    ndvi_before = _read_ndvi(before_path)
    ndvi_after = _read_ndvi(after_path)

    # Simple subtraction — NDVI is already normalized to [-1, 1]
    return ndvi_after - ndvi_before


# =============================================================================
# Output Writing
# =============================================================================

def _write_change_outputs(magnitude: np.ndarray, binary: np.ndarray,
                          profile: dict, output: str, stats: dict):
    """Write change detection results to disk.

    Creates three output files:
        1. {output}.tif         — Change magnitude (float, continuous)
        2. {output}_binary.tif  — Binary change map (1=gain, 0=none, -1=loss)
        3. {output}_stats.json  — Summary statistics report
    """
    # --- Write magnitude raster (float) ---
    mag_profile = profile.copy()
    mag_profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)

    with rasterio.open(output, "w", **mag_profile) as dst:
        dst.write(magnitude.astype(np.float32), 1)
    logger.info(f"Change magnitude: {output}")

    # --- Write binary change map (integer) ---
    bin_path = output.replace(".tif", "_binary.tif")
    bin_profile = profile.copy()
    bin_profile.update(dtype=rasterio.int32, count=1, compress="lzw", nodata=-9999)

    with rasterio.open(bin_path, "w", **bin_profile) as dst:
        dst.write(binary, 1)
    logger.info(f"Binary change map: {bin_path}")

    # --- Write statistics report (JSON) ---
    report_path = output.replace(".tif", "_stats.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats report: {report_path}")


# =============================================================================
# Transition Matrix (Post-Classification Comparison)
# =============================================================================

def transition_matrix(before_classified: str, after_classified: str) -> dict:
    """Compute a land cover transition matrix from two classified rasters.

    This is a post-classification comparison approach: two independently
    classified maps are overlaid to see which classes converted to which.

    The transition matrix shows:
    - Diagonal: pixels that stayed the same class (persistence)
    - Off-diagonal: pixels that changed from one class to another

    Example output:
        From\\To  | Forest | Urban | Water
        Forest   |  5000  |  200  |   10
        Urban    |    50  | 3000  |    5
        Water    |    20  |   30  | 2000

    Args:
        before_classified: Path to earlier classified raster (integer classes).
        after_classified: Path to later classified raster (integer classes).

    Returns:
        Dictionary with:
        - matrix: N x N transition counts
        - persistence_rate: % of area that stayed the same
        - per_class: gains, losses, net change per class
    """
    with rasterio.open(before_classified) as src:
        before = src.read(1)
    with rasterio.open(after_classified) as src:
        after = src.read(1)

    if before.shape != after.shape:
        raise ValueError("Classified rasters must have the same dimensions")

    # Only compare valid pixels (class > 0, since 0 = nodata)
    valid = (before > 0) & (after > 0)
    before_valid = before[valid]
    after_valid = after[valid]

    # Build the transition matrix
    all_classes = sorted(set(np.unique(before_valid)) | set(np.unique(after_valid)))
    n = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    matrix = np.zeros((n, n), dtype=np.int64)
    for bv, av in zip(before_valid, after_valid):
        matrix[class_to_idx[bv], class_to_idx[av]] += 1

    # Summary statistics
    total = matrix.sum()
    persistence = np.diag(matrix).sum()  # Sum of diagonal = unchanged pixels

    result = {
        "classes": [int(c) for c in all_classes],
        "matrix": matrix.tolist(),
        "total_pixels": int(total),
        "persistence_pixels": int(persistence),
        "change_pixels": int(total - persistence),
        "persistence_rate": round(float(persistence / max(total, 1) * 100), 2),
    }

    # Per-class analysis: how much did each class gain or lose?
    per_class = {}
    for i, cls in enumerate(all_classes):
        row_sum = matrix[i, :].sum()   # Total pixels classified as this in 'before'
        col_sum = matrix[:, i].sum()   # Total pixels classified as this in 'after'
        per_class[int(cls)] = {
            "before_count": int(row_sum),
            "after_count": int(col_sum),
            "gain": int(col_sum - matrix[i, i]),    # New pixels entering this class
            "loss": int(row_sum - matrix[i, i]),     # Pixels leaving this class
            "net_change": int(col_sum - row_sum),    # Overall gain or loss
        }
    result["per_class"] = per_class

    logger.info(f"Transition matrix: {result['persistence_rate']}% persistence, "
                f"{result['change_pixels']} changed pixels")
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for change detection."""
    parser = argparse.ArgumentParser(description="Multi-temporal change detection")
    parser.add_argument("--before", required=True, help="Earlier raster")
    parser.add_argument("--after", required=True, help="Later raster")
    parser.add_argument("--output", required=True, help="Output change map")
    parser.add_argument("--method", default="difference",
                        choices=["difference", "ratio", "ndvi_change"])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--band", type=int, default=1)
    args = parser.parse_args()

    detect_changes(
        args.before, args.after, args.output,
        method=args.method, threshold=args.threshold, band=args.band,
    )


if __name__ == "__main__":
    main()
