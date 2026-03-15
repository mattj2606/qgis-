"""
Multi-temporal Change Detection

Detect land use/land cover changes between two dates using raster differencing,
ratio analysis, or post-classification comparison.

Usage:
    python change_detection.py --before 2020_composite.tif \
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


def detect_changes(before: str, after: str, output: str = None,
                   method: str = "difference", threshold: float = 0.3,
                   band: int = 1) -> dict:
    """Detect changes between two raster datasets.

    Args:
        before: Path to the earlier raster.
        after: Path to the later raster.
        output: Optional output path for the change map.
        method: Detection method ('difference', 'ratio', 'ndvi_change').
        threshold: Change magnitude threshold for binary classification.
        band: Band to analyze (1-based).

    Returns:
        Dictionary with change statistics.
    """
    with rasterio.open(before) as src_before:
        data_before = src_before.read(band).astype(np.float64)
        profile = src_before.profile.copy()
        nodata_before = src_before.nodata

    with rasterio.open(after) as src_after:
        data_after = src_after.read(band).astype(np.float64)
        nodata_after = src_after.nodata

        # Reproject 'after' to match 'before' if shapes differ
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

    # Build valid data mask
    valid_mask = np.ones(data_before.shape, dtype=bool)
    if nodata_before is not None:
        valid_mask &= data_before != nodata_before
    if nodata_after is not None:
        valid_mask &= data_after != nodata_after

    # Compute change
    if method == "difference":
        change_magnitude = _difference_change(data_before, data_after, valid_mask)
    elif method == "ratio":
        change_magnitude = _ratio_change(data_before, data_after, valid_mask)
    elif method == "ndvi_change":
        change_magnitude = _ndvi_change(before, after)
        valid_mask = np.isfinite(change_magnitude)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'difference', 'ratio', or 'ndvi_change'.")

    # Binary change map
    change_binary = np.zeros(change_magnitude.shape, dtype=np.int32)
    change_binary[valid_mask & (change_magnitude > threshold)] = 1   # gain
    change_binary[valid_mask & (change_magnitude < -threshold)] = -1  # loss
    change_binary[~valid_mask] = -9999

    # Statistics
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

    if output:
        _write_change_outputs(
            change_magnitude, change_binary, profile, output, stats
        )

    return stats


def _difference_change(before: np.ndarray, after: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """Simple image differencing: after - before."""
    result = np.full(before.shape, np.nan, dtype=np.float64)
    result[mask] = after[mask] - before[mask]

    # Normalize to [-1, 1] based on data range
    valid_values = result[mask]
    if len(valid_values) > 0:
        max_abs = max(abs(valid_values.min()), abs(valid_values.max()), 1)
        result[mask] = result[mask] / max_abs

    return result


def _ratio_change(before: np.ndarray, after: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """Log ratio change detection: log(after / before)."""
    result = np.full(before.shape, np.nan, dtype=np.float64)
    safe_mask = mask & (before > 0) & (after > 0)
    result[safe_mask] = np.log(after[safe_mask] / before[safe_mask])
    return result


def _ndvi_change(before_path: str, after_path: str,
                 red_band: int = 3, nir_band: int = 4) -> np.ndarray:
    """NDVI differencing between two dates."""
    def _read_ndvi(path):
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

    return ndvi_after - ndvi_before


def _write_change_outputs(magnitude: np.ndarray, binary: np.ndarray,
                          profile: dict, output: str, stats: dict):
    """Write change detection outputs."""
    # Write magnitude raster
    mag_profile = profile.copy()
    mag_profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)

    with rasterio.open(output, "w", **mag_profile) as dst:
        dst.write(magnitude.astype(np.float32), 1)
    logger.info(f"Change magnitude: {output}")

    # Write binary change map
    bin_path = output.replace(".tif", "_binary.tif")
    bin_profile = profile.copy()
    bin_profile.update(dtype=rasterio.int32, count=1, compress="lzw", nodata=-9999)

    with rasterio.open(bin_path, "w", **bin_profile) as dst:
        dst.write(binary, 1)
    logger.info(f"Binary change map: {bin_path}")

    # Write stats report
    report_path = output.replace(".tif", "_stats.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats report: {report_path}")


def transition_matrix(before_classified: str, after_classified: str) -> dict:
    """Compute a land cover transition matrix from two classified rasters.

    Args:
        before_classified: Path to earlier classified raster.
        after_classified: Path to later classified raster.

    Returns:
        Dictionary with transition matrix and class-level changes.
    """
    with rasterio.open(before_classified) as src:
        before = src.read(1)
    with rasterio.open(after_classified) as src:
        after = src.read(1)

    if before.shape != after.shape:
        raise ValueError("Classified rasters must have the same dimensions")

    # Only consider valid pixels
    valid = (before > 0) & (after > 0)
    before_valid = before[valid]
    after_valid = after[valid]

    all_classes = sorted(set(np.unique(before_valid)) | set(np.unique(after_valid)))
    n = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    matrix = np.zeros((n, n), dtype=np.int64)
    for bv, av in zip(before_valid, after_valid):
        matrix[class_to_idx[bv], class_to_idx[av]] += 1

    total = matrix.sum()
    persistence = np.diag(matrix).sum()

    result = {
        "classes": [int(c) for c in all_classes],
        "matrix": matrix.tolist(),
        "total_pixels": int(total),
        "persistence_pixels": int(persistence),
        "change_pixels": int(total - persistence),
        "persistence_rate": round(float(persistence / max(total, 1) * 100), 2),
    }

    # Per-class gains and losses
    per_class = {}
    for i, cls in enumerate(all_classes):
        row_sum = matrix[i, :].sum()
        col_sum = matrix[:, i].sum()
        per_class[int(cls)] = {
            "before_count": int(row_sum),
            "after_count": int(col_sum),
            "gain": int(col_sum - matrix[i, i]),
            "loss": int(row_sum - matrix[i, i]),
            "net_change": int(col_sum - row_sum),
        }
    result["per_class"] = per_class

    logger.info(f"Transition matrix: {result['persistence_rate']}% persistence, "
                f"{result['change_pixels']} changed pixels")
    return result


def main():
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
