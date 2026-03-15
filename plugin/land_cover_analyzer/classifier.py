"""
Raster Classification Engine for Land Cover Analysis
=====================================================

This module provides the core classification and accuracy assessment logic
for the Land Cover Analyzer plugin. It is designed to work with multispectral
satellite imagery (e.g., Sentinel-2, Landsat) and can be used both inside
QGIS (via the plugin) and as a standalone library.

Three main components:

1. SpectralIndices   — Computes normalized difference indices (NDVI, NDWI, NDBI)
                       that highlight specific land cover types.

2. UnsupervisedClassifier — Pixel-based unsupervised classification using
                            K-Means or ISO Cluster (Gaussian Mixture Models).

3. AccuracyAssessment — Evaluates classification quality against reference data
                        using confusion matrices, Kappa statistics, and
                        per-class producer's/user's accuracy.

Dependencies: numpy, scikit-learn (lazy-imported to keep startup fast)
"""

import numpy as np


# =============================================================================
# Spectral Index Calculation
# =============================================================================

class SpectralIndices:
    """Compute common remote sensing spectral indices from individual bands.

    Spectral indices are band math formulas that amplify specific surface
    characteristics. They're the bread and butter of remote sensing analysis:

    - NDVI highlights vegetation health/density
    - NDWI highlights open water bodies
    - NDBI highlights built-up/urban areas

    All indices follow the normalized difference formula:
        Index = (Band_A - Band_B) / (Band_A + Band_B)

    This normalizes values to the [-1, +1] range regardless of sensor
    calibration, making results comparable across images and dates.
    """

    @staticmethod
    def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Interpretation:
            -1.0 to 0.0  → Water, bare soil, clouds
             0.0 to 0.2  → Bare soil, rock, sand
             0.2 to 0.4  → Sparse vegetation (shrubs, grassland)
             0.4 to 0.6  → Moderate vegetation (crops, light forest)
             0.6 to 1.0  → Dense vegetation (healthy forest, irrigated crops)

        Args:
            nir: Near-infrared band array (e.g., Sentinel-2 Band 8).
            red: Red band array (e.g., Sentinel-2 Band 4).

        Returns:
            Float32 array with NDVI values in [-1, 1].
        """
        # Upcast to float64 for numerical precision in the division
        nir = nir.astype(np.float64)
        red = red.astype(np.float64)
        denominator = nir + red

        # Where both bands are zero (e.g., nodata), output 0 instead of NaN
        result = np.where(denominator != 0, (nir - red) / denominator, 0.0)

        # Return as float32 to save memory in large rasters
        return result.astype(np.float32)

    @staticmethod
    def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Water Index (McFeeters, 1996).

        NDWI = (Green - NIR) / (Green + NIR)

        Interpretation:
            Values > 0.0  → Likely water
            Values < 0.0  → Likely non-water (vegetation, soil, built-up)

        Args:
            green: Green band array (e.g., Sentinel-2 Band 3).
            nir: Near-infrared band array (e.g., Sentinel-2 Band 8).

        Returns:
            Float32 array with NDWI values in [-1, 1].
        """
        green = green.astype(np.float64)
        nir = nir.astype(np.float64)
        denominator = green + nir
        result = np.where(denominator != 0, (green - nir) / denominator, 0.0)
        return result.astype(np.float32)

    @staticmethod
    def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Built-up Index (Zha et al., 2003).

        NDBI = (SWIR - NIR) / (SWIR + NIR)

        Interpretation:
            Values > 0.0  → Built-up / urban areas
            Values < 0.0  → Vegetation or water

        Args:
            swir: Short-wave infrared band array (e.g., Sentinel-2 Band 11).
            nir: Near-infrared band array (e.g., Sentinel-2 Band 8).

        Returns:
            Float32 array with NDBI values in [-1, 1].
        """
        swir = swir.astype(np.float64)
        nir = nir.astype(np.float64)
        denominator = swir + nir
        result = np.where(denominator != 0, (swir - nir) / denominator, 0.0)
        return result.astype(np.float32)


# =============================================================================
# Unsupervised Classification
# =============================================================================

class UnsupervisedClassifier:
    """Unsupervised pixel-based classification of multispectral rasters.

    "Unsupervised" means the algorithm discovers class structure on its own —
    no training labels required. This is useful when:
    - You don't have ground truth data yet
    - You want to discover natural groupings in the imagery
    - You need a quick first-pass classification

    Supported methods:
        "kmeans"      — MiniBatch K-Means: fast, scalable, good for most cases.
                        Assigns each pixel to the nearest cluster centroid.

        "iso_cluster" — ISO Cluster via Gaussian Mixture Model: more flexible,
                        allows elliptical clusters (K-Means forces spherical).
                        Better for overlapping spectral signatures.

    Output convention:
        0 = nodata (pixels where all bands are zero)
        1..n = class labels
    """

    def __init__(self, n_classes: int = 5, method: str = "kmeans",
                 max_iterations: int = 100, random_state: int = 42):
        """
        Args:
            n_classes: Number of land cover classes to identify.
            method: 'kmeans' or 'iso_cluster'.
            max_iterations: Maximum clustering iterations.
            random_state: Seed for reproducible results.
        """
        self.n_classes = n_classes
        self.method = method
        self.max_iterations = max_iterations
        self.random_state = random_state
        self._model = None     # Stores the fitted sklearn model for inspection
        self._labels = None    # Stores the last classification result

    def fit_predict(self, band_stack: np.ndarray) -> np.ndarray:
        """Classify a stacked raster array into land cover classes.

        The process:
        1. Reshape (bands, H, W) → (pixels, bands) — each pixel becomes a row
        2. Mask out nodata pixels (all-zero across bands)
        3. Run clustering on valid pixels only
        4. Map cluster labels back to the spatial grid

        Args:
            band_stack: Array of shape (bands, height, width).
                        For Sentinel-2: typically 4-13 bands.

        Returns:
            Classified array of shape (height, width) with integer class labels.
            0 = nodata, 1..n_classes = classified land cover types.
        """
        n_bands, height, width = band_stack.shape

        # Reshape from image space to feature space:
        # (4, 1024, 1024) → (1048576, 4) — each pixel is a 4-dimensional point
        pixels = band_stack.reshape(n_bands, -1).T

        # Identify valid pixels (at least one non-zero band value).
        # This prevents nodata areas from influencing the clustering.
        valid_mask = np.any(pixels != 0, axis=1)
        valid_pixels = pixels[valid_mask]

        # If the entire image is nodata, return an all-zero result
        if valid_pixels.shape[0] == 0:
            return np.zeros((height, width), dtype=np.int32)

        # Dispatch to the chosen clustering algorithm
        if self.method == "kmeans":
            labels = self._kmeans(valid_pixels)
        elif self.method == "iso_cluster":
            labels = self._iso_cluster(valid_pixels)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'kmeans' or 'iso_cluster'.")

        # Map cluster labels back to the full pixel array.
        # Add 1 so that class labels start at 1 (0 is reserved for nodata).
        result = np.zeros(pixels.shape[0], dtype=np.int32)
        result[valid_mask] = labels + 1

        # Reshape back to image space: (1048576,) → (1024, 1024)
        self._labels = result.reshape(height, width)
        return self._labels

    def _kmeans(self, pixels: np.ndarray) -> np.ndarray:
        """K-Means clustering using MiniBatch variant for scalability.

        MiniBatchKMeans processes random subsets of data per iteration,
        making it ~10x faster than standard KMeans on large rasters
        with minimal accuracy loss.
        """
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=self.n_classes,
            max_iter=self.max_iterations,
            random_state=self.random_state,
            # Cap batch size to avoid memory issues on small datasets
            batch_size=min(10000, pixels.shape[0]),
        )
        self._model = model
        return model.fit_predict(pixels)

    def _iso_cluster(self, pixels: np.ndarray) -> np.ndarray:
        """ISO Cluster classification using Gaussian Mixture Models.

        ISO Cluster (Iterative Self-Organizing) is a standard remote sensing
        classification method. Unlike K-Means (which assumes spherical clusters),
        GMM models each cluster as a multivariate Gaussian, allowing:
        - Elliptical cluster shapes
        - Different cluster sizes
        - Soft assignment (probability per class, though we use hard labels here)

        The "full" covariance type gives maximum flexibility but requires
        more data points per cluster to estimate reliably.
        """
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=self.n_classes,
            max_iter=self.max_iterations,
            random_state=self.random_state,
            covariance_type="full",  # Each cluster has its own covariance matrix
        )
        self._model = model
        model.fit(pixels)
        return model.predict(pixels)


# =============================================================================
# Accuracy Assessment
# =============================================================================

class AccuracyAssessment:
    """Evaluate classification quality against ground truth reference data.

    Standard remote sensing accuracy metrics following Congalton & Green (2019):

    - Overall Accuracy (OA): fraction of correctly classified pixels
    - Cohen's Kappa: accuracy adjusted for chance agreement (0 = random, 1 = perfect)
    - Producer's Accuracy: per-class recall (how much of class X did we find?)
    - User's Accuracy: per-class precision (when we say class X, how often are we right?)

    These metrics are what journal papers and land management agencies expect
    to see when evaluating a classification.
    """

    def __init__(self, predicted: np.ndarray, reference: np.ndarray):
        """
        Args:
            predicted: Classified raster array (any shape, will be flattened).
            reference: Ground truth raster array (same shape as predicted).

        Note: Pixels with value 0 in either array are treated as nodata
              and excluded from accuracy calculation.
        """
        self.predicted = predicted.ravel()
        self.reference = reference.ravel()

        # Filter to only pixels where both predicted and reference have valid data.
        # Class label 0 is reserved for nodata in our convention.
        valid = (self.predicted > 0) & (self.reference > 0)
        self.predicted = self.predicted[valid]
        self.reference = self.reference[valid]

    def confusion_matrix(self) -> np.ndarray:
        """Generate a confusion matrix (error matrix).

        Rows = reference classes, Columns = predicted classes.
        Diagonal elements = correctly classified pixels.
        Off-diagonal = misclassifications.
        """
        from sklearn.metrics import confusion_matrix
        classes = np.union1d(self.predicted, self.reference)
        return confusion_matrix(self.reference, self.predicted, labels=classes)

    def overall_accuracy(self) -> float:
        """Overall accuracy: fraction of all pixels that are correctly classified.

        OA = (sum of diagonal) / (total pixels)
        """
        if len(self.predicted) == 0:
            return 0.0
        return float(np.mean(self.predicted == self.reference))

    def kappa(self) -> float:
        """Cohen's Kappa coefficient: classification accuracy adjusted for chance.

        Kappa = (OA - Pe) / (1 - Pe)
        where Pe = expected accuracy by random chance.

        Interpretation:
            < 0.20  → Poor
            0.20-0.40 → Fair
            0.40-0.60 → Moderate
            0.60-0.80 → Good
            0.80-1.00 → Excellent
        """
        from sklearn.metrics import cohen_kappa_score
        if len(self.predicted) == 0:
            return 0.0
        return cohen_kappa_score(self.reference, self.predicted)

    def per_class_accuracy(self) -> dict:
        """Compute per-class producer's and user's accuracy.

        Producer's accuracy (recall):
            "Of all reference pixels for class X, how many did we classify correctly?"
            = diagonal / column sum
            Important for: the data *producer* who needs to know detection rates.

        User's accuracy (precision):
            "Of all pixels we *called* class X, how many actually are class X?"
            = diagonal / row sum
            Important for: the map *user* who needs to trust the labels.
        """
        cm = self.confusion_matrix()
        classes = np.union1d(self.predicted, self.reference)
        result = {}
        for i, cls in enumerate(classes):
            col_sum = cm[:, i].sum()  # Total reference pixels for this class
            row_sum = cm[i, :].sum()  # Total predicted pixels for this class
            result[int(cls)] = {
                "producers_accuracy": float(cm[i, i] / col_sum) if col_sum > 0 else 0.0,
                "users_accuracy": float(cm[i, i] / row_sum) if row_sum > 0 else 0.0,
            }
        return result

    def report(self) -> dict:
        """Generate a complete accuracy report as a dictionary.

        Returns a JSON-serializable dict with all metrics, suitable for
        saving to disk or displaying in the QGIS message log.
        """
        return {
            "overall_accuracy": self.overall_accuracy(),
            "kappa": self.kappa(),
            "per_class": self.per_class_accuracy(),
            "confusion_matrix": self.confusion_matrix().tolist(),
        }
