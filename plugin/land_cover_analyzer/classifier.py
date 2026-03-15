"""
Raster classification engine for land cover analysis.

Supports unsupervised classification methods and spectral index computation
on multispectral raster imagery.
"""

import numpy as np


class SpectralIndices:
    """Compute common spectral indices from multispectral bands."""

    @staticmethod
    def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1. Higher values indicate denser vegetation.
        """
        nir = nir.astype(np.float64)
        red = red.astype(np.float64)
        denominator = nir + red
        result = np.where(denominator != 0, (nir - red) / denominator, 0.0)
        return result.astype(np.float32)

    @staticmethod
    def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)
        Range: -1 to 1. Higher values indicate water bodies.
        """
        green = green.astype(np.float64)
        nir = nir.astype(np.float64)
        denominator = green + nir
        result = np.where(denominator != 0, (green - nir) / denominator, 0.0)
        return result.astype(np.float32)

    @staticmethod
    def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Built-up Index.

        NDBI = (SWIR - NIR) / (SWIR + NIR)
        Range: -1 to 1. Higher values indicate built-up areas.
        """
        swir = swir.astype(np.float64)
        nir = nir.astype(np.float64)
        denominator = swir + nir
        result = np.where(denominator != 0, (swir - nir) / denominator, 0.0)
        return result.astype(np.float32)


class UnsupervisedClassifier:
    """Unsupervised pixel-based classification of multispectral rasters."""

    def __init__(self, n_classes: int = 5, method: str = "kmeans",
                 max_iterations: int = 100, random_state: int = 42):
        self.n_classes = n_classes
        self.method = method
        self.max_iterations = max_iterations
        self.random_state = random_state
        self._model = None
        self._labels = None

    def fit_predict(self, band_stack: np.ndarray) -> np.ndarray:
        """Classify a stacked raster array.

        Args:
            band_stack: Array of shape (bands, height, width).

        Returns:
            Classified array of shape (height, width) with integer class labels.
        """
        n_bands, height, width = band_stack.shape
        pixels = band_stack.reshape(n_bands, -1).T  # (n_pixels, n_bands)

        # Mask out nodata (all-zero pixels)
        valid_mask = np.any(pixels != 0, axis=1)
        valid_pixels = pixels[valid_mask]

        if valid_pixels.shape[0] == 0:
            return np.zeros((height, width), dtype=np.int32)

        if self.method == "kmeans":
            labels = self._kmeans(valid_pixels)
        elif self.method == "iso_cluster":
            labels = self._iso_cluster(valid_pixels)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'kmeans' or 'iso_cluster'.")

        result = np.zeros(pixels.shape[0], dtype=np.int32)
        result[valid_mask] = labels + 1  # 0 = nodata, 1..n = classes

        self._labels = result.reshape(height, width)
        return self._labels

    def _kmeans(self, pixels: np.ndarray) -> np.ndarray:
        """K-Means clustering."""
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=self.n_classes,
            max_iter=self.max_iterations,
            random_state=self.random_state,
            batch_size=min(10000, pixels.shape[0]),
        )
        self._model = model
        return model.fit_predict(pixels)

    def _iso_cluster(self, pixels: np.ndarray) -> np.ndarray:
        """ISO Cluster (iterative self-organizing) classification.

        Uses Gaussian Mixture Model as the underlying algorithm, which
        allows clusters to split and merge based on data distribution.
        """
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=self.n_classes,
            max_iter=self.max_iterations,
            random_state=self.random_state,
            covariance_type="full",
        )
        self._model = model
        model.fit(pixels)
        return model.predict(pixels)


class AccuracyAssessment:
    """Compute classification accuracy metrics."""

    def __init__(self, predicted: np.ndarray, reference: np.ndarray):
        self.predicted = predicted.ravel()
        self.reference = reference.ravel()
        # Only compare where both have valid data
        valid = (self.predicted > 0) & (self.reference > 0)
        self.predicted = self.predicted[valid]
        self.reference = self.reference[valid]

    def confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix."""
        from sklearn.metrics import confusion_matrix
        classes = np.union1d(self.predicted, self.reference)
        return confusion_matrix(self.reference, self.predicted, labels=classes)

    def overall_accuracy(self) -> float:
        """Overall classification accuracy."""
        if len(self.predicted) == 0:
            return 0.0
        return float(np.mean(self.predicted == self.reference))

    def kappa(self) -> float:
        """Cohen's Kappa coefficient."""
        from sklearn.metrics import cohen_kappa_score
        if len(self.predicted) == 0:
            return 0.0
        return cohen_kappa_score(self.reference, self.predicted)

    def per_class_accuracy(self) -> dict:
        """Per-class producer's and user's accuracy."""
        cm = self.confusion_matrix()
        classes = np.union1d(self.predicted, self.reference)
        result = {}
        for i, cls in enumerate(classes):
            col_sum = cm[:, i].sum()
            row_sum = cm[i, :].sum()
            result[int(cls)] = {
                "producers_accuracy": float(cm[i, i] / col_sum) if col_sum > 0 else 0.0,
                "users_accuracy": float(cm[i, i] / row_sum) if row_sum > 0 else 0.0,
            }
        return result

    def report(self) -> dict:
        """Full accuracy report."""
        return {
            "overall_accuracy": self.overall_accuracy(),
            "kappa": self.kappa(),
            "per_class": self.per_class_accuracy(),
            "confusion_matrix": self.confusion_matrix().tolist(),
        }
