"""Tests for the land cover classifier module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugin" / "land_cover_analyzer"))
from classifier import SpectralIndices, UnsupervisedClassifier, AccuracyAssessment


class TestSpectralIndices:
    def test_ndvi_basic(self):
        nir = np.array([[0.8, 0.6], [0.4, 0.2]], dtype=np.float32)
        red = np.array([[0.1, 0.3], [0.3, 0.5]], dtype=np.float32)
        result = SpectralIndices.ndvi(nir, red)
        expected = (nir - red) / (nir + red)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_ndvi_zero_denominator(self):
        nir = np.array([[0.0]], dtype=np.float32)
        red = np.array([[0.0]], dtype=np.float32)
        result = SpectralIndices.ndvi(nir, red)
        assert result[0, 0] == 0.0

    def test_ndvi_range(self):
        rng = np.random.RandomState(42)
        nir = rng.rand(100, 100).astype(np.float32)
        red = rng.rand(100, 100).astype(np.float32)
        result = SpectralIndices.ndvi(nir, red)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_ndwi_basic(self):
        green = np.array([[0.5, 0.3]], dtype=np.float32)
        nir = np.array([[0.2, 0.7]], dtype=np.float32)
        result = SpectralIndices.ndwi(green, nir)
        expected = (green - nir) / (green + nir)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_ndbi_basic(self):
        swir = np.array([[0.6]], dtype=np.float32)
        nir = np.array([[0.3]], dtype=np.float32)
        result = SpectralIndices.ndbi(swir, nir)
        expected = np.array([[(0.6 - 0.3) / (0.6 + 0.3)]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestUnsupervisedClassifier:
    def _make_test_data(self, n_classes=3):
        """Create synthetic multispectral data with known clusters."""
        rng = np.random.RandomState(42)
        height, width = 50, 50
        n_bands = 4
        data = np.zeros((n_bands, height, width), dtype=np.float32)

        rows_per_class = height // n_classes
        for i in range(n_classes):
            r_start = i * rows_per_class
            r_end = r_start + rows_per_class
            for b in range(n_bands):
                data[b, r_start:r_end, :] = (i + 1) * 50 + rng.randn(
                    rows_per_class, width
                ).astype(np.float32) * 5

        return data

    def test_kmeans_produces_expected_classes(self):
        data = self._make_test_data(n_classes=3)
        clf = UnsupervisedClassifier(n_classes=3, method="kmeans")
        result = clf.fit_predict(data)
        assert result.shape == (50, 50)
        # Should have classes 1, 2, 3 (0 is nodata)
        unique = set(np.unique(result))
        assert 0 not in unique or len(unique) <= 4
        assert len(unique - {0}) == 3

    def test_iso_cluster(self):
        data = self._make_test_data(n_classes=3)
        clf = UnsupervisedClassifier(n_classes=3, method="iso_cluster")
        result = clf.fit_predict(data)
        assert result.shape == (50, 50)
        assert len(set(np.unique(result)) - {0}) == 3

    def test_all_nodata(self):
        data = np.zeros((4, 10, 10), dtype=np.float32)
        clf = UnsupervisedClassifier(n_classes=3, method="kmeans")
        result = clf.fit_predict(data)
        assert np.all(result == 0)

    def test_invalid_method(self):
        data = self._make_test_data()
        clf = UnsupervisedClassifier(n_classes=3, method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            clf.fit_predict(data)


class TestAccuracyAssessment:
    def test_perfect_accuracy(self):
        pred = np.array([1, 2, 3, 1, 2, 3])
        ref = np.array([1, 2, 3, 1, 2, 3])
        aa = AccuracyAssessment(pred, ref)
        assert aa.overall_accuracy() == 1.0
        assert aa.kappa() == 1.0

    def test_zero_accuracy(self):
        pred = np.array([1, 1, 1])
        ref = np.array([2, 2, 2])
        aa = AccuracyAssessment(pred, ref)
        assert aa.overall_accuracy() == 0.0

    def test_confusion_matrix_shape(self):
        pred = np.array([1, 2, 3, 1, 2, 3])
        ref = np.array([1, 2, 1, 3, 2, 3])
        aa = AccuracyAssessment(pred, ref)
        cm = aa.confusion_matrix()
        assert cm.shape == (3, 3)

    def test_report_keys(self):
        pred = np.array([1, 2, 3])
        ref = np.array([1, 2, 2])
        aa = AccuracyAssessment(pred, ref)
        report = aa.report()
        assert "overall_accuracy" in report
        assert "kappa" in report
        assert "per_class" in report
        assert "confusion_matrix" in report

    def test_nodata_filtered(self):
        pred = np.array([0, 1, 2, 0])
        ref = np.array([0, 1, 2, 0])
        aa = AccuracyAssessment(pred, ref)
        assert aa.overall_accuracy() == 1.0

    def test_empty_data(self):
        pred = np.array([0, 0])
        ref = np.array([0, 0])
        aa = AccuracyAssessment(pred, ref)
        assert aa.overall_accuracy() == 0.0
