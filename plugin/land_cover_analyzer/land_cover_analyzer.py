"""
Main QGIS plugin class for Land Cover Analyzer.

Integrates with the QGIS interface to provide land cover classification
tools via the toolbar and processing framework.
"""

import os
import json

from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsProcessingFeedback,
    QgsMessageLog,
    Qgis,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon

from .land_cover_dialog import LandCoverDialog


class LandCoverAnalyzerPlugin:
    """QGIS Plugin for land cover classification and analysis."""

    PLUGIN_NAME = "Land Cover Analyzer"

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dialog = None

    def initGui(self):
        """Create plugin GUI elements."""
        icon_path = os.path.join(self.plugin_dir, "icon.png")
        self.action = QAction(
            QIcon(icon_path) if os.path.exists(icon_path) else QIcon(),
            self.PLUGIN_NAME,
            self.iface.mainWindow(),
        )
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu(self.PLUGIN_NAME, self.action)

    def unload(self):
        """Remove plugin GUI elements."""
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginRasterMenu(self.PLUGIN_NAME, self.action)

    def run(self):
        """Show the plugin dialog."""
        raster_layers = [
            layer for layer in QgsProject.instance().mapLayers().values()
            if isinstance(layer, QgsRasterLayer)
        ]

        if not raster_layers:
            self.iface.messageBar().pushMessage(
                self.PLUGIN_NAME,
                "No raster layers found. Add a raster layer first.",
                level=Qgis.Warning,
                duration=5,
            )
            return

        self.dialog = LandCoverDialog(
            parent=self.iface.mainWindow(),
            raster_layers=raster_layers,
        )
        self.dialog.classification_requested.connect(self._run_classification)
        self.dialog.show()

    def _run_classification(self, params: dict):
        """Execute classification with the given parameters."""
        from .classifier import UnsupervisedClassifier, SpectralIndices

        layer = params["layer"]
        provider = layer.dataProvider()
        width = layer.width()
        height = layer.height()
        n_bands = layer.bandCount()

        QgsMessageLog.logMessage(
            f"Starting classification: {width}x{height}, {n_bands} bands, "
            f"method={params['method']}, classes={params['n_classes']}",
            self.PLUGIN_NAME,
            Qgis.Info,
        )

        # Read bands into numpy array
        import numpy as np

        band_stack = np.zeros((n_bands, height, width), dtype=np.float32)
        for band_idx in range(n_bands):
            block = provider.block(band_idx + 1, layer.extent(), width, height)
            for row in range(height):
                for col in range(width):
                    band_stack[band_idx, row, col] = block.value(row, col)

        # Classify
        classifier = UnsupervisedClassifier(
            n_classes=params["n_classes"],
            method=params["method"],
        )
        classified = classifier.fit_predict(band_stack)

        # Write output
        self._write_classified_raster(
            classified, layer, params["output_path"]
        )

        # Compute spectral indices if requested
        if params.get("compute_indices") and n_bands >= 4:
            indices = SpectralIndices()
            ndvi = indices.ndvi(band_stack[3], band_stack[2])  # NIR=4, Red=3
            ndvi_path = params["output_path"].replace(".tif", "_ndvi.tif")
            self._write_single_band(ndvi, layer, ndvi_path)

        # Save accuracy report if reference provided
        if params.get("reference_path"):
            self._generate_accuracy_report(
                classified, params["reference_path"], params["output_path"]
            )

        self.iface.messageBar().pushMessage(
            self.PLUGIN_NAME,
            f"Classification complete. Output: {params['output_path']}",
            level=Qgis.Success,
            duration=5,
        )

        # Load result into QGIS
        result_layer = QgsRasterLayer(params["output_path"], "Classified Land Cover")
        if result_layer.isValid():
            QgsProject.instance().addMapLayer(result_layer)

    def _write_classified_raster(self, data, reference_layer, output_path):
        """Write classification result as GeoTIFF."""
        import numpy as np

        try:
            import rasterio
            from rasterio.transform import from_bounds

            extent = reference_layer.extent()
            crs = reference_layer.crs().toWkt()
            height, width = data.shape

            transform = from_bounds(
                extent.xMinimum(), extent.yMinimum(),
                extent.xMaximum(), extent.yMaximum(),
                width, height,
            )

            with rasterio.open(
                output_path, "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=np.int32,
                crs=crs,
                transform=transform,
                compress="lzw",
            ) as dst:
                dst.write(data, 1)

        except ImportError:
            # Fallback: use GDAL directly
            from osgeo import gdal, osr

            extent = reference_layer.extent()
            height, width = data.shape
            x_res = (extent.xMaximum() - extent.xMinimum()) / width
            y_res = (extent.yMaximum() - extent.yMinimum()) / height

            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(output_path, width, height, 1, gdal.GDT_Int32,
                               options=["COMPRESS=LZW"])
            ds.SetGeoTransform([
                extent.xMinimum(), x_res, 0,
                extent.yMaximum(), 0, -y_res,
            ])

            srs = osr.SpatialReference()
            srs.ImportFromWkt(reference_layer.crs().toWkt())
            ds.SetProjection(srs.ExportToWkt())

            ds.GetRasterBand(1).WriteArray(data)
            ds.FlushCache()
            ds = None

    def _write_single_band(self, data, reference_layer, output_path):
        """Write a single-band float raster."""
        import numpy as np

        try:
            import rasterio
            from rasterio.transform import from_bounds

            extent = reference_layer.extent()
            height, width = data.shape
            transform = from_bounds(
                extent.xMinimum(), extent.yMinimum(),
                extent.xMaximum(), extent.yMaximum(),
                width, height,
            )

            with rasterio.open(
                output_path, "w",
                driver="GTiff",
                height=height, width=width, count=1,
                dtype=np.float32,
                crs=reference_layer.crs().toWkt(),
                transform=transform,
                compress="lzw",
            ) as dst:
                dst.write(data, 1)
        except ImportError:
            from osgeo import gdal, osr

            extent = reference_layer.extent()
            height, width = data.shape
            x_res = (extent.xMaximum() - extent.xMinimum()) / width
            y_res = (extent.yMaximum() - extent.yMinimum()) / height

            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32,
                               options=["COMPRESS=LZW"])
            ds.SetGeoTransform([
                extent.xMinimum(), x_res, 0,
                extent.yMaximum(), 0, -y_res,
            ])

            srs = osr.SpatialReference()
            srs.ImportFromWkt(reference_layer.crs().toWkt())
            ds.SetProjection(srs.ExportToWkt())

            ds.GetRasterBand(1).WriteArray(data)
            ds.FlushCache()
            ds = None

    def _generate_accuracy_report(self, classified, reference_path, output_path):
        """Generate accuracy assessment report."""
        import numpy as np
        from .classifier import AccuracyAssessment

        try:
            import rasterio
            with rasterio.open(reference_path) as src:
                reference = src.read(1)
        except ImportError:
            from osgeo import gdal
            ds = gdal.Open(reference_path)
            reference = ds.GetRasterBand(1).ReadAsArray()
            ds = None

        assessment = AccuracyAssessment(classified, reference)
        report = assessment.report()

        report_path = output_path.replace(".tif", "_accuracy.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        QgsMessageLog.logMessage(
            f"Accuracy report saved: {report_path} "
            f"(OA={report['overall_accuracy']:.3f}, Kappa={report['kappa']:.3f})",
            self.PLUGIN_NAME,
            Qgis.Info,
        )
