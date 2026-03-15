"""
Main QGIS Plugin Class for Land Cover Analyzer
================================================

This module wires together the QGIS interface, the classification engine,
and the output I/O layer. It handles:

1. Plugin lifecycle (initGui / unload) — adding/removing toolbar buttons
2. User interaction — launching the config dialog, validating inputs
3. Classification orchestration — reading bands, running the classifier,
   writing results, generating accuracy reports
4. Output I/O — writing GeoTIFF via rasterio (preferred) or GDAL (fallback)

The plugin follows the standard QGIS plugin pattern:
    classFactory() → __init__() → initGui() → run() → unload()
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
    """QGIS Plugin for land cover classification and analysis.

    Provides a toolbar button and raster menu entry that opens a configuration
    dialog. When the user clicks "Classify", this class:
    1. Reads all bands from the selected raster layer
    2. Runs unsupervised classification (K-Means or ISO Cluster)
    3. Optionally computes spectral indices (NDVI, NDWI)
    4. Writes the classified map as a compressed GeoTIFF
    5. Optionally generates an accuracy report against a reference raster
    6. Loads the result back into the QGIS map canvas
    """

    PLUGIN_NAME = "Land Cover Analyzer"

    def __init__(self, iface):
        """
        Args:
            iface: QgisInterface — the main QGIS application interface.
                   Used to access the map canvas, message bar, menus, etc.
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None   # QAction for the toolbar button
        self.dialog = None   # LandCoverDialog instance

    # -------------------------------------------------------------------------
    # Plugin Lifecycle
    # -------------------------------------------------------------------------

    def initGui(self):
        """Called by QGIS when the plugin is loaded.

        Creates the toolbar icon and menu entry. The icon is loaded from
        icon.png if it exists, otherwise a blank QIcon is used.
        """
        icon_path = os.path.join(self.plugin_dir, "icon.png")
        self.action = QAction(
            QIcon(icon_path) if os.path.exists(icon_path) else QIcon(),
            self.PLUGIN_NAME,
            self.iface.mainWindow(),
        )
        self.action.triggered.connect(self.run)

        # Add to both the toolbar and the Raster menu
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu(self.PLUGIN_NAME, self.action)

    def unload(self):
        """Called by QGIS when the plugin is unloaded — clean up GUI elements."""
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginRasterMenu(self.PLUGIN_NAME, self.action)

    # -------------------------------------------------------------------------
    # User Interaction
    # -------------------------------------------------------------------------

    def run(self):
        """Show the classification dialog.

        Scans the current QGIS project for raster layers and populates the
        dialog's layer selector. If no raster layers are loaded, shows a
        warning message instead of the dialog.
        """
        # Collect all raster layers currently loaded in the project
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

        # Create and show the dialog, connecting its signal to our handler
        self.dialog = LandCoverDialog(
            parent=self.iface.mainWindow(),
            raster_layers=raster_layers,
        )
        self.dialog.classification_requested.connect(self._run_classification)
        self.dialog.show()

    # -------------------------------------------------------------------------
    # Classification Pipeline
    # -------------------------------------------------------------------------

    def _run_classification(self, params: dict):
        """Execute the full classification pipeline.

        This is the main workhorse method, triggered when the user clicks
        "Classify" in the dialog.

        Args:
            params: Dictionary from LandCoverDialog containing:
                - layer: QgsRasterLayer to classify
                - method: "kmeans" or "iso_cluster"
                - n_classes: number of classes (2-50)
                - output_path: file path for classified GeoTIFF
                - compute_indices: bool — also compute NDVI/NDWI?
                - reference_path: optional path to ground truth raster
        """
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

        # --- Step 1: Read all bands into a numpy array ---
        # We read the full raster into memory. For very large rasters,
        # consider using the batch_raster_analysis.py script instead,
        # which processes tiles in parallel.
        import numpy as np

        band_stack = np.zeros((n_bands, height, width), dtype=np.float32)
        for band_idx in range(n_bands):
            # QGIS bands are 1-indexed, so we add 1
            block = provider.block(band_idx + 1, layer.extent(), width, height)
            for row in range(height):
                for col in range(width):
                    band_stack[band_idx, row, col] = block.value(row, col)

        # --- Step 2: Run classification ---
        classifier = UnsupervisedClassifier(
            n_classes=params["n_classes"],
            method=params["method"],
        )
        classified = classifier.fit_predict(band_stack)

        # --- Step 3: Write classified raster to disk ---
        self._write_classified_raster(
            classified, layer, params["output_path"]
        )

        # --- Step 4: Compute spectral indices if requested ---
        # Requires at least 4 bands (assumes standard ordering: B,G,R,NIR)
        if params.get("compute_indices") and n_bands >= 4:
            indices = SpectralIndices()
            # Standard Sentinel-2 band ordering: B2=Blue, B3=Green, B4=Red, B8=NIR
            ndvi = indices.ndvi(nir=band_stack[3], red=band_stack[2])
            ndvi_path = params["output_path"].replace(".tif", "_ndvi.tif")
            self._write_single_band(ndvi, layer, ndvi_path)

        # --- Step 5: Generate accuracy report if reference data provided ---
        if params.get("reference_path"):
            self._generate_accuracy_report(
                classified, params["reference_path"], params["output_path"]
            )

        # --- Step 6: Notify user and load result into QGIS ---
        self.iface.messageBar().pushMessage(
            self.PLUGIN_NAME,
            f"Classification complete. Output: {params['output_path']}",
            level=Qgis.Success,
            duration=5,
        )

        result_layer = QgsRasterLayer(params["output_path"], "Classified Land Cover")
        if result_layer.isValid():
            QgsProject.instance().addMapLayer(result_layer)

    # -------------------------------------------------------------------------
    # Output I/O — Two backends for maximum compatibility
    # -------------------------------------------------------------------------

    def _write_classified_raster(self, data, reference_layer, output_path):
        """Write classification result as a LZW-compressed GeoTIFF.

        Tries rasterio first (cleaner API), falls back to GDAL if rasterio
        is not installed. The output inherits CRS and extent from the
        reference (input) layer.
        """
        import numpy as np

        try:
            import rasterio
            from rasterio.transform import from_bounds

            extent = reference_layer.extent()
            crs = reference_layer.crs().toWkt()
            height, width = data.shape

            # Build the affine transform from the layer's geographic extent
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
                compress="lzw",  # Lossless compression, ~50-70% size reduction
            ) as dst:
                dst.write(data, 1)

        except ImportError:
            # Fallback: use GDAL Python bindings (available in all QGIS installs)
            from osgeo import gdal, osr

            extent = reference_layer.extent()
            height, width = data.shape
            x_res = (extent.xMaximum() - extent.xMinimum()) / width
            y_res = (extent.yMaximum() - extent.yMinimum()) / height

            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(output_path, width, height, 1, gdal.GDT_Int32,
                               options=["COMPRESS=LZW"])

            # GeoTransform: [origin_x, pixel_width, rotation, origin_y, rotation, -pixel_height]
            ds.SetGeoTransform([
                extent.xMinimum(), x_res, 0,
                extent.yMaximum(), 0, -y_res,
            ])

            srs = osr.SpatialReference()
            srs.ImportFromWkt(reference_layer.crs().toWkt())
            ds.SetProjection(srs.ExportToWkt())

            ds.GetRasterBand(1).WriteArray(data)
            ds.FlushCache()
            ds = None  # Close the dataset

    def _write_single_band(self, data, reference_layer, output_path):
        """Write a single-band float raster (e.g., NDVI index layer)."""
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

    # -------------------------------------------------------------------------
    # Accuracy Assessment
    # -------------------------------------------------------------------------

    def _generate_accuracy_report(self, classified, reference_path, output_path):
        """Compare classified result against a reference raster and save a report.

        The report includes overall accuracy, Cohen's Kappa, per-class
        producer's/user's accuracy, and the full confusion matrix.
        Saved as a JSON file alongside the classified output.
        """
        import numpy as np
        from .classifier import AccuracyAssessment

        # Read the reference (ground truth) raster
        try:
            import rasterio
            with rasterio.open(reference_path) as src:
                reference = src.read(1)
        except ImportError:
            from osgeo import gdal
            ds = gdal.Open(reference_path)
            reference = ds.GetRasterBand(1).ReadAsArray()
            ds = None

        # Run accuracy assessment
        assessment = AccuracyAssessment(classified, reference)
        report = assessment.report()

        # Save report as JSON alongside the output
        report_path = output_path.replace(".tif", "_accuracy.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        QgsMessageLog.logMessage(
            f"Accuracy report saved: {report_path} "
            f"(OA={report['overall_accuracy']:.3f}, Kappa={report['kappa']:.3f})",
            self.PLUGIN_NAME,
            Qgis.Info,
        )
