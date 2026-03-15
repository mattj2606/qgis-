"""
Configuration Dialog for Land Cover Analyzer
==============================================

PyQt5 dialog that lets the user configure classification parameters
before running the analysis. Provides:

- Raster layer selection (populated from current QGIS project)
- Classification method choice (K-Means vs. ISO Cluster)
- Number of classes slider (2-50)
- Spectral indices toggle
- Output file browser
- Optional reference raster for accuracy assessment

When the user clicks "Classify", the dialog emits a `classification_requested`
signal with all parameters as a dictionary, then closes.
"""

import os

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QGroupBox,
    QFormLayout,
)


class LandCoverDialog(QDialog):
    """Configuration dialog for land cover classification.

    Emits:
        classification_requested(dict): Signal with all user-chosen parameters.
            Keys: layer, method, n_classes, output_path, compute_indices,
                  reference_path (or None)
    """

    # Custom signal emitted when the user clicks "Classify"
    classification_requested = pyqtSignal(dict)

    def __init__(self, parent=None, raster_layers=None):
        """
        Args:
            parent: Parent widget (usually the QGIS main window).
            raster_layers: List of QgsRasterLayer objects to populate
                           the layer selector dropdown.
        """
        super().__init__(parent)
        self.raster_layers = raster_layers or []
        self.setWindowTitle("Land Cover Analyzer")
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self):
        """Construct the dialog layout with three groups: Input, Parameters, Output."""
        layout = QVBoxLayout()

        # --- Input Layer Selection ---
        input_group = QGroupBox("Input")
        input_layout = QFormLayout()

        # Dropdown populated with all raster layers in the current project.
        # Each item stores the QgsRasterLayer object as user data.
        self.layer_combo = QComboBox()
        for layer in self.raster_layers:
            self.layer_combo.addItem(layer.name(), layer)
        input_layout.addRow("Raster Layer:", self.layer_combo)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # --- Classification Parameters ---
        params_group = QGroupBox("Classification Parameters")
        params_layout = QFormLayout()

        # Method selection: K-Means is faster, ISO Cluster is more flexible
        self.method_combo = QComboBox()
        self.method_combo.addItems(["kmeans", "iso_cluster"])
        params_layout.addRow("Method:", self.method_combo)

        # Number of classes — typically 3-10 for land cover
        self.classes_spin = QSpinBox()
        self.classes_spin.setRange(2, 50)
        self.classes_spin.setValue(5)
        params_layout.addRow("Number of Classes:", self.classes_spin)

        # Optional: compute vegetation and water indices alongside classification
        self.indices_check = QCheckBox("Compute spectral indices (NDVI, NDWI)")
        params_layout.addRow(self.indices_check)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- Output Configuration ---
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        # Output file path with browse button
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_btn = QPushButton("Browse...")
        self.output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.output_btn)
        output_layout.addRow("Output File:", output_row)

        # Optional: reference raster for accuracy assessment
        # (e.g., a manually classified ground truth layer)
        ref_row = QHBoxLayout()
        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText("Optional: reference raster for accuracy assessment")
        self.ref_btn = QPushButton("Browse...")
        self.ref_btn.clicked.connect(self._browse_reference)
        ref_row.addWidget(self.ref_edit)
        ref_row.addWidget(self.ref_btn)
        output_layout.addRow("Reference:", ref_row)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Classify")
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.run_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _browse_output(self):
        """Open a file save dialog for the output GeoTIFF."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Classified Raster", "", "GeoTIFF (*.tif)"
        )
        if path:
            # Ensure .tif extension
            if not path.endswith(".tif"):
                path += ".tif"
            self.output_edit.setText(path)

    def _browse_reference(self):
        """Open a file dialog to select a reference (ground truth) raster."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Raster", "", "GeoTIFF (*.tif)"
        )
        if path:
            self.ref_edit.setText(path)

    def _on_run(self):
        """Validate inputs and emit the classification_requested signal."""
        output_path = self.output_edit.text().strip()
        if not output_path:
            return  # Don't proceed without an output path

        # Package all user choices into a dictionary
        layer_idx = self.layer_combo.currentIndex()
        params = {
            "layer": self.raster_layers[layer_idx],
            "method": self.method_combo.currentText(),
            "n_classes": self.classes_spin.value(),
            "output_path": output_path,
            "compute_indices": self.indices_check.isChecked(),
            "reference_path": self.ref_edit.text().strip() or None,
        }

        # Emit signal so the main plugin class can run the classification
        self.classification_requested.emit(params)
        self.accept()  # Close the dialog
