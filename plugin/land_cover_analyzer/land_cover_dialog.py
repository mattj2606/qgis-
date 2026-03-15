"""
Dialog for Land Cover Analyzer plugin configuration.
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
    """Configuration dialog for land cover classification."""

    classification_requested = pyqtSignal(dict)

    def __init__(self, parent=None, raster_layers=None):
        super().__init__(parent)
        self.raster_layers = raster_layers or []
        self.setWindowTitle("Land Cover Analyzer")
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Input layer selection
        input_group = QGroupBox("Input")
        input_layout = QFormLayout()

        self.layer_combo = QComboBox()
        for layer in self.raster_layers:
            self.layer_combo.addItem(layer.name(), layer)
        input_layout.addRow("Raster Layer:", self.layer_combo)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Classification parameters
        params_group = QGroupBox("Classification Parameters")
        params_layout = QFormLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["kmeans", "iso_cluster"])
        params_layout.addRow("Method:", self.method_combo)

        self.classes_spin = QSpinBox()
        self.classes_spin.setRange(2, 50)
        self.classes_spin.setValue(5)
        params_layout.addRow("Number of Classes:", self.classes_spin)

        self.indices_check = QCheckBox("Compute spectral indices (NDVI, NDWI)")
        params_layout.addRow(self.indices_check)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_btn = QPushButton("Browse...")
        self.output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.output_btn)
        output_layout.addRow("Output File:", output_row)

        # Optional reference layer for accuracy
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

        # Buttons
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
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Classified Raster", "", "GeoTIFF (*.tif)"
        )
        if path:
            if not path.endswith(".tif"):
                path += ".tif"
            self.output_edit.setText(path)

    def _browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Raster", "", "GeoTIFF (*.tif)"
        )
        if path:
            self.ref_edit.setText(path)

    def _on_run(self):
        output_path = self.output_edit.text().strip()
        if not output_path:
            return

        layer_idx = self.layer_combo.currentIndex()
        params = {
            "layer": self.raster_layers[layer_idx],
            "method": self.method_combo.currentText(),
            "n_classes": self.classes_spin.value(),
            "output_path": output_path,
            "compute_indices": self.indices_check.isChecked(),
            "reference_path": self.ref_edit.text().strip() or None,
        }

        self.classification_requested.emit(params)
        self.accept()
