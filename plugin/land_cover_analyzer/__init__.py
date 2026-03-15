"""
Land Cover Analyzer — QGIS Plugin
===================================

Classifies multispectral raster imagery into land cover categories
(vegetation, water, built-up, bare soil, etc.) using unsupervised machine
learning methods, and exports results as GeoTIFF with full metadata.

This file is the plugin entry point. QGIS calls classFactory() on startup
to instantiate the plugin. All heavy logic lives in the other modules:

- land_cover_analyzer.py  → Main plugin class (toolbar, menus, I/O)
- land_cover_dialog.py    → PyQt5 configuration dialog
- classifier.py           → Classification engine + accuracy metrics
"""


def classFactory(iface):
    """QGIS plugin entry point — called automatically when the plugin loads.

    Args:
        iface: QgisInterface instance providing access to the QGIS
               application (map canvas, menus, message bar, etc.)

    Returns:
        An instance of LandCoverAnalyzerPlugin.
    """
    from .land_cover_analyzer import LandCoverAnalyzerPlugin
    return LandCoverAnalyzerPlugin(iface)
