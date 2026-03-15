"""
Land Cover Analyzer - QGIS Plugin

Classifies multispectral raster imagery into land cover categories
and exports labeled datasets for machine learning workflows.
"""


def classFactory(iface):
    """QGIS plugin entry point."""
    from .land_cover_analyzer import LandCoverAnalyzerPlugin
    return LandCoverAnalyzerPlugin(iface)
