# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [1.0.0-beta.1] – 2025-01-15

### Added
- First public beta release of RUBIC-AI.
- PyQt5-based graphical user interface for building-stock inspection and labeling.
- Polygon-based sampling workflow for systematic image acquisition.
- Support for Google Street View, local images, and in-field imagery.
- Integration of YOLOv11 object detector for automatic building detection.
- Deep-learning classifiers for building attributes aligned with the GEM taxonomy.
- Extrapolation module using KNN with soft voting.
- Export of results to CSV and GeoPackage formats.

### Known Issues
- Performance may degrade when processing very large polygons.
- Some attributes require manual confirmation for low-quality or occluded images.
- Limited support for rural areas with sparse imagery.
