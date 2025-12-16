# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [1.0.0-beta.1]
### Added
- First beta release of RUBIC-AI.
- PyQt5-based graphical user interface for building-stock inspection and labeling.
- Polygon-based sampling workflow for systematic image acquisition.
- Support for Google Street View, and in-field imagery.
- Integration of YOLOv11 object detector for automatic building detection.
- Deep-learning classifiers for building attributes aligned with the GEM taxonomy.
- Extrapolation module using KNN with soft voting.
- Export of results to CSV and GeoPackage formats.

### Known Issues
- Some attributes require manual confirmation when images are low-quality, partially occluded, or affected by adverse lighting conditions.
- Model performance may degrade for building façades with architectural styles that are significantly underrepresented or absent in the training dataset.
