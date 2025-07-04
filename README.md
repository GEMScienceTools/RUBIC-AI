[![Windows Tests](https://github.com/GEMScienceTools/oq-vmtk/actions/workflows/windows_test.yml/badge.svg)](WIP)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="[https://github.com/dangomezm/GEM_AI_Toolkit]">
    <img src="help_img/RUBIC.png" alt="Logo" >
  </a>

  <h3 align="center">RUBIC-AI – Building Recognition using AI-based Identification Toolkit</h3>

  <p align="center">
    This repository contains an open source comprehensive AI-powered toolkit for risk assessment using facade image analysis. This beta version provides automated building feature prediction and classification through deep learning models, with an intuitive GUI for efficient building inspection workflows.
    <br />
    <a href="https://gemsciencetools.github.io/oq-vmtk/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/GEMScienceTools/oq-vmtk/tree/main/demos">View Demos</a>
    ·
    <a href="https://github.com/GEMScienceTools/oq-vmtk/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/GEMScienceTools/oq-vmtk/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## Features

- **AI-powered building feature prediction** using DenseNet201 with transfer learning
- **Multiple inspection methods** for different data sources and use cases
- **Interactive GUI** for streamlined building assessment workflows
- **Object detection module** to isolate buildings of interest
- **Automated seismic risk assessment** from facade images
- **Flexible data input/output** with CSV support and progress saving

## Quick Start

### Prerequisites

- Python 3.11.13
- Compatible with Windows, macOS, and Linux

### Installation

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd RUBIC-AI
   ```

2. **Create and activate virtual environment**

   **Windows (Anaconda):**
   ```bash
   conda create -n RUBIC-AI python=3.11.13
   conda activate RUBIC-AI
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv RUBIC-AI
   source RUBIC-AI/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Restart your system**

Restarting your system can help resolve potential issues related to environment path changes or incomplete installations.

> 🔁 *This step is usually not required, but recommended if you encounter errors related to newly installed dependencies.*


5. **Launch the application**
   ```bash
   python main.py
   ```

## Inspection Methods

> ⚠️ NOTE:
> The suggestion is to start with the methods we support, and then move to the current unavailable.
> The steps for each method are also simplified. Please check if some steps were skipped.

### 1. Local Images Method ✅ *Available*

**Best for:** Local image collections with known coordinates

**Workflow:**
1. Select the project output folder
2. Choose the "Local Images" inspection method
3. Configure input data:
   - Select the folder containing building images
   - Upload a CSV file with building coordinates and IDs
   - Specify number of images per location (1-3)
4. Process images and classify features
5. Save results as CSV

**Required CSV format:**
```csv
ID,Latitude,Longitude,City,Country
1,10.9639,-74.7964,Barranquilla,Colombia
2,10.9640,-74.7965,Barranquilla,Colombia
```

### 2. Neighbor Extrapolation ✅ *Available*

**Best for:** Expanding known building data to classify unknown buildings

**Workflow:**
1. Select the project output folder
2. Choose the "Neighbour Extrapolation" method
3. Configure extrapolation settings:
   - Select sampling method (KNN with soft voting or stratified sampling)
   - Upload buildings with known information
   - Upload unclassified building locations
4. Process and extrapolate features
5. Save the enhanced dataset

**Features:**
- **KNN with soft voting:** Uses k-nearest neighbors for classification
- **Stratified sampling:** Maintains class distribution in samples
- **Manual or AI-powered classification:** Choose your preferred workflow

### 3. Polygon Method 🚧 *Currently Unavailable*

**Best for:** Area-based building surveys

**Planned features:**
- Rectangle definition by two coordinates
- Define custom polygon areas by uploading vertex coordinates in a CSV file, ordered either clockwise or counterclockwise
- Automated building footprint extraction
- Sample size control for large areas

### 4. Specific Location Method 🚧 *Currently Unavailable*

**Best for:** Targeted building inspections

**Planned features:**
- Direct coordinate-based building selection
- Batch processing of specific locations
- Custom output naming

## AI-Powered Features

### Automatic Feature Prediction

Enable AI assistance by checking the **AI Powered** checkbox. The system will:

- Automatically predict building features using DenseNet201
- Leverage transfer learning from ImageNet
- Provide real-time classifications that you can verify and correct
- Maintain human oversight for quality assurance

### Supported Building Features

*[Additional explanation needed: List the specific building features the AI can classify, such as:]*
- Lateral Load Resisting System (LLRS) 
- LLRS Material
- Code level
- Number of stories
- Occupancy
- Block position
- Roof shape
- Roof material

## Data Management

### Input Requirements

**Image specifications:**
> ⚠️ NOTE:
> Maybe we can add the doc you created explaining the details

- Supported formats: *[JPG, JPEG, PNG]*
- Minimum resolution: *640x480*

**CSV file structure:**
All CSV files should include headers and follow the coordinate format shown in examples above.

### Output Files

Results are automatically saved as CSV files containing:
- All classified building features
- Geographic coordinates
- City and country information
- Image file paths

### Progress Saving

**Important:** Always save your progress before closing the GUI. The application automatically loads previous work when restarted, allowing you to continue from where you left off.

## Installation Details

### Windows Setup

1. **Install Anaconda**
   - Download from [Anaconda.com](https://www.anaconda.com/download/success)
   - Follow [installation instructions](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation)

2. **Create environment using Anaconda Prompt**
   ```bash
   conda create -n RUBIC-AI python=3.11.13
   conda activate RUBIC-AI
   cd /d YOUR_REPO_PATH
   pip install -r requirements.txt
   python main.py
   ```

### macOS Setup

1. **Install Python 3.11.13**
	```bash
   pip install python@3.11.13
   ```
   or you can also try
   ```bash
   brew install python@3.11.13
   ```

2. **Create and activate virtual environment**
   ```bash
   cd ~/your_project_folder
   python3 -m venv GEM_AI
   source GEM_AI/bin/activate
   pip install -r requirements.txt
   python3 main.py
   ```

## Troubleshooting

### Common Issues

**pip installation fails:**
```bash
pip3 install -r requirements.txt  # Try pip3 instead of pip
```

**Python version conflicts:**
Ensure you're using Python 3.11.13 specifically, as the models are optimised for this version.

**GUI won't start:**
- Verify all dependencies are installed
- Check that you're in the correct virtual environment
- Ensure you're in the repository directory when running `python main.py`

## Technical Architecture

### AI Models

- **Base Architecture:** DenseNet201
- **Training Strategy:** Transfer learning from ImageNet with fine-tuning
- **Inference:** Real-time feature prediction with human verification
#### *Currently Unavailable*
- **Performance:** *[Need benchmarks: accuracy, processing time, etc.]*


## Contributing

*pending*

## License

*pending*

## Support

*pending*

## Citation

> ⚠️ NOTE:
> We should also include a reference to your published paper here!

```bibtex
@software{gomez2025gem,
  author = {Daniel Gómez},
  title = {RUBIC-AI: Building Recognition using AI-based Identification Toolkit},
  year = {2025},
  version = {RUBIC-AI_beta1},
  doi = {10.5281/zenodo.14977499},
  url = {https://doi.org/10.5281/zenodo.14977499}
}
```

## Changelog

## [1.0] – 2025-07-03
### Changed
- Updated dependencies to be compatible with Python 3.11.13
