[![Windows Tests](https://github.com/GEMScienceTools/oq-vmtk/actions/workflows/windows_test.yml/badge.svg)](WIP)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="[https://github.com/dangomezm/GEM_AI_Toolkit]">
    <img src="help_img/GUI_RUBIC_LOGO.png" alt="Logo" >
  </a>

  <h3 align="center">RUBIC-AI – Building Recognition using AI-based Identification Toolkit</h3>

  <p align="left">
    This repository contains an open source comprehensive AI-powered toolkit for risk assessment using facade image analysis. This beta version provides automated building feature prediction and classification through deep learning models, with an intuitive GUI for efficient building inspection workflows.
    <a href="https://github.com/dangomezm/GEM_AI_Toolkit/tree/main/demos">View Demos</a>
  </p>
</div>

# ✨ Key Features

- **AI-powered building feature prediction** using DenseNet201 with transfer learning
- **Multiple inspection methods** for different data sources and use cases
- **Interactive GUI** for streamlined building assessment workflows
- **Object detection module** to isolate buildings of interest
- **Automated building stock collection** from facade images
- **Flexible data input/output** with CSV support and progress saving

# 🚀 Get Started

## ⚙️🔧 Prerequisites

Before you begin, ensure you have the following installed:
- [Git](https://git-scm.com/downloads) – Used to clone the repository and manage versions.
- Python 3.11.13
### Check if GIT is installed
Open a terminal and run:
```bash
git --version
```

## 👩‍💻🧑‍💻 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dangomezm/RUBIC-AI.git
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

## 🔍🕵️ Inspection Methods

> ⚠️ NOTE:
> The suggestion is to start with the methods we support, and then move to the current unavailable.
> The steps for each method are also simplified. Please check if some steps were skipped.

### 1. Local Images Method ✅ *Available*

**Best for:** Create a building stock from images stored on your local device

**Workflow:**
1. Select the project output folder
2. Choose the "Local Images" inspection method
3. Configure input data:
   - Select the folder containing building façade images
   - Upload a CSV file with building coordinates and IDs
   - Specify number of images per location (1-3)
4. Process images and classify features
5. Save results as CSV

<img src="help_img/output.gif" alt="Logo" >

**Required typical CSV format:**
```csv
ID,Latitude,Longitude,City,Country
1,10.9639,-74.7964,Barranquilla,Colombia
2,10.9640,-74.7965,Barranquilla,Colombia
```

### 2. Neighbor Extrapolation ✅ *Available*

**Best for:** Expanding known building data to classify unknown buildings

**Workflow:**
1. Select the project output folder
2. Choose the "Neighbor Extrapolation" method
3. Configure extrapolation settings:
   - Select sampling method (KNN with soft voting or stratified sampling)
   - Upload buildings with known information **Required special CSV format:**
   - Upload unclassified building locations
4. Process and extrapolate features
5. Save the enhanced dataset

**Required special CSV format:**
```csv
ID,Latitude,Longitude,Country,City,LLRS Material,LLRS,Code Level,Number of Stories,Occupancy,Block Position,Taxonomy
1,10.92224755,-74.78642608,Colombia,Barranquilla,MCF,LWAL,CDL,1,RES,BP1,MCF/LWAL+CDL/H:1/RES/BP1
2,10.91268031,-74.80288191,Colombia,Soledad,CR,LFM,CDM,3,RES,BP2,CR/LFM+CDM/H:3/RES/BP2
3,10.91968505,-74.79215175,Colombia,Barranquilla,MUR,LWAL,CDL,1,RES,BP1,MUR/LWAL+CDL/H:1/RES/BP1
4,10.91647181,-74.76986198,Colombia,Soledad,CR,LFINF,CDM,2,COM,BP1,CR/LFINF+CDM/H:2/COM/BP1
5,10.90251035,-74.79685532,Colombia,Soledad,CR,LFM,CDL,2,RES,BP2,CR/LFM+CDL/H:2/RES/BP27
```
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

## 🤖🧠 AI-Powered Features

### Automatic Feature Prediction

Enable AI assistance by checking the **AI Powered** checkbox. The system will:

- Automatically predict building features using DenseNet201
- Leverage transfer learning from ImageNet
- Provide real-time classifications that you can verify and correct
- Maintain human oversight for quality assurance

### Supported Building Features
* List the specific building features the AI can classify, such as:
- Lateral Load Resisting System (LLRS) 
- LLRS Material
- Code level
- Number of stories
- Occupancy
- Block position
- Roof shape
- Roof material

## 🗂️ Data Management

### Input Requirements

**Image specifications:**
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

## 📦 Installation Details

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
   python3 -m venv RUBIC-AI
   source RUBIC-AI/bin/activate
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

## 🖥️ Technical Architecture

### AI Models

- **Base Architecture:** DenseNet201
- **Training Strategy:** Transfer learning from ImageNet with fine-tuning
- **Inference:** Real-time feature prediction with human verification
### 🔧 *Currently Working on Improvements*

#### 🏗️ Lateral Load Resistant System (LLRS) Classifier

- **Current Accuracy:** **~75.6%**

<details>
<summary>📊 Confusion Matrices (Click to Expand)</summary>

* **Lateral Load Resistant System (LLRS)**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/LLRS.png)

* **LLRS Material**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/LLRS_Material.png)

* **Number of stories**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/n_stories.png)

* **Occupancy**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/occupancy.png)

* **Code level**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/code_level.png)

* **Block position**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/block_matrix.png)

* **Roof shape**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/roof_shape.png)

* **Roof material**
  
![LLRS](https://github.com/dangomezm/RUBIC-AI/blob/main/help_img/roof_material.png)

</details>


# 🤝 Contributions

[WIP]

# © License

[WIP]

# 🌟 Contributors

[WIP]

## Citation
[WIP]

## Changelog

## [1.0] – 2025-07-03
### Changed
- Updated dependencies to be compatible with Python 3.11.13
