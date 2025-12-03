<!-- [![Windows Tests](https://github.com/GEMScienceTools/oq-vmtk/actions/workflows/windows_test.yml/badge.svg)](WIP) -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="[https://github.com/dangomezm/GEM_AI_Toolkit]">
    <img src="help_img/GUI_RUBIC_LOGO.png" alt="Logo" >
  </a>

  <h3 align="center">RUBIC-AI – Risk and Unified Building Inventory Classifier using AI</h3>

  <p align="left">
    This repository contains an open source comprehensive AI-powered toolkit for image classification using facade image analysis. This beta version provides automated building feature prediction and classification through deep learning models, with an intuitive GUI for efficient building inspection workflows.
    <a href="https://github.com/dangomezm/GEM_AI_Toolkit/tree/main/demos">View Demos</a>
  </p>
</div>

# ✨ Key Features

- **AI-powered building feature prediction** using Deep Learning model e.g.(DenseNet201,ConvNeXt) with transfer learning and fine tuning
- **Multiple usage modes** for different data sources and use cases
- **Interactive GUI** for streamlined building assessment workflows
- **Object detection module** to isolate building of interest
- **Automated building stock collection** from facade images
- **Flexible data input/output** with CSV support and progress saving

# 🚀 Get Started

## ⚙️🔧 Prerequisites

Before you begin, make sure the following are installed on your system:

- [Git](https://git-scm.com/downloads) — Used to clone the repository, manage version control, and install GEM libraries.  
  Git is typically pre-installed on macOS, but on Windows, users need to install it manually. You can verify whether Git is installed by running the following command in the terminal:
  ```bash
  git --version
  ```
- **Python 3.11.13**
- **Anaconda**   
  📥 Download: [Anaconda.com](https://www.anaconda.com/download/success)  
  📖 Installation guide: [Anaconda Installation Instructions](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation)

## 👩‍💻 Installation

1. **Create and activate virtual environment**

   **Windows (Anaconda):**
   ```bash
   conda create -n RUBIC-AI python=3.11.13
   conda activate RUBIC-AI
   ```

   **macOS/Linux:**
   ```bash
   python3.11 -m venv RUBIC-AI
   source RUBIC-AI/bin/activate  # macOS/Linux
   ```
2. **Clone the repository**

   Choose your preferred folder to clone the repository by opening the terminal and navigating to the desired location.
   ```bash
   cd /Users/your-username/Path/To/Your/Repo
   ```

    Clone the ropository
   ```bash
   git clone https://github.com/GEMScienceTools/RUBIC-AI.git
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

## 🕵️ Usage modes

### 1. Polygon Method 
**Best for:** Create a building stock from well defined area such as a neighborhood, city, or similar.

<a href="https://github.com/dangomezm/GEM_AI_Toolkit/tree/main/demos/polygon_method">See detailed instructions in the demos</a>


### 2. Specific Coordinates Method 
**Best for:** Characterizing specific buildings, for example: reviewing all hospitals in the area of analysis, even if they are located in different countries.

<a href="https://github.com/dangomezm/GEM_AI_Toolkit/tree/main/demos/specific_coordinates">See detailed instructions in the demos</a>

### 3. Local Images Method 

**Best for:** Create a building stock from images stored on your local device. Ideal for characterizing buildings in locations where there is no access with GSV and whose images already exist, e.g., inside a factory.

<a href="https://github.com/dangomezm/GEM_AI_Toolkit/tree/main/demos/local_images">See detailed instructions in the demos</a>

### 4. Neighbor Extrapolation

**Best for:** Expanding known building data to classify unknown buildings

<a href="https://github.com/dangomezm/GEM_AI_Toolkit/tree/main/demos/extrapolation">See detailed instructions in the demos</a>

## 🖥️ AI Models and Performance

### AI Models

- **Base Architecture:** DenseNet201
- **Training Strategy:** Transfer learning from ImageNet with fine-tuning
- **Inference:** Real-time feature prediction with or without human verification.  
  > ⚠️ **Warning:**  
  > As this is the first version, we strongly recommend checking a few prediction examples to ensure the desired level of confidence in the model.  
  > However, as shown further below, these models are not perfect — they have certain accuracy limitations and perform better for specific classes and applications.

### 🔧 **Model performance**

#### 🏗️ Lateral Load Resistant System (LLRS) Classifier Performance
- Current Accuracy: **~71.0%**
#### 🧱 LLRS Material Classifier Performance
- Current Accuracy: **~71.5%**
#### 🏢 Number of Stories Classifier Performance
- Current Accuracy: **~73.7%**
#### 🏠 Occupancy Classifier Performance 
-  Current Accuracy: **~82.3%**
#### 🧾 Code Level Classifier Performance 
-  Current Accuracy: **~64.0%**
#### 📍 Block Position Classifier Performance 
-  Current Accuracy: **~59.1%**
#### 🏛️ Roof Shape Classifier Performance 
-  Current Accuracy: **~72.6%**
#### 🔨 Roof Material Classifier Performance 
-  Current Accuracy: **~82.6%**

For each application, there are additional metrics of interest.  
Below is the information from the confusion matrices, which allows users to determine whether these models work for their specific needs.  

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

### Image input specifications:
- Supported formats: *[JPG, JPEG, PNG]*
- Recommended minimum resolution: *640x480*
  
# 🤝 Contributions

[WIP]

# © License

[PRIVATE AND CONFIDENTIAL](./LICENSE.txt)

You should have received a specific license agreement along with
this product.  If you did not, please contact the GEM Foundation
at licensing@globalquakemodel.org.


# Citation
[WIP]
