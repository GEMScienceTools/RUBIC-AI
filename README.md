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

- **AI-powered building feature prediction** using Deep Learning model e.g.(Swin Transformer,ConvNeXt) with transfer learning and fine tuning
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
- [Git LFS](https://git-lfs.com/) — These files may not be downloaded correctly with Git alone, so Git LFS must be installed before cloning the repository. You can verify whether Git LFS is installed by running the following command in the terminal: 
  ```bash
  git lfs --version
  ```
- **Python 3.11**
	which can be download is not native supported in some operative system version, in those cases you can install it from python website:
	- Python 3.11 [macOS](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg)
	
	For windows one recommend option is to use Anaconda which makes the process easier, otherwise you can install directly python from this link 
	- Python 3.11 [windows](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
		
- **Anaconda**   
  📥 Download: [Anaconda.com](https://www.anaconda.com/download/success)  
  📖 Installation guide: [Anaconda Installation Instructions](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation)

## 👩‍💻 Installation

1. **Create and activate virtual environment**

   **Windows (Anaconda):**
   ```bash
   conda create -n RUBIC-AI python=3.11
   conda activate RUBIC-AI
   ```

   **macOS:**
   ```bash
   python3.11 -m venv RUBIC-AI
   source RUBIC-AI/bin/activate  # macOS/Linux
   ```
   
  RUBIC-AI is not available on Linux due to some issue related to the graphical user interface.

2. **Clone the repository**

   Choose your preferred folder to clone the repository by opening the terminal and navigating to the desired location.
   ```bash
   cd /Users/your-username/Path/To/Your/Repo
   ```

    Clone the ropository
   ```bash
   git lfs install
   git clone https://github.com/GEMScienceTools/RUBIC-AI.git
   ```
	> It should be ensured that the artificial intelligence model weights have been downloaded correctly. A common issue is that the model-weight files stored in the **dl_weights** folder have an incorrect file size, which indicates that the download was incomplete or unsuccessful.
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

- **Base Architectures:** SwinTransformer-Tiny, ConvNeXt-Tiny
- **Training Strategy:** Transfer learning from ImageNet with fine-tuning
- **Inference:** Real-time feature prediction with or without human verification.  
  > ⚠️ **Warning:**  
  > As this is the first version, we strongly recommend checking a few prediction examples to ensure the desired level of confidence in the model.  
  > However, as shown further below, these models are not perfect — they have certain accuracy limitations and perform better for specific classes and applications.

### 🔧 **Model performance**

🏗️ **Lateral Load Resisting System (LLRS) Classifier Performance**
- Current Accuracy: **~79%**
- Current Balanced Accuracy: **~80%**
- Current Macro F1 Score: **~80%**
  
🧱 **LLRS Material Classifier Performance**
- Current Accuracy: **~81%**
- Current Balanced Accuracy: **~81%**
- Current Macro F1 Score: **~81%**
  
🏢 **Number of Stories Classifier Performance**
- Current Accuracy: **~85%**
- Current Balanced Accuracy: **~84%**
- Current Macro F1 Score: **~84%**
 
🏠 **Occupancy Classifier Performance** 
-  Current Accuracy: **~83%**
- Current Balanced Accuracy: **~74%**
- Current Macro F1 Score: **~76%**

🧾 **Code Level Classifier Performance** 
- Current Accuracy: **~64%**
- Current Balanced Accuracy: **~64%**
- Current Macro F1 Score: **~63%**
  
📍 **Block Position Classifier Performance** 
- Current Accuracy: **~65%**
- Current Balanced Accuracy: **~68%**
- Current Macro F1 Score: **~67%**

🏛️ **Roof Shape Classifier Performance**
- Current Accuracy: **~79%**
- Current Balanced Accuracy: **~80%**
- Current Macro F1 Score: **~73%**
  
🔨 **Roof Material Classifier Performance** 
-  Current Accuracy: **~81%**
- Current Balanced Accuracy: **~79%**
- Current Macro F1 Score: **~79%**

###  📊 **Data distrubution and confusion matrices** 
For each application, there are additional data and metrics of interest.  
The confusion matrices presented below enable users to evaluate whether these models are suitable for their specific use cases.
Furthermore, it is shown the **distribution of building attributes** in the full database, together with the final number of images used to train each deep-learning model. Minority classes were generally excluded when they contained too few samples for reliable model training. However, some minority classes were retained when they were visually distinct from the remaining classes, thereby preserving a broader range of building attributes within the classification framework.

<details>
<summary>📊 Data Distribution and Confusion Matrices (Click to Expand)</summary>

* **Lateral Load Resistant System (LLRS)**
It is important to note that the wall-system class was divided into low-rise [LWAL(LR)] and high-rise [LWAL(HR)] subclasses because these building types exhibit distinct visual characteristics. In general, low-rise wall buildings are more commonly associated with masonry construction, whereas high-rise wall buildings are typically constructed with reinforced concrete walls. This subdivision was introduced to improve class separability and enhance model performance. However, in the final output, RUBIC-AI maps both subclasses back to the common LWAL class.

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/LWAL_division.png?raw=true"
  alt="Wall system subclasses (left) low-rise (right) high-rise"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/LLRS.png?raw=true"
  alt="LLRS Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/LLRS_dist.png?raw=true"
  alt="LLRS Data Distribution"
  width="700"
/>

---

* **LLRS Material**
  
<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/LLRS_Material.png?raw=true"
  alt="LLRS Material Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/LLRS_Material_dist.png?raw=true"
  alt="LLRS Material Data Distribution"
  width="700"
/>

---

* **Number of stories**
  
<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/n_stories.png?raw=true"
  alt="Number of stories Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/n_stories_dist.png?raw=true"
  alt="Number of stories Data Distribution"
  width="700"
/>

---

* **Occupancy**
  
<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/occupancy.png?raw=true"
  alt="Occupancy Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/occupancy_dist.png?raw=true"
  alt="Occupancy Data Distribution"
  width="700"
/>

---

* **Code level**
  
<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/code_level.png?raw=true"
  alt="Code level Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/code_level_dist.png?raw=true"
  alt="Code level Data Distribution"
  width="700"
/>

---

* **Block position**
  
<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/block_matrix.png?raw=true"
  alt="Block position Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/block_matrix_dist.png?raw=true"
  alt="Block position Data Distribution"
  width="700"
/>

---

* **Roof shape**
  
<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/roof_shape.png?raw=true"
  alt="Roof shape Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/roof_shape_dist.png?raw=true"
  alt="Roof shape Data Distribution"
  width="700"
/>

---

* **Roof material**
  
<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/roof_material.png?raw=true"
  alt="Roof Material Confusion Matrix"
  width="700"
/>

<img
  src="https://github.com/GEMScienceTools/RUBIC-AI/blob/phd/help_img/roof_material_dist.png?raw=true"
  alt="Roof Material Data Distribution"
  width="700"
/>

---

</details>

###  ⚠️ **Manual constraints for taxonomy consistency** 
To ensure consistency among feasible combinations of building attributes, RUBIC-AI incorporates a set of rule-based constraints that prevent the generation of unrealistic building classes. For example, when the primary construction material is classified as unreinforced masonry (MUR), the lateral load-resisting system is automatically assigned as a wall system (LWAL), since unreinforced masonry buildings cannot be represented by other systems. This rule prevents invalid building classes, such as unreinforced masonry moment frame (MUR/LFM), that are not defined in the adopted taxonomy but could otherwise be generated because the individual classification models operate independently and do not exchange information.

<details>
<summary>🔒 Rule-based constraints currently applied (Click to Expand)</summary>

- Buildings classified as unreinforced masonry or confined masonry are assigned to wall system for LLRS.
- Buildings with flat roofs are assigned concrete as the roof material, under the assumption that the roof consists of a concrete slab.
- Buildings with gable or hipped roofs are checked to determine whether the roof material corresponds to metal/asbestos sheets or clay tiles. If neither valid option is predicted, the roof is assigned to the metal/asbestos class. This rule reflects the geographic composition of the training database, which is dominated by South American images, where this roofing material is more common than clay tiles, which are more prevalent in European contexts.
- Buildings with curved roofs are assigned metal sheets as the roof material.
- For unreinforced masonry buildings, the predicted code level is checked to determine whether it corresponds to no-code regulation or low-code regulation. This constraint reflects the fact that many such buildings are either older structures or newer informal constructions that may not fully comply with modern building regulations.
</details>

### Image input specifications
- Supported formats: *[JPG, JPEG, PNG]*
- Recommended minimum resolution: *640x480*
  
# 🤝 Contributions

[WIP]

# © License

The OpenQuake Engine is released under the **[GNU Affero Public License 3](LICENSE)**.

# Citation
[WIP]
