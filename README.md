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

- **AI-powered building feature prediction** using DenseNet201 with transfer learning
- **Multiple usage modes** for different data sources and use cases
- **Interactive GUI** for streamlined building assessment workflows
- **Object detection module** to isolate buildings of interest
- **Automated building stock collection** from facade images
- **Flexible data input/output** with CSV support and progress saving

# 🚀 Get Started

## ⚙️🔧 Prerequisites

Before you begin, make sure the following are installed on your system:

- [Git](https://git-scm.com/downloads) — Used to clone the repository and manage version control.  
  Git is typically pre-installed on macOS, but on Windows, users need to install it manually. You can verify whether Git is installed by running the following command in the terminal:
  ```bash
  git --version
  ```
- **Python 3.11.13**
- **Anaconda** (*only required for Windows users*)  
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
   python3 -m venv RUBIC-AI
   source RUBIC-AI/bin/activate  # macOS/Linux
   ```
2. **Clone the repository**

   Choose your preferred folder to clone the repository by opening the terminal and navigating to the desired location.
   ```bash
   cd /Users/your-username/Path/To/Your/Repo
   ```

    Clone the ropository
   ```bash
   git clone https://github.com/dangomezm/RUBIC-AI.git
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **(Optional) Restart your system**

Restarting your system can help resolve potential issues related to environment path changes or incomplete installations.

> 🔁 *This step is usually not required, but recommended if you encounter errors related to newly installed dependencies.*

5. **Launch the application**
   ```bash
   python main.py
   ```

## 🕵️ Usage modes

### 1. Polygon Method 
**Best for:** Create a building stock from well defined area such as a neighborhood, city, and similar

**Workflow:**
1. **Set the usage mode**  
   Select the **Polygon method** option, and then click the ***Save and continue*** button.
   
2. **Select the output project folder**  
   Click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved.  
   > 📁 *Example:* `demos/polygon_method`
    
3. **Set input files**  
   Click the ***Upload file with coordinates*** button to open a pop-up window and navigate to the file where the vertices of the polygon are stored. The outputs will be saved automatically 
   > 📁 *Example:* `demos/polygon_method/polygon_method_example.csv`  
   > 📝 *Required CSV format:*
   
   ```csv
   id,latitude,longitude
   1,10.9639,-74.7964
   2,10.9640,-74.7965
   ```
   At this point, the file will be previewed in a table so you can verify the selected information.
	
4. Download the available building footprints in the defined area by clicking the ***Get footprints available*** button.  
   **Currently, the footprints are downloaded from OSM, but more options will be available in the future.**

   - **4.1. Footprint Sample**
     - The number of available footprints will be displayed next to the text ***N° footprints***.  
     - The user should define the sample size using the ***Sample size*** field. This value must be equal to or less than the total number of available building footprints.  
     - The user can choose between **Manual** inspection, where classifications are made manually, or **AI-powered**, where the GUI uses a deep learning model to predict each feature.  
	 - Contine by clicking the ***Save and continue*** button.
     > ⚠️ **Important:** *Since this is the first version, it is strongly recommended to verify the results.*

5. Setup Main Interface
   
	Once in the main window, the user should click the ***Next Building*** button to fetch the image from Google Street View.  
	Before starting, the GUI will ask for the most representative epoch of construction in the country under analysis.  
	This must be defined in order to continue the process.  

	After that, the user can either start with manual inspection by selecting each feature from the corresponding combo boxes,  
	or wait for the deep learning models to process the image.

	- **5.1 Manual Box**
	     - If the automatic delimitation of the building is not adequate for proper isolation,  
		   or if the selected building is not the building of interest, the user can define a manual bounding box by clicking on four points.
			<img src="help_img/manual_box.png">
	- **5.2 Search Old Inspection**
	     - By searching with the image ID and clicking the ***Search building*** button,  
		   the GUI will display the corresponding saved inspection.
		   > ⚠️ **Important:** *This only works for inspections that were previously saved.*
	     
	- **5.3 Save Data**
		 - The results of the inspection will be saved in the corresponding folder by clicking the ***Save data*** button.  
			The progress of the work is only saved if this button is clicked.

---

### 2. Specific Coordinates Method 
**Best for:** Ideal for characterizing specific buildings, which may even be located in different countries.

**Workflow:**
1. **Set the usage mode**  
   Select the **Specific coordinates** option, and then click the ***Save and continue*** button.
   
2. **Select the output project folder**  
   Click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved.  
   > 📁 *Example:* `demos/specific_coordinates`
    
3. **Set input files**  
   Click the ***Upload file with coordinates*** button to open a pop-up window and navigate to the file where the vertices of the polygon are stored. The outputs will be saved automatically 
   > 📁 *Example:* `demos/specific_coordinates/specific_coordinates_example_data.csv`  
   > 📝 *Required CSV format:*
   
   ```csv
   id,latitude,longitude
   1,10.9639,-74.7964
   2,10.9640,-74.7965
   ```
	After this, the user should check the input files by using the ***Load data*** button.  
	If everything is correct, please click the ***Save and continue*** button 

4. Setup Main Interface
   
	The process and available feature are the same that the presented in the first method.
---

### 3. Local Images Method 

**Best for:** Create a building stock from images stored on your local device. Ideal for characterizing buildings in locations where there is no access with GSV and whose images already exist, e.g., inside a factory.

**Workflow:**
1. **Set the usage mode**  
   Select the **Local images** option, and then click the ***Save and continue*** button.
   
2. **Select the output project folder**  
   Click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved.  
   > 📁 *Example:* `demos/local_images`
    
3. **Select local image folder**  
   Click the ***Select image folder*** button to open a pop-up window and navigate to the folder where the images are stored.  
   > 📁 *Example:* `demos/local_images/images_ex1`
   
4. **Set Input Files**  
   Click the ***Upload building information*** button to open a pop-up window and navigate to the file containing the building locations.  
   
   > 📁 *Example:* `demos/local_images/data_ex1.csv`  
   > 📝 *Required CSV format:*
   
   ```csv
   id,latitude,longitude
   1,10.9639,-74.7964
   2,10.9640,-74.7965
   ```
	- 4.1. The results will be saved using the name specified in the ***Output name*** field (default: **"Local_images"**) as a `.csv` file. This file will contain all the building features, along with metadata such as city, country, coordinates, and the path to the corresponding building image.
	- 4.2. A maximum of three windows can display the same location (ideally the same building). These will be displayed automatically if they share the same coordinates.  
	- 4.3. Upload the information and check the format by clicking the ***Load data*** button. Once a confirmation message appears, click ***Save and continue*** to proceed to the next step.

5. In this step, the user can classify building features in two ways:  
   **I) Manually** or **II) Using Deep Learning models with verification of predicted attributes.**

	- 4.1. 📝 Manual Classification
		- **4.1.1.** Click the ***Next Building*** button to upload and display the first building image.
		- **4.1.2.** Use the corresponding comboboxes to select the appropriate features based on the displayed image (e.g., select "Concrete" as the LLRS material).
		- **4.1.3.** Click the ***Next Building*** button again to proceed to the next image, and repeat this process until all images have been reviewed.
		- **4.1.4.** Save either all results or a partial set by clicking the ***Save data*** button. When the GUI is launched again, previously saved results will be reloaded, allowing the classification process to resume from where it left off.  
		> ⚠️ **Important:** *If you do not save your data before closing the GUI, your work will be lost.*

	- 4.2. 🤖 AI-Powered Classification
		- **4.2.1.** Click the ***AI Powered*** checkbox to activate the deep learning models.
		- **4.2.2.** Upload the images by clicking the ***Next Building*** button. At this step, the tool will automatically predict the building features.
		- **4.2.3.** Click the ***Next Building*** button again to proceed to the next image, and repeat this process until all images have been reviewed.
		- **4.2.4.** Save either all results or a partial set by clicking the ***Save data*** button. When the GUI is launched again, previously saved results will be reloaded, allowing the classification process to resume from where it left off.  
		> ⚠️ **Important:** *If you do not save your data before closing the GUI, your work will be lost.*

5. ***Review Previous Classifications***
To review or complete unfinished classifications, follow the steps up to **4.1 Manual Classification** to upload the CSV file containing the classification data.
You can navigate through the results using the ***Next Building*** and ***Previous Building*** buttons to move forward or backward between images and review the associated information.
If you want to check a specific image, use the ***Search Building*** button. First, enter the image ID (e.g., `1_1`) in the adjacent field, then click the ***Search Building*** button. The GUI will automatically display the corresponding image and its saved classification.


---

### 4. Neighbor Extrapolation ✅ *Available*

**Best for:** Expanding known building data to classify unknown buildings

**Workflow:**
1. **Select the project folder**  
   Click the ***Project Folder*** button to open a pop-up window and navigate to the folder where the outputs will be saved.  
   > 📁 *Example:* `demos/Extrapolation_data_example`

2. **Set the usage mode**  
   Click the ***Insp. Method*** button, select the **Neighbor extrapolation** option, and then click the ***Save and continue*** button.

3. **Set input files**  
   Click the ***Set Coord.*** button and follow these steps:

	- **3.1.** Click the ***Extrapolation options*** button. This will display a pop-up window with two available options:  
		- **KNN with soft voting**  
		- **Stratified sampling**  

	- Select one of the two options and then click the ***Load Files*** button. A new pop-up window will appear for setting the input files.
		- **3.1.1.** Set the input files using either the **Manual method** or the **Deep Learning model** method.
		     - 3.1.1.1 📄 Upload Data Manually
			- Define the output file name using the ***Output name*** field (default: **"KNN"**).
			- Click the ***Buildings with information*** button and upload a CSV file containing the reference buildings — that is, buildings that have already been classified and include all the features of interest.  
			  > 📁 *Example:* `demos/Extrapolation_data_example/neighbor_building_info.csv`  
			  > 📝 *Required CSV format:*
			   ```csv
			  	ID,Latitude,Longitude,Country,City,LLRS Material,LLRS,Code Level,Number of Stories,Occupancy,Block Position,Taxonomy
				1,10.92224755,-74.78642608,Colombia,Barranquilla,MCF,LWAL,CDL,1,RES,BP1,MCF/LWAL+CDL/H:1/RES/BP1
				2,10.91268031,-74.80288191,Colombia,Soledad,CR,LFM,CDM,3,RES,BP2,CR/LFM+CDM/H:3/RES/BP2
				3,10.91968505,-74.79215175,Colombia,Barranquilla,MUR,LWAL,CDL,1,RES,BP1,MUR/LWAL+CDL/H:1/RES/BP1
				4,10.91647181,-74.76986198,Colombia,Soledad,CR,LFINF,CDM,2,COM,BP1,CR/LFINF+CDM/H:2/COM/BP1
				5,10.90251035,-74.79685532,Colombia,Soledad,CR,LFM,CDL,2,RES,BP2,CR/LFM+CDL/H:2/RES/BP27
			   ```
			- Click the ***Unclassified building locations*** button and upload a CSV file containing ID and coordinates of the building that the user want to classify based on the information of the building of reference due to there is not information availabe for them.
			   > 📁 *Example:* `demos/Extrapolation_data_example/building_with_no_image.csv`
			   > 📝 *Required CSV format:*
			   ```csv
			   ID,Latitude,Longitude
			   1,10.9639,-74.7964
			   2,10.9640,-74.7965
			   ```

		     - 3.1.1.2 🤖 DL Model
			- Define the output file name using the ***Output name*** field (default: **"DL Model"**).
		
			- Click the ***Image Folder*** button to select the folder containing the building images stored locally.  
			  > 📁 *Example:* `demos/Extrapolation_data_example/images_buildings`
		
			- Click the ***Unclassified building locations*** button and upload a CSV file containing the ID and coordinates of the buildings that need to be classified based on the predicted features of the reference buildings. These buildings do not have image data or existing attribute information.
			  > 📁 *Example:* `demos/Extrapolation_data_example/building_data.csv`  
			  > 📝 *Required CSV format:
			   ```csv
			   ID,Latitude,Longitude
			   1,10.9639,-74.7964
			   2,10.9640,-74.7965
			   ```
			- Click the ***Unclassified building locations*** button and upload a CSV file containing the ID and coordinates of the buildings that the user wants to classify based on the information from the reference buildings, as no attribute information is available for them.
			   > 📁 *Example:* `demos/Extrapolation_data_example/building_with_no_image.csv`
			   > 📝 *Required CSV format:*
			   ```csv
			   ID,Latitude,Longitude
			   1,10.9639,-74.7964
			   2,10.9640,-74.7965
			   ```
	- 3.2. Click the ***Save and continue*** button in the ***Setting Input Files*** window, and then click the ***Save and continue*** button in the ***Setting Extrapolation Method*** window.
	- 3.3. In the ***Setting Polygon Coordinates*** window, click the ***Load data*** button to check the format, then click the ***Save and continue*** button.

4. Click the ***Next Building*** button to perform the extrapolation. The results will be saved in the output path selected in Step 1.  
   > 📁 *Example:* `demos/Extrapolation_data_example/Extrapolation.csv`





## 🖥️ Technical Architecture

### AI Models

- **Base Architecture:** DenseNet201
- **Training Strategy:** Transfer learning from ImageNet with fine-tuning
- **Inference:** Real-time feature prediction with human verification
### 🔧 *Currently Working on Improvements*

#### 🏗️ Lateral Load Resistant System (LLRS) Classifier Performance
- **Current Accuracy:** **~75.6%**
#### 🧱 LLRS Material Classifier Performance
- **Current Accuracy:** **~51.1%**
#### 🏢 Number of Stories Classifier Performance
- **Current Accuracy:** **~79.6%**
#### 🏠 Occupancy Classifier Performance 
-  **Current Accuracy:** **~61.7%**
#### 🧾 Code Level Classifier Performance 
-  **Current Accuracy:** **~70.0%**
#### 📍 Block Position Classifier Performance 
-  **Current Accuracy:** **~64.7%**
#### 🏛️ Roof Shape Classifier Performance 
-  **Current Accuracy:** **~87.3%**
#### 🔨 Roof Material Classifier Performance 
-  **Current Accuracy:** **~84.9%**

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

### Image input specifications:**
- Supported formats: *[JPG, JPEG, PNG]*
- Minimum resolution: *640x480*
  
# 🤝 Contributions

[WIP]

# © License

[PRIVATE AND CONFIDENTIAL](./LICENSE.txt)

You should have received a specific license agreement along with
this product.  If you did not, please contact the GEM Foundation
at licensing@globalquakemodel.org.


# Citation
[WIP]
