# GEM AI Toolkit
This is the beta version of the toolkit for building feature prediction using facade images, for seismic risk assessment with AI. The Graphic User Interface enables a fast assessment of the building, including an object detection module, to isolate the building of interest.

# How to use
This guideline utilizes [Anaconda](https://www.anaconda.com/) to facilitate the configuration of the recommended virtual environment for working with AI. It can be downloaded for various operating systems from  
[Download Anaconda](https://www.anaconda.com/download/success), and installation instructions can be found in [Installing Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation). Once Anaconda has been installed, it is necessary to create a virtual environment. *Note: Python 3.9.13 can also be installed directly, and the virtual environment can be created using your preferred method* 

## 1. Create a virtual environment 
### *Windows*
* Clone this repository
* Open Anaconda Prompt
* *Create a virtual environment:* conda create -n MY_VIRTUAL_ENVIRONMENT_NAME python=3.9.13 (e.g., conda create -n GEM_AI python=3.9.13)

```bash
conda create -n YOUR_VIRTUAL_ENVIRONMENT_NAME python=3.9.13
``` 
* Activate the virtual environment
```bash
conda activate YOUR_VIRTUAL_ENVIRONMENT_NAME 
```

* Assign path to the cloned repository
```bash
cd /d YOUR_REPO_PATH
```

* Install dependencies
```bash
pip install -r requirements.txt
```
* Execute GUI
```bash
python main.py
```

### *MacOS*
The process would be the same as the one executed for Windows, up until the activation of the virtual environment
* Assign path to the cloned repository
```bash
cd ~/YOUR_REPO_PATH
```
* Install dependencies
```bash
pip install -r requirements.txt
```
In case pip is not working, you can also try:
```bash
pip3 install -r requirements.txt
```
* Execute GUI
```bash
python3 main.py
```
## 2. Inspection methods available
### 2.1 Polygon method
#### *Currently unavailable*
* Select the folder where the outputs will be saved by clicking the ***Project Folder*** button.
* Set the inspection method by clicking the ***Insp. Method*** button and choosing the polygon method
* Click on the ***Set Coord.*** button and provide the input files. There are two options for this method:
  #### ***1. Rectangle by two points***
  The user provides two points in the **format (lat, lon)**, and the tool creates a rectangle, extracting all the building footprints available within it for subsequent inspection.
  #### ***2. Polygon coordinates using a csv file***
  The user provides a CSV file with the vertex coordinates (which must be uploaded in either clockwise or counterclockwise order). The tool extracts all the building footprints available within the polygon in order to performs virtual inspections. The required file format is attached below.
  
  ![image](https://github.com/user-attachments/assets/2e4e29a9-ebc9-455f-8c60-63f54ff9cb61)
  
* The number of inspections is based on the ***sample size***, which should be provided by the user. The available footprint population is displayed after clicking the ***Load Data*** button, allowing the user to choose the sample size or decide if the inspection should cover the entire building population within the area. The format of the file is presented below.
* Upload the images by clicking the ***Next Building*** button
* Chose building feature for each combobox (e.g., concrete as LLRS material)
* Continue the process by clicking the ***Next Building*** button until the inspections are complete, or save your progress by clicking the ***Save Inspection*** button.
* The results will be saved with the name of the city and contry (e.g., Barranquilla_Colombia).
* The saved file will contain all the building features, city, country, coordinates and the path to the building image. Furthermore, it will be uploaded when the GUI starts again, allowing the inspection process to continue from where it was finished. This is why it is important to save the inspection before closing the GUI.
  
### 2.2 Specific location method
#### *Currently unavailable*
* Select the folder where the outputs will be saved by clicking the ***Project Folder*** button.
* Set the inspection method by clicking the ***Insp. Method*** button and choosing the specific coordinates method
* Click on the ***Set Coord.*** button and provide a *.csv file containing the ID and coordinates of the building images (with the same structure presented in polygon method, by clicking the ***Upload building coordinates*** button.
* Upload the images by clicking the ***Next Building*** button
* Chose building feature for each combobox (e.g., concrete as LLRS material)
* Continue the process by clicking the ***Next Building*** button until the inspection is complete, or save your progress by clicking the ***Save Inspection*** button.
* The results will be saved with the name given in the ***Output name*** cell, which by default is "Specific", in a *.csv file. This file will contain all the building features, city, country, coordinates and the path to the building image. Furthermore, it will be uploaded when the GUI starts again, allowing the inspection process to continue from where it was finished. This is why it is important to save the inspection before closing the GUI.

### 2.3 Local images method
* Select the folder where the outputs will be saved by clicking the ***Project Folder*** button.
* Set the inspection method by clicking the ***Insp. Method*** button and choosing the local method
* Click on the ***Set Coord.*** button and provide the input files by clicking the Select ***Image Folder*** button and selecting the folder where the building images are stored on the local device. Then, by clicking the ***Upload Building Information*** button, the user should provide a *.csv file containing the ID and coordinates of the building images (as shown in polygon method). Finally, select the number of images available per location (1 to 3).
* Upload the images by clicking the ***Next Building*** button
* Chose building feature for each combobox (e.g., concrete as LLRS material)
* Continue the process by clicking the ***Next Building*** button until the inspection is complete, or save your progress by clicking the ***Save Inspection*** button.
* The results will be saved with the name given in the ***Output name*** cell, which by default is "Local", in a *.csv file. This file will contain all the building features, city, country, coordinates and the path to the building image. Furthermore, it will be uploaded when the GUI starts again, allowing the inspection process to continue from where it was finished. This is why it is important to save the inspection before closing the GUI.

### AI powered Option 
This option is enabled by clicking the ***AI Powered*** checkbox. Then, each time the ***Next Building*** button is clicked, all the building features will be predicted using a deep learning model. Currently, all models use [DenseNet201](https://pytorch.org/vision/0.20/models/generated/torchvision.models.densenet201.html) and leverage transfer learning from [ImageNet](https://www.image-net.org/), followed by fine-tuning. However, the user should still verify that these features match the ground truth labels and choose the image quality.

# Citation
Daniel Gómez. (2025). dangomezm/GEM_AI_Toolkit: GEM_V1 (GEM_V1). Zenodo. https://doi.org/10.5281/zenodo.14977499
