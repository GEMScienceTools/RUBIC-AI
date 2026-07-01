# Specific Coordinates Method 
**Best for:** Characterizing specific buildings, for example: reviewing all hospitals in the area of analysis, even if they are located in different countries.

**Workflow:**
1. **Set the usage mode**  
   Select the **Specific coordinates** option, and then click the ***Save and continue*** button.
   
2. **Select the output project folder**  
   Click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved.  
   > 📁 *Example:* `demos/specific_coordinates`
    
3. **Set input files**  
   Click the ***Upload file with coordinates*** button to open a pop-up window and navigate to the file where the coordinates of the building of interest are stored.
   > 📁 *Example:* `demos/specific_coordinates/specific_coordinates_example_data.csv`  
   > 📝 *Required CSV format:*
   
   ```csv
   id,latitude,longitude
   1,10.9639,-74.7964
   2,10.9640,-74.7965
   ```
	  At this point, the file will be previewed in a table so you can verify the selected information.  
	  After this, the user should check the input files by clicking the ***Save and continue*** button.  
	  If any field is missing or contains an error in its name, the GUI will display a message indicating which fields are missing.  
   
	- 3.1. The results will be saved using the name specified in the ***Output name*** field (default: **"specific_coord"**) as a `.csv` file. This file will contain all the building features, along with metadata such as city, country, coordinates, and the path to the corresponding building image.

4. **Classification options, the user can classify building features in two ways:**
     
   **I)** Manually or **II)** Using Deep Learning models with verification of predicted attributes.

	- 4.1. 📝 ***Manual Classification***
		- **4.1.1.** Click the ***Next Building*** button to upload and display the first building image.
		- **4.1.2.** Specify the construction epoch that is most relevant to the area under analysis (this step is only required for the first analysis).
		- **4.1.3.** Use the corresponding combo boxes to select the appropriate features based on the displayed image (e.g., select "Concrete" as the LLRS material).
		- **4.1.4.** Click the ***Next Building*** button again to proceed to the next image, and repeat this process until all images have been reviewed.
		- **4.1.5.** Save either all results or a partial set by clicking the ***Save data*** button. When the GUI is launched again, previously saved results will be reloaded, allowing the classification process to resume from where it left off.  
		> ⚠️ **Important:** *If you do not save your data before closing the GUI, your work will be lost.*

	- 4.2. 🤖 ***AI-Powered Classification***
		- **4.2.1.** The ***AI Powered*** checkbox will be activated, which means the feature will be predicted using AI.  
However, the user can easily switch back to manual inspection by clicking the checkbox again.
		- **4.2.2.** Upload the images by clicking the ***Next Building*** button. At this step, the tool will automatically predict the building features.
		- **4.2.3.** The user should manually define the epoch of construction and image quality, since there is currently no model available for these features
		- **4.2.4.** Click the ***Next Building*** button again to proceed to the next image, and repeat this process until all images have been reviewed.
		- **4.2.5.** Save either all results or a partial set by clicking the ***Save data*** button. When the GUI is launched again, previously saved results will be reloaded, allowing the classification process to resume from where it left off.  
		> ⚠️ **Important:** *If you do not save your data before closing the GUI, your work will be lost.*

5. **Main interface features**
   
	- **5.1 Manual Box**
	     - If the automatic delimitation of the building is not adequate for proper isolation,  
		   or if the selected building is not the building of interest, the user can define a manual bounding box by clicking on four points.

			<img src="../../help_img/manual_box.png">
	- **5.2 Set Image Angle**  
		- The user can define the **pitch** angle, which controls the **vertical inclination of the camera**, allowing a full view of tall buildings. This angle is limited between **0°** and **60°**, with **5°** as the default value.  
		<img src="../../help_img/pitch_angle.png" height="350">

		- The user can also define the **heading** angle, which refers to the **horizontal camera angle**, controlling the direction of the camera and even allowing a view of buildings on the opposite side of the street (180°). This angle is limited between **–180°** and **180°**, with **0°** as the default value.

		- The user can also define the **FOV** (field of view). The FOV controls the **camera zoom**: smaller values zoom in, while larger values zoom out. By default, this value is set to **120**, which is the maximum allowed. This helps create a natural zoom effect without losing too much resolution in distant building images.

	- **5.3 Review Previous Classifications**
	     - If you want to check a specific image, use the ***Search Building*** button. First, enter the image ID (e.g., `1_1`) in the adjacent field, then click the ***Search Building*** button. The GUI will automatically display the corresponding image and its saved classification.
		   > ⚠️ **Important:** *This only works for inspections that were previously saved.*
    - **5.4 Building Attribute Prediction**
	     - Even when up to three images are displayed, only one image is used to classify the building. The selected image is identified by a highlighted border.
   		 <img src="../../help_img/selected_building_img.png" height="350">

6. **API Key Configuration**  

The user must create two Google API keys:  
- **Google Street View Static API** → saved as ***gsv_api_key.txt***  
- **Google Roads API** → saved as ***roads_api_key.txt***  

Both files should be placed inside the `methods` folder.  

> 📁📝 *Example:* `methods/gsv_api_key.txt`  
> 📁📝 *Example:* `methods/roads_api_key.txt`
