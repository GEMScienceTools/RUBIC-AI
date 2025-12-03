### Polygon Method 
**Best for:** Create a building stock from well defined area such as a neighborhood, city, or similar

**Workflow:**
1. **Set the usage mode**  
   Select the **Polygon method** option, and then click the ***Save and continue*** button.
   
2. **Select the output project folder**  
   Click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved.  
   > 📁 *Example:* `demos/polygon_method`
    
3. **Set input files**  

   - **3.1 New polygon**
      Click the ***Upload file with coordinates*** button to open a pop-up window and navigate to the file where the vertices of the polygon are stored. The outputs will be saved automatically 
	   > 📁 *Example:* `demos/polygon_method/polygon_method_example.csv`  
	   > 📝 *Required CSV format:*
	   
	   ```csv
	   id,latitude,longitude
	   1,10.9639,-74.7964
	   2,10.9640,-74.7965
	   ```
	   At this point, the file will be previewed in a table so you can verify the selected information.

	- **3.2 Existing Polygon**  
  Check the ***Perform analysis using an existing polygon*** option to enable the ***Upload existing polygon*** button.  
  Clicking this button will open a pop-up window where you can select and upload a `.gpkg` or `.shp` file.

4. **Define the Source of the Building Footprint**  
Select the desired option from the menu next to ***Footprint source***.  
By default, the selected option is **OpenStreetMap**, but **Overture Maps** is also available.  
Overture combines different sources of information, such as Google Open Buildings, Microsoft Building Footprints, and OpenStreetMap.  
However, this option may take more time to retrieve the footprints.
   
5. **Download the available building footprints in the defined area by clicking the ***Get footprints available*** button**  

   - **5.1. Footprint Sample**
     - The number of available footprints will be displayed next to the text ***N° footprints***.  
     - The user should define the sample size using the ***Sample size*** field. This value must be equal to or less than the total number of available building footprints.  

6. **Classification options, the user can classify building features in two ways:**
     
   **I)** Manually** or **II)** Using Deep Learning models with verification of predicted attributes.

	- 6.1. 📝 ***Manual Classification***
		- **6.1.1.** Click the ***Next Building*** button to upload and display the first building image.
		- **6.1.2.** Specify the construction epoch that is most relevant to the area under analysis (this step is only required for the first analysis).
		- **6.1.3.** Use the corresponding combo boxes to select the appropriate features based on the displayed image (e.g., select "Concrete" as the LLRS material).
		- **6.1.4.** Click the ***Next Building*** button again to proceed to the next image, and repeat this process until all images have been reviewed.
		- **6.1.5.** Save either all results or a partial set by clicking the ***Save data*** button. When the GUI is launched again, previously saved results will be reloaded, allowing the classification process to resume from where it left off.  
		> ⚠️ **Important:** *If you do not save your data before closing the GUI, your work will be lost.*

	- 6.2. 🤖 ***AI-Powered Classification***
		- **6.2.1.** The ***AI Powered*** checkbox will be activated, which means the feature will be predicted using AI.  
However, the user can easily switch back to manual inspection by clicking the checkbox again.
		- **6.2.2.** Upload the images by clicking the ***Next Building*** button. At this step, the tool will automatically predict the building features.
		- **6.2.3.** The user should manually define the epoch of construction and image quality, since there is currently no model available for these features
		- **6.2.4.** Click the ***Next Building*** button again to proceed to the next image, and repeat this process until all images have been reviewed.
		- **6.2.5.** Save either all results or a partial set by clicking the ***Save data*** button. When the GUI is launched again, previously saved results will be reloaded, allowing the classification process to resume from where it left off.  
		> ⚠️ **Important:** *If you do not save your data before closing the GUI, your work will be lost.*
	
	- 6.3. The results will be saved using the name specified in the ***Output name*** field (default: **"polygon_building"**) as a `.csv` file. This file will contain all the building features, along with metadata such as city, country, coordinates, and the path to the corresponding building image.
		> ⚠️ **Important:** *Since this is the first version, it is strongly recommended to verify the results from the AI-powered mode.*

7. **Main interface features**
   
	- **7.1 Manual Box**
	     - If the automatic delimitation of the building is not adequate for proper isolation,  
		   or if the selected building is not the building of interest, the user can define a manual bounding box by clicking on four points.

			<img src="help_img/manual_box.png">
	- **7.2 Review Previous Classifications**
	     - If you want to check a specific image, use the ***Search Building*** button. First, enter the image ID (e.g., `1_1`) in the adjacent field, then click the ***Search Building*** button. The GUI will automatically display the corresponding image and its saved classification.
		   > ⚠️ **Important:** *This only works for inspections that were previously saved.*

8. **API Key Configuration**  

The user must create two Google API keys:  
- **Google Street View Static API** → saved as ***gsv_api_key.txt***  
- **Google Roads API** → saved as ***roads_api_key.txt***  

Both files should be placed inside the `methods` folder.  

> 📁📝 *Example:* `methods/gsv_api_key.txt`  
> 📁📝 *Example:* `methods/roads_api_key.txt`
