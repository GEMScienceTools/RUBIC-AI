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
   Click the ***Upload building information*** button to open a pop-up window and navigate to the file containing the building coordinates.  
   
   > 📁 *Example:* `demos/local_images/data_ex1.csv`  
   > 📝 *Required CSV format:*
   
   ```csv
   id,latitude,longitude
   1,10.9639,-74.7964
   2,10.9640,-74.7965
   ```
   At this point, the file will be previewed in a table so you can verify the selected information.
   
	- 4.1. The results will be saved using the name specified in the ***Output name*** field (default: **"Local_images"**) as a `.csv` file. This file will contain all the building features, along with metadata such as city, country, coordinates, and the path to the corresponding building image.
	- 4.2. A maximum of three windows can display the same location (ideally the same building). These will be displayed automatically if they share the same coordinates.  
	- 4.3. Upload the information and check the format by clicking the ***Load data*** button. Once a confirmation message appears, click ***Save and continue*** to proceed to the next step.

5. **Classification options and setup main interface**

   - The process and available features are the same as those presented in sections 5 and 6 of the ***Polygon Method***.
   


