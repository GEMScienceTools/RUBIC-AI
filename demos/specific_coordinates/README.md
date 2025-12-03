### 2. Specific Coordinates Method 
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
	  After this, the user should check the input files by clicking the ***Load data*** button.  
	  If any field is missing or contains an error in its name, the GUI will display a message indicating which fields are missing.  
	  If everything is correct, click the ***Save and continue*** button.
   
	- 3.1. The results will be saved using the name specified in the ***Output name*** field (default: **"specific_coord"**) as a `.csv` file. This file will contain all the building features, along with metadata such as city, country, coordinates, and the path to the corresponding building image.

4. **Classification options and setup main interface**

   - The process and available features are the same as those presented in sections 5 and 6 of the ***Polygon Method***.

5. **API Key Configuration**

   - Same process as explained in the ***Polygon Method***.
