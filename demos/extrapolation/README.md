# Neighbor Extrapolation

**Best for:** Expanding known building data to classify unknown buildings

**Workflow:**
1. **Set the usage mode**  
   Select the **Neighbor extrapolation** option, and then click the ***Input Requerements*** button.

2. **Set the Extrapolation Mode**  
   Currently, there are two options available:  

   - **KNN with soft voting** – a basic extrapolation strategy based on the distance to the closest examples.  
   - **Stratified sampling** – a statistical method that ensures representative data by dividing a population into homogeneous subgroups (strata) and sampling from each.  

   Select one of the two options and then click the ***Load Files*** button. A new pop-up window will appear for setting the input files.
   
3. **Set input files - KNN with soft voting**
   Set the input files using either the **Manual** (upload data manually) or the **AI** (deep learning model) method.
	- 3.1. 📄 ***Upload data manually***
		- 3.1.1. Define the output file name using the ***Output name*** field (default: **"KNN_manual"**).
	 	- 3.1.2. Define output folder by click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved.  
		    > 📁 *Example:* `demos/local_images`
		- 3.1.3. Click the ***Buildings with information*** button and upload a CSV file containing the reference buildings, this mean, buildings that have already been classified and include all the features of interest.  
			> 📁 *Example:* `demos/extrapolation/neighbor_building_info.csv`  
			> 📝 *Required CSV format:*
			
			```
			id,latitude,longitude,country,city,material,llrs,code_level,n_stories,occupancy,block_position,taxonomy
			1,10.92224755,-74.78642608,Colombia,Barranquilla,MCF,LWAL,CDL,1,RES,BP1,MCF/LWAL+CDL/H:1/RES/BP1
			2,10.91268031,-74.80288191,Colombia,Soledad,CR,LFM,CDM,3,RES,BP2,CR/LFM+CDM/H:3/RES/BP2
			3,10.91968505,-74.79215175,Colombia,Barranquilla,MUR,LWAL,CDL,1,RES,BP1,MUR/LWAL+CDL/H:1/RES/BP1
			4,10.91647181,-74.76986198,Colombia,Soledad,CR,LFINF,CDM,2,COM,BP1,CR/LFINF+CDM/H:2/COM/BP1
			5,10.90251035,-74.79685532,Colombia,Soledad,CR,LFM,CDL,2,RES,BP2,CR/LFM+CDL/H:2/RES/BP2
			```
		
		- 3.1.4. Click the ***Unclassified building coords*** button and upload a CSV file containing the ID and coordinates of the buildings that you want to classify based on the information from the reference buildings, since no information is available for them.
			> 📁 *Example:* `demos/Extrapolation_data_example/unclassified_building_coord.csv`
			> 📝 *Required CSV format:*
			```csv
			id,latitude,longitude
			1,10.9639,-74.7964
			2,10.9640,-74.7965
			```
   		- 3.1.5 Define **K-value**, which represents the number of nearest reference buildings used to predict the attributes of each target building.
   		- 3.1.5 Click the ***Save and Continue*** button and follow the instructions provided by the GUI.  
  Then, click the ***Next Building*** button in the main panel to start the extrapolation analysis.

	- 3.2 🤖 ***Deep learning model***
		- 3.2.1. Define the output file name using the ***Output name*** field (default: **"KNN_dl"**).
		- 3.2.2. Define output folder by click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved. 
	 	- 3.2.3. Click the **Upload building coordinates** button and select the file containing the coordinates of the reference buildings for the extrapolation process.
			> 📁 *Example:* `demos/Extrapolation_data_example/building_coordinates_example.csv`
			> 📝 *Required CSV format:*
			```csv
			id,Latitude,Longitude
			1,10.9559,-74.7964
			2,10.9550,-74.7965
			```
	
		- 3.2.4. Click the ***Unclassified building locations*** button and upload a CSV file containing the ID and coordinates of the buildings that need to be classified based on the predicted features of the reference buildings. These buildings do not have image data or existing attribute information.
			> 📁 *Example:* `demos/Extrapolation_data_example/unclassified_building_coord.csv`  
			> 📝 *Required CSV format:
			```csv
			id,Latitude,Longitude
			1,10.9639,-74.7964
			2,10.9640,-74.7965
			```
   		- 3.2.5 Define **K-value**, which represents the number of nearest reference buildings used to predict the attributes of each target building.
   		- 3.2.6 Click the ***Save and continue*** button and follow the instructions provided by the GUI.  
  Then, click the ***Next Building*** button in the main panel to start the extrapolation analysis.

**4. Set Input Files – Stratified Sampling**  
  - 4.1. Define the output file name using the ***Output name*** field (default: **"rubic_ai"**).  
  - 4.2. Define output folder by click the ***Select output folder*** button to open a pop-up window and navigate to the folder where outputs will be saved  
  - 4.3. Define the extrapolation mode: **(I)** Deep Learning Model or **(II)** Manual.  
  - 4.4. Provide population data:  
    - For the **deep learning model**, this input corresponds to the coordinates of the reference buildings, which are classified using an approach similar to the one described in the *Specific Coordinates* method.
    	> 📁 *Example:* `demos/extrapolation/building_coordinates_example.csv` 
    - For the **manual option**, this input corresponds to the complete information for the reference buildings that have already been classified.  
	    > 📁 *Example:* `demos/extrapolation/manual_existing_distribution_example.csv`  
	- Select the **feature strata** of interest (e.g., material). The selected feature will then appear in the Selected section below.
		<details>
		<summary>📚 Click to expand: Learn how strata influence the results</summary>
		
		In stratified sampling, the population is divided into homogeneous subgroups, called **strata**, based on a relevant characteristic.  
		For example, if we want to analyze the number of stories of buildings in a city, we can divide the entire building stock into several strata according to their height:
		
		- **Stratum 1:** 1–2-story buildings (e.g., 6,000 buildings)  
		- **Stratum 2:** 3–5-story buildings (e.g., 3,000 buildings)  
		- **Stratum 3:** More than 5 stories (e.g., 1,000 buildings)  
		
		If we decide to sample 10% of all buildings, stratified sampling ensures that the sample maintains the same proportions as the population. Therefore, we would select approximately:  
		
		- 600 buildings from Stratum 1  
		- 300 buildings from Stratum 2  
		- 100 buildings from Stratum 3  
		
		This way, the final sample (1,000 buildings) accurately represents the city’s building-height distribution.  
		Without stratification, a simple random sample might over- or under-represent certain strata (for instance, selecting too many low-rise buildings), leading to biased results.
		
		</details>
	
	- 4.5 Define the parameters for stratified sampling or keep the default values (e.g. initial fraction, step)  
	
	- 4.6 Click the ***Save and Continue*** button and follow the instructions provided by the GUI.  
	  Then, click the ***Next Building*** button in the main panel to start the extrapolation analysis.
	
	- 4.7. **API Key Configuration**  
	
	The user must create two Google API keys:  
	- **Google Street View Static API** → saved as ***gsv_api_key.txt***  
	- **Google Roads API** → saved as ***roads_api_key.txt***  
	
	Both files should be placed inside the `methods` folder.  
	
	> 📁📝 *Example:* `methods/gsv_api_key.txt`  
	> 📁📝 *Example:* `methods/roads_api_key.txt`
