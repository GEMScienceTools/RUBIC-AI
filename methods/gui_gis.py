# GUI pyqt5 libraries
from PyQt5.QtWidgets import QMessageBox

# Libries for shape and geopackage creation
import osmnx as ox
import geopandas as gpd

# Utilities libraries
import time
import os

class GUI_geofiles:
    def __init__(self, ui):
        self.ui = ui  # Link to the UI components
        
    ############ Normalize input method ################
    # def normalize_input (self):
    #     """
    #     Normalize the input text for city and country names.
        
    #     This method retrieves the city and country names entered in the user interface (UI),
    #     normalizes them by capitalizing the first letter of each word, and updates the respective
    #     fields in the UI with the normalized text.
        
    #     Args:
    #         None. The method operates on UI elements and updates instance attributes 
    #         `country_method` and `city_method`.
        
    #     Returns:
    #         None. Normalized city and country names are saved to instance attributes 
    #         and updated in the UI.
        
    #     Effects:
    #         - Updates `self.country_method` and `self.city_method` with the normalized values.
    #         - Updates the UI fields `country_value` and `city_value` with normalized text.
    #     """
    #     # Normalize country input
    #     country = str(self.ui.country_value.text()) # Retrieve text from QLineEdit
    #     # Normalize the text
    #     normalized_text = ' '.join(word.capitalize() for word in country.strip().split())
    #     # Update the QLineEdit with normalized text
    #     self.country_method = normalized_text
    #     self.ui.country_value.setText(normalized_text)
        
    #     # Normalize city input
    #     city = self.ui.city_value.text()
    #     # Normalize the text
    #     normalized_text = ' '.join(word.capitalize() for word in city.strip().split())
    #     # Update the QLineEdit with normalized text
    #     self.city_method = normalized_text
    #     self.ui.city_value.setText(normalized_text)
                
    # ############ City boundary shape file ################
    # def get_administrative_boundary(self):
    #     """
    #     Retrieve and save the administrative boundary of a specified city as a GeoPackage file.
        
    #     This method checks for a defined project folder and avoids creating duplicate files 
    #     if the boundary file already exists. If no boundary file is found, it downloads the 
    #     city's administrative boundary using OpenStreetMap data, then saves it to a GeoPackage.
    
    #     Args:
    #         None. The method relies on instance attributes such as `city_method`, `country_method`, 
    #         and the output folder path provided in the UI.
    
    #     Returns:
    #         None. The administrative boundary is saved to a GeoPackage file in the specified 
    #         output folder.
    
    #     Raises:
    #         QMessageBox.Warning: If no project folder is defined in the UI.
    #     """
    
    #     # # Conditional to avoid executing the method if there is no project folder
    #     # if self.output_folder_value.text() == "-":
    #     #     QMessageBox.warning(self, "Project Error", "Please select project folder")
    #     # # Conditional to avoid executing the method if there is no country name
    #     # elif self.country_value.text() == "-":
    #     #     QMessageBox.warning(self, "Country Error", "Please sets country name")
    #     # # Conditional to avoid executing the method if there is no city name 
    #     # elif self.city_value.text() == "-":
    #     #     QMessageBox.warning(self, "City Error", "Please sets city name")
    #     # else:
    #     # Concatenate city and country to save data
    #     self.city_method = self.city_value.text()
    #     self.country_method = self.country_value.text()
    #     self.city_name = self.city_method +" , " + self.country_method
    #     # Create output file for city's administrative boundary 
    #     self.boundary_path = self.output_folder_value.text()+"/"+self.city_method+"_"+self.country_method+"_boundary.gpkg"
    #     # Conditionional checks for an existing boundary file, and if it exists, avoids creating a duplicate
    #     if os.path.exists(self.boundary_path):
    #         self.method_progress.setText("Boundary is available")
    #     else:
    #         # Download the city's administrative boundary 
    #         print("---------------- " + self.city_name + " -------------------")
    #         boundary = ox.geocode_to_gdf(self.city_name)             
    #         # Export the boundary as geopackage file 
    #         boundary.to_file(self.boundary_path, driver="GPKG", layer="Boundary")
    #         print("Done. Boundary is available")
    
    
    ############ City boundary shape file ################
    def download_building_footprints(self):
        """
        Download and save building footprints within a defined city boundary as a GeoPackage.
    
        This method checks for an existing project folder and boundary file before downloading
        building footprints from OpenStreetMap (OSM). It ensures that the coordinate reference 
        system (CRS) is in EPSG:4326 and filters out invalid geometries before saving the 
        building footprints to a GeoPackage.
    
        Args:
            None. The method relies on instance attributes such as `boundary_path`, `city_method`, 
            and the output folder path provided in the UI.
    
        Returns:
            None. The building footprints are saved to a GeoPackage file in the specified 
            output folder.
    
        Effects:
            - Displays a progress bar in the UI during the download process.
            - Updates the UI with the progress and status of the operation.
    
        Raises:
            - Skips execution if no project folder or boundary file is defined.
            - Logs messages if no building footprints are found or valid geometries are unavailable.
        """
    
        # # Conditional to avoid executing the method if there is no project folder
        # if self.output_folder_value.text() == "-":
        #     pass
        # # Conditional to avoid executing the method if there is no country name
        # elif self.country_value.text() == "-":
        #     pass
        # # Conditional to avoid executing the method if there is no city name 
        # elif self.city_value.text() == "-":
        #     pass
        # else:
        # Create output file for building footprints
        self.city_method = self.city
        self.country_method = self.country
        output_file = self.method.output_folder_value+"/"+self.city_method+"_"+self.country_method+"_buildings_footprint.gpkg"
        # Conditionional checks for an existing boundary file, and if it exists, avoids creating a duplicate
        if os.path.exists(output_file):
            buildings = gpd.read_file(output_file)
        else:
            # Check if a boundary file exists to download the building footprints within it
            if os.path.exists(self.boundary_path):
                # Create a progress bar for users so they know the GUI is processing tasks in the backend
                # for i in range (100):
                #     time.sleep(0.0001)
                #     self.progress_bar_method.setValue(i)
                #     self.method_progress.setText("Method in progress ....")
                
                # Define input and output file paths
                geopackage_path = self.boundary_path
                # Load the single layer from the GeoPackage
                gdf = gpd.read_file(geopackage_path)
                # Ensure the CRS is EPSG:4326
                if gdf.crs.to_string() != "EPSG:4326":
                    print("Reprojecting to EPSG:4326...")
                    gdf = gdf.to_crs("EPSG:4326")
                # Download building footprints from OSM
                polygon = gdf.unary_union
                # For latest osmnx versions
                try:
                    buildings = ox.geometries_from_polygon(polygon, tags={"building": True})
                except AttributeError:
                    buildings = ox.features_from_polygon(polygon, tags={"building": True})
                # Save the downloaded footprints to a new GeoPackage
                if buildings.empty:
                    return None
                # Filter only Polygon and MultiPolygon geometries
                buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])]
                if buildings.empty:
                    return None   
                # Drop the AREA column if it exists
                if "AREA" in buildings.columns:
                    buildings = buildings.drop(columns=["AREA"])
                # Save to GeoPackage
                buildings.to_file(output_file, driver="GPKG")
        
        return len(buildings)
    
    ############ Random subset buildings ################  
    def extract_random_subset(self , sample_size):
        """
        Extract a random subset of building footprints and save it as a GeoPackage.
    
        This method selects a random sample of building footprints from a previously saved GeoPackage 
        and saves the subset to a new GeoPackage file. It ensures that the sample size does not 
        exceed the number of features in the dataset.
    
        Args:
            None. The method operates on instance attributes such as `city_method`, `country_method`, 
            and the output folder path provided in the UI.
    
        Returns:
            None. The random subset is saved to a GeoPackage file in the specified output folder.
    
        Effects:
            - Saves a randomly selected subset of building footprints to a new GeoPackage.
            - Logs messages about the saving process.
    
        Raises:
            ValueError: If the sample size exceeds the number of features in the dataset.
    
        Notes:
            - The random sample is controlled by a predefined sample size (`sample_size`) and seed 
              (`seed`) for reproducibility.
        """
    
        # # Conditional to avoid executing the method if there is no project folder
        # if self.output_folder_value.text() == "-":
        #     pass
        # # Conditional to avoid executing the method if there is no country name
        # elif self.country_value.text() == "-":
        #     pass
        # # Conditional to avoid executing the method if there is no city name 
        # elif self.city_value.text() == "-":
        #     pass
        # else:
        # Load buildng footprints
        footprint = self.method.output_folder_value+"/"+self.city_method+"_"+self.country_method+"_buildings_footprint.gpkg"
        # Create output file for building footprints
        output_file= self.method.output_folder_value+"/"+self.city_method+"_"+self.country_method+"_subset_footprints.gpkg"
        # Ensure sample size is not greater than the number of points in the dataset
        seed=10
        # Check if a subset file exists
        if os.path.exists(output_file):
            QMessageBox.warning(
                self,
                "Sample Size Warning",
                "A sample size file already exists.\n\n"
                "Any changes to the sample size have not been applied.\n"
                "If you want to use a different sample size, please delete the existing files first."
            )
        else:
            sample_size = int(sample_size)
            # Load the input point layer
            gdf = gpd.read_file(footprint)
            
            print("Sample size: ", sample_size)
            print("N° Buildings: ", len(gdf))
            
            if sample_size > len(gdf):
                QMessageBox.warning(
                    self,
                    "Sample Size Warning",
                    f"Sample size {sample_size} exceeds the number of features in the dataset ({len(gdf)}).")
                
            # Extract a random sample
            subset = gdf.sample(n=sample_size, random_state=seed)
            # Save the subset to a GeoPackage
            print(f"Saving random subset to: {output_file}")
            subset.to_file(output_file, driver="GPKG", layer="random_subset")

    ############ Create a point layer and extract the coordinates of a subset of buildings ################ 
    def create_centroid_layer(self):
        """
        Create a GeoPackage layer of centroids from building footprints, including latitude and longitude.
        
        This method calculates the centroids of building footprints from a subset GeoPackage file, 
        extracts their latitude and longitude, and saves the resulting data to a new GeoPackage layer. 
        If the centroid file already exists, it skips the execution.
        
        Args:
            None. The method operates on instance attributes such as `city_method`, `country_method`, 
            and the output folder path provided in the UI.
        
        Returns:
            None. The centroid data is saved to a GeoPackage file in the specified output folder.
        
        Effects:
            - Computes the centroids of building footprints.
            - Extracts latitude and longitude from centroid geometries.
            - Saves the centroids to a GeoPackage file.
        
        Notes:
            - The method checks for an existing centroid GeoPackage to avoid duplicate processing.
            - Ensures the output retains the same CRS as the input building footprints.
        """

        # # Conditional to avoid executing the method if there is no project folder
        # if self.output_folder_value.text() == "-":
        #     pass
        # # Conditional to avoid executing the method if there is no country name
        # elif self.country_value.text() == "-":
        #     pass
        # # Conditional to avoid executing the method if there is no city name 
        # elif self.city_value.text() == "-":
        #     pass
        # else:
        # Load selected subset building
        subset_file=self.method.output_folder_value+"/"+self.city_method+"_"+self.country_method+"_subset_footprints.gpkg"
        # Create output file for building footprints
        output_file=self.method.output_folder_value+"/"+self.city_method+"_"+self.country_method+"_subset_centroids.gpkg"
        # Check if a centroid file exists
        if os.path.exists(output_file):
            pass
        else:
            # Load the building footprints
            buildings = gpd.read_file(subset_file)
            # Calculate centroids
            print("Calculating centroids...")
            buildings["centroid"] = buildings.geometry.centroid
            # Create a new GeoDataFrame for the centroids
            centroids = gpd.GeoDataFrame(
                buildings.drop(columns="geometry"),  # Drop original geometry column
                geometry=buildings["centroid"],  # Set centroid as the active geometry
                crs=buildings.crs ) # Use the same CRS as the input  
            # Drop any remaining references to the original geometry column
            if "centroid" in centroids.columns:
                centroids = centroids.drop(columns=["centroid"])
            # Add latitude and longitude columns
            print("Extracting latitude and longitude...")
            centroids["latitude"] = centroids.geometry.y
            centroids["longitude"] = centroids.geometry.x
            # Save the centroids to a GeoPackage
            print(f"Saving centroids to: {output_file}")
            centroids.to_file(output_file, driver="GPKG", layer="centroids")
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    