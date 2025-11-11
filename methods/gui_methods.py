# GUI pyqt5 libraries
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5 import QtCore, QtGui

# Libries for shape and geopackage creation
import geopandas as gpd

# Utilities libraries
import time
import os
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from geopy.geocoders import Nominatim
import torch

# Taxonomy check
from methods.taxonomy import check_taxonomy
import re  

# Google Street Maps libry
import requests

# *.py scripts with complex methods
from methods.dl_prediction_models import predict_llrs_img, predict_material_img, predict_code_img, predict_roof_shape_img
from methods.dl_prediction_models import predict_occupancy_img, predict_block_position_img, predict_n_stories_img, predict_roof_material_img
from methods.get_building_orientation import get_street_view_image , get_road_orientation
from methods.bounding_box_manual import BoundingBoxWindow
from methods.epoch_construction import EpochSelectionDialog
from methods.help_window import HelpDialog
from methods.neighbor_building_extrapolation_feature import find_nearest_neighbors_geodesic, compute_taxonomy_distribution_full_structure
from methods.dl_extrapolation import create_database, dl_models, inspection_database, extrapolation_existing_reference

from methods.dl_stratified import iterative_distribution_stability_manual , iterative_label_discovery_cached_fractional, labeling_function


class GUIMethods:
    def __init__(self, ui):
        self.ui = ui  # Link to the UI components
        # Initialize attributes to avoid AttributeError
        self.click_count = -1           # Building ID aux varible     
        self.start = True               # Building ID aux varible  
        self.data_building = None       # Building sample information
        self.data_ai = None             # AI inspection
        self.data_old = None            # Existing inspection swicth
        self.sw_insp = True             # Existing inspection swicth           
        self.save_id = True             # Existing inspection swicth      
        self.box_id = None              # Manual bounding box
        self.epoch_const= True          # Epoch of construction
        self.data_old_local = False
        self.sw_local_previous = True
        self.cropped_image = ["","",""]
        self.start_click = True
        self.aux_previous =  True 
        self.aux_ai_check = True
        self.limit_local = True
        """Get screen resolution to adapt to different screen sizes"""
        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()

        # Scale the GUI based on resolution
        self.sf_x = screen_width / 1920
        

    ############ Counts the number of clicks made on the next button ################ 
    def count_clicks_next(self):
        """
        Increment the click counter and update the inspection dataset.
    
        This method increments the click counter to navigate through building inspections. 
        It ensures that a project folder, country, and city name are defined before execution. 
        If necessary, it loads the subset of buildings from a CSV file based on the selected 
        inspection method. The method also verifies that the current building ID does not 
        exceed the number of available samples.
        """
    
        """Increment the click counter and update the label."""
        # load the dataset of the subset buildings
        if self.data_building is None:
            
            if self.ui.insp_method == 0:
                path=self.ui.output_folder_value+"/"+self.ui.file_name+"_building_info.csv"
                self.data_building = pd.read_csv(path)
            elif self.ui.insp_method == 1:
                path= self.ui.output_folder_value+"/"+self.ui.file_name+"_building_info.csv"
                self.data_building = pd.read_csv(path)
            elif self.ui.insp_method == 2:
                self.data_building = pd.read_csv(self.ui.file_local_csv)
            elif self.ui.insp_method == 3:
                self.data_building = pd.DataFrame(np.zeros((2,1)))
                
        # Verify that the building ID is less than the number of sample
        if self.ui.insp_method != 2:
            limit_insp = self.data_building.shape[0] - 1
        elif self.ui.insp_method == 2:
            if self.limit_local == True:
                limit_insp = self.data_building.shape[0] - 1
            else:
                limit_insp = len(self.index_id)-1
        self.limit_local = False
        
        if self.click_count >= limit_insp:
            QMessageBox.warning(self.ui, "Database Error", "No further inspections are available")
        else:
            self.ui.method_progress.setText("Loading images ...")
            
            # Save inspection for first click after save results or start the script               
            if self.click_count >= 0:
                self.inspection_database()
            
            if self.ui.insp_method == 2:
                if self.sw_local_previous == True:
                    valid_coords = self.ui.data_method[['latitude', 'longitude']].dropna()
                    # Drop duplicates to get unique coordinate pairs
                    unique_coords = valid_coords.drop_duplicates()
                    self.index_id = unique_coords.index.tolist()
                    df = self.ui.data_method
                    # Create a coordinate pair column
                    df["coord_pair"] = list(zip(df["latitude"], df["longitude"]))
                    # Keep the first occurrence of each coordinate
                    self.unique_coords = df.drop_duplicates(subset="coord_pair", keep="first").reset_index()
                    self.sw_local_previous = False
                    
                # Check if there is inspection already done
                if self.data_old is not None:
                    self.n_insp = int(self.data_ai.dropna(how='all').shape[0])
                    self.data_old = None  # Only give the number of inspection one time per saved button clicked
            else:
                if self.data_old is not None:
                    self.n_insp = int(self.data_ai.dropna(how='all').shape[0])
                    self.data_old = None  # Only give the number of inspection one time per saved button clicked
            
            # Calculates the number of inspections saved
            try:
                # Conditional for only update the number of click and the ID cont one time
                if self.n_insp > 0 and self.sw_insp == True:
                    if self.ui.insp_method != 2:
                        # self.click_count = int(self.n_insp/3 - 1)
                        self.click_count = self.n_insp - 1
                        self.sw_insp = False
                    elif self.ui.insp_method == 2:
                        # Drop rows with missing coordinates
                        valid_coords = self.data_ai_existing[['latitude', 'longitude']].dropna()
                        # Drop duplicates to get unique coordinate pairs
                        unique_coords = valid_coords.drop_duplicates()
                        # Count unique coordinate pairs
                        num_unique_coords = len(unique_coords)
                        self.click_count = num_unique_coords - 1
                        self.sw_insp = False
            except:
                pass
            # ID increaser
            self.click_count += 1
            
            if self.aux_ai_check == True:
                try:
                    self.ui.ai_check.setChecked(self.ui.ai_value)
                except:
                    pass
                self.aux_ai_check = False
                
            if self.start_click == True:
                if self.click_count >= self.data_building.shape[0] - 1:
                    self.click_count = self.data_building.shape[0] - 1
                    QMessageBox.information(self.ui, "Inspections available", "The next building displayed is the final one in the database")
                    self.start_click = False
                
        
    ############ Counts the number of clicks made on the previous button ################ 
    def count_clicks_previous(self):
        """
        Decrement the click counter to navigate to the previous building inspection.
    
        This method decreases the click counter, allowing the user to move back to a 
        previous inspection record. It ensures that a project folder, country, and 
        city name are defined before execution. If the dataset has not been initialized, 
        it prompts the user to click the "Next" button first.
        """

        if self.data_building is None:
            QMessageBox.warning(self.ui, "GUI Error", "Please click the Next button to start the GUI")
            self.n_images_local=0
        else:
            """Increment the click counter and update the label."""
            if self.click_count > 0:
                if self.ui.insp_method == 2:
                    self.click_count += -1
                                          
                elif self.ui.insp_method == 0 or self.ui.insp_method == 1:
                    self.click_count += -1
                        

    ############ Get city name using coordinates ################
    def get_city_name(self):
        """
        Get the city name for a given latitude and longitude using reverse geocoding.
        
        Args:
            lat (float): latitude of the point.
            lon (float): longitude of the point.
        
        Returns:
            str: The name of the city, including the country, or an error message.
        """
        
        if self.click_count >= 0:
            if self.ui.insp_method == 2:

                if self.data_old_local == True:
                    self.cont_local = self.n_insp
                    self.data_old_local = False
                
                self.cont_local = int(self.index_id[self.click_count])
                if 'latitude' in self.ui.data_method.columns:
                    lat = float(self.ui.data_method.loc[self.cont_local, 'latitude'])
                else:
                    raise ValueError("The 'latitude' column is missing from the data.")
                    
                if 'longitude' in self.ui.data_method.columns:
                    lon = float(self.ui.data_method.loc[self.cont_local, 'longitude'])
                else:
                    raise ValueError("The 'latitude' column is missing from the data.")
                
                self.ui.lat_value.setText(str(round(lat,8)))
                self.ui.lon_value.setText(str(round(lon,8)))
                
                df = self.ui.data_method
                matching_rows = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
            
                self.n_images_local = len(matching_rows)
                self.old_local = self.cont_local
                self.cont_local = self.cont_local + self.n_images_local
                    
                geolocator = Nominatim(user_agent="city_name_locator")
                location = geolocator.reverse((lat, lon), exactly_one=True, language="en", timeout=3)
                
                if location and 'address' in location.raw:
                    address = location.raw['address']
                    self.city = (address.get("city") or address.get("town") or address.get("village")
                        or address.get("municipality") or address.get("county") or address.get("state_district")
                        or "Unknown")
                    self.country = address.get('country', 'Unknown')
                    self.city_name_manual = self.city+"_"+self.country
                    self.ui.city_value.setText(self.city) 
                    self.ui.country_value.setText(self.country) 
                    return (self.city , self.country)
                
                return "City not found"
                 
            elif self.ui.insp_method == 0 or self.ui.insp_method == 1:                
                self.ui.lat_value.setText(str(round(self.data_building.loc[self.click_count, 'latitude'], 8)))
                self.ui.lon_value.setText(str(round(self.data_building.loc[self.click_count, 'longitude'], 8)))

                
                lat = float(self.ui.lat_value.text())
                lon = float(self.ui.lon_value.text())
                    
                geolocator = Nominatim(user_agent="city_name_locator")
                location = geolocator.reverse((lat, lon), exactly_one=True, language="en", timeout=3)
                
                if location and 'address' in location.raw:
                    address = location.raw['address']
                    self.city = (address.get("city") or address.get("town") or address.get("village")
                        or address.get("municipality") or address.get("county") or address.get("state_district")
                        or "Unknown")
                    self.country = address.get('country', 'Unknown')
                    self.city_name_manual = self.city+"_"+self.country
                    self.ui.city_value.setText(self.city) 
                    self.ui.country_value.setText(self.country) 
                    return (self.city , self.country)
                
                return "City not found"
        else:
            pass
     
        
    ############ Create building dataset for upload images from GSV ################ 
    def create_database(self):
        """
        Create a CSV dataset of building information for uploading images from Google Street View (GSV).
    
        This method extracts relevant data (ID, latitude, and longitude) from a GeoPackage file containing 
        building centroids and saves it to a CSV file. If the CSV file already exists, the method skips execution.
    
        Args:
            None. The method operates on instance attributes such as `city_method`, `country_method`, 
            and the output folder path provided in the UI.
    
        Returns:
            None. The filtered building dataset is saved as a CSV file in the specified output folder.
    
        Effects:
            - Filters the GeoDataFrame to retain only the `id`, `latitude`, and `longitude` columns.
            - Exports the filtered data to a CSV file.
    
        Notes:
            - Checks for the existence of the CSV file to avoid creating duplicate files.
            - The exported CSV can be used for further processing, such as batch uploading to GSV.
        """       
        
        if self.ui.insp_method == 0:
            self.city_method = self.ui.city
            self.country_method = self.ui.country
            # Input and output for the method
            centroid_file=self.ui.output_folder_value+"/"+self.ui.file_name+"_subset_centroids.gpkg"
            database_file=self.ui.output_folder_value+"/"+self.ui.file_name+"_building_info.csv"
            # Check if a database file exists
            if os.path.exists(database_file):
                pass
            else:
                # Load the GeoPackage
                gdf = gpd.read_file(centroid_file)
                # Filter columns
                filtered_gdf = gdf[['id', 'latitude', 'longitude']]
                # Export to CSV
                filtered_gdf.to_csv(database_file, index=False)             
                print("Filtered CSV exported successfully!")
                
        elif self.ui.insp_method == 1:
            
            self.city_method = self.ui.city_value.text()
            self.country_method = self.ui.country_value.text()
            
            centroid_file = self.ui.output_folder_value+"/"+self.ui.file_name+".gpkg"
            database_file = self.ui.output_folder_value+"/"+self.ui.file_name+"_building_info.csv"
            # Check if a database file exists
            # if os.path.exists(database_file):
            #     pass
            # else:
            # Load the GeoPackage
            gdf = gpd.read_file(centroid_file)
            # Filter columns
            filtered_gdf = gdf[['id', 'latitude', 'longitude']]
            # Export to CSV
            filtered_gdf.to_csv(database_file, index=False)             
        
        elif self.ui.insp_method == 2:  
            pass # There is already the information in the csv with building information
            
                
        # Upload the create building info to get the size of the inspection dataset
        # Craete an empty dataframe with the exact size
        if self.data_building is None:
            # Load the footprint database           
            if self.ui.insp_method == 0:
                footprint_data = pd.read_csv(self.ui.output_folder_value+"/"+self.ui.file_name+"_building_info.csv")
            elif self.ui.insp_method == 1:
                footprint_data = pd.read_csv(self.ui.output_folder_value+"/"+self.ui.file_name+"_building_info.csv")
            elif self.ui.insp_method == 2:
                footprint_data = pd.read_csv(self.ui.file_local_csv)
            elif self.ui.insp_method == 3:
                pass
            
            # Define the column namesfor the inspection database
            column_names = ["id", 
                            "latitude", 
                            "longitude",
                            "country",
                            "city",
                            "material",
                            "llrs",
                            "code_level",
                            "n_stories",
                            "occupancy",
                            "block_position",
                            "epoch_construction",
                            "roof_shape",
                            "roof_material",
                            "image_quality",
                            "taxonomy",
                            "image filename or link"]
            
            # Create an empty DataFrame for number of footprint available
            try:
                if  self.data_ai == None:
                    self.data_ai = pd.DataFrame(np.full((footprint_data.shape[0]*3, len(column_names)), None), columns=column_names)
            except:
                pass
        
            
    ############ Create building dataset for upload images from GSV ################     
    def load_existing_insp(self):
        """
        Load existing inspection data from CSV files based on the selected inspection method.

        This method checks whether a project folder, country, and city name are defined before 
        attempting to load previously saved inspection data. It retrieves existing AI and 
        exposure model inspection records from CSV files and integrates them into the 
        current dataset.
        """
        # Upload existing inspections
        if self.start == True:
            try:
                if self.ui.insp_method == 0:
                    output_folder = self.ui.output_folder_value
                    insp_path = f"{output_folder}/{self.ui.file_name}"
                    # Upload the existing inspections for AI 
                    self.data_ai_existing = pd.read_csv(insp_path+"_AI_aux_cont.csv")
                    # Replace empty rows with the existing information
                    self.data_ai.iloc[:self.data_ai_existing.shape[0], :] = self.data_ai_existing.iloc[:self.data_ai_existing.shape[0], :]
                    
                    self.data_old = "OK"  # THERE IS EXISTING DATA
                    self.start = False
                    
                elif self.ui.insp_method == 1:
                    insp_path = self.ui.output_folder_value+"/"+self.ui.file_name
                    # Upload the existing inspections for AI 
                    self.data_ai_existing = pd.read_csv(insp_path+"_AI_aux_cont.csv")
                    # Replace empty rows with the existing information
                    self.data_ai.iloc[:self.data_ai_existing.shape[0], :] = self.data_ai_existing.iloc[:self.data_ai_existing.shape[0], :]
                    
                    self.data_old = "OK"  # THERE IS EXISTING DATA
                    self.start = False
                    
                elif self.ui.insp_method == 2:
                    insp_path = self.ui.output_folder_value+"/"+self.ui.file_name_local.text()
                    # Upload the existing inspections for AI 
                    self.data_ai_existing = pd.read_csv(insp_path+"_AI_classification.csv")
                    self.cont_local_data = pd.read_csv(insp_path+"_AI_aux_cont.csv")
                    # Replace empty rows with the existing information
                    self.data_ai.iloc[:self.data_ai_existing.shape[0], :] = self.data_ai_existing.iloc[:self.data_ai_existing.shape[0], :]
                    self.cont_local_data.iloc[:self.cont_local_data.shape[0], :] = self.cont_local_data.iloc[:self.cont_local_data.shape[0], :] 
                    print("Upload existing data sucessfully")
                    self.data_old = "OK"  # THERE IS EXISTING DATA
                    self.data_old_local = True
                    self.start = False
                    
                elif self.ui.insp_method == 3:
                    pass
            except:
                pass

 
    ############ Checks if there is GSV availability ################  
    def check_street_view(self):
        """
        Check if Google Street View is available at the building's location.
    
        This method sends a request to the Google Street View API to determine 
        whether Street View imagery is available for the latitude and longitude 
        of the currently selected building. It ensures that a project folder, 
        country, and city name are defined before execution.
    
        Returns:
            bool: 
                - `True` if Street View imagery is available at the given location.
                - `False` if no Street View coverage exists.
    
        Effects:
            - Sends an HTTP request to the Google Street View API.
            - Retrieves metadata about Street View availability.
    
        Notes:
            - Requires a valid Google Street View API key.
            - The API key used in this function is hardcoded, which may pose security risks.
            - Ensures execution only if project details are correctly set.
        """
        # Input parameters
        with open("methods/gsv_api_key.txt", "r") as f:
            api_key = f.read().strip()

        lat= self.ui.lat_value.text()
        lon= self.ui.lon_value.text() 
        url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            "location": f"{lat},{lon}",
            "key": api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        # Check status
        if data.get("status") == "OK":
            return True  # Street View is available
        else:
            return False  # No Street View coverage
        
            
    ############# Downnload GSV building images ################   
    def fetch_three_step_views(self):
        """
        Get in memory three directional Google Street View (GSV) images for a building's location.
    
        This method retrieves the latitude and longitude of a building, fetches three images from 
        Google Street View at angles of -30°, 0°, and +30°, and keep them in memory. The method updates 
        the UI with the image IDs for the current building. 
        If images already exist, it skips them.
    
        Args:
            None. The method relies on instance attributes such as `city_method`, `country_method`, 
            `data_building`, and UI elements for user input and display.
    
        Effects:
            - Fetches images from Google Street View using the provided API key.
            - Updates UI fields with image IDs.
    
        Notes:
            - Requires a valid Google Street View API key to fetch images.
            - Checks for Street View availability before attempting to fetch images.
            - Skips execution if no project folder is defined or if images already exist.
        """
           
        # Image ID displayed values
        # left image
        self.ui.img_id_value_1.setText(str(self.click_count+1)+"_1")
        # central image
        self.ui.img_id_value_2.setText(str(self.click_count+1)+"_2")
        # right image
        self.ui.img_id_value_3.setText(str(self.click_count+1)+"_3")
        
        if self.ui.insp_method == 0 or self.ui.insp_method == 1:  

            # Building coordinates
            location = (float(self.ui.lat_value.text()), float(self.ui.lon_value.text()))
            # API key is required; without it, access to GSV is not possible
            with open("methods/gsv_api_key.txt", "r") as f:
                api_key = f.read().strip() 
                                    
            # angles for taking the images
            angle = (-30,0,30)
            self.img_url = ["","",""]
            for aux in range (3):
                if self.check_street_view() == True:
                    # Get image from GSV
                    if aux == 0:
                        self.img_url[aux] , self.img_original_1, self.year_left = get_street_view_image(location, api_key, angle[aux])
                    elif aux == 1:
                        self.img_url[aux] , self.img_original_2, self.year_center = get_street_view_image(location, api_key, angle[aux])
                    else:
                        self.img_url[aux] , self.img_original_3, self.year_right = get_street_view_image(location, api_key, angle[aux])
                else:
                    print("Street View not available")
                    self.img_original_1, self.year_left = ["",""]
                    self.img_original_2, self.year_center = ["",""]
                    self.img_original_3, self.year_right = ["",""]
                    
            self.ui.year_value_1.setText(str(self.year_left))
            self.ui.year_value_2.setText(str(self.year_center))
            self.ui.year_value_3.setText(str(self.year_right))
        else:
            pass

        
        
    ############ Building detector model ################
    def object_detector_building(self):
        """
        Detect and isolate buildings from Google Street View (GSV) images using a YOLO-based object detector.
        
        This method processes three directional GSV images (-30°, 0°, +30°) for a building's location and 
        applies a YOLO-based object detection model to identify and isolate buildings. The detected building 
        with the highest confidence is cropped and displayed in the User Interface. If no building is detected, a message 
        is displayed instead. The method handles images both in memory and from the local device.
        
        Args:
            None. The method relies on instance attributes such as `ui` for image display elements, 
            `img_orginial_1`, `img_orginial_2`, `img_orginial_3` for input images, and `data_building` 
            for project-specific data.
        
        Effects:
            - Loads a YOLO model for building detection.
            - Fetches and processes GSV images.
            - Identifies the most confident bounding box for buildings.
            - Crops and displays detected buildings in the UI.
            - Handles scenarios where no buildings are detected or Street View is unavailable.
        
        Notes:
            - Requires a trained YOLO model and corresponding weight file (`building_detector.pt`).
            - Checks if Street View coverage is available before performing detection.
        """

        # Class mapping (update this with your actual mappings)
        class_map = {0: "building-xzyh"}  # Replace with the correct mapping
        weight_path = "methods/building_detector.pt" # Replace with your YOLO .pt file
        # Load the YOLO model
        model = YOLO(weight_path)
        # Set device GPU or CPU
        device= "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        # List of the frame
        self.ui.left_gsv_img.clear()
        self.ui.central_gsv_img.clear()
        self.ui.right_gsv_img.clear()
        img_frames = [self.ui.left_gsv_img,self.ui.central_gsv_img,self.ui.right_gsv_img]
        sw = True
        #Check inspection mode
        if self.ui.insp_method == 0 or self.ui.insp_method == 1:
            try:
                # Getting the images from GSV
                org_img = [self.img_original_1,self.img_original_2,self.img_original_3]
                # Vector for check if the building is detected
                self.predicted_img = [0,0,0]
                # Check GSV availability
                
                for aux in range (3):
                    if sw == True:
                        for i in range (100):
                            time.sleep(0.0001)
                            self.ui.progress_bar_method.setValue(i)
                            self.ui.method_progress.setText("Isolating building ....")
                        sw = False
            
                    # Ensure the image is in RGB format
                    image_rgb = org_img[aux]
            
                    # Run inference
                    results = model.predict(image_rgb)
            
                    highest_conf = 0
                    highest_conf_box = None
            
                    # Process the results
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
                        confs = result.boxes.conf.cpu().numpy()  # Confidence scores
                        classes = result.boxes.cls.cpu().numpy()  # Class IDs
            
                        for box, conf, cls in zip(boxes, confs, classes):
                            cls = int(cls)
                            # Getting the building image with higher confidence as selected bounding box
                            if class_map.get(cls) == "building-xzyh" and conf > highest_conf:
                                highest_conf = conf
                                highest_conf_box = box
                    
                    # Create image for cropped and displayed
                    display_image = org_img[aux].copy()
                    
                    # Extracting selecting bounding box coordinates witin the image
                    if highest_conf_box is not None:
                        x1, y1, x2, y2 = map(int, highest_conf_box)
                        
                        # Crop the area within the selected bounding box
                        # self.cropped_image[aux] = org_img[aux][y1:y2, x1:x2]
                        # self.org_img_bp = org_img[aux]
                        self.cropped_image[aux] = org_img[1][y1:y2, x1:x2]
                        self.org_img_bp = org_img[1]
                        # Draw a dashed rectangle for the highest confidence box
                        for i in range(x1, x2, 14):
                            cv2.line(display_image, (i, y1), (min(i + 5, x2), y1), (0, 0, 255), 3)
                            cv2.line(display_image, (i, y2), (min(i + 5, x2), y2), (0, 0, 255), 3)
                        for i in range(y1, y2, 14):
                            cv2.line(display_image, (x1, i), (x1, min(i + 5, y2)), (0, 0, 255), 3)
                            cv2.line(display_image, (x2, i), (x2, min(i + 5, y2)), (0, 0, 255), 3)
                            
                        if display_image is not None:
                            # Convert BGR image (OpenCV) to RGB format
                            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                            
                            # Convert the RGB image to QImage
                            height, width, channel = display_image_rgb.shape
                            bytes_per_line = 3 * width
                            qimage = QtGui.QImage(display_image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                            
                            # Convert QImage to QPixmap
                            building_pixmap = QtGui.QPixmap.fromImage(qimage)
                            self.predicted_img[aux] = 1
                            # Set the pixmap to QLabel
                            img_frames[aux].setPixmap(
                                building_pixmap.scaled(
                                    img_frames[aux].width(),
                                    img_frames[aux].height(),
                                    QtCore.Qt.IgnoreAspectRatio,  # Adjust scaling mode as needed
                                    QtCore.Qt.SmoothTransformation))  # Ensure high-quality scaling
                    else: 
                        # No building dectection 
                        self.no_image = "No Building detected"
                        # Skipping prection for this image
                        self.predicted_img[aux] = 0
                        font = QtGui.QFont()
                        font.setPointSize(int(16 * self.sf_x ))
                        font.setBold(True)
                        font.setWeight(75)
                        img_frames[aux].setFont(font)
                        img_frames[aux].setText(self.no_image)
                        img_frames[aux].setAlignment(QtCore.Qt.AlignCenter)  # Center-align text
            
            # There is not GSV image coverage
            except:
                self.no_image = "Street View not available" 
                for aux in range (3):
                    font = QtGui.QFont()
                    font.setPointSize(int(16 * self.sf_x))
                    font.setBold(True)
                    font.setWeight(75)
                    img_frames[aux].setFont(font)
                    img_frames[aux].setText(self.no_image)
                    img_frames[aux].setAlignment(QtCore.Qt.AlignCenter)  # Center-align text
        
        # Checking Inspection method (manual option)
        elif self.ui.insp_method == 2:
            
            # Image frames
            img_frames = [self.ui.left_gsv_img, self.ui.central_gsv_img, self.ui.right_gsv_img]
            # Loop for the number of image displayed selected with the option in the coordinates pop-up
            for aux in range (self.n_images_local):
                # Load the image for drawing
                try:
                    img_path = self.ui.folder_path+"/"+str(self.data_building.iloc[self.old_local + aux, 0])
                except:
                    QMessageBox.warning(self.ui, "Input Error", "No further inspections are available")
            
                aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                +str(self.data_building.iloc[self.old_local + aux, 0]))
                cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                
                aux_displayed_path = (self.ui.folder_path+"/displayed_images/"
                                +str(self.data_building.iloc[self.old_local + aux, 0]))
                displayed_path = os.path.splitext(aux_displayed_path)[0]+"_displayed.jpg"

                # Display building image
                if os.path.exists(displayed_path):
                    # Display an already isolated image
                    building_pixmap = QtGui.QPixmap(displayed_path)
                    # Selection of image frame using "aux" variable
                    img_frames[aux].setPixmap(
                        building_pixmap.scaled(
                            img_frames[aux].width(),
                            img_frames[aux].height(),
                            QtCore.Qt.IgnoreAspectRatio,  # Adjust scaling mode as needed
                            QtCore.Qt.SmoothTransformation))  # Ensure high-quality scaling
                else:
                    # Display a new image
                    if sw == True:
                        # Star progress bar until 99%
                        for i in range (100):
                            time.sleep(0.0001)
                            self.ui.progress_bar_method.setValue(i)
                            self.ui.method_progress.setText("Isolating building ....")
                        sw = False  
                    # Check and/or create cropped folder
                    if not os.path.exists(self.ui.folder_path+"/Cropped_images"):
                        os.makedirs(self.ui.folder_path+"/Cropped_images")
                    
                    self.gap = None
                    try:
                        # Image results
                        results = model(img_path)
                        image_rgb = cv2.imread(img_path)
                        # Adapting line weight depending of image size, in order to have an appropiate thickness
                        height, width, channels = image_rgb.shape
                        area = height*width
                        ratio = int(area*3/307200)
                        # Lines ratio
                        if area <= 600000:
                            # Small images
                            self.gap = int(area*14/307200)
                        elif area < 1000000:
                            # Medium images
                            self.gap = int(area*14/307200 * 3/4)
                        else:
                            # Large images
                            self.gap = int(area*14/307200 * 3/8)
                            ratio = int(area*3/307200 * 5/8)
                        highest_conf = 0
                        highest_conf_box = None
                        
                        for result in results:
                            boxes = result.boxes.xyxy  # Bounding box coordinates
                            confs = result.boxes.conf  # Confidence scores
                            classes = result.boxes.cls  # Class IDs
                        
                            for box, conf, cls in zip(boxes, confs, classes):
                                cls = int(cls)  # Ensure the class ID is an integer
                                # Getting the building image with higher confidence as selected bounding box
                                if class_map[cls] == "building-xzyh" and conf > highest_conf:
                                    highest_conf = conf
                                    highest_conf_box = box
                        
                        if highest_conf_box is not None:
                            x1, y1, x2, y2 = map(int, highest_conf_box)
                            
                            # Crop the area within the bounding box
                            cropped_image = image_rgb[y1:y2, x1:x2]
                            # Save image in local device
                            cv2.imwrite(cropped_path, cropped_image)
                        
                            # Draw a dashed red rectangle for the highest confidence box
                            if self.gap == 0:
                                self.gap = 1
                            for i in range(x1, x2, self.gap):
                                cv2.line(image_rgb, (i, y1), (min(i + 5, x2), y1), (0, 0, 255), max(1, int(ratio)))  # Top edge
                                cv2.line(image_rgb, (i, y2), (min(i + 5, x2), y2), (0, 0, 255), max(1, int(ratio)))  # Bottom edge
                            for i in range(y1, y2, self.gap):
                                cv2.line(image_rgb, (x1, i), (x1, min(i + 5, y2)), (0, 0, 255), max(1, int(ratio)))  # Left edge
                                cv2.line(image_rgb, (x2, i), (x2, min(i + 5, y2)), (0, 0, 255), max(1, int(ratio)))  # Right edge
                                
                            # Check and/or create diplayed folder              
                            if not os.path.exists(self.ui.folder_path+"/displayed_images"):
                                os.makedirs(self.ui.folder_path+"/displayed_images")
                            cv2.imwrite(displayed_path, image_rgb)
                            
                            if image_rgb is not None:
                                # Convert BGR image (OpenCV) to RGB format
                                display_image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
                                # display_image_rgb = image_rgb.copy()
                                # Convert the RGB image to QImage
                                height, width, channel = display_image_rgb.shape
                                bytes_per_line = 3 * width
                                qimage = QtGui.QImage(display_image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                                
                                # Convert QImage to QPixmap
                                building_pixmap = QtGui.QPixmap.fromImage(qimage)

                    except FileNotFoundError:
                        self.no_image = f"""
                                        <b><u>No image found</u></b><br><br>
                                        Please check that the image file exists at the specified path:<br>
                                        <code>{img_path}</code>
                                        """
                        font = QtGui.QFont()
                        font.setPointSize(int(12 * self.sf_x))
                        font.setBold(True)
                        font.setWeight(75)
                    
                        img_frames[aux].setFont(font)
                        img_frames[aux].setTextFormat(QtCore.Qt.RichText)  # Enable rich text (HTML)
                        img_frames[aux].setText(self.no_image)
                        img_frames[aux].setAlignment(QtCore.Qt.AlignCenter)
                        img_frames[aux].setWordWrap(True)
                        
                    try:
                        # Displayed image in corresponding frames
                        img_frames[aux].setPixmap(
                            building_pixmap.scaled(
                                img_frames[aux].width(),
                                img_frames[aux].height(),
                                QtCore.Qt.IgnoreAspectRatio,  # Adjust scaling mode as needed
                                QtCore.Qt.SmoothTransformation))  # Ensure high-quality scaling
                    except:
                        if self.gap == None:
                            pass
                        else:
                            # Displayed image in corresponding frames
                            self.no_image = """
                            <b><u>NO BUILDING DETECTED</u></b><br><br>
                            There is no building detected by the tool. However, if you believe there is a building in the image,<br>
                            <b><u>PLEASE CLICK THE "MANUAL BOX" BUTTON</u></b> and manually select the building.<br>
                            <b><u>The building detector has a precision of 93%</u></b>; therefore, you may ignore this message <br>
                            and simply click <b><u>Next Building</u></b> to continue classifying.
                            """
                            
                            font = QtGui.QFont()
                            font.setPointSize(int(12 * self.sf_x))    
                            font.setBold(True)
                            font.setWeight(75)
                            
                            img_frames[aux].setFont(font)
                            img_frames[aux].setTextFormat(QtCore.Qt.RichText)  # Enable rich text (HTML)
                            img_frames[aux].setText(self.no_image)
                            img_frames[aux].setAlignment(QtCore.Qt.AlignCenter)
                            img_frames[aux].setWordWrap(True)  # Wrap long text
                        
                        
                        
        # Chance progress bar to complete
        self.ui.progress_bar_method.setValue(100)
        self.ui.method_progress.setText("Done!")
    
       
    ############ Left Bounding Box Manual Selection ################
    def bounding_box_frame_left(self):
        """
        Select the left frame image for building detection and retrieve its corresponding path.
    
        This method determines the appropriate source image for the left frame based on the selected 
        inspection mode. If the inspection mode is not manual (`insp_method != 2`), the image is 
        retrieved from Google Street View (GSV). Otherwise, the image is loaded from the local device.
        """
        try:
            # Getting the image depending of the inspection mode selected.
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                # From GSV (Polygon and Specific method)
                self.image_bb = self.img_original_1
            elif self.ui.insp_method == 2:
                # From local device 
                self.image_bb = self.ui.folder_path+"/"+str(self.data_building.iloc[self.old_local, 0])
            # Left Frame to display
            self.frame_bb_disp = self.ui.left_gsv_img
            
            # Getting the path for image prediction
            if self.ui.insp_method == 2:
                try:   
                    aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local, 0]))
                    self.cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg" 
                except:
                    QMessageBox.warning(self.ui, "File Error", "This option is only available if there is a previous building detection.")
            else:
                self.cropped_path = None
        
            # Bounding box ID for prediction models
            self.box_id = 0
        except:
            pass
    
    ############ Left Bounding Box Manual Selection ################       
    def bounding_box_frame_central(self):
        """
        Select the central frame image for building detection and retrieve its corresponding path.
    
        This method determines the appropriate source image for the left frame based on the selected 
        inspection mode. If the inspection mode is not manual (`insp_method != 2`), the image is 
        retrieved from Google Street View (GSV). Otherwise, the image is loaded from the local device.
        """
        try:
            # Getting the image depending of the inspection mode selected.
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                # From GSV (Polygon and Specific method)
                self.image_bb = self.img_original_2
            elif self.ui.insp_method == 2:
                # From local device 
                self.image_bb = self.ui.folder_path+"/"+str(self.data_building.iloc[self.old_local + 1, 0])
            # Central Frame to display 
            self.frame_bb_disp = self.ui.central_gsv_img
            
            # Getting the path for image prediction
            if self.ui.insp_method == 2:
                try:
                    # Cropped image path
                    aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + 1, 0]))
                    self.cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                except:
                    QMessageBox.warning(self.ui, "File Error", "This option is only available if there is a previous building detection.")
            else:
                self.cropped_path = None
        
            # Bounding box ID for prediction models
            self.box_id = 1
        except:
            pass
            
    ############ Right Bounding Box Manual Selection ################
    def bounding_box_frame_right(self):
        """
        Select the right frame image for building detection and retrieve its corresponding path.
    
        This method determines the appropriate source image for the left frame based on the selected 
        inspection mode. If the inspection mode is not manual (`insp_method != 2`), the image is 
        retrieved from Google Street View (GSV). Otherwise, the image is loaded from the local device.
        """
        try:
            # Getting the image depending of the inspection mode selected.
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                # From GSV (Polygon and Specific method)
                self.image_bb = self.img_original_3
            elif self.ui.insp_method == 2:
                # From local device 
                self.image_bb = self.ui.folder_path+"/"+str(self.data_building.iloc[self.old_local + 2, 0])
            
            self.frame_bb_disp = self.ui.right_gsv_img
            # Getting the path for image prediction
            if self.ui.insp_method == 2:
                try:
                    # Cropped image path
                    aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + 2, 0]))
                    self.cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                except:
                    QMessageBox.warning(self.ui, "File Error", "This option is only available if there is a previous building detection.")
            else:
                self.cropped_path = None
            # Bounding box ID for prediction models
            self.box_id = 2
        except:
            pass  
    ############ Folder Selection ################
    def bounding_box(self):
        """
        Open a bounding box selection pop-up for manual annotation.
    
        This method allows the user to manually define a bounding box around a building 
        in an image by selecting four points. The bounding box is then displayed in the UI 
        frame. Before execution, the method verifies that the project folder, country, and 
        city name are correctly set.
    
        Notes:
            - If the AI-powered option is enabled, users must manually label the building 
              or verify that existing labels are correct.
            - Ensures a valid PyQt5 `QApplication` instance exists before opening the pop-up.
        """
        # Conditional to avoid executing the method if there is no project folder
        if self.ui.output_folder_value == "-":
            QMessageBox.warning(self.ui, "File Error", "This option is only available once the building image is displayed.")
        # Conditional to avoid executing the method if there is no country name
        elif self.ui.country_value.text() == "-":
            QMessageBox.warning(self.ui, "File Error", "This option is only available once the building image is displayed.")
        # Conditional to avoid executing the method if there is no city name 
        elif self.ui.city_value.text() == "-":
            QMessageBox.warning(self.ui, "File Error", "This option is only available once the building image is displayed.")
        else:
            
            """Open the bounding box selection pop-up window."""
            app = QApplication.instance()  # Ensure PyQt instance exists
            if app is None:
                app = QApplication([])
                
            # Called function where the user creates a manual bounding box by clicking four points, which is then displayed in the UI frame.
            dialog = BoundingBoxWindow(self.image_bb, self.frame_bb_disp, self.ui.insp_method, self.cropped_path, 
                                       parent=self.ui, main_window=self.ui, gui_methods=self)
    
            dialog.exec_()  # Open the pop-up
        
            try:
                # Image path
                image_file = dialog.prediction_img
                
                if self.ui.ai_check.isChecked():
                    
                    PENDING = 0
                    
                    # Comboboxes for each image label
                    material_id = [self.ui.material_cb_1,self.ui.material_cb_1,self.ui.material_cb_1]
                    material_index = predict_material_img(image_file, self.ui.insp_method, self.box_id, self)
                    # LLRS building image sets prediction
                    class_names_mat = ['Concrete', 'Hybrid - Confined and Unreinforced masonry', 'Informal materials', 
                                   'Masonry - Confined', 'Masonry - Reinforced', 'Masonry - Unreinforced', 
                                   'Steel','Wood']                       
                    material_id[self.box_id].setCurrentText(class_names_mat[material_index])
                    
                    # Comboboxes for each image label
                    llrs_id = [self.ui.llrs_cb_1,self.ui.llrs_cb_1,self.ui.llrs_cb_1]
                    # LLRS building image prediction
                    llrs_index = predict_llrs_img(image_file, self.ui.insp_method, self.box_id, self)
                    class_names_llrs = ['Dual System', 'Braced Frames', 'Infilled Frames', 'Moment Frames', 
                                   'No lateral load-resisting system', 'Walls', 'Walls']             
                    llrs_id[self.box_id].setCurrentText(class_names_llrs[llrs_index])
                         
                    llrs_pred = self.ui.llrs_cb_1.currentData()
                    if self.pred_mat_value == "MCF":
                        llrs_id[self.box_id].setCurrentIndex(4) 
                    elif self.pred_mat_value == "MUR":
                        llrs_id[self.box_id].setCurrentIndex(4) 
                    elif self.pred_mat_value == "MR":
                        llrs_id[self.box_id].setCurrentIndex(4)
                    elif self.pred_mat_value == "INF":
                        llrs_id[self.box_id].setCurrentIndex(6)
                    elif self.pred_mat_value == "CR":
                        if llrs_pred in ("LDUAL", "LFM", "LFINF"):
                            pass
                        else:
                            llrs_id[self.box_id].setCurrentIndex(3)
                    elif self.pred_mat_value == "S":
                        if llrs_pred in ("LFM", "LFBR"):
                            pass
                        else:
                            llrs_id[self.box_id].setCurrentIndex(3)
                    
                    # Comboboxes for each image label
                    code_level_id = [self.ui.age_cb_1,self.ui.age_cb_1,self.ui.age_cb_1]
                    # LLRS building image prediction
                    code_level_index = predict_code_img(image_file, self.ui.insp_method, self.box_id, self)
                    # LLRS building image sets prediction
                    class_names_code = ['High-Code','Low-Code', 'Moderate-code', 'No-Code']
                    code_level_id[self.box_id].setCurrentText(class_names_code[code_level_index])
                    
                    # Comboboxes for each image label
                    n_stories_id = [self.ui.n_stories_value_1,self.ui.n_stories_value_1,self.ui.n_stories_value_1]
                    # LLRS building image prediction
                    n_stories_index = predict_n_stories_img(image_file, self.ui.insp_method, self.box_id, self)
                    # LLRS building image sets prediction
                    class_names = ['10-12', '13+', '1', '2', '3', '4', '5', '6-7', '8-9']
                    n_stories_id[self.box_id].setCurrentText(class_names[n_stories_index])
                    
                    # Comboboxes for each image label
                    occupancy_id = [self.ui.occup_cb_1,self.ui.occup_cb_1,self.ui.occup_cb_1]
                    # LLRS building image prediction
                    occupancy_index = predict_occupancy_img(image_file, self.ui.insp_method, self.box_id, self)
                    # LLRS building image sets prediction
                    occupancy_class = [ 'Commercial' , 'Industrial' ,'Mixed (Residential + Commercial)', 'Residential']
                    occupancy_id[self.box_id].setCurrentText(occupancy_class[occupancy_index])  
                    
                    # Comboboxes for each image label
                    block_position_id = [self.ui.bck_pos_cb_1,self.ui.bck_pos_cb_1,self.ui.bck_pos_cb_1]    
                    # block_position building image prediction
                    block_position_index = predict_block_position_img(image_file, self.ui.insp_method, self.box_id, self)
                    # block_position building image sets prediction
                    block_position_id[self.box_id].setCurrentIndex(block_position_index+1)
                    
                    # Comboboxes for each image label
                    roof_shape_id = [self.ui.roof_shape_cb_1,self.ui.roof_shape_cb_1,self.ui.roof_shape_cb_1]   
                    # roof_shape building image prediction
                    roof_shape_index = predict_roof_shape_img(image_file, self.ui.insp_method, self.box_id, self)
                    # roof_shape building image sets prediction
                    roof_shape_id[self.box_id].setCurrentIndex(roof_shape_index+1)
                    
                    # Comboboxes for each image label
                    roof_material_id = [self.ui.roof_material_cb_1,self.ui.roof_material_cb_1,self.ui.roof_material_cb_1]   
                    # roof_material building image prediction
                    roof_material_index = predict_roof_material_img(image_file, self.ui.insp_method, self.box_id, self)
                    # roof_material building image sets prediction
                    roof_material_id[self.box_id].setCurrentIndex(roof_material_index+1)
                    
                    self.box_id = None
                # Message with special format
                message = """
                If you are making predictions using the AI-powered option, when a manual bounding box is created, 
                <b><u>NEW PREDICTION EXECUTION OCCURS</u></b>. Therefore, you should verify that the current labels 
                correspond to the true labels.
                """
                
                # Create a QMessageBox instance
                msg_box = QMessageBox(self.ui)
                msg_box.setTextFormat(QtCore.Qt.RichText)
                msg_box.setText(message)
                msg_box.setWindowTitle("AI-powered inspection form")
                
                # Show the message box
                msg_box.exec_()
            
            except:
                pass
            
    def tax_check(self, tax_value):
        # 1) Create a small DataFrame with taxonomy strings
        df = pd.DataFrame({"TAXONOMY": [tax_value]})
        
        try:
            tax = check_taxonomy(df, taxo_col="TAXONOMY")
        except ValueError as e:
            print("There are invalid taxonomies ❌")
            # Convert the error to string
            err_str = str(e)
            
            # Extract the canonical value from the string using regex
            match = re.search(r"'canonical': '([^']+)'", err_str)
            if match:
                tax_canonical = match.group(1)
                print("Canonical taxonomy:", tax_canonical)
            else:
                tax_canonical = None
                print("No canonical value found.")
            

    ############ Obtain value of the form of each building image ################       
    def inspection_database (self):
        """
        Populate the inspection database with data extracted from building images and user inputs.
    
        This method collects information for three building images (left, central, right) and appends 
        the data to the inspection database. Data includes geographic coordinates, building attributes, 
        and user-provided metadata from the UI fields. Each building's data is treated as a separate 
        entry in the database.
    
        Args:
            None. The method operates on instance attributes such as `click_count`, `data_building`, 
            and various UI elements for user inputs and data display.
    
        Returns:
            None. The collected data is appended to `self.ui.database`.
    
        Effects:
            - Updates `self.ui.database` with building attributes and metadata for the current 
              set of images.
            - Gathers inputs such as material type, lateral load-resisting system (LLRS), code level, 
              number of stories, occupancy, block position, and image quality.
    
        Notes:
            - The database is populated only if `click_count` is greater than zero.
            - Assumes the `data_building` DataFrame contains valid latitude and longitude values.
            - Each entry in the database corresponds to a specific building image.
            - Requires properly configured UI components to retrieve and store data.
        """
        if self.ui.city_value.text() == "-":
            QMessageBox.warning(self.ui,"File Error", "This option is only available once the building image is displayed.\n"
                                                      "Please click the *Next Building* button.")
        else:
            # Save inspection function
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                base_url = "https://www.google.com/maps/@?api=1&map_action=pano&viewpoint="
                coord = str(self.ui.lat_value.text()) + "," + str(self.ui.lon_value.text())
                heading = get_road_orientation((float(self.ui.lat_value.text()), float(self.ui.lon_value.text())))
                
            # -------------------  Left building image ---------------------- 
            
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                self.data_ai.iloc[self.click_count * 3 , 0] = self.ui.img_id_value_1.text()[:-2]                 # ID
                self.data_ai.iloc[self.click_count * 3 , 1] = self.data_building.loc[self.click_count, 'latitude']    # latitude
                self.data_ai.iloc[self.click_count * 3 , 2] = self.data_building.loc[self.click_count, 'longitude']    # longitude
                self.data_ai.iloc[self.click_count * 3 , 3] = self.ui.country_value.text()                   # Country
                self.data_ai.iloc[self.click_count * 3 , 4] = self.ui.city_value.text()                      # City
                self.data_ai.iloc[self.click_count * 3 , 5] = self.ui.material_cb_1.currentData()            # LLRS Material
                self.data_ai.iloc[self.click_count * 3 , 6] = self.ui.llrs_cb_1.currentData()                # LLRS 
                self.data_ai.iloc[self.click_count * 3 , 7] = self.ui.age_cb_1.currentData()                 # Code Level 
                self.data_ai.iloc[self.click_count * 3 , 8] = self.ui.n_stories_value_1.currentData()        # Number of Stories 
                self.data_ai.iloc[self.click_count * 3 , 9] = self.ui.occup_cb_1.currentData()               # Occupancy
                self.data_ai.iloc[self.click_count * 3 , 10] = self.ui.bck_pos_cb_1.currentData()            # Block Position
                self.data_ai.iloc[self.click_count * 3 , 11] = self.ui.epc_const_cb_1.currentText()          # Epoch of construction
                self.data_ai.iloc[self.click_count * 3 , 12] = self.ui.roof_shape_cb_1.currentData()         # Roof shape
                self.data_ai.iloc[self.click_count * 3 , 13] = self.ui.roof_material_cb_1.currentData()      # Roof material
                self.data_ai.iloc[self.click_count * 3 , 14] = self.ui.img_q_cb_1.currentData()              # Image Quality
                
                try:
                    self.data_ai.iloc[self.click_count * 3 , 15] = (self.ui.material_cb_1.currentData()+"/"+
                                                                    self.ui.llrs_cb_1.currentData()+"/"+
                                                                    self.ui.age_cb_1.currentData()+"/H:"+
                                                                    self.ui.n_stories_value_1.currentText()+"/"+
                                                                    self.ui.bck_pos_cb_1.currentData()+"/"+
                                                                    self.ui.roof_shape_cb_1.currentData()+"+"+
                                                                    self.ui.roof_material_cb_1.currentData()+"/"+
                                                                    self.ui.occup_cb_1.currentData())
                                                                    
                    # Taxonomy
                    self.tax_check(self.data_ai.iloc[self.click_count * 3 , 15])

                except:
                    pass
                
                if self.img_url[0]  != "":
                    self.data_ai.iloc[self.click_count * 3 , 16] = self.img_url[0]                           # Image URL
                else:
                    if isinstance(heading, int):
                        self.data_ai.iloc[self.click_count * 3 , 16] = base_url + coord +"&heading="+str((heading+ 180) % 360)+"&pitch=5&fov=120"
            
            # ------------------- Local  -----------------------
            elif self.ui.insp_method == 2:
                # Left building image
                self.data_ai.iloc[self.old_local , 0] = self.ui.img_id_value_1.text()[:-2]                  # ID
                self.data_ai.iloc[self.old_local , 1] = self.data_building.loc[self.old_local,'latitude']      # latitude
                self.data_ai.iloc[self.old_local , 2] = self.data_building.loc[self.old_local,'longitude']      # longitude
                self.data_ai.iloc[self.old_local , 3] = self.ui.country_value.text()                   # Country
                self.data_ai.iloc[self.old_local , 4] = self.ui.city_value.text()                      # City
                self.data_ai.iloc[self.old_local , 5] = self.ui.material_cb_1.currentData()            # LLRS Material
                self.data_ai.iloc[self.old_local , 6] = self.ui.llrs_cb_1.currentData()                # LLRS 
                self.data_ai.iloc[self.old_local , 7] = self.ui.age_cb_1.currentData()                 # Code Level 
                self.data_ai.iloc[self.old_local , 8] = self.ui.n_stories_value_1.currentData()        # Number of Stories 
                self.data_ai.iloc[self.old_local , 9] = self.ui.occup_cb_1.currentData()               # Occupancy
                self.data_ai.iloc[self.old_local , 10] = self.ui.bck_pos_cb_1.currentData()            # Block Position
                self.data_ai.iloc[self.old_local , 11] = self.ui.epc_const_cb_1.currentText()          # Epoch of construction
                self.data_ai.iloc[self.old_local , 12] = self.ui.roof_shape_cb_1.currentData()         # Roof shape
                self.data_ai.iloc[self.old_local , 13] = self.ui.roof_material_cb_1.currentData()      # Roof material
                self.data_ai.iloc[self.old_local , 14] = self.ui.img_q_cb_1.currentData()              # Image Quality
                
                try:
                    self.data_ai.iloc[self.old_local , 15] = (self.ui.material_cb_1.currentData()+"/"+
                                                                    self.ui.llrs_cb_1.currentData()+"+"+
                                                                    self.ui.age_cb_1.currentData()+"/H:"+
                                                                    self.ui.n_stories_value_1.currentText()+"/"+
                                                                    self.ui.occup_cb_1.currentData()+"/"+
                                                                    self.ui.bck_pos_cb_1.currentData())            # Taxonomy
                except:
                    pass
                
                self.data_ai.iloc[self.old_local , 16] = self.data_building.iloc[self.old_local , 0] 
            
        
            
    ############ Saves the data from the inspections that were conducted ################       
    def save_database (self):
        """
        Save the inspection data to a CSV file.
    
        This method consolidates new inspection data with any previously saved data and exports 
        the combined dataset to a CSV file. If no prior data exists, it creates a new CSV file 
        containing only the current inspections. The CSV file is named using the city and country 
        information and stored in the specified output folder.
    
        Args:
            None. The method operates on the `self.ui.database` attribute and the UI-provided output folder path.
    
        Returns:
            None. The inspection data is saved or updated in the CSV file.
    
        Effects:
            - Reads previous inspection data from an existing CSV file (if available).
            - Appends the new inspection data to the existing dataset.
            - Exports the combined dataset to a CSV file in the specified output folder.
    
        Notes:
            - The CSV file is named with the pattern `<city>_<country>_inspection.csv`.
            - Handles exceptions gracefully when no previous CSV file exists.
            - Calls `self.inspection_database()` to gather new inspection data before saving.
        """
        # Load path 
        output_folder = self.ui.output_folder_value

        # Create CSV with new inspections
        self.inspection_database()
        
        # Star progress bar
        self.ui.method_progress.setText("Saving inspections ...")
        for j in range (101):
            time.sleep(0.0001)
            self.ui.progress_bar_method.setValue(j)
        # Save inspections
        ############################## Polygon #######################################
        if self.ui.insp_method == 0:
            try:
                # Save the AI inspection data to a CSV file
                img_prefix = f"{output_folder}/{self.ui.file_name}"
                self.data_ai.to_csv(img_prefix + "_AI_aux_cont.csv", index=False)
                final_df = self.data_ai
                filtered_df = final_df[final_df['n_stories'].notna() | final_df['llrs'].notna()]
                filtered_df.to_csv(img_prefix + "_AI_classification.csv", index=False)
                # Update the progress message in the GUI
                self.ui.method_progress.setText("Inspections exported successfully!")
            except:
                # Show a warning message box if there's a permission error
                QMessageBox.warning(self.ui, "File Error", "The file is open or the folder is inaccessible."
                                    +" Please close the file or check folder permissions.")
        ############################## Specific #######################################
        elif self.ui.insp_method == 1:
            try:
                # Save the AI inspection data to a CSV file
                self.data_ai.to_csv(self.ui.output_folder_value+"/"+self.ui.file_name+"_AI_aux_cont.csv", index=False)
                final_df = self.data_ai
                filtered_df = final_df[final_df['n_stories'].notna() | final_df['llrs'].notna()]
                filtered_df.to_csv(self.ui.output_folder_value+"/"+self.ui.file_name+ "_AI_classification.csv", index=False)
                # Update the progress message in the GUI
                self.ui.method_progress.setText("Inspections exported successfully!")
            except:
                # Show a warning message box if there's a permission error
                QMessageBox.warning(self.ui, "File Error", "The file is open or the folder is inaccessible."
                                    +"Please close the file or check folder permissions.")
        ############################## Local #######################################
        elif self.ui.insp_method == 2:
            # Save the AI inspection data to a CSV file
            try:
                self.data_ai.to_csv(self.ui.output_folder_value+"/"+self.ui.file_name_local.text()+"_AI_aux_cont.csv", index=False)
                final_df = self.data_ai
                filtered_df = final_df[final_df['n_stories'].notna() | final_df['llrs'].notna()]
                filtered_def = filtered_df.drop_duplicates(subset='id', keep='first') 
                filtered_def = filtered_def.drop_duplicates(subset='image filename or link', keep='first') 
                filtered_def.to_csv(self.ui.output_folder_value+"/"+self.ui.file_name_local.text()+ "_AI_classification.csv", index=False)
                # Update the progress message in the GUI
                self.ui.method_progress.setText("Inspections exported successfully!")
            except:
                # Show a warning message box if there's a permission error
                QMessageBox.warning(self.ui, "File Error", "The file is open or the folder is inaccessible. "
                                    +"Please close the file or check folder permissions.")
         ############################## Local #######################################
        elif self.ui.insp_method == 3:
            pass
        
        self.data_old = "OK" # TO BE SAVED THERE IS EXISTING DATA
        self.sw_insp = True
        self.save_id = True
        self.start = True
            
            
    def setComboBoxByData(self, comboBox, data):
        """
        Set the index of a QComboBox based on its associated data value.
    
        This method iterates through the items in a `QComboBox` and selects the index 
        corresponding to the provided `data` value. If a match is found, the combo box 
        is updated to that index. If no match is found, the default return value is an 
        empty string.
    
        Args:
            comboBox (QComboBox): The combo box to update.
            data (Any): The data value to search for within the combo box items.
    
        Returns:
            str: Returns an empty string if no match is found; otherwise, returns the 
                 result of `setCurrentIndex(i)`, though `setCurrentIndex` does not 
                 explicitly return a value.
    
        Effects:
            - Updates the `comboBox` selection if a matching data value is found.
    
        Notes:
            - If no match is found, the combo box remains unchanged.
            - The method assumes `comboBox.itemData(i)` correctly retrieves stored data values.
        """
        # Iterate through the comboBox items to find the one with matching data
        match_data = [""]
        for i in range(comboBox.count()):
            if comboBox.itemData(i) == data:
                match_data = comboBox.setCurrentIndex(i)
                break
        return match_data
        
        
    ############ Restart default value of each building feature ################  
    def clean_database (self):
        """
        Reset the UI fields for building features to their default values.
    
        This method clears and resets all inputs related to the building features for the left, 
        central, and right building images. It sets dropdown menus to default selections, 
        numeric fields to zero, and other UI components to their initial states.
    
        Args:
            None. The method operates on UI elements for user inputs.
    
        Returns:
            None. The UI fields for building features are reset to their default values.
    
        Effects:
            - Resets dropdowns for material type, lateral load-resisting system (LLRS), code level, 
              occupancy type, block position, and image quality to default values.
            - Resets numeric fields for the number of stories to zero.
    
        Notes:
            - This method ensures that the UI is cleared and ready for new input after processing 
              or a reset action.
            - Requires properly configured UI elements to work as intended.
        """
        if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
            # ----------------------- LEFT -----------------------------
            # Material
            if self.data_ai.iloc[self.click_count * 3 , 5] is None:
                self.ui.material_cb_1.setCurrentText("Select Material")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 5]) == True:
                self.ui.material_cb_1.setCurrentText("Select Material")
            else:
                self.setComboBoxByData(self.ui.material_cb_1 , self.data_ai.iloc[self.click_count * 3 , 5])
    
            # LLRS
            if self.data_ai.iloc[self.click_count * 3 , 6] is None :
                self.ui.llrs_cb_1.setCurrentText("Select LLRS")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 6]) == True:
                self.ui.llrs_cb_1.setCurrentText("Select LLRS")
            else:
                self.setComboBoxByData(self.ui.llrs_cb_1 , self.data_ai.iloc[self.click_count * 3 , 6])
                
            # Code level
            if self.data_ai.iloc[self.click_count * 3 , 7] is None :
                self.ui.age_cb_1.setCurrentText("Select Code Level")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 7]) == True:
                self.ui.age_cb_1.setCurrentText("Select Code Level")
            else:
                self.setComboBoxByData(self.ui.age_cb_1 , self.data_ai.iloc[self.click_count * 3 , 7])
            
            # Number of stories
            if self.data_ai.iloc[self.click_count * 3 , 8] is None :
                self.ui.n_stories_value_1.setCurrentText("Select Number of Stories")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 8]) == True:
                self.ui.n_stories_value_1.setCurrentText("Select Number of Stories")
            else:
                self.setComboBoxByData(self.ui.n_stories_value_1, self.data_ai.iloc[self.click_count * 3 , 8])
                
            # Occupancy
            if self.data_ai.iloc[self.click_count * 3 , 9] is None :
                self.ui.occup_cb_1.setCurrentText("Select Occupancy Type")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 9]) == True:
                self.ui.occup_cb_1.setCurrentText("Select Occupancy Type")
            else:
                self.setComboBoxByData(self.ui.occup_cb_1 , self.data_ai.iloc[self.click_count * 3 , 9])
            
            # Block Position
            if self.data_ai.iloc[self.click_count * 3 , 10] is None :
                self.ui.bck_pos_cb_1.setCurrentText("Select Block Position")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 10]) == True:
                self.ui.bck_pos_cb_1.setCurrentText("Select Block Position")
            else:
                self.setComboBoxByData(self.ui.bck_pos_cb_1 , self.data_ai.iloc[self.click_count * 3 , 10])
                
            # Epoch of construction
            if self.data_ai.iloc[self.click_count * 3 , 11] is None :
                self.ui.epc_const_cb_1.setCurrentIndex(0)
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 11]) == True:
                self.ui.epc_const_cb_1.setCurrentIndex(0)
            else:
                self.setComboBoxByData(self.ui.epc_const_cb_1 , self.data_ai.iloc[self.click_count * 3 , 11])
                
            # Roof Shape
            if self.data_ai.iloc[self.click_count * 3 , 12] is None :
                self.ui.roof_shape_cb_1.setCurrentText("Select Roof Shape")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 12]) == True:
                self.ui.roof_shape_cb_1.setCurrentText("Select Roof Shape")
            else:
                self.setComboBoxByData(self.ui.roof_shape_cb_1 , self.data_ai.iloc[self.click_count * 3 , 12])
                
            # Roof Material
            if self.data_ai.iloc[self.click_count * 3 , 13] is None :
                self.ui.roof_material_cb_1.setCurrentText("Select Roof Material")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 13]) == True:
                self.ui.roof_material_cb_1.setCurrentText("Select Roof Material")
            else:
                self.setComboBoxByData(self.ui.roof_material_cb_1 , self.data_ai.iloc[self.click_count * 3 , 13])
    
            # Image quality
            if self.data_ai.iloc[self.click_count * 3 , 14] is None :
                self.ui.img_q_cb_1.setCurrentText("Select Image Quality")
            elif pd.isna(self.data_ai.iloc[self.click_count * 3 , 14]) == True:
                self.ui.img_q_cb_1.setCurrentText("Select Image Quality")
            else:
                self.setComboBoxByData(self.ui.img_q_cb_1 , self.data_ai.iloc[self.click_count * 3 , 14])
                
        
        elif self.ui.insp_method == 2: 
            #########################################################
            ##########============  Local images ===========#########
            #########################################################
            # Material
                if self.data_ai.iloc[self.old_local , 5] is None:
                    self.ui.material_cb_1.setCurrentText("Select Material")
                elif pd.isna(self.data_ai.iloc[self.old_local , 5]) == True:
                    self.ui.material_cb_1.setCurrentText("Select Material")
                else:
                    self.setComboBoxByData(self.ui.material_cb_1 , self.data_ai.iloc[self.old_local , 5])
        
                # LLRS
                if self.data_ai.iloc[self.old_local , 6] is None :
                    self.ui.llrs_cb_1.setCurrentText("Select LLRS")
                elif pd.isna(self.data_ai.iloc[self.old_local , 6]) == True:
                    self.ui.llrs_cb_1.setCurrentText("Select LLRS")
                else:
                    self.setComboBoxByData(self.ui.llrs_cb_1 , self.data_ai.iloc[self.old_local , 6])
                    
                # Code level
                if self.data_ai.iloc[self.old_local , 7] is None :
                    self.ui.age_cb_1.setCurrentText("Select Code Level")
                elif pd.isna(self.data_ai.iloc[self.old_local , 7]) == True:
                    self.ui.age_cb_1.setCurrentText("Select Code Level")
                else:
                    self.setComboBoxByData(self.ui.age_cb_1 , self.data_ai.iloc[self.old_local , 7])
                
                # Number of stories
                if self.data_ai.iloc[self.old_local , 8] is None :
                    self.ui.n_stories_value_1.setCurrentText("Select Number of Stories")
                elif pd.isna(self.data_ai.iloc[self.old_local , 8]) == True:
                    self.ui.n_stories_value_1.setCurrentText("Select Number of Stories")
                else:
                    self.setComboBoxByData(self.ui.n_stories_value_1, str(self.data_ai.iloc[self.old_local , 8]))
                    
                # Occupancy
                if self.data_ai.iloc[self.old_local , 9] is None :
                    self.ui.occup_cb_1.setCurrentText("Select Occupancy Type")
                elif pd.isna(self.data_ai.iloc[self.old_local , 9]) == True:
                    self.ui.occup_cb_1.setCurrentText("Select Occupancy Type")
                else:
                    self.setComboBoxByData(self.ui.occup_cb_1 , self.data_ai.iloc[self.old_local , 9])
                
                # Block Position
                if self.data_ai.iloc[self.old_local , 10] is None :
                    self.ui.bck_pos_cb_1.setCurrentText("Select Block Position")
                elif pd.isna(self.data_ai.iloc[self.old_local , 10]) == True:
                    self.ui.bck_pos_cb_1.setCurrentText("Select Block Position")
                else:
                    self.setComboBoxByData(self.ui.bck_pos_cb_1 , self.data_ai.iloc[self.old_local , 10])
                    
                # Epoch of construction
                if self.data_ai.iloc[self.old_local , 11] is None :
                    self.ui.epc_const_cb_1.setCurrentIndex(0)
                elif pd.isna(self.data_ai.iloc[self.old_local , 11]) == True:
                    self.ui.epc_const_cb_1.setCurrentIndex(0)
                else:
                    self.setComboBoxByData(self.ui.epc_const_cb_1 , str(self.data_ai.iloc[self.old_local , 11]))
                    
                # Roof Shape
                if self.data_ai.iloc[self.old_local , 12] is None :
                    self.ui.roof_shape_cb_1.setCurrentText("Select Roof Shape")
                elif pd.isna(self.data_ai.iloc[self.old_local , 12]) == True:
                    self.ui.roof_shape_cb_1.setCurrentText("Select Roof Shape")
                else:
                    self.setComboBoxByData(self.ui.roof_shape_cb_1 , self.data_ai.iloc[self.old_local , 12])
                    
                # Roof Material
                if self.data_ai.iloc[self.old_local , 13] is None :
                    self.ui.roof_material_cb_1.setCurrentText("Select Roof Material")
                elif pd.isna(self.data_ai.iloc[self.old_local , 13]) == True:
                    self.ui.roof_material_cb_1.setCurrentText("Select Roof Material")
                else:
                    self.setComboBoxByData(self.ui.roof_material_cb_1 , self.data_ai.iloc[self.old_local , 13])
        
                # Image quality
                if self.data_ai.iloc[self.old_local , 14] is None :
                    self.ui.img_q_cb_1.setCurrentText("Select Image Quality")
                elif pd.isna(self.data_ai.iloc[self.old_local , 14]) == True:
                    self.ui.img_q_cb_1.setCurrentText("Select Image Quality")
                else:
                    self.setComboBoxByData(self.ui.img_q_cb_1 , self.data_ai.iloc[self.old_local , 14])
                    
        
    ############ Deep learning model for predict the LLRS Material ################
    def material_prediction (self):
        """
        Predict the Material of the Lateral Load Resisting System (LLRS) of a building using a deep learning model.
    
        This method applies a deep learning model to predict the LLRS Material for three building images 
        (e.g., from Google Street View). If the AI-powered checkbox is activated in the UI, the 
        method processes each image, retrieves predictions, and updates the corresponding UI 
        comboboxes with the predicted LLRS Material values.
    
        Returns:
            None. The predicted LLRS values are set in the UI comboboxes.
    
        Effects:
            - Loads and applies a deep learning model to predict LLRS for the images.
            - Updates the comboboxes (`material_cb_1`, `material_cb_2`, `material_cb_3`) in the UI with predictions.
            - Updates the progress bar and status message in the UI.
    
        Notes:
            - The progress bar provides visual feedback during model loading and prediction.
            - Handles exceptions silently if predictions or UI updates fail.
            - Requires the AI-powered checkbox (`ai_check`) to be selected for predictions to proceed.
            - Assumes a predefined function `predict_material_img` for making predictions.
        """
        # Comboboxes for each image label
        material_id = [self.ui.material_cb_1,self.ui.material_cb_1,self.ui.material_cb_1]
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # Polygon and Specific method
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range (100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)
                # for i in range (3):
                # Image path
                pred_img = False
                for aux_img in range (3):
                    if self.predicted_img[1] == 1:
                        pred_img = True
                        j = 1
                    elif self.predicted_img[0] == 1:
                        pred_img = True
                        j = 0
                    elif self.predicted_img[2] == 1:
                        pred_img = True
                        j = 2
                if pred_img == True:
                    image_file = self.cropped_image[j]
                    # LLRS building image prediction
                    box_aux = None
                    material_index = predict_material_img(image_file, self.ui.insp_method, box_aux, self.ui)
              
                    # Set DL model prediction
                    # LLRS building image sets prediction
                    if material_index is None:
                        pass
                    else:
                        i=1
                        class_names_mat = ['Concrete', 'Hybrid - Confined and Unreinforced masonry', 'Informal materials', 
                                       'Masonry - Confined', 'Masonry - Reinforced', 'Masonry - Unreinforced', 
                                       'Steel','Wood']                         
                        material_id[i].setCurrentText(class_names_mat[material_index])
                        self.pred_mat_value = self.ui.material_cb_1.currentData()

                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")
            
            # Local method
            elif self.ui.insp_method == 2:
                
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range (100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)
                    
                # for aux in range (self.n_images_local):
                # Local cropped image path
                aux = 1
                aux_path = (self.ui.folder_path+"/Cropped_images/"
                                +str(self.data_building.iloc[self.old_local + aux, 0]))
                
                cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                
                try:
                    image = cv2.imread(cropped_path)
                except:
                    aux = 0
                    aux_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    
                    cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
    
                # LLRS building image prediction
                material_index = predict_material_img(cropped_path, self.ui.insp_method, self.box_id, self.ui)
                # LLRS building image sets prediction
                if material_index is None:
                    pass
                else:
                    class_names_mat = ['Concrete', 'Hybrid - Confined and Unreinforced masonry', 'Informal materials', 
                                   'Masonry - Confined', 'Masonry - Reinforced', 'Masonry - Unreinforced', 
                                   'Steel','Wood']                       
                    material_id[aux].setCurrentText(class_names_mat[material_index])
                    self.pred_mat_value = self.ui.material_cb_1.currentData()
                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")
                    
                        
    ############ Deep learning model for predict the LLRS ################
    def llrs_prediction (self):
        """
        Predict the Lateral Load Resisting System (LLRS) of a building using a deep learning model.
    
        This method applies a deep learning model to predict the LLRS for three building images 
        (e.g., from Google Street View). If the AI-powered checkbox is activated in the UI, the 
        method processes each image, retrieves predictions, and updates the corresponding UI 
        comboboxes with the predicted LLRS values.
    
        Effects:
            - Loads and applies a deep learning model to predict LLRS for the images.
            - Updates the comboboxes (`llrs_cb_1`, `llrs_cb_2`, `llrs_cb_3`) in the UI with predictions.
            - Updates the progress bar and status message in the UI.
    
        Notes:
            - The progress bar provides visual feedback during model loading and prediction.
            - Handles exceptions silently if predictions or UI updates fail.
            - Requires the AI-powered checkbox (`ai_check`) to be selected for predictions to proceed.
            - Assumes a predefined function `predict_llrs_img` for making predictions.
        """
        # Comboboxes for each image label
        llrs_id = [self.ui.llrs_cb_1,self.ui.llrs_cb_1,self.ui.llrs_cb_1]
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                # for i in range (3):
                pred_img = False
                for aux_img in range (3):
                    if self.predicted_img[1] == 1:
                        pred_img = True
                        j = 1
                    elif self.predicted_img[0] == 1:
                        pred_img = True
                        j = 0
                    elif self.predicted_img[2] == 1:
                        pred_img = True
                        j = 2
                if pred_img == True:
                    # Image path
                    image_file = self.cropped_image[j]       
                    # LLRS building image prediction
                    box_aux = None
                    llrs_index = predict_llrs_img(image_file, self.ui.insp_method, box_aux, self.ui)
                    # LLRS building image sets prediction
                    if llrs_index is None:
                        pass
                    else:
                        i=1
                        class_names_llrs = ['Dual System', 'Braced Frames', 'Infilled Frames', 'Moment Frames', 
                                       'No lateral load-resisting system', 'Walls', 'Walls']
                        
                        llrs_id[i].setCurrentText(class_names_llrs[llrs_index])
                        llrs_pred = self.ui.llrs_cb_1.currentData()
                        
                        if self.pred_mat_value == "MCF":
                            llrs_id[i].setCurrentText(class_names_llrs[5]) 
                        elif self.pred_mat_value == "MUR":
                            llrs_id[i].setCurrentText(class_names_llrs[5]) 
                        elif self.pred_mat_value == "MR":
                            llrs_id[i].setCurrentText(class_names_llrs[5])
                        elif self.pred_mat_value == "INF":
                            llrs_id[i].setCurrentText(class_names_llrs[4])
                        elif self.pred_mat_value == "CR":
                            if llrs_pred in ("LDUAL", "LFM", "LFINF"):
                                pass
                            else:
                                llrs_id[i].setCurrentText(class_names_llrs[3])
                        elif self.pred_mat_value == "S":
                            if llrs_pred in ("LFM", "LFBR"):
                                pass
                            else:
                                llrs_id[i].setCurrentText(class_names_llrs[3]) 
                        # Progress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")
            
            elif self.ui.insp_method == 2:
                
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range (100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)
                    
                # for aux in range (self.n_images_local):
                aux = 1
                # Local cropped image path
                
                aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                +str(self.data_building.iloc[self.old_local + aux, 0]))
                cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"                  
              
                try:
                    image = cv2.imread(cropped_path)
                except:
                    aux = 0
                    aux_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    
                    cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                # LLRS building image prediction
                llrs_index = predict_llrs_img(cropped_path, self.ui.insp_method, self.box_id, self.ui)
                # LLRS building image sets prediction
                if llrs_index is None:
                    pass
                else:
                    class_names_llrs = ['Dual System', 'Braced Frames', 'Infilled Frames', 'Moment Frames', 
                                   'No lateral load-resisting system', 'Walls', 'Walls']
                    
                    llrs_id[aux].setCurrentText(class_names_llrs[llrs_index])
                    llrs_pred = self.ui.llrs_cb_1.currentData()
                    
                    if self.pred_mat_value == "MCF":
                        llrs_id[aux].setCurrentText(class_names_llrs[5]) 
                    elif self.pred_mat_value == "MUR":
                        llrs_id[aux].setCurrentText(class_names_llrs[5]) 
                    elif self.pred_mat_value == "MR":
                        llrs_id[aux].setCurrentText(class_names_llrs[5])
                    elif self.pred_mat_value == "INF":
                        llrs_id[aux].setCurrentText(class_names_llrs[4])
                    elif self.pred_mat_value == "CR":
                        if llrs_pred in ("LDUAL", "LFM", "LFINF"):
                            pass
                        else:
                            llrs_id[aux].setCurrentText(class_names_llrs[3])
                    elif self.pred_mat_value == "S":
                        if llrs_pred in ("LFM", "LFBR"):
                            pass
                        else:
                            llrs_id[aux].setCurrentText(class_names_llrs[3])       
                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")
                    
                        
    ############ Deep learning model for predict the Code level ################
    def code_level_prediction (self):
        """
        Predict and assign a code level to a building based on its image.
    
        This method utilizes an AI model to predict the structural code level of a building 
        from an image. It first ensures that the necessary project details (folder, country, 
        and city) are set before execution. If AI-powered prediction is enabled, it processes 
        the images and updates the corresponding UI elements with the predicted code level.
    
        Effects:
            - Uses an AI model to predict the structural code level of the building.
            - Updates the UI combo boxes (`age_cb_1`, `age_cb_2`, `age_cb_3`) with the predicted values.
            - Displays progress updates via the UI progress bar.
            - Handles both Google Street View (GSV) and local image-based inspections.
    
        Notes:
            - AI-based prediction is performed only if the AI checkbox (`ai_check`) is enabled.
            - In polygon-based and specific inspection modes (`insp_method != 2`), 
              cropped images are used for prediction.
            - For manual inspection mode (`insp_method == 2`), predictions are performed 
              on local cropped images.
            - The `predict_code_img` function is called to generate predictions.
        """
        # Comboboxes for each image label
        code_level_id = [self.ui.age_cb_1,self.ui.age_cb_1,self.ui.age_cb_1]
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                # for i in range (3):
                pred_img = False
                for aux_img in range (3):
                    if self.predicted_img[1] == 1:
                        pred_img = True
                        j = 1
                    elif self.predicted_img[0] == 1:
                        pred_img = True
                        j = 0
                    elif self.predicted_img[2] == 1:
                        pred_img = True
                        j = 2
                if pred_img == True:
                    # Image path
                    image_file = self.cropped_image[j]

                    # LLRS building image prediction
                    box_aux = None
                    code_level_index = predict_code_img(image_file, self.ui.insp_method, box_aux, self.ui)

                    # LLRS building image sets prediction
                    if code_level_index is None:
                        pass
                    else:
                        i=1
                        class_names_code = ['High-Code','Low-Code', 'Moderate-code', 'No-Code']
                        code_level_id[i].setCurrentText(class_names_code[code_level_index])
                 
                        if self.pred_mat_value == "MUR":
                            code_level_id[i].setCurrentText(class_names_code[1]) 
                        elif self.pred_mat_value == "INF":
                            code_level_id[i].setCurrentText(class_names_code[3]) 
                        
                        # Progress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")
            
            elif self.ui.insp_method == 2:
                
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range (100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)
                    
                # for aux in range (self.n_images_local):
                aux = 1
                # Local cropped image path
                
                aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                +str(self.data_building.iloc[self.old_local + aux, 0]))
                cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                             
                try:
                    image = cv2.imread(cropped_path)
                except:
                    aux = 0
                    aux_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    
                    cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                
                # LLRS building image prediction
                code_level_index = predict_code_img(cropped_path, self.ui.insp_method, self.box_id, self.ui)
                # LLRS building image sets prediction
                if code_level_index is None:
                    pass
                else:
                    class_names_code = ['High-Code','Low-Code', 'Moderate-code', 'No-Code']
                    code_level_id[aux].setCurrentText(class_names_code[code_level_index])
             
                    if self.pred_mat_value == "MUR":
                        code_level_id[aux].setCurrentText(class_names_code[1]) 
                    elif self.pred_mat_value == "INF":
                        code_level_id[aux].setCurrentText(class_names_code[3])
                        
                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")
                
     
    ############ Deep learning model for predict the Number of Stories ################
    def n_stories_prediction (self):
        """
        Predict the Lateral Load Resisting System (LLRS) of a building using a deep learning model.
    
        This method applies a deep learning model to predict the LLRS for three building images 
        (e.g., from Google Street View). If the AI-powered checkbox is activated in the UI, the 
        method processes each image, retrieves predictions, and updates the corresponding UI 
        comboboxes with the predicted LLRS values.
    
        Returns:
            None. The predicted LLRS values are set in the UI comboboxes.
    
        Effects:
            - Loads and applies a deep learning model to predict LLRS for the images.
            - Updates the comboboxes (`llrs_cb_1`, `llrs_cb_2`, `llrs_cb_3`) in the UI with predictions.
            - Updates the progress bar and status message in the UI.
    
        Notes:
            - The progress bar provides visual feedback during model loading and prediction.
            - Handles exceptions silently if predictions or UI updates fail.
            - Requires the AI-powered checkbox (`ai_check`) to be selected for predictions to proceed.
            - Assumes a predefined function `predict_llrs_img` for making predictions.
        """
        # Comboboxes for each image label
        n_stories_id = [self.ui.n_stories_value_1,self.ui.n_stories_value_1,self.ui.n_stories_value_1]
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                # for i in range (3):
                pred_img = False
                for aux_img in range (3):
                    if self.predicted_img[1] == 1:
                        pred_img = True
                        j = 1
                    elif self.predicted_img[0] == 1:
                        pred_img = True
                        j = 0
                    elif self.predicted_img[2] == 1:
                        pred_img = True
                        j = 2
                if pred_img == True:
                    # Image path
                    image_file = self.cropped_image[j]

                    # LLRS building image prediction
                    box_aux = None
                    n_stories_index = predict_n_stories_img(image_file, self.ui.insp_method, box_aux, self.ui)

                    # LLRS building image sets prediction
                    class_names = ['10-12', '13+', '1', '2', '3', '4', '5', '6-7', '8-9']
                    if n_stories_index is None:
                        pass
                    else:
                        i=1
                        n_stories_id[i].setCurrentText(class_names[n_stories_index])    
                    
                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")
            
            elif self.ui.insp_method == 2:
                
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range (100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)
                    
                # for aux in range (self.n_images_local):
                aux = 1
                # Local cropped image path
                
                aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                +str(self.data_building.iloc[self.old_local + aux, 0]))
                cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                
                try:
                    image = cv2.imread(cropped_path)
                except:
                    aux = 0
                    aux_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    
                    cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                
                # LLRS building image prediction
                n_stories_index = predict_n_stories_img(cropped_path, self.ui.insp_method, self.box_id, self.ui)
                # LLRS building image sets prediction
                class_names = ['10-12', '13+', '1', '2', '3', '4', '5', '6-7', '8-9']
                if n_stories_index is None:
                    pass
                else:
                    n_stories_id[aux].setCurrentText(class_names[n_stories_index])
      
                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")
                    
                        
    ############ Deep learning model for predict the Occupancy type ################
    def occupancy_prediction (self):
        """
        Predict and assign an occupancy classification to a building based on its image.
    
        This method utilizes an AI model to predict the occupancy class of a building 
        from an image. It ensures that necessary project details (folder, country, 
        and city) are set before execution. If AI-powered prediction is enabled, it 
        processes the images and updates the corresponding UI elements with the 
        predicted occupancy class.
    
        Effects:
            - Uses an AI model to predict the occupancy classification of the building.
            - Updates the UI combo boxes (`occup_cb_1`, `occup_cb_2`, `occup_cb_3`) 
              with the predicted occupancy class.
            - Displays progress updates via the UI progress bar.
            - Handles both Google Street View (GSV) and local image-based inspections.
    
        Notes:
            - AI-based prediction is performed only if the AI checkbox (`ai_check`) is enabled.
            - In polygon-based and specific inspection modes (`insp_method != 2`), 
              cropped images are used for prediction.
            - For manual inspection mode (`insp_method == 2`), predictions are performed 
              on local cropped images.
        """
        # Comboboxes for each image label
        occupancy_id = [self.ui.occup_cb_1,self.ui.occup_cb_1,self.ui.occup_cb_1]
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                # for i in range (3):
                pred_img = False
                for aux_img in range (3):
                    if self.predicted_img[1] == 1:
                        pred_img = True
                        j = 1
                    elif self.predicted_img[0] == 1:
                        pred_img = True
                        j = 0
                    elif self.predicted_img[2] == 1:
                        pred_img = True
                        j = 2
                if pred_img == True:
                    # Image path
                    image_file = self.cropped_image[j]
                    # LLRS building image prediction
                    box_aux = None
                    occupancy_index = predict_occupancy_img(image_file, self.ui.insp_method, box_aux, self.ui)

                    # LLRS building image sets prediction
                    occupancy_class = [ 'Commercial' , 'Industrial' ,'Mixed (Residential + Commercial)', 'Residential']
                    if occupancy_index is None:
                        pass
                    else:
                        i=1
                        occupancy_id[i].setCurrentText(occupancy_class[occupancy_index])                      
                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")
            
            elif self.ui.insp_method == 2:
                
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range (100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)
                    
                # for aux in range (self.n_images_local):
                aux = 1
                
                aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                +str(self.data_building.iloc[self.old_local + aux, 0]))
                cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"      
                
                try:
                    image = cv2.imread(cropped_path)
                except:
                    aux = 0
                    aux_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    
                    cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                
                # LLRS building image prediction
                occupancy_index = predict_occupancy_img(cropped_path, self.ui.insp_method, self.box_id, self.ui)
                # LLRS building image sets prediction
                occupancy_class = [ 'Commercial' , 'Industrial' ,'Mixed (Residential + Commercial)', 'Residential']
                if occupancy_index is None:
                    pass
                else:
                    occupancy_id[aux].setCurrentText(occupancy_class[occupancy_index])     
      
                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")
                        
                        
    ############ Deep learning model for predict the block_position ################
    def block_position_prediction (self):
        """
        Predict and assign a block position classification to a building based on its image.
    
        This method utilizes an AI model to predict the block position of a building 
        from an image. It ensures that necessary project details (folder, country, 
        and city) are set before execution. If AI-powered prediction is enabled, it 
        processes the images and updates the corresponding UI elements with the 
        predicted block position.
    
        Effects:
            - Uses an AI model to predict the block position classification of the building.
            - Updates the UI combo boxes (`bck_pos_cb_1`, `bck_pos_cb_2`, `bck_pos_cb_3`) 
              with the predicted values.
            - Displays progress updates via the UI progress bar.
            - Handles both Google Street View (GSV) and local image-based inspections.
    
        Notes:
            - AI-based prediction is performed only if the AI checkbox (`ai_check`) is enabled.
            - In polygon-based and specific inspection modes (`insp_method != 2`), 
              cropped images are used for prediction.
            - For manual inspection mode (`insp_method == 2`), predictions are performed 
              on local cropped images.
            - The `predict_block_position_img` function is called to generate predictions.
            - The predicted index is incremented by 1 before being assigned to the combo box.
        """

        # Comboboxes for each image label
        block_position_id = [self.ui.bck_pos_cb_1,self.ui.bck_pos_cb_1,self.ui.bck_pos_cb_1]
        if self.box_id == None:
            # Checkbox for the AI powered activation
            if self.ui.ai_check.isChecked():
                if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                    # for i in range (3):
                    pred_img = False
                    for aux_img in range (3):
                        if self.predicted_img[1] == 1:
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            pred_img = True
                            j = 2
                    if pred_img == True:
                        # Image path
                        # image_file = self.cropped_image[i]  
                        image_file = self.org_img_bp
                        # block_position building image prediction
                        box_aux = None
                        block_position_index = predict_block_position_img(image_file, self.ui.insp_method, box_aux, self.ui)
                        # block_position building image sets prediction
                        if block_position_index is None:
                            pass
                        else:
                            i=1
                            block_position_id[i].setCurrentIndex(block_position_index+1)                       
                            # Peogress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                
                elif self.ui.insp_method == 2:
                    
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range (100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)
                        
                    # for aux in range (self.n_images_local):
                    aux = 1
                    # Local cropped image path
                    
                    aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                    
                    try:
                        image = cv2.imread(cropped_path)
                    except:
                        aux = 0
                        aux_path = (self.ui.folder_path+"/Cropped_images/"
                                        +str(self.data_building.iloc[self.old_local + aux, 0]))
                        
                        cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                    
                    org_path = (self.ui.folder_path+"/" +str(self.data_building.iloc[self.old_local + aux, 0]))
                    # block_position building image prediction
                    block_position_index = predict_block_position_img(org_path, self.ui.insp_method, self.box_id, self.ui)
                    # block_position building image sets prediction
                    if block_position_index is None:
                        pass
                    else:
                        block_position_id[aux].setCurrentIndex(block_position_index+1)
          
                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")      
        else:
            self.box_id = None
                
            
    ############ Deep learning model for predict the Roof shape ################
    def roof_shape_prediction (self):
        # Comboboxes for each image label
        roof_shape_id = [self.ui.roof_shape_cb_1,self.ui.roof_shape_cb_1,self.ui.roof_shape_cb_1]
        if self.box_id == None:
            # Checkbox for the AI powered activation
            if self.ui.ai_check.isChecked():
                if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                    # for i in range (3):
                    pred_img = False
                    for aux_img in range (3):
                        if self.predicted_img[1] == 1:
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            pred_img = True
                            j = 2
                    if pred_img == True:
                        # Image path
                        image_file = self.cropped_image[j]      
                        # roof_shape building image prediction
                        box_aux = None
                        roof_shape_index = predict_roof_shape_img(image_file, self.ui.insp_method, box_aux, self.ui)
                        # roof_shape building image sets prediction
                        if roof_shape_index is None:
                            pass
                        else:
                            i=1
                            roof_shape_id[i].setCurrentIndex(roof_shape_index+1) 
                            self.pred_roof_shape = self.ui.roof_shape_cb_1.currentData()
                            # Peogress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                
                elif self.ui.insp_method == 2:
                    
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range (100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)
                        
                    # for aux in range (self.n_images_local):
                    aux = 1
                    # Local cropped image path
                    
                    aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                    
                    try:
                        image = cv2.imread(cropped_path)
                    except:
                        aux = 0
                        aux_path = (self.ui.folder_path+"/Cropped_images/"
                                        +str(self.data_building.iloc[self.old_local + aux, 0]))
                        
                        cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                    
                    # roof_shape building image prediction
                    roof_shape_index = predict_roof_shape_img(cropped_path, self.ui.insp_method, self.box_id, self.ui)
                    # roof_shape building image sets prediction
                    if roof_shape_index is None:
                        pass
                    else:
                        roof_shape_id[aux].setCurrentIndex(roof_shape_index+1)
                        self.pred_roof_shape = self.ui.roof_shape_cb_1.currentData()
                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")      
        else:
            self.box_id = None
            
    ############ Deep learning model for predict the Roof shape ################
    def roof_material_prediction (self):
        # Comboboxes for each image label
        roof_material_id = [self.ui.roof_material_cb_1,self.ui.roof_material_cb_1,self.ui.roof_material_cb_1]
        if self.box_id == None:
            # Checkbox for the AI powered activation
            if self.ui.ai_check.isChecked():
                if self.ui.insp_method == 0 or self.ui.insp_method == 1: 
                    # for i in range (3):
                    pred_img = False
                    for aux_img in range (3):
                        if self.predicted_img[1] == 1:
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            pred_img = True
                            j = 2
                    if pred_img == True:
                        # Image path
                        image_file = self.cropped_image[j]      
                        # roof_material building image prediction
                        box_aux = None
                        roof_material_index = predict_roof_material_img(image_file, self.ui.insp_method, box_aux, self.ui)
                        # roof_material building image sets prediction
                        if roof_material_index is None:
                            pass
                        else:
                            i=1
                            roof_material_id[i].setCurrentIndex(roof_material_index+1)
                            roof_mat_pred = self.ui.roof_material_cb_1.currentData()
                            
                            if self.pred_roof_shape == "RSH1":
                                roof_material_id[i].setCurrentIndex(1) 
                            elif self.pred_roof_shape == "RSH7":
                                roof_material_id[i].setCurrentIndex(3)
                            elif self.pred_roof_shape == "RSH2":
                                if roof_mat_pred in ("RMT1", "RMT6"):
                                    pass
                                else:
                                    roof_material_id[i].setCurrentIndex(3)
                            elif self.pred_roof_shape == "RSH3":
                                if roof_mat_pred in ("RMT1", "RMT6"):
                                    pass
                                else:
                                    roof_material_id[i].setCurrentIndex(3)
                            elif self.pred_roof_shape == "RSH5":
                                if roof_mat_pred in ("RMT1", "RMT6"):
                                    pass
                                else:
                                    roof_material_id[i].setCurrentIndex(3)
                                              
                            # Peogress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                
                elif self.ui.insp_method == 2:
                    
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range (100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)
                        
                    # for aux in range (self.n_images_local):
                    aux = 1
                    # Local cropped image path
                    aux_cropped_path = (self.ui.folder_path+"/Cropped_images/"
                                    +str(self.data_building.iloc[self.old_local + aux, 0]))
                    cropped_path = os.path.splitext(aux_cropped_path)[0]+"_cropped.jpg"
                         
                    try:
                        image = cv2.imread(cropped_path)
                    except:
                        aux = 0
                        aux_path = (self.ui.folder_path+"/Cropped_images/"
                                        +str(self.data_building.iloc[self.old_local + aux, 0]))
                        
                        cropped_path = os.path.splitext(aux_path)[0]+"_cropped.jpg"
                           
                    # roof_material building image prediction
                    roof_material_index = predict_roof_material_img(cropped_path, self.ui.insp_method, self.box_id, self.ui)
                    # roof_material building image sets prediction
                    if roof_material_index is None:
                        pass
                    else:
                        roof_material_id[aux].setCurrentIndex(roof_material_index+1)
                        roof_mat_pred = self.ui.roof_material_cb_1.currentData()
                        
                        if self.pred_roof_shape == "RSH1":
                            roof_material_id[aux].setCurrentIndex(1) 
                        elif self.pred_roof_shape == "RSH7":
                            roof_material_id[aux].setCurrentIndex(3)
                        elif self.pred_roof_shape == "RSH2":
                            if roof_mat_pred in ("RMT1", "RMT6"):
                                pass
                            else:
                                roof_material_id[aux].setCurrentIndex(3)
                        elif self.pred_roof_shape == "RSH3":
                            if roof_mat_pred in ("RMT1", "RMT6"):
                                pass
                            else:
                                roof_material_id[aux].setCurrentIndex(3)
                        elif self.pred_roof_shape == "RSH5":
                            if roof_mat_pred in ("RMT1", "RMT6"):
                                pass
                            else:
                                roof_material_id[aux].setCurrentIndex(3)
                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")      
        else:
            self.box_id = None                      
                
                
    ############ Search and load existing inspections ################                  
    def search_inspection(self):
        """
        Search for a specific building inspection record by its ID.
    
        This method retrieves a building inspection record from the dataset based on 
        the ID entered in the UI search field. It ensures that a project folder, country, 
        and city name are defined before execution. If the search value is empty, an 
        error message is displayed. If the inspection database is not loaded, the user 
        is prompted to upload it.
    
        Notes:
            - The search is performed on the `data_ai` DataFrame using the column 'id'.
            - If no valid ID is entered, a message is set in the UI field instead of executing a search.
            - If the database is missing, the user is advised to upload it using the "Next Building" button.
        """
        # Get the value from the QLineEdit
        search_value = self.ui.search_img_value.text()
        # Check if the value is not empty
        if not search_value.strip():
            self.ui.search_img_value.setText("Please enter a value to search.")
            return

        # Search in the DataFrame
        try:
            try:
                result = self.data_ai[self.data_ai['id'] == int(search_value)]
                if result.shape[0]<0:
                    result = self.data_ai[self.data_ai['id'] == search_value]
            except:
                result = self.data_ai[self.data_ai['id'] == search_value]
                if result.shape[0]<0:
                    result = self.data_ai[self.data_ai['id'] == int(search_value)]
            
            n_building = result.iloc[0,0]
        except:
            QMessageBox.warning(self.ui, "Data Error", "Please click the Next Building button to upload the inspection database")
        
        try:
            if self.ui.insp_method == 0:
                self.click_count = int(n_building) - 1
            if self.ui.insp_method == 1:
                self.click_count = int(n_building) - 1
            if self.ui.insp_method == 2:
                self.click_count = int(n_building) - 1
        except:
            pass
        
        try:
            self.get_city_name()
            self.fetch_three_step_views()
            self.object_detector_building()
            self.clean_database()
        except:
            pass

    def neighbor_extrapolation(self):
        
        if self.ui.insp_method == 3:
            if self.ui.extrapolation_mode == 2:
                #####################################
                ######## KNN mode ############
                #####################################
                if self.ui.coord_reference is not True:
                    #######===========  Function results =========###########
                    data_existing_dl = create_database(self.ui.coord_reference)
                    dl_models()
                    inspection_database(data_existing_dl)
                    predicted_path =  self.ui.output_path+"/"+self.ui.coord_reference_building_feature_path
                    data_existing_dl.to_csv(predicted_path, index= False)
                    extra_path =  self.ui.output_path+"/"+self.ui.knn_dl_saved_path
                    n_neighbors = self.ui.k_value
                    extrapolation_existing_reference(data_existing_dl , self.ui.building_extra_path, extra_path, n_neighbors)
                else:
                    building_no_info = self.ui.building_extra_path
                    building_reference = self.ui.example_building_path
                    final_distribution_list_full = []
                       
                    # Iterate over each building with no image
                    for idx, input_row in building_no_info.iterrows():
                        # Find 3 nearest neighbors using geodesic distance
                        n_neighbors = self.ui.k_value
                        nearest_neighbors = find_nearest_neighbors_geodesic(input_row, building_reference, n_neighbors)
                        # Compute taxonomy-based distributions with full structure
                        distribution_rows = compute_taxonomy_distribution_full_structure(nearest_neighbors, input_row)
                        
                        # Append to final result
                        final_distribution_list_full.extend(distribution_rows)
               
                    # Convert final list to DataFrame
                    final_distribution_df_full = pd.DataFrame(final_distribution_list_full)
                    # Export to CSV
                    saved_path = self.ui.output_path+"/"+self.ui.extrapolation_name+".csv"
                    
                    final_distribution_df_full.to_csv(saved_path, index=False)
                    self.ui.method_progress.setText("Successful extrapolation process")
             
            #####################################
            ######## Stratified mode ############
            #####################################
            else:
                if self.ui.extrapolation_mode == 0:
                    building_data = self.ui.data_population
                    self.lat_dl = building_data.loc[0, "latitude"]
                    self.lon_dl = building_data.loc[0, "longitude"]
                    # ========== Run sampling for each feature ==========
                    analysis_features = self.ui.feature_strata
                    sample_size_def = []
                    for aux in analysis_features:
                        print(" ========== " + aux + " ===========")
                        final_sample, class_dist, final_size = iterative_label_discovery_cached_fractional(
                            data=building_data,
                            labeling_function=lambda x: labeling_function(x, aux, building_data),
                            id_column='id',
                            id_feature=aux,
                            initial_fraction = self.ui.initial_fraction,
                            step_fraction = self.ui.step_fraction,
                            max_fraction = self.ui.max_fraction,
                            max_iterations = self.ui.max_iterations,
                            stability_threshold = self.ui.stability_threshold
                        )
                        sample_size_def.append(len(final_sample))
                        final_sample.to_csv(f"{self.ui.folder_path_new}/stratified_dl_{aux}.csv", index=False)
                        print("")
                        
                    print("Sample size definitive: ", np.max(sample_size_def))
                    
                elif self.ui.extrapolation_mode == 1:
                    
                    building_data = self.ui.data_population
                    # ========== Run sampling for each feature ==========
                    analysis_features = self.ui.feature_strata
                    
                    for feature in analysis_features:
                        print(" ========== " + feature + " ===========")
                        final_sample, class_dist, final_size = iterative_distribution_stability_manual(
                            data=building_data,
                            id_feature=feature,
                            id_column='id',
                            initial_fraction = self.ui.initial_fraction,
                            step_fraction = self.ui.step_fraction,
                            max_fraction = self.ui.max_fraction,
                            max_iterations = self.ui.max_iterations,
                            stability_threshold = self.ui.stability_threshold
                        )
                        
                        final_sample.to_csv(f"{self.ui.folder_path_new}/stratified_{feature}.csv", index=False)
                        print("")
        
    def epoch_construction(self):
        if self.ui.insp_method != 3:
            path = self.ui.output_folder_value+"/epoch_value.csv"
          
            try:
                epoch = pd.read_csv(path)
                if self.epoch_const == True:
                    self.epoch_const = False
                    for i in range(len(epoch)):
                        self.ui.epc_const_cb_1.addItem(str(epoch.iloc[i,0]))

            except:
                self.epoch_const = False
                
                # Message with special format
                message = """
                The construction epoch varies by country and is typically linked to the implementation 
                of specific building code regulations. As a result, each country has its own relevant 
                periods. <b><u>PLEASE SELECT AND ENTER THE APPROPRIATE CONSTRUCTION EPOCH FOR YOUR COUNTRY</u></b>.
                """
                
                # Create a QMessageBox instance
                msg_box = QMessageBox(self.ui)
                msg_box.setTextFormat(QtCore.Qt.RichText)
                msg_box.setText(message)
                msg_box.setWindowTitle("Epoch of construction values")
                
                # Show the message box
                msg_box.exec_()
                
                """Open the bounding box selection pop-up window."""
                app = QApplication.instance()  # Ensure PyQt instance exists
                if app is None:
                    app = QApplication([])
                    
                # Called function where the user creates a manual bounding box by clicking four points, which is then displayed in the UI frame.
                dialog = EpochSelectionDialog(parent=self.ui, main_window=self.ui)
                dialog.exec_()  # Open the pop-up
                epoch = dialog.get_epochs()
                for i in range(len(epoch)):
                    self.ui.epc_const_cb_1.addItem(epoch[i])
                    
                epoch = pd.DataFrame(epoch)
                epoch.columns = ["Epochs"]
                epoch.to_csv(path, index=False)
            
    def help_block_position(self):
        # Paths to your example images for each roof shape 
        self.images = {"Block position options": "help_img/block_position.png" }
        
        help_window = HelpDialog(self.images, w_size_width=720, w_size_height=500, w_title= "Block position - visual example",
                                 img_width=640, img_height=480, parent=self.ui, main_window=self.ui)
        help_window.exec_()
        
    def help_roof_shape(self):
        # Paths to your example images for each roof shape 
        self.images = {
            "Flat": "help_img/flat_roof.png",
            "Pitched with gable ends": "help_img/gable_roof.png",
            "Pitched and hipped": "help_img/hipped_roof.png",
            "Pitched with dormers": "help_img/dormes_roof.png",
            "Monopitch": "help_img/monoslope_roof.png",
            "Sawtooth": "help_img/Sawtooth_roof.png",
            "Curved": "help_img/curved.png",
            "Complex regular": "help_img/complex_regular.png",
            "Complex irregular": "help_img/complex_irregular.png"
        }
        
        help_window = HelpDialog(self.images, w_size_width=700, w_size_height=700, w_title= "Roof Shape - visual example",
                                 img_width=180, img_height=180, parent=self.ui, main_window=self.ui)
        help_window.exec_()
        
    def help_roof_material(self):
        # Paths to your example images for each roof shape 
        self.images = {
            "Concrete": "help_img/concrete.jpg",
            "Clay or concrete tile": "help_img/clay_tile.jpg",
            "Metal or asbestos sheets": "help_img/asbesto.png",
            "Wooden and asphalt shingles": "help_img/asphalt_shingles.jpg",
            "Slate": "help_img/Slate.png",
            "Solar panelled roofs": "help_img/solar_panel.png"
        }
        
        help_window = HelpDialog(self.images, w_size_width=700, w_size_height=500, w_title= "Roof Material - visual example",
                                 img_width=180, img_height=180, parent=self.ui, main_window=self.ui)
        help_window.exec_()