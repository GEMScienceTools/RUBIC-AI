"""Provide GUI workflow methods for RUBIC-AI.

This module coordinates image acquisition, building detection, attribute
prediction, database management, extrapolation, and user-interface actions.
"""

# GUI pyqt5 libraries
import contextlib
import os
import re
import sys

# Utilities libraries
import time

import cv2

# Libries for shape and geopackage creation
import geopandas as gpd
import numpy as np
import pandas as pd

# Google Street Maps libry
import requests
import torch
from geopy.geocoders import ArcGIS, Nominatim, Photon
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMessageBox
from ultralytics import YOLO

from methods.bounding_box_manual import BoundingBoxWindow

# *.py scripts with complex methods
from methods.dl_prediction_models import (
    predict_block_position_img,
    predict_code_img,
    predict_llrs_img,
    predict_material_img,
    predict_n_stories_img,
    predict_occupancy_img,
    predict_roof_material_img,
    predict_roof_shape_img,
)
from methods.epoch_construction import EpochSelectionDialog
from methods.get_building_orientation import get_road_orientation, get_street_view_image
from methods.gsv_image_angle import gsv_angle_setting
from methods.help_window import HelpDialog
from methods.knn_extrapolation_feature import (
    compute_taxonomy_distribution_full_structure,
    extrapolation_existing_reference,
    find_nearest_neighbors_geodesic,
)
from methods.mapillary_images import mapillary_image_source
from methods.stratified_extrapolation_feature import (
    iterative_distribution_stability_manual,
    iterative_label_discovery_cached_fractional,
    labeling_function,
)

# Taxonomy check
from methods.taxonomy import check_taxonomy
from methods.vulnerability_plot import VulnerabilityDialog


class GUIMethods:
    """Coordinate GUI workflows for inspection and image processing."""

    def __init__(self, ui):
        self.ui = ui  # Link to the UI components
        # Initialize attributes to avoid AttributeError
        self.click_count = -1  # Building ID aux varible
        self.start = True  # Building ID aux varible
        self.data_building = None  # Building sample information
        self.data_ai = None  # AI inspection
        self.data_old = None  # Existing inspection swicth
        self.sw_insp = True  # Existing inspection swicth
        self.save_id = True  # Existing inspection swicth
        self.box_id = None  # Manual bounding box
        self.epoch_const = True  # Epoch of construction
        self.data_old_local = False
        self.sw_local_previous = True
        self.cropped_image = ["", "", ""]
        self.start_click = True
        self.aux_previous = True
        self.aux_ai_check = True
        self.limit_local = True
        self.search_count = False
        self.sw_angle = None
        self.sw_extrapolation = False
        self.color_control = False
        """Get screen resolution to adapt to different screen sizes"""
        # Get screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        #
        design_width = 1920
        design_height = 1080
        design_dpi = 96 * 1.25  # 125% Windows baseline -> 120 DPI

        # Scale the GUI based on resolution
        sf_x = screen_width / design_width
        sf_y = screen_height / design_height
        sf_factor = np.sqrt(sf_x * sf_y)
        self.sf_factor = sf_factor
        # DPI-based scale
        # Get a reliable DPI value
        if sys.platform.startswith("win"):
            # Windows: use ctypes to get real DPI
            import ctypes

            log_pixels_x = 88
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, log_pixels_x)
            ctypes.windll.user32.ReleaseDC(0, hdc)
        else:
            # macOS / Linux: start with logical DPI
            dpi = screen.logicalDotsPerInch()
            # If logical DPI looks weird, fallback to physical
            if dpi < 60 or dpi > 200:
                dpi = screen.physicalDotsPerInch()

        # Normalize to your design environment (Windows @ 125% = 120 DPI)
        # If dpi == 120 => scale_dpi = 1 (your original machine)
        scale_dpi = design_dpi / dpi

        # For geometry: mainly resolution-based
        sf_x = sf_factor
        sf_y = sf_factor
        # Scale the GUI based on resolution
        self.sf_font = sf_factor * scale_dpi

    ############ Counts the number of clicks made on the next button ################
    def count_clicks_next(self):  # noqa: C901
        """Advance to the next available inspection record.

        Load the building data when needed, update the inspection counter,
        and manage navigation and progress messages.
        """
        # load the dataset of the subset buildings
        if self.data_building is None:
            if self.ui.insp_method == 0:
                # Polygon method
                path = (
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name
                    + "_building_info.csv"
                )
                self.data_building = pd.read_csv(path)
            elif self.ui.insp_method == 1:
                # Specific coordinates
                path = (
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name
                    + "_building_info.csv"
                )
                self.data_building = pd.read_csv(path)
            elif self.ui.insp_method == 2:
                # Local images
                self.data_building = pd.read_csv(self.ui.file_local_csv)
            elif self.ui.insp_method == 3:
                # Extrapolation
                pass

        # Verify that the building ID is less than the number of sample
        if self.ui.insp_method in (0, 1):
            # Polygon and Specific Coordinates
            limit_insp = self.data_building.shape[0] - 1
        elif self.ui.insp_method == 2:
            # Local images
            if self.limit_local is True:
                limit_insp = self.data_building.shape[0] - 1
            else:
                limit_insp = len(self.index_id) - 1
        elif self.ui.insp_method == 3:
            # Local images
            limit_insp = 1

        self.limit_local = False

        # Warning about last inspection
        if self.click_count >= limit_insp:
            QMessageBox.warning(
                self.ui, "Database Error", "No further inspections are available"
            )
        else:
            self.ui.method_progress.setText("Loading images ...")

            # Save inspection for first click after save results or start the script
            if self.click_count >= 0:
                self.inspection_database()

            # ==============================================================
            # Local images existing inspection counter
            # ==============================================================
            if self.ui.insp_method == 2:
                if self.sw_local_previous is True:
                    valid_coords = self.ui.data_method[
                        ["latitude", "longitude"]
                    ].dropna()
                    # Drop duplicates to get unique coordinate pairs
                    unique_coords = valid_coords.drop_duplicates()
                    self.index_id = unique_coords.index.tolist()
                    df = self.ui.data_method
                    # Create a coordinate pair column
                    df["coord_pair"] = list(
                        zip(df["latitude"], df["longitude"], strict=False)
                    )
                    # Keep the first occurrence of each coordinate
                    self.unique_coords = df.drop_duplicates(
                        subset="coord_pair", keep="first"
                    ).reset_index()
                    self.sw_local_previous = False

                # Check if there is inspection already done
                if self.data_old is not None:
                    self.n_insp = int(self.data_ai.dropna(how="all").shape[0])
                    self.data_old = None
                    # Use the saved inspection count only once.
            else:
                # ==============================================================
                # Other methods existing inspection counter
                # ==============================================================
                if self.data_old is not None:
                    self.n_insp = int(self.data_ai.dropna(how="all").shape[0])
                    self.data_old = None
                    # Use the saved inspection count only once.

            # Calculates the number of inspections saved
            if self.search_count is False:
                try:
                    # Conditional for only update the number of click and the ID cont
                    # one time
                    if self.n_insp > 0 and self.sw_insp is True:
                        if self.ui.insp_method != 2:
                            # self.click_count = int(self.n_insp/3 - 1)
                            self.click_count = self.n_insp - 1
                            self.sw_insp = False
                        elif self.ui.insp_method == 2:
                            # Drop rows with missing coordinates
                            valid_coords = self.data_ai_existing[
                                ["latitude", "longitude"]
                            ].dropna()
                            # Drop duplicates to get unique coordinate pairs
                            unique_coords = valid_coords.drop_duplicates()
                            # Count unique coordinate pairs
                            num_unique_coords = len(unique_coords)
                            self.click_count = num_unique_coords - 1
                            self.sw_insp = False
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass

            # ID increaser
            self.click_count += 1

            # Turn AI-powered mode on or off
            if self.aux_ai_check is True:
                with contextlib.suppress(AttributeError, TypeError):
                    self.ui.ai_check.setChecked(self.ui.ai_value)
                self.aux_ai_check = False

            # Check the size of inspection available
            if (
                self.start_click
                and self.ui.insp_method != 3
                and self.click_count >= self.data_building.shape[0] - 1
            ):
                self.click_count = self.data_building.shape[0] - 1
                QMessageBox.information(
                    self.ui,
                    "Inspections available",
                    ("The next building displayed is the final one in the database"),
                )
                self.start_click = False

    ############ Counts the number of clicks made on the previous button #####
    def count_clicks_previous(self):
        """Decrement the click counter to navigate to the previous building inspection.

        This method decreases the click counter, allowing the user to move back to a
        previous inspection record. It ensures that a project folder, country, and
        city name are defined before execution. If the dataset has not been initialized,
        it prompts the user to click the "Next" button first.
        """
        if self.data_building is None:
            QMessageBox.warning(
                self.ui, "GUI Error", "Please click the Next button to start the GUI"
            )
            self.n_images_local = 0
        else:
            """Increment the click counter and update the label."""
            if self.click_count > 0 and self.ui.insp_method in (0, 1, 2):
                self.click_count -= 1

    ############ Gets city name ################
    def get_location_with_fallback(self, lat, lon, ui_instance=None):
        """Try multiple geocoding providers with automatic fallback."""
        providers = [
            ("Nominatim", self.get_location_nominatim),
            ("ArcGIS", self.get_location_arcgis),
            ("Photon", self.get_location_photon),
        ]

        for provider_name, provider_func in providers:
            try:
                city, country = provider_func(lat, lon)
                if city and country and city != "Unknown":
                    if ui_instance:
                        print(f"Successfully geocoded using {provider_name}")
                    return city, country
            except Exception as e:
                print(f"{provider_name} failed: {e}")
                continue

        return "Unknown", "Unknown"

    def get_location_nominatim(self, lat, lon):
        """Try Nominatim provider."""
        geolocator = Nominatim(user_agent="city_name_locator")
        time.sleep(1)  # Rate limiting
        location = geolocator.reverse(
            (lat, lon), exactly_one=True, language="en", timeout=3
        )

        if location and "address" in location.raw:
            address = location.raw["address"]
            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
                or address.get("county")
                or address.get("state_district")
                or "Unknown"
            )
            country = address.get("country", "Unknown")
            return city, country

        return "Unknown", "Unknown"

    def get_location_photon(self, lat, lon):
        """Try Photon provider."""
        geolocator = Photon(user_agent="city_name_locator")
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=3)

        if location and hasattr(location, "raw") and "properties" in location.raw:
            props = location.raw["properties"]
            city = (
                props.get("city")
                or props.get("town")
                or props.get("village")
                or props.get("county")
                or "Unknown"
            )
            country = props.get("country", "Unknown")
            return city, country

        return "Unknown", "Unknown"

    def get_location_arcgis(self, lat, lon):
        """Try ArcGIS provider (most reliable)."""
        geolocator = ArcGIS(user_agent="city_name_locator")
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=5)

        if location and hasattr(location, "raw") and "address" in location.raw:
            address = location.raw["address"]
            city = address.get("City") or address.get("Subregion") or "Unknown"
            country = address.get("CountryCode") or "Unknown"
            return city, country

        return "Unknown", "Unknown"

    ############ Assign city name using coordinates ################
    def get_city_name(self):
        """Retrieve the city and country for the current coordinates.

        Update the interface for local-image, polygon, specific-coordinate,
        and extrapolation workflows.
        """
        if self.click_count >= 0:
            # ==============================================================
            # Local images city name
            # ==============================================================
            if self.ui.insp_method == 2:
                try:
                    if self.data_old_local is True:
                        self.cont_local = self.n_insp
                        self.data_old_local = False

                    if len(self.index_id) > self.click_count:
                        self.cont_local = int(self.index_id[self.click_count])
                    else:
                        self.cont_local = int(self.index_id[-1])
                        self.click_count = self.click_count - 1

                    if "latitude" in self.ui.data_method.columns:
                        lat = float(
                            self.ui.data_method.loc[self.cont_local, "latitude"]
                        )
                    else:
                        raise ValueError(
                            "The 'latitude' column is missing from the data."
                        )

                    if "longitude" in self.ui.data_method.columns:
                        lon = float(
                            self.ui.data_method.loc[self.cont_local, "longitude"]
                        )
                    else:
                        raise ValueError(
                            "The 'latitude' column is missing from the data."
                        )

                    self.ui.lat_value.setText(str(round(lat, 8)))
                    self.ui.lon_value.setText(str(round(lon, 8)))

                    df = self.ui.data_method
                    matching_rows = df[
                        (df["latitude"] == lat) & (df["longitude"] == lon)
                    ]

                    self.n_images_local = len(matching_rows)
                    self.old_local = self.cont_local
                    self.cont_local = self.cont_local + self.n_images_local

                    try:
                        # Try multiple providers automatically
                        city, country = self.get_location_with_fallback(lat, lon)

                    except (AttributeError, ValueError, ConnectionError, TimeoutError):
                        QMessageBox.warning(
                            self.ui,
                            "Geocoding Error",
                            "The city and country could not be retrieved."
                            + "Please try again.",
                        )
                        city, country = "Unknown", "Unknown"

                    self.city, self.country = city, country
                    self.ui.city_value.setText(self.city)
                    self.ui.country_value.setText(self.country)
                    return self.city, self.country

                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    QMessageBox.warning(
                        self.ui,
                        "Input Error",
                        (
                            "Some required inputs are missing or invalid. "
                            "Please review all fields and check the coordinates "
                            "file for inconsistencies."
                        ),
                    )
            # ==============================================================
            # Poligon and Specific Coordinates city name
            # ==============================================================
            elif self.ui.insp_method == 0 or self.ui.insp_method == 1:
                self.ui.lat_value.setText(
                    str(round(self.data_building.loc[self.click_count, "latitude"], 8))
                )
                self.ui.lon_value.setText(
                    str(round(self.data_building.loc[self.click_count, "longitude"], 8))
                )

                lat = float(self.ui.lat_value.text())
                lon = float(self.ui.lon_value.text())

                try:
                    # Try multiple providers automatically
                    city, country = self.get_location_with_fallback(lat, lon)

                except (AttributeError, ValueError, ConnectionError, TimeoutError):
                    QMessageBox.warning(
                        self.ui,
                        "Geocoding Error",
                        "The city and country could not be retrieved."
                        + "Please try again.",
                    )
                    city, country = "Unknown", "Unknown"

                self.city, self.country = city, country
                self.ui.city_value.setText(self.city)
                self.ui.country_value.setText(self.country)
                return self.city, self.country

            # ==============================================================
            # Extrapolation city name
            # ==============================================================
            elif self.ui.insp_method == 3:
                try:
                    city, country = self.get_location_with_fallback(
                        self.lat_extrapolation, self.lon_extrapolation
                    )
                except (AttributeError, ValueError, ConnectionError, TimeoutError):
                    city = "Unknown"
                    country = "Unknown"
                return city, country
        else:
            pass

    ############ Create building dataset for upload images from GSV ################
    def create_database(self):
        """Create the inspection database for the selected workflow.

        Export building coordinates when needed and initialize the results
        table for the available buildings.
        """
        # ==============================================================
        # Polygon method database
        # ==============================================================
        if self.ui.insp_method == 0:
            # Input and output for the method
            centroid_file = (
                self.ui.output_folder_value
                + "/"
                + self.ui.file_name
                + "_subset_centroids.gpkg"
            )
            database_file = (
                self.ui.output_folder_value
                + "/"
                + self.ui.file_name
                + "_building_info.csv"
            )

            # Check if a database file exists
            if os.path.exists(database_file):
                pass
            else:
                # Load the GeoPackage
                gdf = gpd.read_file(centroid_file)
                # Filter columns
                filtered_gdf = gdf[["id", "latitude", "longitude"]]
                # Export to CSV
                filtered_gdf.to_csv(database_file, index=False)
                print("Filtered CSV exported successfully!")

        # ==============================================================
        # Specific coordinates method database
        # ==============================================================
        elif self.ui.insp_method == 1:
            self.city_method = self.ui.city_value.text()
            self.country_method = self.ui.country_value.text()

            centroid_file = (
                self.ui.output_folder_value + "/" + self.ui.file_name + ".gpkg"
            )
            database_file = (
                self.ui.output_folder_value
                + "/"
                + self.ui.file_name
                + "_building_info.csv"
            )
            # Load the GeoPackage
            gdf = gpd.read_file(centroid_file)
            # Filter columns
            filtered_gdf = gdf[["id", "latitude", "longitude"]]
            # Export to CSV
            filtered_gdf.to_csv(database_file, index=False)

        # Upload the create building info to get the size of the inspection dataset
        # Craete an empty dataframe with the exact size
        if self.data_building is None:
            # Load the footprint database
            if self.ui.insp_method == 0:
                # Polygon method
                footprint_data = pd.read_csv(
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name
                    + "_building_info.csv"
                )
            elif self.ui.insp_method == 1:
                # Specific Coordinates method
                footprint_data = pd.read_csv(
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name
                    + "_building_info.csv"
                )
            elif self.ui.insp_method == 2:
                # Local Images method
                footprint_data = pd.read_csv(self.ui.file_local_csv)
            elif self.ui.insp_method == 3:
                # Extrapolation
                self.ui.progress_bar_method.setValue(10)
                self.ui.method_progress.setText("Extrapolation in progress...")
                if self.ui.extrapolation_mode == 2:
                    # KNN
                    footprint_data = self.ui.coord_reference

            # Define the column namesfor the inspection database
            column_names = [
                "id",
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
                "v_irregularity",
                "n_bays",
                "image_quality",
                "taxonomy",
                "image filename or link",
            ]

            # Create an empty DataFrame for number of footprint available
            try:
                if self.data_ai is None:
                    self.data_ai = pd.DataFrame(
                        np.full((footprint_data.shape[0], len(column_names)), None),
                        columns=column_names,
                    )
            except (
                FileNotFoundError,
                PermissionError,
                TypeError,
                ValueError,
                UnboundLocalError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                AttributeError,
            ):
                pass

    ############ Uploads existing database ################
    def load_existing_insp(self):
        """Load previously saved inspection results.

        Restore existing records and progress for polygon, specific-coordinate,
        and local-image workflows.
        """
        # Upload existing inspections when the GUI is started for first time
        if self.start is True:
            try:
                if self.ui.insp_method == 0:
                    # ==============================================================
                    # Polygon method existing inpection
                    # ==============================================================
                    output_folder = self.ui.output_folder_value
                    insp_path = f"{output_folder}/{self.ui.file_name}"
                    # Upload the existing inspections for AI
                    self.data_ai_existing = pd.read_csv(insp_path + "_AI_aux_cont.csv")
                    # Replace empty rows with the existing information
                    self.data_ai.iloc[: self.data_ai_existing.shape[0], :] = (
                        self.data_ai_existing.iloc[: self.data_ai_existing.shape[0], :]
                    )

                    self.data_old = "OK"  # THERE IS EXISTING DATA
                    self.start = False

                elif self.ui.insp_method == 1:
                    # ==============================================================
                    # Specific coordinates existing inpection
                    # ==============================================================
                    insp_path = self.ui.output_folder_value + "/" + self.ui.file_name
                    # Upload the existing inspections for AI
                    self.data_ai_existing = pd.read_csv(insp_path + "_AI_aux_cont.csv")
                    # Replace empty rows with the existing information
                    self.data_ai.iloc[: self.data_ai_existing.shape[0], :] = (
                        self.data_ai_existing.iloc[: self.data_ai_existing.shape[0], :]
                    )

                    self.data_old = "OK"  # THERE IS EXISTING DATA
                    self.start = False

                elif self.ui.insp_method == 2:
                    # ==============================================================
                    # Local images existing inpection
                    # ==============================================================
                    insp_path = (
                        self.ui.output_folder_value
                        + "/"
                        + self.ui.file_name_local.text()
                    )
                    # Upload the existing inspections for AI
                    self.data_ai_existing = pd.read_csv(
                        insp_path + "_AI_classification.csv"
                    )
                    self.cont_local_data = pd.read_csv(insp_path + "_AI_aux_cont.csv")
                    # Replace empty rows with the existing information
                    self.data_ai.iloc[: self.data_ai_existing.shape[0], :] = (
                        self.data_ai_existing.iloc[: self.data_ai_existing.shape[0], :]
                    )
                    self.cont_local_data.iloc[: self.cont_local_data.shape[0], :] = (
                        self.cont_local_data.iloc[: self.cont_local_data.shape[0], :]
                    )
                    print("Upload existing data sucessfully")
                    self.data_old = "OK"  # THERE IS EXISTING DATA
                    self.data_old_local = True
                    self.start = False
            except (
                FileNotFoundError,
                PermissionError,
                AttributeError,
                TypeError,
                ValueError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ):
                pass

    ############ Checks if there is GSV availability ################
    def check_street_view(self):
        """Check whether Google Street View coverage is available.

        Query the Street View metadata service for the current coordinates.
        """
        if self.ui.insp_method in (0, 1) and self.ui.img_source == 1:
            # Google Street View requires a paid API key.
            try:
                with open("methods/gsv_api_key.txt") as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                api_key = None
            lat = self.ui.lat_value.text()
            lon = self.ui.lon_value.text()
            url = "https://maps.googleapis.com/maps/api/streetview/metadata"
            params = {"location": f"{lat},{lon}", "key": api_key}
            response = requests.get(url, params=params)
            data = response.json()
            return data.get("status") == "OK"

        return False

    ############# Retrieve GSV building images ################
    def fetch_three_step_views(self):  # noqa: C901
        """Retrieve and store up to three street-level images.

        Update image identifiers and capture years for standard inspection and
        extrapolation workflows.
        """
        # Image ID displayed values
        # left image
        self.ui.img_id_value_1.setText(str(self.click_count + 1) + "_1")
        # central image
        self.ui.img_id_value_2.setText(str(self.click_count + 1) + "_2")
        # right image
        self.ui.img_id_value_3.setText(str(self.click_count + 1) + "_3")

        # ==============================================================
        # Polygon method or Specific Coordinates
        # ==============================================================

        if self.ui.insp_method == 0 or self.ui.insp_method == 1:
            # Image Source
            # 1: Google Street View (Paid API needed)
            # 2: Mapillary (Free API is needed, but it provides less coverage and, in
            # some cases, lower image quality)
            if self.ui.img_source == 1:
                # 1: Google Street View
                # Building coordinates
                location = (
                    float(self.ui.lat_value.text()),
                    float(self.ui.lon_value.text()),
                )
                # API key is required; without it, access to GSV is not possible
                try:
                    with open("methods/gsv_api_key.txt") as f:
                        api_key = f.read().strip()
                except FileNotFoundError as error:
                    QMessageBox.warning(
                        self.ui,
                        "API Key Error",
                        (
                            "The Google Street View API key file is missing.\n\n"
                            f"Details: {error}"
                        ),
                    )
                    return
                # angles for taking the images
                angle = (-30, 0, 30)
                self.img_url = ["", "", ""]
                for aux in range(3):
                    if self.check_street_view() is True:
                        # Get image from GSV
                        if aux == 0:
                            self.img_url[aux], self.img_original_1, self.year_left = (
                                get_street_view_image(
                                    location, api_key, angle[aux], 5, 120
                                )
                            )
                        elif aux == 1:
                            self.img_url[aux], self.img_original_2, self.year_center = (
                                get_street_view_image(
                                    location, api_key, angle[aux], 5, 120
                                )
                            )
                        else:
                            self.img_url[aux], self.img_original_3, self.year_right = (
                                get_street_view_image(
                                    location, api_key, angle[aux], 5, 120
                                )
                            )
                    else:
                        print("Street View not available")
                        self.img_original_1, self.year_left = ["", ""]
                        self.img_original_2, self.year_center = ["", ""]
                        self.img_original_3, self.year_right = ["", ""]

                # Year of the GSV IMAGE
                self.ui.year_value_1.setText(str(self.year_left))
                self.ui.year_value_2.setText(str(self.year_center))
                self.ui.year_value_3.setText(str(self.year_right))

            # 2: Mapillary
            else:
                try:
                    with open("methods/mapillary_api_key.txt") as f:
                        mapillary_access_token = f.read().strip()
                except FileNotFoundError as error:
                    QMessageBox.warning(
                        self.ui,
                        "API Key Error",
                        (
                            "The Mapillary API key file is missing.\n\n"
                            f"Details: {error}"
                        ),
                    )
                    return
                img_name = str(self.click_count + 1) + ".jpg"
                self.img_url = [img_name, img_name, img_name]

                img_path = (
                    self.ui.output_folder_value
                    + "/Mapillary/orthophotos/"
                    + str(self.click_count + 1)
                    + ".jpg"
                )
                try:
                    orthophoto_matrix = cv2.imread(img_path, cv2.IMREAD_COLOR)
                except FileNotFoundError:
                    orthophoto_matrix = None

                if orthophoto_matrix is not None:
                    for aux in range(3):
                        if aux == 0:
                            self.img_original_1 = orthophoto_matrix
                        elif aux == 1:
                            self.img_original_2 = orthophoto_matrix
                        else:
                            self.img_original_3 = orthophoto_matrix
                else:
                    # ==============================================================
                    # Get Images From Mapillary
                    try:
                        orthophoto_matrix, _info = mapillary_image_source(
                            access_token=mapillary_access_token,
                            latitude=self.data_building.loc[
                                self.click_count, "latitude"
                            ],
                            longitude=self.data_building.loc[
                                self.click_count, "longitude"
                            ],
                            point_id=self.data_building.loc[self.click_count, "id"],
                            image_number=self.click_count + 1,
                            output_dir=self.ui.output_folder_value + "/Mapillary",
                            return_info=True,
                            return_color_order="BGR",
                        )
                    except (
                        AttributeError,
                        ConnectionError,
                        IndexError,
                        KeyError,
                        TimeoutError,
                        TypeError,
                        ValueError,
                    ) as error:
                        print(f"Could not retrieve the Mapillary image: {error}")

                        img_path = "help_img/mapillary_error.jpg"
                        orthophoto_matrix = cv2.imread(
                            img_path,
                            cv2.IMREAD_COLOR,
                        )

                    angle = (-30, 0, 30)
                    img_name = str(self.click_count + 1) + ".jpg"
                    self.img_url = [img_name, img_name, img_name]
                    for aux in range(3):
                        if aux == 0:
                            self.img_original_1 = orthophoto_matrix
                        elif aux == 1:
                            self.img_original_2 = orthophoto_matrix
                        else:
                            self.img_original_3 = orthophoto_matrix

        # ==============================================================
        # Extrapolation
        # ==============================================================
        elif self.ui.insp_method == 3:
            if self.sw_extrapolation is False:
                pass
            else:
                # Building coordinates
                lat, lon = self.lat_extrapolation, self.lon_extrapolation
                location = (lat, lon)
                # API key is required; without it, access to GSV is not possible
                try:
                    with open("methods/gsv_api_key.txt") as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    api_key = None
                try:
                    angle = 0
                    url_gsv, img_gsv, _year = get_street_view_image(
                        location, api_key, angle, 5, 120
                    )
                except (
                    AttributeError,
                    ConnectionError,
                    TimeoutError,
                    TypeError,
                    ValueError,
                ) as error:
                    print(f"Street View is not available: {error}")
                    url_gsv = "Street View not available"
                    img_gsv = []

                return img_gsv, url_gsv

    ############ Camara angle setting for better building image perspective ##
    def img_angle_left(self):
        """Open the image angle setting pop-up window."""
        if self.ui.insp_method == 0 or self.ui.insp_method == 1:
            # Image Source
            # 1: Google Street View (Paid API needed)
            # 2: Mapillary (Free API is needed, but it provides less coverage and, in
            # some cases, lower image quality)
            if self.ui.img_source == 1:
                try:
                    # Building coordinates
                    location = (
                        float(self.ui.lat_value.text()),
                        float(self.ui.lon_value.text()),
                    )
                    # API key is required; without it, access to GSV is not possible
                    try:
                        with open("methods/gsv_api_key.txt") as f:
                            api_key = f.read().strip()
                    except FileNotFoundError:
                        api_key = None
                    app = QApplication.instance()  # Ensure PyQt instance exists
                    if app is None:
                        app = QApplication([])

                    # Called function where the GSV image can be adjusted
                    gsv_dialog = gsv_angle_setting(
                        parent=self.ui, main_window=self.ui, gui_methods=self
                    )
                    gsv_dialog.exec_()  # Open the pop-up

                    # Get the feature values provide by the user
                    self.pitch_left = gsv_dialog.pitch_value.value()
                    self.heading_left = gsv_dialog.heading_value.value()
                    self.fov_left = gsv_dialog.fov_value.value()

                    # Left image identifier
                    self.img_original_1 = get_street_view_image(
                        location,
                        api_key,
                        self.heading_left,
                        self.pitch_left,
                        self.fov_left,
                    )[1]
                    self.sw_angle = 0

                    # Prepare new image
                    display_image_rgb = cv2.cvtColor(
                        self.img_original_1, cv2.COLOR_BGR2RGB
                    )
                    h, w, _ch = display_image_rgb.shape
                    bytes_per_line = w * 3
                    qimg = QtGui.QImage(
                        display_image_rgb.data,
                        w,
                        h,
                        bytes_per_line,
                        QtGui.QImage.Format_RGB888,
                    )

                    # Plot new img in the GUI
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    img_frames = [
                        self.ui.left_gsv_img,
                        self.ui.central_gsv_img,
                        self.ui.right_gsv_img,
                    ]
                    img_frames[0].setPixmap(
                        pixmap.scaled(
                            img_frames[0].width(),
                            img_frames[0].height(),
                            QtCore.Qt.IgnoreAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    )

                    # Building detector function
                    self.object_detector_building(None)

                except (
                    AttributeError,
                    FileNotFoundError,
                    IndexError,
                    OSError,
                    TypeError,
                    ValueError,
                    cv2.error,
                ):
                    QMessageBox.warning(
                        self.ui,
                        "Image Error",
                        (
                            "No image is currently displayed. Please click "
                            "*Next Building* to load an image first."
                        ),
                    )
            else:
                app = QApplication.instance()  # Ensure PyQt instance exists
                if app is None:
                    app = QApplication([])
                try:
                    with open("methods/mapillary_api_key.txt") as f:
                        mapillary_access_token = f.read().strip()
                except FileNotFoundError:
                    mapillary_access_token = None
                # Called function where the GSV image can be adjusted
                gsv_dialog = gsv_angle_setting(
                    parent=self.ui, main_window=self.ui, gui_methods=self
                )
                gsv_dialog.exec_()  # Open the pop-up

                # Get the feature values provide by the user
                self.pitch_left = gsv_dialog.pitch_value.value()
                self.heading_left = gsv_dialog.heading_value.value()
                self.fov_left = gsv_dialog.fov_value.value()
                self.color_control = True

                # Left image identifier
                orthophoto_matrix, _info = mapillary_image_source(
                    access_token=mapillary_access_token,
                    latitude=self.data_building.loc[self.click_count, "latitude"],
                    longitude=self.data_building.loc[self.click_count, "longitude"],
                    point_id=self.data_building.loc[self.click_count, "id"],
                    image_number=self.click_count + 1,
                    output_dir=self.ui.output_folder_value + "/Mapillary",
                    return_info=True,
                    # return_color_order="RGB",
                    ortho_fov_deg=self.fov_left,
                    ortho_pitch_deg=-self.pitch_left,
                    ortho_base_yaw_deg=180 + self.heading_left,
                )

                self.img_original_1 = orthophoto_matrix
                self.sw_angle = 0

                # Prepare new image
                display_image_rgb = cv2.cvtColor(self.img_original_1, cv2.COLOR_BGR2RGB)

                img_path = (
                    self.ui.output_folder_value
                    + "/Mapillary/displayed_images/"
                    + str(self.click_count + 1)
                    + ".jpg"
                )
                cv2.imwrite(img_path, display_image_rgb)
                h, w, _ch = display_image_rgb.shape
                bytes_per_line = w * 3
                qimg = QtGui.QImage(
                    display_image_rgb.data,
                    w,
                    h,
                    bytes_per_line,
                    QtGui.QImage.Format_RGB888,
                )

                # Plot new img in the GUI
                pixmap = QtGui.QPixmap.fromImage(qimg)
                img_frames = [
                    self.ui.left_gsv_img,
                    self.ui.central_gsv_img,
                    self.ui.right_gsv_img,
                ]
                img_frames[0].setPixmap(
                    pixmap.scaled(
                        img_frames[0].width(),
                        img_frames[0].height(),
                        QtCore.Qt.IgnoreAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )

                # Building detector function
                self.object_detector_building(None)
        else:
            QMessageBox.warning(
                self.ui,
                "Method Error",
                "This option is not available for local images.",
            )

    ############ Camara angle setting for better building image perspective ##
    def img_angle_central(self):
        """Open the image angle setting pop-up window."""
        if self.ui.insp_method == 0 or self.ui.insp_method == 1:
            if self.ui.img_source == 1:
                try:
                    # Building coordinates
                    location = (
                        float(self.ui.lat_value.text()),
                        float(self.ui.lon_value.text()),
                    )
                    # API key is required; without it, access to GSV is not possible
                    try:
                        with open("methods/gsv_api_key.txt") as f:
                            api_key = f.read().strip()
                    except FileNotFoundError:
                        api_key = None
                    app = QApplication.instance()  # Ensure PyQt instance exists
                    if app is None:
                        app = QApplication([])

                    # Called function where the user creates a manual bounding box by
                    # clicking four points, which is then displayed in the UI frame.
                    gsv_dialog = gsv_angle_setting(
                        parent=self.ui, main_window=self.ui, gui_methods=self
                    )
                    gsv_dialog.exec_()  # Open the pop-up

                    # Get the feature values provide by the user
                    self.pitch_central = gsv_dialog.pitch_value.value()
                    self.heading_central = gsv_dialog.heading_value.value()
                    self.fov_central = gsv_dialog.fov_value.value()

                    self.img_original_2 = get_street_view_image(
                        location,
                        api_key,
                        self.heading_central,
                        self.pitch_central,
                        self.fov_central,
                    )[1]
                    self.sw_angle = 1
                    display_image_rgb = cv2.cvtColor(
                        self.img_original_2, cv2.COLOR_BGR2RGB
                    )
                    h, w, _ch = display_image_rgb.shape
                    bytes_per_line = w * 3
                    qimg = QtGui.QImage(
                        display_image_rgb.data,
                        w,
                        h,
                        bytes_per_line,
                        QtGui.QImage.Format_RGB888,
                    )

                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    img_frames = [
                        self.ui.left_gsv_img,
                        self.ui.central_gsv_img,
                        self.ui.right_gsv_img,
                    ]
                    img_frames[1].setPixmap(
                        pixmap.scaled(
                            img_frames[1].width(),
                            img_frames[1].height(),
                            QtCore.Qt.IgnoreAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    )

                    # Building detector function
                    self.object_detector_building(None)

                except (
                    AttributeError,
                    FileNotFoundError,
                    IndexError,
                    OSError,
                    TypeError,
                    ValueError,
                    cv2.error,
                ):
                    QMessageBox.warning(
                        self.ui,
                        "Image Error",
                        (
                            "No image is currently displayed. Please click "
                            "*Next Building* to load an image first."
                        ),
                    )
            else:
                self.sw_angle = 1
        else:
            QMessageBox.warning(
                self.ui,
                "Method Error",
                "This option is not available for local images.",
            )

    ############ Camara angle setting for better building image perspective ##
    def img_angle_right(self):
        """Open the image angle setting pop-up window."""
        if self.ui.insp_method == 0 or self.ui.insp_method == 1:
            if self.ui.img_source == 1:
                try:
                    # Building coordinates
                    location = (
                        float(self.ui.lat_value.text()),
                        float(self.ui.lon_value.text()),
                    )
                    # API key is required; without it, access to GSV is not possible
                    try:
                        with open("methods/gsv_api_key.txt") as f:
                            api_key = f.read().strip()
                    except FileNotFoundError:
                        api_key = None
                    app = QApplication.instance()  # Ensure PyQt instance exists
                    if app is None:
                        app = QApplication([])

                    # Called function where the user creates a manual bounding box by
                    # clicking four points, which is then displayed in the UI frame.
                    gsv_dialog = gsv_angle_setting(
                        parent=self.ui, main_window=self.ui, gui_methods=self
                    )
                    gsv_dialog.exec_()  # Open the pop-up

                    self.pitch_right = gsv_dialog.pitch_value.value()
                    self.heading_right = gsv_dialog.heading_value.value()
                    self.fov_right = gsv_dialog.fov_value.value()

                    self.img_original_3 = get_street_view_image(
                        location,
                        api_key,
                        self.heading_right,
                        self.pitch_right,
                        self.fov_right,
                    )[1]
                    self.sw_angle = 2
                    display_image_rgb = cv2.cvtColor(
                        self.img_original_3, cv2.COLOR_BGR2RGB
                    )
                    h, w, _ch = display_image_rgb.shape
                    bytes_per_line = w * 3
                    qimg = QtGui.QImage(
                        display_image_rgb.data,
                        w,
                        h,
                        bytes_per_line,
                        QtGui.QImage.Format_RGB888,
                    )

                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    img_frames = [
                        self.ui.left_gsv_img,
                        self.ui.right_gsv_img,
                        self.ui.right_gsv_img,
                    ]
                    img_frames[2].setPixmap(
                        pixmap.scaled(
                            img_frames[2].width(),
                            img_frames[2].height(),
                            QtCore.Qt.IgnoreAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    )

                    # Building detector function
                    self.object_detector_building(None)

                except (
                    AttributeError,
                    FileNotFoundError,
                    IndexError,
                    OSError,
                    TypeError,
                    ValueError,
                    cv2.error,
                ):
                    QMessageBox.warning(
                        self.ui,
                        "Image Error",
                        (
                            "No image is currently displayed. Please click "
                            "*Next Building* to load an image first."
                        ),
                    )
            else:
                self.sw_angle = 2
        else:
            QMessageBox.warning(
                self.ui,
                "Method Error",
                "This option is not available for local images.",
            )

    ############ Building detector model bounding box ################
    def _prepare_display_with_bbox(self, img_bgr, bbox_xyxy, target_w, target_h):
        """Return a resized BGR image with a dashed bounding box.

        Draw the bounding box consistently in display-pixel coordinates.
        """
        x1, y1, x2, y2 = map(int, bbox_xyxy)

        h0, w0 = img_bgr.shape[:2]

        # Resize for the QLabel
        disp_bgr = cv2.resize(
            img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

        # Scale bbox coords to display size
        sx = target_w / float(w0)
        sy = target_h / float(h0)
        dx1 = round(x1 * sx)
        dy1 = round(y1 * sy)
        dx2 = round(x2 * sx)
        dy2 = round(y2 * sy)

        # Consistent style in DISPLAY pixels (tune once)

        short_side = max(1, min(target_w, target_h))
        long_side = max(1, max(target_w, target_h))

        thickness = int(np.clip(round(short_side / 180), 1, 5))
        dash_len = int(np.clip(round(long_side / 90), 8, 45))
        gap_len = int(np.clip(round(long_side / 140) * 2.5, 5, 30))

        self._draw_dashed_rect(
            disp_bgr,
            dx1,
            dy1,
            dx2,
            dy2,
            color=(0, 0, 255),
            thickness=thickness,
            dash_len=dash_len,
            gap_len=gap_len,
        )
        return disp_bgr

    def _draw_dashed_rect(
        self,
        img_bgr,
        x1,
        y1,
        x2,
        y2,
        color=(0, 0, 255),
        thickness=2,
        dash_len=10,
        gap_len=6,
    ):
        """Draw dashed rectangle using 4 dashed lines."""
        # Ensure proper ordering
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))

        # Top & bottom edges
        x = x1
        while x < x2:
            x_end = min(x + dash_len, x2)
            cv2.line(
                img_bgr, (x, y1), (x_end, y1), color, thickness, lineType=cv2.LINE_AA
            )
            cv2.line(
                img_bgr, (x, y2), (x_end, y2), color, thickness, lineType=cv2.LINE_AA
            )
            x = x_end + gap_len

        # Left & right edges
        y = y1
        while y < y2:
            y_end = min(y + dash_len, y2)
            cv2.line(
                img_bgr, (x1, y), (x1, y_end), color, thickness, lineType=cv2.LINE_AA
            )
            cv2.line(
                img_bgr, (x2, y), (x2, y_end), color, thickness, lineType=cv2.LINE_AA
            )
            y = y_end + gap_len

    def object_detector_building(self, aux):  # noqa: C901
        """Detect the target building in the available images.

        Update the image frames, save cropped and displayed images, and support
        polygon, specific-coordinate, local-image, and extrapolation workflows.
        """
        # Class mapping (update this with your actual mappings)
        weight_path = (
            "dl_weights/building_detector.pt"  # Replace with your YOLO .pt file
        )
        # Load the YOLO model
        model = YOLO(weight_path)
        # Classes
        class_names = model.names
        target_class = "building-xzyh"
        # Set device GPU or CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.ui.insp_method in (0, 1, 2):
            ##########################################################################
            # --------------- Polygon method ----------------
            # --------- Specific coordinates method ---------
            # ------------ Local images method --------------
            ##########################################################################
            # Clear old image
            if self.sw_angle == 0:
                self.ui.left_gsv_img.clear()
                n_img = 1
            elif self.sw_angle == 1:
                self.ui.central_gsv_img.clear()
                n_img = 1
            elif self.sw_angle == 2:
                self.ui.right_gsv_img.clear()
                n_img = 1
            else:
                self.ui.left_gsv_img.clear()
                self.ui.central_gsv_img.clear()
                self.ui.right_gsv_img.clear()
                n_img = 3

            # List of the frame
            img_frames = [
                self.ui.left_gsv_img,
                self.ui.central_gsv_img,
                self.ui.right_gsv_img,
            ]
            sw = True

            # ==============================================================
            # Polygon method and Specific coordinates building detector
            # ==============================================================

            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                # Image Source
                # 1: Google Street View (Paid API needed)
                # 2: Mapillary (Free API is needed, but it provides less coverage and,
                # in some cases, lower image quality)
                if self.ui.img_source == 1:
                    # ==============================================================
                    # Google Street View
                    # ==============================================================
                    try:
                        # Getting the images from GSV
                        org_img = [
                            self.img_original_1,
                            self.img_original_2,
                            self.img_original_3,
                        ]
                        # Vector for check if the building is detected
                        self.predicted_img = [0, 0, 0]
                        # Check GSV availability

                        for aux in range(n_img):
                            if sw is True:
                                for i in range(100):
                                    self.ui.progress_bar_method.setValue(i)
                                    QApplication.processEvents()
                                    self.ui.method_progress.setText(
                                        "Isolating building ...."
                                    )
                                sw = False

                            # Ensure the image is in RGB format
                            if self.sw_angle == 0:
                                image_rgb = self.img_original_1
                                self.sw_angle = None
                                aux = 0
                            elif self.sw_angle == 1:
                                image_rgb = self.img_original_2
                                self.sw_angle = None
                                aux = 1
                            elif self.sw_angle == 2:
                                image_rgb = self.img_original_3
                                self.sw_angle = None
                                aux = 2
                            else:
                                image_rgb = org_img[aux]

                            # Run inference
                            results = model.predict(image_rgb, device=device)

                            best_box = None
                            best_score = 0.0

                            # for box in results.boxes:
                            for box in results[0].boxes:
                                cls_id = int(box.cls)
                                cls_name = class_names[cls_id]
                                score = float(box.conf)  # confidence score
                                if (
                                    cls_name == target_class
                                    and score > best_score
                                    and score > 0.5
                                ):
                                    best_score = score
                                    best_box = box

                            if best_box is None:
                                # No building dectection
                                image_rgb = self.add_not_detected_overlay(
                                    image_rgb, opacity=0.5
                                )

                                # Convert BGR image (OpenCV) to RGB format
                                display_image_rgb = cv2.cvtColor(
                                    image_rgb, cv2.COLOR_BGR2RGB
                                )
                                # display_image_rgb = image_rgb.copy()
                                # Convert the RGB image to QImage
                                height, width, _channel = display_image_rgb.shape
                                bytes_per_line = 3 * width
                                qimage = QtGui.QImage(
                                    display_image_rgb.data,
                                    width,
                                    height,
                                    bytes_per_line,
                                    QtGui.QImage.Format_RGB888,
                                )

                                # Convert QImage to QPixmap
                                building_pixmap = QtGui.QPixmap.fromImage(qimage)

                                img_frames[aux].setPixmap(
                                    building_pixmap.scaled(
                                        img_frames[aux].width(),
                                        img_frames[aux].height(),
                                        QtCore.Qt.IgnoreAspectRatio,
                                        QtCore.Qt.SmoothTransformation,
                                    )
                                )  # Ensure high-quality scaling
                                continue

                            # Bounding box coordinates
                            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                            # Crop the area within the selected bounding box
                            self.cropped_image[aux] = org_img[1][y1:y2, x1:x2]
                            self.org_img_bp = org_img[1]

                            # Create image for cropped and displayed
                            display_image = org_img[aux].copy()

                            for i in range(x1, x2, 14):
                                cv2.line(
                                    display_image,
                                    (i, y1),
                                    (min(i + 5, x2), y1),
                                    (0, 0, 255),
                                    3,
                                )
                                cv2.line(
                                    display_image,
                                    (i, y2),
                                    (min(i + 5, x2), y2),
                                    (0, 0, 255),
                                    3,
                                )

                            for i in range(y1, y2, 14):
                                cv2.line(
                                    display_image,
                                    (x1, i),
                                    (x1, min(i + 5, y2)),
                                    (0, 0, 255),
                                    3,
                                )
                                cv2.line(
                                    display_image,
                                    (x2, i),
                                    (x2, min(i + 5, y2)),
                                    (0, 0, 255),
                                    3,
                                )

                            display_image_rgb = cv2.cvtColor(
                                display_image, cv2.COLOR_BGR2RGB
                            )
                            h, w, ch = display_image_rgb.shape
                            bytes_per_line = w * 3
                            qimg = QtGui.QImage(
                                display_image_rgb.data,
                                w,
                                h,
                                bytes_per_line,
                                QtGui.QImage.Format_RGB888,
                            )

                            pixmap = QtGui.QPixmap.fromImage(qimg)
                            self.predicted_img[aux] = 1

                            img_frames[aux].setPixmap(
                                pixmap.scaled(
                                    img_frames[aux].width(),
                                    img_frames[aux].height(),
                                    QtCore.Qt.IgnoreAspectRatio,
                                    QtCore.Qt.SmoothTransformation,
                                )
                            )

                    # There is not GSV image coverage
                    except (
                        AttributeError,
                        IndexError,
                        KeyError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        FileNotFoundError,
                        cv2.error,
                    ):
                        self.no_image = "Street View not available"
                        for aux in range(3):
                            font = QtGui.QFont()
                            font.setPointSize(int(16 * self.sf_font))
                            font.setBold(True)
                            font.setWeight(75)
                            img_frames[aux].setFont(font)
                            img_frames[aux].setText(self.no_image)
                            img_frames[aux].setAlignment(
                                QtCore.Qt.AlignCenter
                            )  # Center-align text

                else:
                    # ==============================================================
                    # Mapillary
                    # ==============================================================
                    # Getting the images from Mapillary
                    org_img = [
                        self.img_original_1,
                        self.img_original_2,
                        self.img_original_3,
                    ]
                    # Vector for check if the building is detected
                    self.predicted_img = [0, 0, 0]

                    # Ensure the image is in RGB format
                    if self.sw_angle == 0:
                        image_rgb = self.img_original_1
                        self.sw_angle = None
                        aux = 0
                    elif self.sw_angle == 1:
                        image_rgb = self.img_original_2
                        self.sw_angle = None
                        aux = 1
                    elif self.sw_angle == 2:
                        image_rgb = self.img_original_3
                        self.sw_angle = None
                        aux = 2
                    else:
                        image_rgb = org_img[aux]

                    # Getting the images from Mapillary
                    org_img = [
                        self.img_original_1,
                        self.img_original_2,
                        self.img_original_3,
                    ]
                    # Vector for check if the building is detected
                    self.predicted_img = [0, 0, 0]
                    # Load the image for drawing

                    img_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/orthophotos/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    displayed_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/displayed_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )

                    # Display building image
                    if os.path.exists(displayed_path):
                        # Display an already isolated image
                        building_pixmap = QtGui.QPixmap(displayed_path)
                        # Selection of image frame using "aux" variable
                        img_frames[aux].setPixmap(
                            building_pixmap.scaled(
                                img_frames[aux].width(),
                                img_frames[aux].height(),
                                QtCore.Qt.IgnoreAspectRatio,
                                QtCore.Qt.SmoothTransformation,
                            )
                        )  # Ensure high-quality scaling
                    else:
                        # Display a new image
                        if sw is True:
                            # Star progress bar until 99%
                            for i in range(100):
                                time.sleep(0.0001)
                                self.ui.progress_bar_method.setValue(i)
                                self.ui.method_progress.setText(
                                    "Isolating building ...."
                                )
                            sw = False
                        # Check and/or create cropped folder
                        if not os.path.exists(
                            self.ui.output_folder_value + "/Mapillary/Cropped_images"
                        ):
                            os.makedirs(
                                self.ui.output_folder_value
                                + "/Mapillary/Cropped_images"
                            )

                        self.gap = None
                        try:
                            # Run inference
                            results = model.predict(img_path, device=device)

                            img_bgr = cv2.imread(img_path)
                            if img_bgr is None:
                                raise RuntimeError(f"Could not read image: {img_path}")

                            _h0, _w0 = img_bgr.shape[:2]

                            best_box = None
                            best_score = 0.0

                            for box in results[0].boxes:
                                cls_id = int(box.cls)
                                cls_name = class_names[cls_id]
                                score = float(box.conf)

                                if (
                                    cls_name == target_class
                                    and score > best_score
                                    and score > 0.5
                                ):
                                    best_score = score
                                    best_box = box

                            # If no building detected, trigger your "overlay" logic
                            if best_box is None:
                                self.gap = 1
                                raise RuntimeError(
                                    "No building detected with confidence > 0.5"
                                )

                            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                            cropped_image = img_bgr[y1:y2, x1:x2]
                            cv2.imwrite(cropped_path, cropped_image)

                            label_w = img_frames[aux].width()
                            label_h = img_frames[aux].height()

                            disp_bgr = self._prepare_display_with_bbox(
                                img_bgr, (x1, y1, x2, y2), label_w, label_h
                            )

                            if not os.path.exists(
                                self.ui.output_folder_value
                                + "/Mapillary/displayed_images"
                            ):
                                os.makedirs(
                                    self.ui.output_folder_value
                                    + "/Mapillary/displayed_images"
                                )

                            cv2.imwrite(displayed_path, disp_bgr)

                            disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
                            h, w, ch = disp_rgb.shape
                            bytes_per_line = ch * w
                            qimage = QtGui.QImage(
                                disp_rgb.data,
                                w,
                                h,
                                bytes_per_line,
                                QtGui.QImage.Format_RGB888,
                            )
                            self.predicted_img[aux] = 1
                            building_pixmap = QtGui.QPixmap.fromImage(qimage)

                        except (
                            AttributeError,
                            FileNotFoundError,
                            IndexError,
                            KeyError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            cv2.error,
                        ):
                            self.no_image = f"""
                                            <b><u>No image found</u></b><br><br>
                                            Please check that the image file exists<br>
                                            at the specified path:<br>
                                            <code>{img_path}</code>
                                            """
                            font = QtGui.QFont()
                            font.setPointSize(int(12 * self.sf_font))
                            font.setBold(True)
                            font.setWeight(75)

                            img_frames[aux].setFont(font)
                            img_frames[aux].setTextFormat(
                                QtCore.Qt.RichText
                            )  # Enable rich text (HTML)
                            img_frames[aux].setText(self.no_image)
                            img_frames[aux].setAlignment(QtCore.Qt.AlignCenter)
                            img_frames[aux].setWordWrap(True)

                        try:
                            # Displayed image in corresponding frames
                            img_frames[aux].setPixmap(
                                building_pixmap.scaled(
                                    img_frames[aux].width(),
                                    img_frames[aux].height(),
                                    QtCore.Qt.IgnoreAspectRatio,
                                    QtCore.Qt.SmoothTransformation,
                                )
                            )  # Ensure high-quality scaling
                        except (AttributeError, IndexError, TypeError):
                            if self.gap is not None:
                                # Displayed image in corresponding frames
                                image_rgb = self.add_not_detected_overlay(
                                    img_bgr, opacity=0.5
                                )

                                # Convert BGR image (OpenCV) to RGB format
                                display_image_rgb = cv2.cvtColor(
                                    image_rgb, cv2.COLOR_BGR2RGB
                                )
                                # display_image_rgb = image_rgb.copy()
                                # Convert the RGB image to QImage
                                height, width, _channel = display_image_rgb.shape
                                bytes_per_line = 3 * width
                                qimage = QtGui.QImage(
                                    display_image_rgb.data,
                                    width,
                                    height,
                                    bytes_per_line,
                                    QtGui.QImage.Format_RGB888,
                                )

                                # Convert QImage to QPixmap
                                building_pixmap = QtGui.QPixmap.fromImage(qimage)

                                img_frames[aux].setPixmap(
                                    building_pixmap.scaled(
                                        img_frames[aux].width(),
                                        img_frames[aux].height(),
                                        QtCore.Qt.IgnoreAspectRatio,
                                        QtCore.Qt.SmoothTransformation,
                                    )
                                )  # Ensure high-quality scaling

            # ==============================================================
            # Local images building detector
            # ==============================================================

            elif self.ui.insp_method == 2:
                # Image frames
                img_frames = [
                    self.ui.left_gsv_img,
                    self.ui.central_gsv_img,
                    self.ui.right_gsv_img,
                ]
                # Loop for the number of image displayed selected with the option in the
                # coordinates pop-up
                for aux in range(self.n_images_local):
                    # Load the image for drawing
                    try:
                        img_path = (
                            self.ui.folder_path
                            + "/"
                            + str(self.data_building.iloc[self.old_local + aux, 0])
                        )
                    except (AttributeError, IndexError, TypeError):
                        QMessageBox.warning(
                            self.ui,
                            "Input Error",
                            "No further inspections are available",
                        )

                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local + aux, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )

                    aux_displayed_path = (
                        self.ui.folder_path
                        + "/displayed_images/"
                        + str(self.data_building.iloc[self.old_local + aux, 0])
                    )
                    displayed_path = (
                        os.path.splitext(aux_displayed_path)[0] + "_displayed.jpg"
                    )

                    # Display building image
                    if os.path.exists(displayed_path):
                        # Display an already isolated image
                        building_pixmap = QtGui.QPixmap(displayed_path)
                        # Selection of image frame using "aux" variable
                        img_frames[aux].setPixmap(
                            building_pixmap.scaled(
                                img_frames[aux].width(),
                                img_frames[aux].height(),
                                QtCore.Qt.IgnoreAspectRatio,
                                QtCore.Qt.SmoothTransformation,
                            )
                        )  # Ensure high-quality scaling
                    else:
                        # Display a new image
                        if sw is True:
                            # Star progress bar until 99%
                            for i in range(100):
                                time.sleep(0.0001)
                                self.ui.progress_bar_method.setValue(i)
                                self.ui.method_progress.setText(
                                    "Isolating building ...."
                                )
                            sw = False
                        # Check and/or create cropped folder
                        if not os.path.exists(self.ui.folder_path + "/Cropped_images"):
                            os.makedirs(self.ui.folder_path + "/Cropped_images")

                        self.gap = None
                        try:
                            # Run inference
                            results = model.predict(img_path, device=device)

                            img_bgr = cv2.imread(img_path)
                            if img_bgr is None:
                                raise RuntimeError(f"Could not read image: {img_path}")

                            _h0, _w0 = img_bgr.shape[:2]

                            best_box = None
                            best_score = 0.0

                            for box in results[0].boxes:
                                cls_id = int(box.cls)
                                cls_name = class_names[cls_id]
                                score = float(box.conf)

                                if (
                                    cls_name == target_class
                                    and score > best_score
                                    and score > 0.5
                                ):
                                    best_score = score
                                    best_box = box

                            # If no building detected, trigger your "overlay" logic
                            if best_box is None:
                                self.gap = 1
                                raise RuntimeError(
                                    "No building detected with confidence > 0.5"
                                )

                            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                            cropped_image = img_bgr[y1:y2, x1:x2]
                            cv2.imwrite(cropped_path, cropped_image)

                            label_w = img_frames[aux].width()
                            label_h = img_frames[aux].height()

                            disp_bgr = self._prepare_display_with_bbox(
                                img_bgr, (x1, y1, x2, y2), label_w, label_h
                            )

                            if not os.path.exists(
                                self.ui.folder_path + "/displayed_images"
                            ):
                                os.makedirs(self.ui.folder_path + "/displayed_images")

                            cv2.imwrite(displayed_path, disp_bgr)

                            disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
                            h, w, ch = disp_rgb.shape
                            bytes_per_line = ch * w
                            qimage = QtGui.QImage(
                                disp_rgb.data,
                                w,
                                h,
                                bytes_per_line,
                                QtGui.QImage.Format_RGB888,
                            )

                            building_pixmap = QtGui.QPixmap.fromImage(qimage)

                        except (
                            AttributeError,
                            FileNotFoundError,
                            IndexError,
                            KeyError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            cv2.error,
                        ):
                            self.no_image = f"""
                                            <b><u>No image found</u></b><br><br>
                                            Please check that the image file exists<br>
                                            at the specified path:<br>
                                            <code>{img_path}</code>
                                            """
                            font = QtGui.QFont()
                            font.setPointSize(int(12 * self.sf_font))
                            font.setBold(True)
                            font.setWeight(75)

                            img_frames[aux].setFont(font)
                            img_frames[aux].setTextFormat(
                                QtCore.Qt.RichText
                            )  # Enable rich text (HTML)
                            img_frames[aux].setText(self.no_image)
                            img_frames[aux].setAlignment(QtCore.Qt.AlignCenter)
                            img_frames[aux].setWordWrap(True)

                        try:
                            # Displayed image in corresponding frames
                            img_frames[aux].setPixmap(
                                building_pixmap.scaled(
                                    img_frames[aux].width(),
                                    img_frames[aux].height(),
                                    QtCore.Qt.IgnoreAspectRatio,
                                    QtCore.Qt.SmoothTransformation,
                                )
                            )  # Ensure high-quality scaling
                        except (
                                AttributeError,
                                IndexError,
                                TypeError,
                                UnboundLocalError
                                ):
                            if self.gap is not None:
                                # Displayed image in corresponding frames
                                image_rgb = self.add_not_detected_overlay(
                                    img_bgr, opacity=0.5
                                )

                                # Convert BGR image (OpenCV) to RGB format
                                display_image_rgb = cv2.cvtColor(
                                    image_rgb, cv2.COLOR_BGR2RGB
                                )
                                # display_image_rgb = image_rgb.copy()
                                # Convert the RGB image to QImage
                                height, width, _channel = display_image_rgb.shape
                                bytes_per_line = 3 * width
                                qimage = QtGui.QImage(
                                    display_image_rgb.data,
                                    width,
                                    height,
                                    bytes_per_line,
                                    QtGui.QImage.Format_RGB888,
                                )

                                # Convert QImage to QPixmap
                                building_pixmap = QtGui.QPixmap.fromImage(qimage)

                                img_frames[aux].setPixmap(
                                    building_pixmap.scaled(
                                        img_frames[aux].width(),
                                        img_frames[aux].height(),
                                        QtCore.Qt.IgnoreAspectRatio,
                                        QtCore.Qt.SmoothTransformation,
                                    )
                                )  # Ensure high-quality scaling

            # Chance progress bar to complete
            self.ui.progress_bar_method.setValue(100)
            self.ui.method_progress.setText("Done!")

        else:
            ##########################################################################
            ########################## --------- Extrapolation method ---------------#
            ##########################################################################
            if self.sw_extrapolation is False:
                self.sw_extrapolation = True
            else:
                conf_threshold = 0.5
                self.lat_extrapolation = float(
                    self.ui.coord_reference.loc[aux, "latitude"]
                )
                self.lon_extrapolation = float(
                    self.ui.coord_reference.loc[aux, "longitude"]
                )
                img_gsv, url_gsv = self.fetch_three_step_views()
                try:
                    # Run inference
                    results = model.predict(img_gsv, device=device)[0]
                    h, w, _ = img_gsv.shape
                    # Get class names
                    class_names = model.names
                    best_box = None
                    best_conf = 0

                    # Loop through detected boxes
                    if results.boxes is not None:
                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            label = class_names[cls_id]
                            conf = float(box.conf[0])

                            if (
                                label == target_class
                                and conf > conf_threshold
                                and conf > best_conf
                            ):
                                best_conf = conf
                                best_box = box.xyxy[0].cpu().numpy().astype(int)

                    if best_box is None:
                        print(f"❌ No '{target_class}' detected in image.")
                        return

                    x1, y1, x2, y2 = best_box

                    # ✅ Ensure values inside image
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    # ✅ Crop image
                    cropped_image = img_gsv[y1:y2, x1:x2]
                    return cropped_image, url_gsv
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    cropped_image = []

    ############ Opacity function ################
    def add_not_detected_overlay(
        self,
        image_bgr,
        opacity=0.5,
        text="BUILDING NOT DETECTED",
        output_size=(490, 310),
    ):
        """Resize the image and draw a consistent detection overlay.

        Display ``BUILDING NOT DETECTED`` over the resized image.

        Parameters
        ----------
        image_bgr : np.ndarray
            Input image in BGR format.
        opacity : float
            Opacity of the original image over the white background.
        text : str
            Message to display.
        output_size : tuple
            Fixed output size as (width, height). Default: (490, 310).

        Returns
        -------
        blended : np.ndarray
            Output image with fixed size and consistent overlay.
        """
        if image_bgr is None:
            raise ValueError("Image is None in add_not_detected_overlay")

        # Fixed display size
        out_w, out_h = output_size

        # Resize image FIRST so all overlay elements are drawn
        # in the same coordinate system
        resized = cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # White background
        background = np.full_like(resized, 255)

        # Blend original image with white background
        blended = cv2.addWeighted(resized, opacity, background, 1 - opacity, 0)

        # Fixed text settings for the fixed display size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.85
        thickness = 2
        padding_x = 12
        padding_y = 10

        # Measure text
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Center text
        x = (out_w - text_w) // 2
        y = (out_h + text_h) // 2

        # Rectangle coordinates
        top_left = (x - padding_x, y - text_h - padding_y)
        bottom_right = (x + text_w + padding_x, y + baseline + padding_y)

        # Draw white rectangle
        cv2.rectangle(blended, top_left, bottom_right, (255, 255, 255), -1)

        # Draw red text
        cv2.putText(
            blended,
            text,
            (x, y),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )

        return blended

    ############ Left Bounding Box Manual Selection ################
    def bounding_box_frame_left(self):
        """Select the left frame image and retrieve its path.

        Use a Google Street View image for polygon and specific-coordinate
        workflows. Use a local image when the local-image workflow is active.
        """
        # Highlight the image used for AI-prediction
        self.ui.left_gsv_bb.setContentsMargins(8, 8, 8, 8)
        self.ui.central_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.ui.right_gsv_bb.setContentsMargins(0, 0, 0, 0)
        try:
            # Getting the image depending of the inspection mode selected.
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                # From GSV (Polygon and Specific method)
                self.image_bb = self.img_original_1
            elif self.ui.insp_method == 2:
                # From local device (original)
                self.image_bb = (
                    self.ui.folder_path
                    + "/"
                    + str(self.data_building.iloc[self.old_local, 0])
                )
            # Left Frame to display
            self.frame_bb_disp = self.ui.left_gsv_img

            # Getting the path for image prediction
            if self.ui.insp_method == 2:
                try:
                    # Auxiliary cropped-image path
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )

                    # Left cropped image
                    self.cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )

                except (AttributeError, IndexError, TypeError):
                    QMessageBox.warning(
                        self.ui,
                        "File Error",
                        (
                            "This option is only available when a previous "
                            "building detection exists."
                        ),
                    )
            else:
                self.cropped_path = None

            # Bounding box ID for prediction models
            self.box_id = 0
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
        ):
            pass

    ############ Central Bounding Box Manual Selection ################
    def bounding_box_frame_central(self):
        """Select the central frame image and retrieve its path.

        Use a Google Street View image for polygon and specific-coordinate
        workflows. Use a local image when the local-image workflow is active.
        """
        # Highlight the image used for AI-prediction
        self.ui.left_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.ui.central_gsv_bb.setContentsMargins(8, 8, 8, 8)
        self.ui.right_gsv_bb.setContentsMargins(0, 0, 0, 0)
        try:
            # Getting the image depending of the inspection mode selected.
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                # From GSV (Polygon and Specific method)
                self.image_bb = self.img_original_2
            elif self.ui.insp_method == 2:
                # From local device (original)
                self.image_bb = (
                    self.ui.folder_path
                    + "/"
                    + str(self.data_building.iloc[self.old_local + 1, 0])
                )
            # Central Frame to display
            self.frame_bb_disp = self.ui.central_gsv_img

            # Getting the path for image prediction
            if self.ui.insp_method == 2:
                try:
                    # Cropped image path
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local + 1, 0])
                    )
                    self.cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                except (AttributeError, IndexError, TypeError):
                    QMessageBox.warning(
                        self.ui,
                        "File Error",
                        (
                            "This option is only available if there is a previous "
                            "building detection."
                        ),
                    )
            else:
                self.cropped_path = None

            # Bounding box ID for prediction models
            self.box_id = 1
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
        ):
            pass

    ############ Right Bounding Box Manual Selection ################
    def bounding_box_frame_right(self):
        """Select the right frame image and retrieve its path.

        Use a Google Street View image for polygon and specific-coordinate
        workflows. Use a local image when the local-image workflow is active.
        """
        # Highlight the image used for AI-prediction
        self.ui.left_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.ui.central_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.ui.right_gsv_bb.setContentsMargins(8, 8, 8, 8)
        try:
            # Getting the image depending of the inspection mode selected.
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                # From GSV (Polygon and Specific method)
                self.image_bb = self.img_original_3
            elif self.ui.insp_method == 2:
                # From local device (original)
                self.image_bb = (
                    self.ui.folder_path
                    + "/"
                    + str(self.data_building.iloc[self.old_local + 2, 0])
                )

            self.frame_bb_disp = self.ui.right_gsv_img
            # Getting the path for image prediction
            if self.ui.insp_method == 2:
                try:
                    # Cropped image path
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local + 2, 0])
                    )
                    self.cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                except (AttributeError, IndexError, TypeError):
                    QMessageBox.warning(
                        self.ui,
                        "File Error",
                        (
                            "This option is only available if there is a previous "
                            "building detection."
                        ),
                    )
            else:
                self.cropped_path = None
            # Bounding box ID for prediction models
            self.box_id = 2
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
        ):
            pass

    ############ Lists of features ################
    def feature_comboboxes_values(self):
        """Define the class labels used by the deep-learning models.

        Store the human-readable class labels associated with each predicted
        building attribute, including material, lateral load-resisting system,
        code level, number of stories, occupancy, block position, roof shape,
        and roof material.
        """
        # Material
        self.class_mat = ["Concrete", "Masonry - Confined", "Masonry - Unreinforced"]
        # LLRS
        self.class_llrs = [
            "Dual System",
            "Infilled Frames",
            "Moment Frames",
            "Walls",
            "Walls",
        ]
        # Code level
        self.class_code = ["High-Code", "Low-Code", "Moderate-code", "No-Code"]
        # Number of stories
        self.class_ns = ["10-12", "13+", "1", "2", "3", "4", "5", "6-7", "8-9"]
        # Occupancy
        self.class_occ = [
            "Commercial",
            "Industrial",
            "Mixed (Residential + Commercial)",
            "Residential",
        ]
        # Block position
        self.class_bp = [
            "Adjoining building(s) one side",
            "Adjoining building(s) two side",
            "Adjoining building(s) three side",
            "Detached building",
        ]
        # Roof Shape
        self.class_r_shape = [
            "Flat",
            "Pitched with gable ends",
            "Pitched and hipped",
            "Curved",
        ]
        # Roof material
        self.class_r_mat = [
            "Concrete",
            "Clay or concrete tile",
            "Metal or asbestos sheets",
        ]

    ############ Manual bounding box Selection ################
    def bounding_box(self):
        """Define a building bounding box manually.

        When AI-powered mode is enabled, run the classification models,
        update the interface, and apply engineering-based adjustments.
        """
        # Conditional to avoid executing the method if there is no project folder
        if (
            self.ui.output_folder_value == "-"
            or self.ui.country_value.text() == "-"
            or self.ui.city_value.text() == "-"
        ):
            QMessageBox.warning(
                self.ui,
                "File Error",
                "This option is only available once the building image is displayed.",
            )
        else:
            """Open the bounding box selection pop-up window."""
            app = QApplication.instance()  # Ensure PyQt instance exists
            if app is None:
                app = QApplication([])

            # Called function where the user creates a manual bounding box by clicking
            # four points, which is then displayed in the UI frame.
            dialog = BoundingBoxWindow(
                self.image_bb,
                self.frame_bb_disp,
                self.ui.insp_method,
                self.cropped_path,
                parent=self.ui,
                main_window=self.ui,
                gui_methods=self,
            )

            dialog.exec_()  # Open the pop-up

            try:
                # Image path
                image_file = dialog.prediction_img

                if self.ui.ai_check.isChecked():
                    self.feature_comboboxes_values()
                    # LLRS building image sets prediction
                    material_index = predict_material_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.material_cb_1.setCurrentText(self.class_mat[material_index])

                    # LLRS building image prediction
                    llrs_index = predict_llrs_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.llrs_cb_1.setCurrentText(self.class_llrs[llrs_index])

                    # LLRS building image prediction
                    code_level_index = predict_code_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.age_cb_1.setCurrentText(self.class_code[code_level_index])

                    # LLRS building image prediction
                    n_stories_index = predict_n_stories_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.n_stories_value_1.setCurrentText(
                        self.class_ns[n_stories_index]
                    )

                    # LLRS building image prediction
                    occupancy_index = predict_occupancy_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.occup_cb_1.setCurrentText(self.class_occ[occupancy_index])

                    # block_position building image prediction
                    block_position_index = predict_block_position_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.bck_pos_cb_1.setCurrentText(
                        self.class_bp[block_position_index]
                    )

                    # roof_shape building image prediction
                    roof_shape_index = predict_roof_shape_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.roof_shape_cb_1.setCurrentText(
                        self.class_r_shape[roof_shape_index]
                    )

                    # roof_material building image prediction
                    roof_material_index = predict_roof_material_img(
                        image_file, self.ui.insp_method, self.box_id, self
                    )
                    self.ui.roof_material_cb_1.setCurrentText(
                        self.class_r_mat[roof_material_index]
                    )

                    # Taxonomy adjustments
                    self.pred_mat_value = self.ui.material_cb_1.currentData()
                    self.llrs_pred = self.ui.llrs_cb_1.currentData()
                    self.pred_roof_shape = self.ui.roof_shape_cb_1.currentData()
                    self.roof_mat_pred = self.ui.roof_material_cb_1.currentData()
                    self.code_level_pred = self.ui.age_cb_1.currentData()

                    # LLRS adjusments based on material
                    if self.pred_mat_value == "MCF" or self.pred_mat_value == "MUR":
                        self.ui.llrs_cb_1.setCurrentText(self.class_llrs[4])

                    # Roof material adjument based on roof shape
                    if self.pred_roof_shape == "RSH1":
                        self.ui.roof_material_cb_1.setCurrentText(self.class_r_mat[0])
                    elif self.pred_roof_shape == "RSH7":
                        self.ui.roof_material_cb_1.setCurrentText(self.class_r_mat[2])
                    elif self.pred_roof_shape == "RSH2":
                        if self.roof_mat_pred in ("RMT1", "RMT6"):
                            pass
                        else:
                            # This depends on the country
                            self.ui.roof_material_cb_1.setCurrentText(
                                self.class_r_mat[2]
                            )
                    elif self.pred_roof_shape == "RSH3":
                        if self.roof_mat_pred in ("RMT1", "RMT6"):
                            pass
                        else:
                            self.ui.roof_material_cb_1.setCurrentText(
                                self.class_r_mat[1]
                            )

                    # Code level adjustment based on material
                    if self.pred_mat_value == "MUR":
                        if self.code_level_pred in ("CDL", "CDN"):
                            pass
                        else:
                            self.ui.age_cb_1.setCurrentText(self.class_code[3])

                    # For conditional within dl models
                    self.box_id = None

                # Message with special format
                message = """
                When using the AI-powered option, creating a manual bounding
                box triggers <b><u>A NEW PREDICTION</u></b>. Verify that the
                current labels
                correspond to the true labels.
                """

                # Create a QMessageBox instance
                msg_box = QMessageBox(self.ui)
                msg_box.setTextFormat(QtCore.Qt.RichText)
                msg_box.setText(message)
                msg_box.setWindowTitle("AI-powered inspection form")

                # Show the message box
                msg_box.exec_()

            except (
                AttributeError,
                FileNotFoundError,
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                pass

    ############ Taxonomy check ################
    def tax_check(self, tax_value):
        """Validate a taxonomy string.

        Report the canonical suggestion extracted from the validation error.
        """
        # 1) Create a small DataFrame with taxonomy strings
        df = pd.DataFrame({"TAXONOMY": [tax_value]})
        try:
            check_taxonomy(df, taxo_col="TAXONOMY")
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
    def inspection_database(self):  # noqa: C901
        """Store the current workflow results in the inspection database.

        Save the location, assigned attributes, taxonomy, and image reference
        for all supported inspection workflows.
        """
        ##########################################################################
        # --------------- Polygon method ----------------
        # --------- Specific coordinates method ---------
        # ------------ Local images method --------------
        ##########################################################################
        if self.ui.insp_method in (0, 1, 2):
            # Check the stage to avoid trying to use the feature button before all the
            # information is properly set up
            if self.ui.city_value.text() == "-":
                QMessageBox.warning(
                    self.ui,
                    "File Error",
                    "This option is only available once the building image "
                    "is displayed.\n"
                    "Please click the *Next Building* button.",
                )
            else:
                # ==============================================================
                # Polygon and Specific coordinates method
                # ==============================================================
                if self.ui.insp_method in (0, 1) and self.ui.img_source == 1:
                    base_url = (
                        "https://www.google.com/maps/@?api=1&map_action=pano&viewpoint="
                    )
                    coord = (
                        str(self.ui.lat_value.text())
                        + ","
                        + str(self.ui.lon_value.text())
                    )
                    heading = get_road_orientation(
                        (
                            float(self.ui.lat_value.text()),
                            float(self.ui.lon_value.text()),
                        )
                    )

                if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                    self.data_ai.iloc[self.click_count, 0] = (
                        self.ui.img_id_value_1.text()[:-2]
                    )
                    self.data_ai.iloc[self.click_count, 1] = self.data_building.loc[
                        self.click_count, "latitude"
                    ]
                    self.data_ai.iloc[self.click_count, 2] = self.data_building.loc[
                        self.click_count, "longitude"
                    ]
                    self.data_ai.iloc[self.click_count, 3] = (
                        self.ui.country_value.text()
                    )
                    self.data_ai.iloc[self.click_count, 4] = self.ui.city_value.text()
                    self.data_ai.iloc[self.click_count, 5] = (
                        self.ui.material_cb_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 6] = (
                        self.ui.llrs_cb_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 7] = (
                        self.ui.age_cb_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 8] = (
                        self.ui.n_stories_value_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 9] = (
                        self.ui.occup_cb_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 10] = (
                        self.ui.bck_pos_cb_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 11] = (
                        self.ui.epc_const_cb_1.currentText()
                    )
                    self.data_ai.iloc[self.click_count, 12] = (
                        self.ui.roof_shape_cb_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 13] = (
                        self.ui.roof_material_cb_1.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 14] = (
                        self.ui.irregularity_cb.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 15] = (
                        self.ui.n_bay_cb.currentData()
                    )
                    self.data_ai.iloc[self.click_count, 16] = (
                        self.ui.img_q_cb_1.currentData()
                    )

                    # Taxonomy (safe + partial)
                    def _s(v):
                        return "" if v is None else str(v).strip()

                    def _add(out, v):
                        v = _s(v)
                        if v:
                            out.append(v)

                    parts = []
                    _add(parts, self.ui.material_cb_1.currentData())
                    _add(parts, self.ui.llrs_cb_1.currentData())
                    _add(parts, self.ui.age_cb_1.currentData())
                    st = _s(self.ui.n_stories_value_1.currentText())
                    if st:
                        parts.append(f"H:{st}")
                    _add(parts, self.ui.bck_pos_cb_1.currentData())

                    roof_shape = _s(self.ui.roof_shape_cb_1.currentData())
                    roof_mat = _s(self.ui.roof_material_cb_1.currentData())
                    if roof_shape and roof_mat:
                        parts.append(f"{roof_shape}+{roof_mat}")
                    elif roof_shape:
                        parts.append(roof_shape)
                    elif roof_mat:
                        parts.append(roof_mat)

                    _add(parts, self.ui.occup_cb_1.currentData())

                    tax = "/".join(parts)
                    self.data_ai.iloc[self.click_count, 17] = tax
                    if tax:
                        self.tax_check(tax)

                    if self.img_url[0] != "":
                        self.data_ai.iloc[self.click_count, 18] = self.img_url[0]
                    else:
                        if isinstance(heading, int):
                            self.data_ai.iloc[self.click_count, 18] = (
                                base_url
                                + coord
                                + "&heading="
                                + str((heading + 180) % 360)
                                + "&pitch=5&fov=120"
                            )

                # ==============================================================
                # Local images
                # ==============================================================
                elif self.ui.insp_method == 2:
                    # Left building image
                    self.data_ai.iloc[self.old_local, 0] = (
                        self.ui.img_id_value_1.text()[:-2]
                    )  # ID
                    self.data_ai.iloc[self.old_local, 1] = self.data_building.loc[
                        self.old_local, "latitude"
                    ]  # latitude
                    self.data_ai.iloc[self.old_local, 2] = self.data_building.loc[
                        self.old_local, "longitude"
                    ]  # longitude
                    self.data_ai.iloc[self.old_local, 3] = (
                        self.ui.country_value.text()
                    )  # Country
                    self.data_ai.iloc[self.old_local, 4] = (
                        self.ui.city_value.text()
                    )  # City
                    self.data_ai.iloc[self.old_local, 5] = (
                        self.ui.material_cb_1.currentData()
                    )  # LLRS Material
                    self.data_ai.iloc[self.old_local, 6] = (
                        self.ui.llrs_cb_1.currentData()
                    )  # LLRS
                    self.data_ai.iloc[self.old_local, 7] = (
                        self.ui.age_cb_1.currentData()
                    )  # Code Level
                    self.data_ai.iloc[self.old_local, 8] = (
                        self.ui.n_stories_value_1.currentData()
                    )  # Number of Stories
                    self.data_ai.iloc[self.old_local, 9] = (
                        self.ui.occup_cb_1.currentData()
                    )  # Occupancy
                    self.data_ai.iloc[self.old_local, 10] = (
                        self.ui.bck_pos_cb_1.currentData()
                    )  # Block Position
                    self.data_ai.iloc[self.old_local, 11] = (
                        self.ui.epc_const_cb_1.currentText()
                    )  # Epoch of construction
                    self.data_ai.iloc[self.old_local, 12] = (
                        self.ui.roof_shape_cb_1.currentData()
                    )  # Roof shape
                    self.data_ai.iloc[self.old_local, 13] = (
                        self.ui.roof_material_cb_1.currentData()
                    )  # Roof material
                    self.data_ai.iloc[self.old_local, 14] = (
                        self.ui.irregularity_cb.currentData()
                    )  # Vertical irregularity
                    self.data_ai.iloc[self.old_local, 15] = (
                        self.ui.n_bay_cb.currentData()
                    )  # Vertical irregularity
                    self.data_ai.iloc[self.old_local, 16] = (
                        self.ui.img_q_cb_1.currentData()
                    )  # Image Quality

                    # Taxonomy (works with missing fields)
                    def _s(v):
                        return "" if v is None else str(v).strip()

                    parts = []

                    for v in (
                        _s(self.ui.material_cb_1.currentData()),
                        _s(self.ui.llrs_cb_1.currentData()),
                        _s(self.ui.age_cb_1.currentData()),
                        f"H:{_s(self.ui.n_stories_value_1.currentText())}"
                        if _s(self.ui.n_stories_value_1.currentText())
                        else "",
                        _s(self.ui.bck_pos_cb_1.currentData()),
                    ):
                        if v:
                            parts.append(v)

                    roof_shape = _s(self.ui.roof_shape_cb_1.currentData())
                    roof_mat = _s(self.ui.roof_material_cb_1.currentData())
                    if roof_shape and roof_mat:
                        parts.append(f"{roof_shape}+{roof_mat}")
                    elif roof_shape:
                        parts.append(roof_shape)
                    elif roof_mat:
                        parts.append(roof_mat)

                    occup = _s(self.ui.occup_cb_1.currentData())
                    if occup:
                        parts.append(occup)

                    tax = "/".join(parts)
                    self.data_ai.iloc[self.old_local, 17] = tax  # Taxonomy
                    if tax:
                        self.tax_check(tax)

                    self.data_ai.iloc[self.old_local, 18] = self.data_building.iloc[
                        self.old_local, 0
                    ]

        ##########################################################################
        # --------------- Extrapolation -----------------
        ##########################################################################
        elif self.ui.insp_method == 3:
            for i in range(self.data_ai.shape[0]):
                self.data_ai.iloc[i, 0] = self.ui.coord_reference.loc[i, "id"]  # ID
                self.data_ai.iloc[i, 1] = self.ui.coord_reference.loc[
                    i, "latitude"
                ]  # Latitude
                self.data_ai.iloc[i, 2] = self.ui.coord_reference.loc[i, "longitude"]
                try:
                    image_file, url_gsv = self.object_detector_building(i)
                    if image_file is None:
                        pass
                    else:
                        city, country = self.get_city_name()

                        material_classes = ["CR", "MCF", "MUR"]
                        llrs_classes = ["LDUAL", "LFINF", "LFM", "LWAL", "LWAL"]
                        code_level_classes = ["CDH", "CDL", "CDM", "CDN"]
                        ns_classes = [
                            "10-12",
                            "13+",
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6-7",
                            "8-9",
                        ]
                        occupancy_class = ["COM", "IND", "MIX(RES;COM)", "RES"]
                        block_position_classes = ["BP1", "BP2", "BP3", "BPD"]
                        roof_shape_classes = ["RSH1", "RSH2", "RSH3", "RSH7"]
                        roof_material_classes = ["RMN", "RMT1", "RMT6"]

                        self.data_ai.iloc[i, 3], self.data_ai.iloc[i, 4] = country, city
                        self.data_ai.iloc[i, 5] = material_classes[
                            predict_material_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]  # LLRS Material
                        self.data_ai.iloc[i, 6] = llrs_classes[
                            predict_llrs_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]  # LLRS
                        self.data_ai.iloc[i, 7] = code_level_classes[
                            predict_code_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]  # Code Level
                        self.data_ai.iloc[i, 8] = ns_classes[
                            predict_n_stories_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]  # Number of Stories
                        self.data_ai.iloc[i, 9] = occupancy_class[
                            predict_occupancy_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]
                        self.data_ai.iloc[i, 10] = block_position_classes[
                            predict_block_position_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]  # Block Position
                        self.data_ai.iloc[i, 12] = roof_shape_classes[
                            predict_roof_shape_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]  # Roof shape
                        self.data_ai.iloc[i, 13] = roof_material_classes[
                            predict_roof_material_img(
                                image_file, self.ui.insp_method, None, self.ui
                            )
                        ]

                        self.data_ai.iloc[i, 16] = (
                            self.data_ai.iloc[i, 5]
                            + "/"
                            + self.data_ai.iloc[i, 6]
                            + "+"
                            + self.data_ai.iloc[i, 7]
                            + "/H:"
                            + str(self.data_ai.iloc[i, 8])
                            + "/"
                            + self.data_ai.iloc[i, 9]
                            + "/"
                            + self.data_ai.iloc[i, 10]
                            + "/"
                            + self.data_ai.iloc[i, 12]
                            + "+"
                            + self.data_ai.iloc[i, 13]
                        )  # Taxonomy

                        self.data_ai.iloc[i, 17] = url_gsv
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    print(f"Could not process building {i}")

                print(
                    "Inspection: "
                    + str(i + 1)
                    + "/"
                    + str(self.data_ai.shape[0])
                    + " -------------------------------------"
                )

    ############ Saves the data from the inspections that were conducted #####
    def save_database(self):
        """Save the inspection database to CSV files.

        Export auxiliary and filtered classification results, update progress
        indicators, and resets the internal flags related to saved inspections.
        """
        # Load path
        output_folder = self.ui.output_folder_value

        # Create CSV with new inspections
        self.inspection_database()

        # Star progress bar
        self.ui.method_progress.setText("Saving inspections ...")
        for j in range(101):
            time.sleep(0.0001)
            self.ui.progress_bar_method.setValue(j)

        # ==============================================================
        # Polygon method
        # ==============================================================
        if self.ui.insp_method == 0:
            try:
                # Save the AI inspection data to a CSV file
                img_prefix = f"{output_folder}/{self.ui.file_name}"
                self.data_ai.to_csv(img_prefix + "_AI_aux_cont.csv", index=False)
                final_df = self.data_ai
                filtered_df = final_df[
                    final_df["n_stories"].notna() | final_df["llrs"].notna()
                ]
                filtered_df.to_csv(img_prefix + "_AI_classification.csv", index=False)
                # Update the progress message in the GUI
                self.ui.method_progress.setText("Inspections exported successfully!")
            except (OSError, KeyError, AttributeError, TypeError, ValueError):
                # Show a warning message box if there's a permission error
                QMessageBox.warning(
                    self.ui,
                    "File Error",
                    "The file is open or the folder is inaccessible."
                    + " Please close the file or check folder permissions.",
                )
        # ==============================================================
        # Specific coordinates
        # ==============================================================
        elif self.ui.insp_method == 1:
            try:
                # Save the AI inspection data to a CSV file
                self.data_ai.to_csv(
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name
                    + "_AI_aux_cont.csv",
                    index=False,
                )
                final_df = self.data_ai
                filtered_df = final_df[
                    final_df["n_stories"].notna() | final_df["llrs"].notna()
                ]
                filtered_df.to_csv(
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name
                    + "_AI_classification.csv",
                    index=False,
                )
                # Update the progress message in the GUI
                self.ui.method_progress.setText("Inspections exported successfully!")
            except (OSError, KeyError, AttributeError, TypeError, ValueError):
                # Show a warning message box if there's a permission error
                QMessageBox.warning(
                    self.ui,
                    "File Error",
                    "The file is open or the folder is inaccessible."
                    + "Please close the file or check folder permissions.",
                )
        # ==============================================================
        # Local images
        # ==============================================================
        elif self.ui.insp_method == 2:
            # Save the AI inspection data to a CSV file
            try:
                self.data_ai.to_csv(
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name_local.text()
                    + "_AI_aux_cont.csv",
                    index=False,
                )
                final_df = self.data_ai
                filtered_df = final_df[
                    final_df["n_stories"].notna() | final_df["llrs"].notna()
                ]
                filtered_def = filtered_df.drop_duplicates(subset="id", keep="first")
                filtered_def = filtered_def.drop_duplicates(
                    subset="image filename or link", keep="first"
                )
                filtered_def.to_csv(
                    self.ui.output_folder_value
                    + "/"
                    + self.ui.file_name_local.text()
                    + "_AI_classification.csv",
                    index=False,
                )
                # Update the progress message in the GUI
                self.ui.method_progress.setText("Inspections exported successfully!")
            except (OSError, KeyError, AttributeError, TypeError, ValueError):
                # Show a warning message box if there's a permission error
                QMessageBox.warning(
                    self.ui,
                    "File Error",
                    "The file is open or the folder is inaccessible. "
                    + "Please close the file or check folder permissions.",
                )

            # Replicate inspections
            classification_df = filtered_def
            coordinates_df = self.data_building

            lat_col = "latitude"
            lon_col = "longitude"
            filename_col = "image filename or link"
            data_image_id_col = "id"

            # Check required columns
            required_classification_cols = {lat_col, lon_col, filename_col}
            required_coordinates_cols = {data_image_id_col, lat_col, lon_col}

            missing_classification = required_classification_cols - set(
                classification_df.columns
            )
            missing_coordinates = required_coordinates_cols - set(
                coordinates_df.columns
            )

            if missing_classification:
                raise ValueError(
                    "Missing columns in classification dataframe: "
                    f"{sorted(missing_classification)}"
                )

            if missing_coordinates:
                raise ValueError(
                    "Missing columns in coordinates dataframe: "
                    f"{sorted(missing_coordinates)}"
                )

            # Keep original order so the output follows the input classification order
            # and, within each classification row, the order in data_ex2.csv.
            classification_df = classification_df.copy()
            coordinates_df = coordinates_df.copy()

            classification_df["__classification_order"] = range(len(classification_df))
            coordinates_df["__coordinates_order"] = range(len(coordinates_df))

            # ------------------------------------------------------------
            # Convert coordinate columns to numeric before rounding.
            # This prevents: TypeError: Expected numeric dtype, got object instead.
            # ------------------------------------------------------------
            for df_name, df in {
                "classification_df": classification_df,
                "coordinates_df": coordinates_df,
            }.items():
                for col in [lat_col, lon_col]:
                    # Convert possible string coordinates to numeric values.
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                    # Check if any coordinate could not be converted.
                    if df[col].isna().any():
                        invalid_rows = df[df[col].isna()]

                        raise ValueError(
                            f"Invalid or missing values in column '{col}' "
                            f"of {df_name}. Please check these rows:\n{invalid_rows}"
                        )

            # Create rounded coordinate keys for robust matching.
            classification_df["__lat_key"] = classification_df[lat_col].round(8)
            classification_df["__lon_key"] = classification_df[lon_col].round(8)

            coordinates_df["__lat_key"] = coordinates_df[lat_col].round(8)
            coordinates_df["__lon_key"] = coordinates_df[lon_col].round(8)

            # Rename data_ex2.csv id column to avoid conflict with the classification
            # id.
            coordinates_df = coordinates_df.rename(
                columns={data_image_id_col: "__matched_filename"}
            )

            # Merge: one classification row is repeated for every coordinate match.
            expanded_df = classification_df.merge(
                coordinates_df[
                    [
                        "__matched_filename",
                        "__lat_key",
                        "__lon_key",
                        "__coordinates_order",
                    ]
                ],
                on=["__lat_key", "__lon_key"],
                how="inner",
            )

            # Replace the image filename/link with the matching filename from
            # data_ex2.csv.
            expanded_df[filename_col] = expanded_df["__matched_filename"]

            # Restore stable order.
            expanded_df = expanded_df.sort_values(
                by=["__classification_order", "__coordinates_order"],
                kind="stable",
            )

            # Remove helper columns and preserve the original classification CSV
            # columns.
            original_classification_cols = classification_df.drop(
                columns=[
                    "__classification_order",
                    "__lat_key",
                    "__lon_key",
                ]
            ).columns

            expanded_df = expanded_df[original_classification_cols]

            # Save result.
            output_csv = (
                self.ui.output_folder_value
                + "/"
                + self.ui.file_name_local.text()
                + "_All_images.csv"
            )

            expanded_df.to_csv(output_csv, index=False)

        # restart varibles
        self.data_old = "OK"  # TO BE SAVED THERE IS EXISTING DATA
        self.sw_insp = True
        self.save_id = True
        self.start = True

    def setComboBoxByData(self, combo_box, data):  # noqa: N802
        """Select the combobox item matching the given data value."""
        # Iterate through the comboBox items to find the one with matching data
        match_data = [""]
        for i in range(combo_box.count()):
            if combo_box.itemData(i) == data:
                match_data = combo_box.setCurrentIndex(i)
                break
        return match_data

    ############ Restart default value of each building feature ################
    def clean_database(self):  # noqa: C901
        """Restore inspection fields from the saved database.

        Reset empty values and repopulate the corresponding
        comboboxes for polygon, specific coordinates, or local image workflows.
        """
        if self.ui.insp_method == 0 or self.ui.insp_method == 1:
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            # Material
            if (
                self.data_ai.iloc[self.click_count, 5] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 5]) is True
            ):
                self.ui.material_cb_1.setCurrentText("Select Material")
            else:
                self.setComboBoxByData(
                    self.ui.material_cb_1, self.data_ai.iloc[self.click_count, 5]
                )

            # LLRS
            if (
                self.data_ai.iloc[self.click_count, 6] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 6]) is True
            ):
                self.ui.llrs_cb_1.setCurrentText("Select LLRS")
            else:
                self.setComboBoxByData(
                    self.ui.llrs_cb_1, self.data_ai.iloc[self.click_count, 6]
                )

            # Code level
            if (
                self.data_ai.iloc[self.click_count, 7] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 7]) is True
            ):
                self.ui.age_cb_1.setCurrentText("Select Code Level")
            else:
                self.setComboBoxByData(
                    self.ui.age_cb_1, self.data_ai.iloc[self.click_count, 7]
                )

            # Number of stories
            if (
                self.data_ai.iloc[self.click_count, 8] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 8]) is True
            ):
                self.ui.n_stories_value_1.setCurrentText("Select Number of Stories")
            else:
                n_value = self.data_ai.iloc[self.click_count, 8]
                if n_value == "1.0" or n_value == 1.0:
                    n_value = "1"
                elif n_value == "2.0" or n_value == 2.0:
                    n_value = "2"
                elif n_value == "3.0" or n_value == 3.0:
                    n_value = "3"
                elif n_value == "4.0" or n_value == 4.0:
                    n_value = "4"
                elif n_value == "5.0" or n_value == 5.0:
                    n_value = "5"
                self.setComboBoxByData(self.ui.n_stories_value_1, n_value)

            # Occupancy
            if (
                self.data_ai.iloc[self.click_count, 9] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 9]) is True
            ):
                self.ui.occup_cb_1.setCurrentText("Select Occupancy Type")
            else:
                self.setComboBoxByData(
                    self.ui.occup_cb_1, self.data_ai.iloc[self.click_count, 9]
                )

            # Block Position
            if (
                self.data_ai.iloc[self.click_count, 10] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 10]) is True
            ):
                self.ui.bck_pos_cb_1.setCurrentText("Select Block Position")
            else:
                self.setComboBoxByData(
                    self.ui.bck_pos_cb_1, self.data_ai.iloc[self.click_count, 10]
                )

            # Epoch of construction
            if (
                self.data_ai.iloc[self.click_count, 11] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 11]) is True
            ):
                self.ui.epc_const_cb_1.setCurrentIndex(0)
            else:
                self.setComboBoxByData(
                    self.ui.epc_const_cb_1, self.data_ai.iloc[self.click_count, 11]
                )

            # Roof Shape
            if (
                self.data_ai.iloc[self.click_count, 12] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 12]) is True
            ):
                self.ui.roof_shape_cb_1.setCurrentText("Select Roof Shape")
            else:
                self.setComboBoxByData(
                    self.ui.roof_shape_cb_1, self.data_ai.iloc[self.click_count, 12]
                )

            # Roof Material
            if (
                self.data_ai.iloc[self.click_count, 13] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 13]) is True
            ):
                self.ui.roof_material_cb_1.setCurrentText("Select Roof Material")
            else:
                self.setComboBoxByData(
                    self.ui.roof_material_cb_1, self.data_ai.iloc[self.click_count, 13]
                )

            # Vertical irregularity
            if (
                self.data_ai.iloc[self.click_count, 14] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 14]) is True
            ):
                self.ui.irregularity_cb.setCurrentText("Select Irregularity")
            else:
                self.setComboBoxByData(
                    self.ui.irregularity_cb, self.data_ai.iloc[self.click_count, 14]
                )

            # Number of bays
            if (
                self.data_ai.iloc[self.click_count, 15] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 15]) is True
            ):
                self.ui.n_bay_cb.setCurrentText("Select Number of Bays")
            else:
                self.setComboBoxByData(
                    self.ui.n_bay_cb, self.data_ai.iloc[self.click_count, 15]
                )

            # Image quality
            if (
                self.data_ai.iloc[self.click_count, 16] is None
                or pd.isna(self.data_ai.iloc[self.click_count, 16]) is True
            ):
                self.ui.img_q_cb_1.setCurrentText("Select Image Quality")
            else:
                self.setComboBoxByData(
                    self.ui.img_q_cb_1, self.data_ai.iloc[self.click_count, 16]
                )

        elif self.ui.insp_method == 2:
            # ==============================================================
            # Local images
            # ==============================================================
            # Material
            if (
                self.data_ai.iloc[self.old_local, 5] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 5]) is True
            ):
                self.ui.material_cb_1.setCurrentText("Select Material")
            else:
                self.setComboBoxByData(
                    self.ui.material_cb_1, self.data_ai.iloc[self.old_local, 5]
                )

            # LLRS
            if (
                self.data_ai.iloc[self.old_local, 6] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 6]) is True
            ):
                self.ui.llrs_cb_1.setCurrentText("Select LLRS")
            else:
                self.setComboBoxByData(
                    self.ui.llrs_cb_1, self.data_ai.iloc[self.old_local, 6]
                )

            # Code level
            if (
                self.data_ai.iloc[self.old_local, 7] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 7]) is True
            ):
                self.ui.age_cb_1.setCurrentText("Select Code Level")
            else:
                self.setComboBoxByData(
                    self.ui.age_cb_1, self.data_ai.iloc[self.old_local, 7]
                )

            # Number of stories
            if (
                self.data_ai.iloc[self.old_local, 8] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 8]) is True
            ):
                self.ui.n_stories_value_1.setCurrentText("Select Number of Stories")
            else:
                n_value = str(self.data_ai.iloc[self.old_local, 8])
                if n_value == "1.0" or n_value == 1.0:
                    n_value = "1"
                elif n_value == "2.0" or n_value == 2.0:
                    n_value = "2"
                elif n_value == "3.0" or n_value == 3.0:
                    n_value = "3"
                elif n_value == "4.0" or n_value == 4.0:
                    n_value = "4"
                elif n_value == "5.0" or n_value == 5.0:
                    n_value = "5"
                self.setComboBoxByData(self.ui.n_stories_value_1, n_value)

            # Occupancy
            if (
                self.data_ai.iloc[self.old_local, 9] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 9]) is True
            ):
                self.ui.occup_cb_1.setCurrentText("Select Occupancy Type")
            else:
                self.setComboBoxByData(
                    self.ui.occup_cb_1, self.data_ai.iloc[self.old_local, 9]
                )

            # Block Position
            if (
                self.data_ai.iloc[self.old_local, 10] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 10]) is True
            ):
                self.ui.bck_pos_cb_1.setCurrentText("Select Block Position")
            else:
                self.setComboBoxByData(
                    self.ui.bck_pos_cb_1, self.data_ai.iloc[self.old_local, 10]
                )

            # Epoch of construction
            if (
                self.data_ai.iloc[self.old_local, 11] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 11]) is True
            ):
                self.ui.epc_const_cb_1.setCurrentIndex(0)
            else:
                self.setComboBoxByData(
                    self.ui.epc_const_cb_1, str(self.data_ai.iloc[self.old_local, 11])
                )

            # Roof Shape
            if (
                self.data_ai.iloc[self.old_local, 12] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 12]) is True
            ):
                self.ui.roof_shape_cb_1.setCurrentText("Select Roof Shape")
            else:
                self.setComboBoxByData(
                    self.ui.roof_shape_cb_1, self.data_ai.iloc[self.old_local, 12]
                )

            # Roof Material
            if (
                self.data_ai.iloc[self.old_local, 13] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 13]) is True
            ):
                self.ui.roof_material_cb_1.setCurrentText("Select Roof Material")
            else:
                self.setComboBoxByData(
                    self.ui.roof_material_cb_1, self.data_ai.iloc[self.old_local, 13]
                )

            # Image quality
            if (
                self.data_ai.iloc[self.old_local, 14] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 14]) is True
            ):
                self.ui.irregularity_cb.setCurrentText("Select Irregularity")
            else:
                self.setComboBoxByData(
                    self.ui.irregularity_cb, self.data_ai.iloc[self.old_local, 14]
                )

            # Image quality
            if (
                self.data_ai.iloc[self.old_local, 15] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 15]) is True
            ):
                self.ui.n_bay_cb.setCurrentText("Select Number of Bays")
            else:
                self.setComboBoxByData(
                    self.ui.n_bay_cb, self.data_ai.iloc[self.old_local, 15]
                )

            # Image quality
            if (
                self.data_ai.iloc[self.old_local, 16] is None
                or pd.isna(self.data_ai.iloc[self.old_local, 16]) is True
            ):
                self.ui.img_q_cb_1.setCurrentText("Select Image Quality")
            else:
                self.setComboBoxByData(
                    self.ui.img_q_cb_1, self.data_ai.iloc[self.old_local, 16]
                )

    ############ Deep learning model for predict the LLRS Material ################
    def material_prediction(self):  # noqa: C901
        """Run material classification on the selected building image.

        Update the material field and refresh the progress indicators
        for polygon, specific coordinates, or local image workflows.
        """
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # Create list of feature
            self.feature_comboboxes_values()
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                            # Highlight the image used for AI-prediction
                            self.ui.left_gsv_bb.setContentsMargins(0, 0, 0, 0)
                            self.ui.central_gsv_bb.setContentsMargins(8, 8, 8, 8)
                            self.ui.right_gsv_bb.setContentsMargins(0, 0, 0, 0)
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                            # Highlight the image used for AI-prediction
                            self.ui.left_gsv_bb.setContentsMargins(8, 8, 8, 8)
                            self.ui.central_gsv_bb.setContentsMargins(0, 0, 0, 0)
                            self.ui.right_gsv_bb.setContentsMargins(0, 0, 0, 0)
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2
                            # Highlight the image used for AI-prediction
                            self.ui.left_gsv_bb.setContentsMargins(0, 0, 0, 0)
                            self.ui.central_gsv_bb.setContentsMargins(0, 0, 0, 0)
                            self.ui.right_gsv_bb.setContentsMargins(8, 8, 8, 8)

                    if pred_img is True:
                        image_file = self.cropped_image[j]
                        box_aux = None
                        material_index = predict_material_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )

                        # Set DL model prediction
                        if material_index is None:
                            pass
                        else:
                            self.ui.material_cb_1.setCurrentText(
                                self.class_mat[material_index]
                            )
                            self.pred_mat_value = self.ui.material_cb_1.currentData()

                            # Progress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    material_index = predict_material_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # LLRS building image sets prediction
                    if material_index is None:
                        pass
                    else:
                        self.ui.material_cb_1.setCurrentText(
                            self.class_mat[material_index]
                        )
                        self.pred_mat_value = self.ui.material_cb_1.currentData()
                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            # ==============================================================
            # Local images
            # ==============================================================
            elif self.ui.insp_method == 2:
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                # Local cropped image path
                try:
                    aux_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = os.path.splitext(aux_path)[0] + "_cropped.jpg"
                    # Highlight the image used for AI-prediction
                    self.ui.left_gsv_bb.setContentsMargins(8, 8, 8, 8)
                    self.ui.central_gsv_bb.setContentsMargins(0, 0, 0, 0)
                    self.ui.right_gsv_bb.setContentsMargins(0, 0, 0, 0)
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                # LLRS building image prediction
                material_index = predict_material_img(
                    cropped_path, self.ui.insp_method, self.box_id, self.ui
                )
                # LLRS building image sets prediction
                if material_index is None:
                    pass
                else:
                    self.ui.material_cb_1.setCurrentText(self.class_mat[material_index])
                    self.pred_mat_value = self.ui.material_cb_1.currentData()
                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Deep learning model for predict the LLRS ################
    def llrs_prediction(self):  # noqa: C901
        """Run LLRS classification on the selected building image.

        Update the LLRS field and refresh the progress indicators
        for polygon, specific coordinates, or local image workflows.
        """
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # Create list of feature
            self.feature_comboboxes_values()
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2
                    if pred_img is True:
                        # Image path
                        image_file = self.cropped_image[j]
                        # LLRS building image prediction
                        box_aux = None
                        llrs_index = predict_llrs_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )
                        # LLRS building image sets prediction
                        if llrs_index is None:
                            pass
                        else:
                            self.ui.llrs_cb_1.setCurrentText(
                                self.class_llrs[llrs_index]
                            )
                            self.llrs_pred = self.ui.llrs_cb_1.currentData()

                            # LLRS adjusments based on material
                            if (
                                self.pred_mat_value == "MCF"
                                or self.pred_mat_value == "MUR"
                            ):
                                self.ui.llrs_cb_1.setCurrentText(self.class_llrs[4])

                            if self.pred_mat_value == "CR":
                                if self.llrs_pred in ("LDUAL", "LFM", "LFINF"):
                                    pass
                                else:
                                    self.ui.llrs_cb_1.setCurrentText(self.class_llrs[2])
                            # Progress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    llrs_index = predict_llrs_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # LLRS building image sets prediction
                    if llrs_index is None:
                        pass
                    else:
                        self.ui.llrs_cb_1.setCurrentText(self.class_llrs[llrs_index])
                        self.llrs_pred = self.ui.llrs_cb_1.currentData()

                        # LLRS adjusments based on material
                        if self.pred_mat_value == "MCF" or self.pred_mat_value == "MUR":
                            self.ui.llrs_cb_1.setCurrentText(self.class_llrs[4])

                        if self.pred_mat_value == "CR":
                            if self.llrs_pred in ("LDUAL", "LFM", "LFINF"):
                                pass
                            else:
                                self.ui.llrs_cb_1.setCurrentText(self.class_llrs[2])

                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            elif self.ui.insp_method == 2:
                # ==============================================================
                # Local images
                # ==============================================================
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                try:
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                # LLRS building image prediction
                llrs_index = predict_llrs_img(
                    cropped_path, self.ui.insp_method, self.box_id, self.ui
                )
                # LLRS building image sets prediction
                if llrs_index is None:
                    pass
                else:
                    self.ui.llrs_cb_1.setCurrentText(self.class_llrs[llrs_index])
                    self.llrs_pred = self.ui.llrs_cb_1.currentData()

                    # LLRS adjusments based on material
                    if self.pred_mat_value == "MCF" or self.pred_mat_value == "MUR":
                        self.ui.llrs_cb_1.setCurrentText(self.class_llrs[4])

                    if self.pred_mat_value == "CR":
                        if self.llrs_pred in ("LDUAL", "LFM", "LFINF"):
                            pass
                        else:
                            self.ui.llrs_cb_1.setCurrentText(self.class_llrs[2])

                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Deep learning model for predict the Code level ################
    def code_level_prediction(self):  # noqa: C901
        """Run code-level classification on the selected building image.

        Update the code-level field and refresh the progress indicators
        for polygon, specific coordinates, or local image workflows.
        """
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2
                    if pred_img is True:
                        # Image path
                        image_file = self.cropped_image[j]

                        # LLRS building image prediction
                        box_aux = None
                        code_level_index = predict_code_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )

                        # LLRS building image sets prediction
                        if code_level_index is None:
                            pass
                        else:
                            self.ui.age_cb_1.setCurrentText(
                                self.class_code[code_level_index]
                            )
                            self.code_level_pred = self.ui.age_cb_1.currentData()

                            # Code level adjustment based on material
                            if self.pred_mat_value == "MUR":
                                if self.code_level_pred in ("CDL", "CDN"):
                                    pass
                                else:
                                    self.ui.age_cb_1.setCurrentText(self.class_code[3])

                            # Progress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    code_level_index = predict_code_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # LLRS building image sets prediction
                    if code_level_index is None:
                        pass
                    else:
                        self.ui.age_cb_1.setCurrentText(
                            self.class_code[code_level_index]
                        )
                        self.code_level_pred = self.ui.age_cb_1.currentData()

                        # Code level adjustment based on material
                        if self.pred_mat_value == "MUR":
                            if self.code_level_pred in ("CDL", "CDN"):
                                pass
                            else:
                                self.ui.age_cb_1.setCurrentText(self.class_code[3])

                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            elif self.ui.insp_method == 2:
                # ==============================================================
                # Local images
                # ==============================================================
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                try:
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                # LLRS building image prediction
                code_level_index = predict_code_img(
                    cropped_path, self.ui.insp_method, self.box_id, self.ui
                )
                # LLRS building image sets prediction
                if code_level_index is None:
                    pass
                else:
                    self.ui.age_cb_1.setCurrentText(self.class_code[code_level_index])
                    self.code_level_pred = self.ui.age_cb_1.currentData()

                    # Code level adjustment based on material
                    if self.pred_mat_value == "MUR":
                        if self.code_level_pred in ("CDL", "CDN"):
                            pass
                        else:
                            self.ui.age_cb_1.setCurrentText(self.class_code[3])

                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Deep learning model for predict the Number of Stories ################
    def n_stories_prediction(self):  # noqa: C901
        """Predict the number of stories for the selected building image."""
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2
                    if pred_img is True:
                        # Image path
                        image_file = self.cropped_image[j]

                        # LLRS building image prediction
                        box_aux = None
                        n_stories_index = predict_n_stories_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )

                        if n_stories_index is None:
                            pass
                        else:
                            self.ui.n_stories_value_1.setCurrentText(
                                self.class_ns[n_stories_index]
                            )

                            # Progress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    # LLRS building image prediction
                    n_stories_index = predict_n_stories_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # LLRS building image sets prediction
                    if n_stories_index is None:
                        pass
                    else:
                        self.ui.n_stories_value_1.setCurrentText(
                            self.class_ns[n_stories_index]
                        )

                        # Progress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            elif self.ui.insp_method == 2:
                # ==============================================================
                # Local images
                # ==============================================================
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                try:
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                # LLRS building image prediction
                n_stories_index = predict_n_stories_img(
                    cropped_path, self.ui.insp_method, self.box_id, self.ui
                )
                # LLRS building image sets prediction
                if n_stories_index is None:
                    pass
                else:
                    self.ui.n_stories_value_1.setCurrentText(
                        self.class_ns[n_stories_index]
                    )

                    # Progress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Deep learning model for predict the Occupancy type ################
    def occupancy_prediction(self):  # noqa: C901
        """Predict the occupancy type for the selected building image."""
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2
                    if pred_img is True:
                        # Image path
                        image_file = self.cropped_image[j]
                        # LLRS building image prediction
                        box_aux = None
                        occupancy_index = predict_occupancy_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )

                        # LLRS building image sets prediction
                        if occupancy_index is None:
                            pass
                        else:
                            self.ui.occup_cb_1.setCurrentText(
                                self.class_occ[occupancy_index]
                            )

                            self.pred_occ_value = self.ui.occup_cb_1.currentData()
                            # Material and LLRS adjustment based on occupancy
                            if self.pred_occ_value == "IND":
                                if self.pred_mat_value in ("CR"):
                                    pass
                                else:
                                    self.ui.material_cb_1.setCurrentText(self.class_mat[0])
                                    self.ui.llrs_cb_1.setCurrentText(self.class_llrs[2])

                            # Progress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    occupancy_index = predict_occupancy_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # LLRS building image sets prediction
                    if occupancy_index is None:
                        pass
                    else:
                        self.ui.occup_cb_1.setCurrentText(
                            self.class_occ[occupancy_index]
                        )

                        self.pred_occ_value = self.ui.occup_cb_1.currentData()
                        # Material and LLRS adjustment based on occupancy
                        if self.pred_occ_value == "IND":
                            if self.pred_mat_value in ("CR"):
                                pass
                            else:
                                self.ui.material_cb_1.setCurrentText(self.class_mat[0])
                                self.ui.llrs_cb_1.setCurrentText(self.class_llrs[2])

                        # Progress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            elif self.ui.insp_method == 2:
                # ==============================================================
                # Local images
                # ==============================================================
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                try:
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                # LLRS building image prediction
                occupancy_index = predict_occupancy_img(
                    cropped_path, self.ui.insp_method, self.box_id, self.ui
                )
                # LLRS building image sets prediction
                if occupancy_index is None:
                    pass
                else:
                    self.ui.occup_cb_1.setCurrentText(self.class_occ[occupancy_index])

                    self.pred_occ_value = self.ui.occup_cb_1.currentData()
                    # Material and LLRS adjustment based on occupancy
                    if self.pred_occ_value == "IND":
                        if self.pred_mat_value in ("CR"):
                            pass
                        else:
                            self.ui.material_cb_1.setCurrentText(self.class_mat[0])
                            self.ui.llrs_cb_1.setCurrentText(self.class_llrs[2])

                    # Progress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Deep learning model for predict the block_position ################
    def block_position_prediction(self):  # noqa: C901
        """Predict the block position for the selected building image."""
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    # ==============================================================
                    # Polygon and Specific coordinates
                    # ==============================================================
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2
                    if pred_img is True:
                        # Image path
                        # image_file = self.cropped_image[i]
                        image_file = self.org_img_bp
                        # block_position building image prediction
                        box_aux = None
                        block_position_index = predict_block_position_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )
                        # block_position building image sets prediction
                        if block_position_index is None:
                            pass
                        else:
                            self.ui.bck_pos_cb_1.setCurrentText(
                                self.class_bp[block_position_index]
                            )
                            # Progress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/orthophotos/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    block_position_index = predict_block_position_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # block_position building image sets prediction
                    if block_position_index is None:
                        pass
                    else:
                        self.ui.bck_pos_cb_1.setCurrentText(
                            self.class_bp[block_position_index]
                        )

                        # Progress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            elif self.ui.insp_method == 2:
                # ==============================================================
                # Local images
                # ==============================================================
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                try:
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                org_path = (
                    self.ui.folder_path
                    + "/"
                    + str(self.data_building.iloc[self.old_local, 0])
                )
                # block_position building image prediction
                block_position_index = predict_block_position_img(
                    org_path, self.ui.insp_method, self.box_id, self.ui
                )
                # block_position building image sets prediction
                if block_position_index is None:
                    pass
                else:
                    self.ui.bck_pos_cb_1.setCurrentText(
                        self.class_bp[block_position_index]
                    )

                    # Progress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Deep learning model for predict the Roof shape ################
    def roof_shape_prediction(self):  # noqa: C901
        """Predict the roof shape for the selected building image."""
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2

                    if pred_img is True:
                        # Image path
                        image_file = self.cropped_image[j]
                        # roof_shape building image prediction
                        box_aux = None
                        roof_shape_index = predict_roof_shape_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )
                        # roof_shape building image sets prediction
                        if roof_shape_index is None:
                            pass
                        else:
                            self.ui.roof_shape_cb_1.setCurrentText(
                                self.class_r_shape[roof_shape_index]
                            )
                            self.pred_roof_shape = self.ui.roof_shape_cb_1.currentData()
                            # Progress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    roof_shape_index = predict_roof_shape_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # roof_shape building image sets prediction
                    if roof_shape_index is None:
                        pass
                    else:
                        self.ui.roof_shape_cb_1.setCurrentText(
                            self.class_r_shape[roof_shape_index]
                        )
                        self.pred_roof_shape = self.ui.roof_shape_cb_1.currentData()
                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            elif self.ui.insp_method == 2:
                # ==============================================================
                # Local images
                # ==============================================================
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                try:
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                # roof_shape building image prediction
                roof_shape_index = predict_roof_shape_img(
                    cropped_path, self.ui.insp_method, self.box_id, self.ui
                )

                # roof_shape building image sets prediction
                if roof_shape_index is None:
                    pass
                else:
                    self.ui.roof_shape_cb_1.setCurrentText(
                        self.class_r_shape[roof_shape_index]
                    )
                    self.pred_roof_shape = self.ui.roof_shape_cb_1.currentData()
                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Deep learning model for predict the Roof shape ################
    def roof_material_prediction(self):  # noqa: C901
        """Predict the roof material for the selected building image."""
        # Checkbox for the AI powered activation
        if self.ui.ai_check.isChecked():
            # ==============================================================
            # Polygon and Specific coordinates
            # ==============================================================
            if self.ui.insp_method == 0 or self.ui.insp_method == 1:
                if self.ui.img_source == 1:
                    # Image for prediction: central image by default, followed by the
                    # left image, and finally the right image
                    pred_img = False
                    for _aux_img in range(3):
                        if self.predicted_img[1] == 1:
                            # Central image
                            pred_img = True
                            j = 1
                        elif self.predicted_img[0] == 1:
                            # Left image
                            pred_img = True
                            j = 0
                        elif self.predicted_img[2] == 1:
                            # Right image
                            pred_img = True
                            j = 2
                    if pred_img is True:
                        # Image path
                        image_file = self.cropped_image[j]
                        # roof_material building image prediction
                        box_aux = None
                        roof_material_index = predict_roof_material_img(
                            image_file, self.ui.insp_method, box_aux, self.ui
                        )
                        # roof_material building image sets prediction
                        if roof_material_index is None:
                            pass
                        else:
                            self.ui.roof_shape_cb_1.setCurrentText(
                                self.class_r_mat[roof_material_index]
                            )
                            self.roof_mat_pred = (
                                self.ui.roof_material_cb_1.currentData()
                            )

                            # Roof material adjument based on roof shape
                            if self.pred_roof_shape == "RSH1":
                                self.ui.roof_material_cb_1.setCurrentText(
                                    self.class_r_mat[0]
                                )
                            elif self.pred_roof_shape == "RSH7":
                                self.ui.roof_material_cb_1.setCurrentText(
                                    self.class_r_mat[2]
                                )
                            elif self.pred_roof_shape == "RSH2":
                                if self.roof_mat_pred in ("RMT1", "RMT6"):
                                    pass
                                else:
                                    # This depends on the country
                                    self.ui.roof_material_cb_1.setCurrentText(
                                        self.class_r_mat[2]
                                    )
                            elif self.pred_roof_shape == "RSH3":
                                if self.roof_mat_pred in ("RMT1", "RMT6"):
                                    pass
                                else:
                                    self.ui.roof_material_cb_1.setCurrentText(
                                        self.class_r_mat[1]
                                    )
                            # Peogress bar update
                            self.ui.progress_bar_method.setValue(100)
                            self.ui.method_progress.setText("Prediction complete!")
                else:
                    self.ui.method_progress.setText("Loading AI model ...")
                    for j in range(100):
                        time.sleep(0.0001)
                        self.ui.progress_bar_method.setValue(j)

                    cropped_path = (
                        self.ui.output_folder_value
                        + "/Mapillary/cropped_images/"
                        + str(self.click_count + 1)
                        + ".jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")

                    roof_material_index = predict_roof_material_img(
                        image, self.ui.insp_method, self.box_id, self.ui
                    )
                    # roof_material building image sets prediction
                    if roof_material_index is None:
                        pass
                    else:
                        self.ui.roof_shape_cb_1.setCurrentText(
                            self.class_r_mat[roof_material_index]
                        )
                        self.roof_mat_pred = self.ui.roof_material_cb_1.currentData()

                        # Roof material adjument based on roof shape
                        if self.pred_roof_shape == "RSH1":
                            self.ui.roof_material_cb_1.setCurrentText(
                                self.class_r_mat[0]
                            )
                        elif self.pred_roof_shape == "RSH7":
                            self.ui.roof_material_cb_1.setCurrentText(
                                self.class_r_mat[2]
                            )
                        elif self.pred_roof_shape == "RSH2":
                            if self.roof_mat_pred in ("RMT1", "RMT6"):
                                pass
                            else:
                                # This depends on the country
                                self.ui.roof_material_cb_1.setCurrentText(
                                    self.class_r_mat[2]
                                )
                        elif self.pred_roof_shape == "RSH3":
                            if self.roof_mat_pred in ("RMT1", "RMT6"):
                                pass
                            else:
                                self.ui.roof_material_cb_1.setCurrentText(
                                    self.class_r_mat[1]
                                )

                        # Peogress bar update
                        self.ui.progress_bar_method.setValue(100)
                        self.ui.method_progress.setText("Prediction complete!")

            elif self.ui.insp_method == 2:
                self.ui.method_progress.setText("Loading AI model ...")
                for j in range(100):
                    time.sleep(0.0001)
                    self.ui.progress_bar_method.setValue(j)

                try:
                    aux_cropped_path = (
                        self.ui.folder_path
                        + "/cropped_images/"
                        + str(self.data_building.iloc[self.old_local, 0])
                    )
                    cropped_path = (
                        os.path.splitext(aux_cropped_path)[0] + "_cropped.jpg"
                    )
                    image = cv2.imread(cropped_path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError("Unable to read image")
                except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
                    print(f"Error building path: {e}")
                    cropped_path = None

                # roof_material building image prediction
                roof_material_index = predict_roof_material_img(
                    cropped_path, self.ui.insp_method, self.box_id, self.ui
                )
                # roof_material building image sets prediction
                if roof_material_index is None:
                    pass
                else:
                    self.ui.roof_shape_cb_1.setCurrentText(
                        self.class_r_mat[roof_material_index]
                    )
                    self.roof_mat_pred = self.ui.roof_material_cb_1.currentData()

                    # Roof material adjument based on roof shape
                    if self.pred_roof_shape == "RSH1":
                        self.ui.roof_material_cb_1.setCurrentText(self.class_r_mat[0])
                    elif self.pred_roof_shape == "RSH7":
                        self.ui.roof_material_cb_1.setCurrentText(self.class_r_mat[2])
                    elif self.pred_roof_shape == "RSH2":
                        if self.roof_mat_pred in ("RMT1", "RMT6"):
                            pass
                        else:
                            # This depends on the country
                            self.ui.roof_material_cb_1.setCurrentText(
                                self.class_r_mat[2]
                            )
                    elif self.pred_roof_shape == "RSH3":
                        if self.roof_mat_pred in ("RMT1", "RMT6"):
                            pass
                        else:
                            self.ui.roof_material_cb_1.setCurrentText(
                                self.class_r_mat[1]
                            )

                    # Peogress bar update
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Prediction complete!")

    ############ Search and load existing inspections ################
    def search_inspection(self):
        """Search for and load a saved inspection by image ID."""
        # Clean the image frame
        self.ui.left_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.ui.central_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.ui.right_gsv_bb.setContentsMargins(0, 0, 0, 0)
        search_value = self.ui.search_img_value.text()
        # Check if the value is not empty
        if not search_value.strip():
            self.ui.search_img_value.setText("Please enter a value to search.")
            return

        # Search in the DataFrame
        try:
            try:
                result = self.data_ai[self.data_ai["id"] == int(search_value)]
                if result.empty:
                    result = self.data_ai[self.data_ai["id"] == search_value]
            except (ValueError, TypeError):
                result = self.data_ai[self.data_ai["id"] == search_value]
                if result.empty:
                    result = self.data_ai[self.data_ai["id"] == int(search_value)]

            try:
                n_building = result.iloc[0, 0]
            except IndexError:
                self.data_ai.to_csv(
                    self.ui.output_folder_value + "/" + "search_aux.csv", index=False
                )
                self.data_ai = pd.read_csv(
                    self.ui.output_folder_value + "/" + "search_aux.csv"
                )
                try:
                    result = self.data_ai[self.data_ai["id"] == int(search_value)]
                    if result.empty:
                        result = self.data_ai[self.data_ai["id"] == search_value]
                except (ValueError, TypeError):
                    result = self.data_ai[self.data_ai["id"] == search_value]
                    if result.empty:
                        result = self.data_ai[self.data_ai["id"] == int(search_value)]
                # Remove auxiliar file
                os.remove(self.ui.output_folder_value + "/" + "search_aux.csv")

            n_building = result.iloc[0, 0]

            # Get image ID
            try:
                if self.ui.insp_method in (0, 1, 2):
                    self.click_count = int(n_building) - 1

            except (TypeError, ValueError):
                pass

            try:
                self.get_city_name()
                self.fetch_three_step_views()
                self.object_detector_building(None)
                self.clean_database()
            except (
                AttributeError,
                ConnectionError,
                IndexError,
                KeyError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ):
                pass

            self.search_count = True

        except (
            AttributeError,
            IndexError,
            KeyError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            QMessageBox.warning(
                self.ui, "Data Error", "There is no saved inspection for this Image ID."
            )

    ############ Extrapolation functions ################
    def neighbor_extrapolation(self):
        """Run the selected building-feature extrapolation workflow."""
        if self.ui.insp_method == 3:
            if self.ui.extrapolation_mode == 2:
                #####################################
                ######## KNN mode ###################
                #####################################
                if self.ui.use_coord_reference is True:
                    #######===========  Function results =========###########
                    self.create_database()
                    data_existing_dl = self.data_ai
                    self.inspection_database()

                    predicted_path = (
                        self.ui.output_path
                        + "/"
                        + self.ui.coord_reference_building_feature_path
                    )
                    data_existing_dl = data_existing_dl.dropna(subset=["llrs"])
                    data_existing_dl.to_csv(predicted_path, index=False)
                    extra_path = self.ui.output_path + "/" + self.ui.knn_dl_saved_path
                    n_neighbors = self.ui.k_value
                    extrapolation_existing_reference(
                        data_existing_dl,
                        self.ui.building_extra_path,
                        extra_path,
                        n_neighbors,
                    )
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Successful extrapolation process")
                else:
                    building_no_info = self.ui.building_extra_path
                    building_reference = self.ui.example_building_path
                    final_distribution_list_full = []

                    # Iterate over each building with no image
                    for _idx, input_row in building_no_info.iterrows():
                        # Find 3 nearest neighbors using geodesic distance
                        n_neighbors = self.ui.k_value
                        nearest_neighbors = find_nearest_neighbors_geodesic(
                            input_row, building_reference, n_neighbors
                        )
                        # Compute taxonomy-based distributions with full structure
                        distribution_rows = (
                            compute_taxonomy_distribution_full_structure(
                                nearest_neighbors, input_row
                            )
                        )

                        # Append to final result
                        final_distribution_list_full.extend(distribution_rows)

                    # Convert final list to DataFrame
                    final_distribution_df_full = pd.DataFrame(
                        final_distribution_list_full
                    )
                    # Export to CSV
                    saved_path = (
                        self.ui.output_path + "/" + self.ui.extrapolation_name + ".csv"
                    )

                    final_distribution_df_full.to_csv(saved_path, index=False)
                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Successful extrapolation process")

            #####################################
            ######## Stratified mode ############
            #####################################
            else:
                if self.ui.extrapolation_mode == 0:
                    self.ui.progress_bar_method.setValue(10)
                    self.ui.method_progress.setText("Extrapolation in process ...")

                    building_data = self.ui.data_population
                    self.lat_dl = building_data.loc[0, "latitude"]
                    self.lon_dl = building_data.loc[0, "longitude"]
                    # ========== Run sampling for each feature ==========
                    analysis_features = self.ui.feature_strata
                    for aux in analysis_features:
                        print(" ========== " + aux + " ===========")
                        final_sample, class_dist, final_size = (
                            iterative_label_discovery_cached_fractional(
                                data=building_data,
                                labeling_function=lambda x, feature=aux: (
                                    labeling_function(x, feature, building_data)
                                ),
                                id_column="id",
                                id_feature=aux,
                                initial_fraction=self.ui.initial_fraction,
                                step_fraction=self.ui.step_fraction,
                                max_fraction=self.ui.max_fraction,
                                max_iterations=self.ui.max_iterations,
                                stability_threshold=self.ui.stability_threshold,
                            )
                        )
                        final_sample.to_csv(
                            f"{self.ui.folder_path_new}/{self.ui.prefix_strata}_stratified_dl_{aux}.csv",
                            index=False,
                        )
                        dist_por = pd.DataFrame(
                            list(class_dist.items()), columns=["Class", "Proportion"]
                        )
                        dist_por.to_csv(
                            f"{self.ui.folder_path_new}/{self.ui.prefix_strata}_stratified_dl_dist_{aux}.csv",
                            index=False,
                        )
                        print("")

                    print("Sample size definitive: ", final_size)

                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Successful extrapolation process")

                elif self.ui.extrapolation_mode == 1:
                    building_data = self.ui.data_population
                    # ========== Run sampling for each feature ==========
                    analysis_features = self.ui.feature_strata

                    for feature in analysis_features:
                        print(" ========== " + feature + " ===========")
                        final_sample, class_dist, final_size = (
                            iterative_distribution_stability_manual(
                                data=building_data,
                                id_feature=feature,
                                id_column="id",
                                initial_fraction=self.ui.initial_fraction,
                                step_fraction=self.ui.step_fraction,
                                max_fraction=self.ui.max_fraction,
                                max_iterations=self.ui.max_iterations,
                                stability_threshold=self.ui.stability_threshold,
                            )
                        )
                        dist_por = pd.DataFrame(
                            list(class_dist.items()), columns=["Class", "Proportion"]
                        )
                        dist_por.to_csv(
                            f"{self.ui.folder_path_new}/{self.ui.prefix_strata}_stratified_dist_{feature}.csv",
                            index=False,
                        )
                        final_sample.to_csv(
                            f"{self.ui.folder_path_new}/{self.ui.prefix_strata}_stratified_{feature}.csv",
                            index=False,
                        )
                        print("")

                    self.ui.progress_bar_method.setValue(100)
                    self.ui.method_progress.setText("Successful extrapolation process")

    ############ Epoch of construction loader ################
    def epoch_construction(self):
        """Load or request the construction-epoch values."""
        if self.ui.insp_method != 3:
            path = self.ui.output_folder_value + "/epoch_value.csv"

            try:
                epoch = pd.read_csv(path)
                if self.epoch_const is True:
                    self.epoch_const = False
                    for i in range(len(epoch)):
                        self.ui.epc_const_cb_1.addItem(str(epoch.iloc[i, 0]), 
                                                       str(epoch.iloc[i, 0]))

            except (
                AttributeError,
                FileNotFoundError,
                OSError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ):
                self.epoch_const = False

                # Message with special format
                message = """
                The construction epoch varies by country and is typically linked to the
                implementation
                of specific building code regulations. As a result, each country has its
                own relevant
                periods. <b><u>PLEASE SELECT AND ENTER THE APPROPRIATE CONSTRUCTION
                EPOCH FOR YOUR COUNTRY</u></b>.
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

                # Called function where the user creates a manual bounding box by
                # clicking four points, which is then displayed in the UI frame.
                dialog = EpochSelectionDialog(parent=self.ui, main_window=self.ui)
                dialog.exec_()  # Open the pop-up
                epoch = dialog.get_epochs()
                for i in range(len(epoch)):
                    self.ui.epc_const_cb_1.addItem(epoch[i])

                epoch = pd.DataFrame(epoch)
                epoch.columns = ["Epochs"]
                epoch.to_csv(path, index=False)

    ############ Help button ################
    def help_llrs_material(self):
        """Open the LLRS-material visual help dialog."""
        description = (
            "Use these examples as a visual reference to identify the dominant "
            "material"
            "of the"
            "lateral load-resisting system (LLRS). When more than one material is "
            "visible, select"
            "the material that best represents the main structural system of the "
            "building."
        )

        # Paths to your example images for each LLRS material
        self.images = {
            "Concrete": {
                "path": "help_img/concrete_llrs.jpg",
                "description": (
                    "Building whose main lateral load-resisting system is made of "
                    "reinforced concrete,"
                    "such as concrete frames, shear walls, or concrete structural "
                    "elements."
                ),
            },
            "Masonry - Confined": {
                "path": "help_img/mcf.jpg",
                "description": (
                    "Masonry building where walls are confined by reinforced concrete "
                    "elements,"
                    "typically vertical tie-columns and horizontal bond beams."
                ),
            },
            "Masonry - Reinforced": {
                "path": "help_img/mr.jpg",
                "description": (
                    "Masonry building with walls strengthened using reinforcement, "
                    "such"
                    "as steel bars"
                    "to improve structural resistance."
                ),
            },
            "Masonry - Unreinforced": {
                "path": "help_img/mur.jpg",
                "description": (
                    "Masonry building made of brick, block, or stone walls without "
                    "visible or expected"
                    "reinforced concrete confinement or internal reinforcement."
                ),
            },
            "Steel": {
                "path": "help_img/steel.jpg",
                "description": (
                    "Building whose main structural system is made of steel elements, "
                    "such as steel"
                    "frames, columns, beams, or bracing systems."
                ),
            },
            "Hybrid - MCF/MUR": {
                "path": "help_img/hby.jpg",
                "description": (
                    "Building that combines confined masonry and unreinforced masonry "
                    "characteristics,"
                    "or where different parts of the building appear to use different "
                    "masonry systems."
                ),
            },
            "Informal materials": {
                "path": "help_img/inf_mat.jpg",
                "description": (
                    "Building constructed with informal, temporary, lightweight, or "
                    "low-quality materials,"
                    "often showing irregular construction practices or non-engineered "
                    "solutions."
                ),
            },
            "Wood": {
                "path": "help_img/wood.jpg",
                "description": (
                    "Building whose main structural system is made primarily of timber "
                    "or wooden elements,"
                    "such as wood frames, posts, beams, or panels."
                ),
            },
            "Adobe": {
                "path": "help_img/adobe.jpg",
                "description": (
                    "Building constructed mainly with adobe or earthen masonry units, "
                    "commonly identified"
                    "by thick walls and traditional construction techniques."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(800 * self.sf_factor),
            w_size_height=int(750 * self.sf_factor),
            w_title="LLRS Material - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_llrs(self):
        """Open the LLRS visual help dialog."""
        description = (
            "Use these examples as a visual reference to identify the dominant "
            "lateral load-resisting system (LLRS). When more than one system is "
            "visible, select"
            "the system that best represents the main structural behavior of the "
            "building."
        )

        # Paths to your example images for each LLRS type
        self.images = {
            "Dual System": {
                "path": "help_img/LDUAL.jpg",
                "description": (
                    "Structural system that combines moment frames with shear walls or "
                    "braced frames."
                    "Both systems contribute to resisting lateral loads such as "
                    "earthquake or wind forces."
                ),
            },
            "Infilled frames": {
                "path": "help_img/LFINF.jpg",
                "description": (
                    "Frame structure, usually concrete or steel, where the spaces "
                    "between columns and beams"
                    "are filled with masonry walls. The infill walls may influence the "
                    "lateral behavior."
                ),
            },
            "Moment frames": {
                "path": "help_img/LFM.jpg",
                "description": (
                    "Structural system composed of beams and columns connected with "
                    "rigid or semi-rigid joints,"
                    "allowing the frame to resist lateral loads through bending action."
                ),
            },
            "Walls": {
                "path": "help_img/LWAL.jpg",
                "description": (
                    "Structural system where walls are the main elements resisting "
                    "lateral loads."
                    "These may include masonry walls, reinforced concrete shear walls, "
                    "or load-bearing walls."
                ),
            },
            "Braced frames": {
                "path": "help_img/LFBR.jpg",
                "description": (
                    "Frame system with diagonal bracing elements that provide lateral "
                    "stiffness and strength."
                    "This system is commonly found in steel structures."
                ),
            },
            "No lateral load-resisting system": {
                "path": "help_img/inf_mat.jpg",
                "description": (
                    "Building with no clear or identifiable lateral load-resisting "
                    "system."
                    "This may apply to very weak, informal, temporary, or highly "
                    "irregular structures."
                ),
            },
            "Flat slab/plate or waffle slab": {
                "path": "help_img/WAFFLE.jpg",
                "description": (
                    "Structural system where floor slabs are directly supported by "
                    "columns, without deep beams."
                    "This includes flat plates, flat slabs, or waffle slabs."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(700 * self.sf_factor),
            w_size_height=int(750 * self.sf_factor),
            w_title="LLRS - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )
        help_window.exec_()

    ############ Help button ################
    def help_occupancy(self):
        """Open the occupancy visual help dialog."""
        description = (
            "Use these examples as a visual reference to identify the dominant "
            "building"
            "occupancy."
            "When more than one use is visible, select the option that best represents "
            "the main use of the building."
        )

        # Paths to your example images for each occupancy type
        self.images = {
            "Residential": {
                "path": "help_img/mur.jpg",
                "description": (
                    "Building mainly used for housing, such as houses, apartment "
                    "buildings, or residential towers."
                    "Typical visual cues include balconies, repeated windows, private "
                    "entrances, and domestic-scale façades."
                ),
            },
            "Commercial": {
                "path": "help_img/LFBR.jpg",
                "description": (
                    "Building mainly used for business, retail, offices, restaurants, "
                    "or services."
                    "Typical visual cues include storefronts, large display windows, "
                    "signs, offices, or public entrances."
                ),
            },
            "Industrial": {
                "path": "help_img/steel.jpg",
                "description": (
                    "Building mainly used for manufacturing, storage, logistics, or "
                    "industrial activities."
                    "Typical visual cues include large volumes, wide doors, loading "
                    "bays, metal cladding, few windows, or warehouse-like forms."
                ),
            },
            "Mixed (Residential + Commercial)": {
                "path": "help_img/res_com.jpg",
                "description": (
                    "Building combining residential and commercial uses, commonly with "
                    "shops or services on the ground floor"
                    "and apartments or housing units on the upper floors."
                ),
            },
            "Educational": {
                "path": "help_img/edu.jpg",
                "description": (
                    "Building mainly used for education, such as schools, "
                    "universities,"
                    "or training centers."
                    "Typical visual cues include large institutional layouts, "
                    "classrooms, playgrounds, campus areas, or school signage."
                ),
            },
            "Healthcare": {
                "path": "help_img/healthcare.jpg",
                "description": (
                    "Building mainly used for medical or health services, such as "
                    "hospitals, clinics, or health centers."
                    "Typical visual cues include emergency entrances, medical signs, "
                    "large institutional façades, or ambulance access."
                ),
            },
            "Government": {
                "path": "help_img/government.jpg",
                "description": (
                    "Building mainly used for public administration or government "
                    "services, such as municipal offices, courts,"
                    "police stations, or other official public institutions."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(700 * self.sf_factor),
            w_size_height=int(750 * self.sf_factor),
            w_title="Occupancy type - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_code(self):
        """Open the building-code visual help dialog."""
        description = (
            "Use these examples as a visual reference to identify the likely building "
            "code level."
            "The code level should be selected based on visible construction quality, "
            "structural regularity,"
            "materials, detailing, and the apparent level of engineering design."
        )

        # Paths to your example images for each building code level
        self.images = {
            "High-code": {
                "path": "help_img/healthcare.jpg",
                "description": (
                    "Building that appears to follow modern seismic or structural "
                    "design provisions."
                    "Typical visual cues include regular geometry, good construction "
                    "quality, engineered materials,"
                    "and well-defined structural elements."
                ),
            },
            "Moderate-code": {
                "path": "help_img/sos.jpg",
                "description": (
                    "Building that appears to have been designed or constructed under "
                    "earlier engineering standards, which"
                    "may increase vulnerability in specific aspects, typically with "
                    "acceptable material quality."
                ),
            },
            "Low-code": {
                "path": "help_img/POP.jpg",
                "description": (
                    "Building that appears to have been designed or constructed under "
                    "early engineering standards, where"
                    "structural provisions are limited and some details now prohibited "
                    "in complex buildings may still be present,"
                    "although they may remain acceptable for simple structures."
                ),
            },
            "No-code": {
                "path": "help_img/inf_mat.jpg",
                "description": (
                    "Building that appears to be non-engineered or informal, with "
                    "little or no evidence of code-based"
                    "construction. Typical visual cues include improvised materials, "
                    "poor workmanship, high irregularity,"
                    "or temporary construction features."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(700 * self.sf_factor),
            w_size_height=int(750 * self.sf_factor),
            w_title="Building code level - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_image_quality(self):
        """Open the image-quality visual help dialog."""
        description = (
            "Use these examples as a visual reference to assess the quality of the "
            "image for building"
            "attribute identification. Select the option that best represents how "
            "clearly the relevant"
            "visual cues can be observed."
        )

        # Paths to your example images for each image quality level
        self.images = {
            "Excellent": {
                "path": "help_img/res_com.jpg",
                "description": (
                    "Image with a clear and unobstructed view of the building, where "
                    "most or all visual cues"
                    "needed to identify the building attributes are clearly visible."
                ),
            },
            "Good": {
                "path": "help_img/good.jpg",
                "description": (
                    "Image with a good view of the building, with only minor "
                    "obstacles,"
                    "noise, or perspective"
                    "issues that do not significantly affect the identification of "
                    "building attributes."
                ),
            },
            "Intermediate": {
                "path": "help_img/LFBR.jpg",
                "description": (
                    "Image where some building attributes can be identified clearly, "
                    "but others are uncertain"
                    "due to partial obstruction, limited resolution, shadows, angle, "
                    "or"
                    "image noise."
                ),
            },
            "Bad": {
                "path": "help_img/bad_quality.jpg",
                "description": (
                    "Image where it is difficult to identify building attributes "
                    "because of strong obstruction,"
                    "poor resolution, excessive noise, bad angle, blur, or very "
                    "limited"
                    "visibility of the building."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(950 * self.sf_factor),
            w_size_height=int(800 * self.sf_factor),
            w_title="Image quality - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_n_bay(self):
        """Open the number-of-bays visual help dialog."""
        description = (
            "Use this example as a visual reference to assign the image quality."
        )

        # Path to the example image for counting the number of bays
        self.images = {
            "Number of bays": {
                "path": "help_img/n_bay.png",
                "description": (
                    "Visual reference showing how to identify and count façade bays. "
                    "Count the repeated horizontal divisions between main vertical "
                    "elements, such as columns,"
                    "structural walls, or clearly visible façade modules."
                ),
            }
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(850 * self.sf_factor),
            w_size_height=int(750 * self.sf_factor),
            w_title="Number of bays - visual example",
            img_width=int(640 * self.sf_factor),
            img_height=int(480 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_block_position(self):
        """Open the block-position visual help dialog."""
        description = (
            "Use this example as a visual reference to identify the position of the "
            "building within the block."
            "Select the option that best represents how the building is connected to "
            "neighboring buildings."
        )

        # Path to the example image for block position options
        self.images = {
            "Detached building": {
                "path": "help_img/isolated.jpg",
                "description": (
                    "The building is detached, with no adjoining or "
                    "attached buildings; all exterior walls are exposed"
                ),
            },
            "Adjoining building(s) one side": {
                "path": "help_img/bp1.jpg",
                "description": (
                    "The building has adjoining or attached building(s) on one side"
                ),
            },
            "Adjoining building(s) two side": {
                "path": "help_img/bp2.jpg",
                "description": (
                    "The building has adjoining or attached building(s) on two sides"
                ),
            },
            "Adjoining building(s) three side": {
                "path": "help_img/bp3.jpg",
                "description": (
                    "The building has adjoining or attached building(s) on three sides"
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(1200 * self.sf_factor),
            w_size_height=int(800 * self.sf_factor),
            w_title="Block position - visual example",
            img_width=int(350 * self.sf_factor),
            img_height=int(350 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_roof_shape(self):
        """Open the roof-shape visual help dialog."""
        description = (
            "Use these examples as a visual reference to identify the dominant roof "
            "shape."
            "When the roof has multiple visible shapes, select the option that best "
            "represents the main roof configuration."
        )

        # Paths to your example images for each roof shape
        self.images = {
            "Flat": {
                "path": "help_img/flat_roof.png",
                "description": (
                    "Roof with little or no visible slope, commonly "
                    "found in modern, commercial, or high-rise "
                    "buildings."
                ),
            },
            "Pitched with gable ends": {
                "path": "help_img/gable_roof.png",
                "description": (
                    "Roof with two sloping sides that meet at a ridge, "
                    "forming triangular gable ends."
                ),
            },
            "Pitched and hipped": {
                "path": "help_img/hipped_roof.png",
                "description": (
                    "Pitched roof where all sides slope downward toward "
                    "the walls, without vertical gable ends."
                ),
            },
            "Pitched with dormers": {
                "path": "help_img/dormes_roof.png",
                "description": (
                    "Pitched roof containing dormer windows or small "
                    "roof projections emerging from the main roof "
                    "surface."
                ),
            },
            "Monopitch": {
                "path": "help_img/monoslope_roof.png",
                "description": (
                    "Single-sloped roof surface, also known as a shed or "
                    "mono-slope roof."
                ),
            },
            "Sawtooth": {
                "path": "help_img/Sawtooth_roof.png",
                "description": (
                    "Roof composed of a series of repeated slopes, often "
                    "resembling saw teeth and commonly used in "
                    "industrial buildings."
                ),
            },
            "Curved": {
                "path": "help_img/curved.png",
                "description": (
                    "Roof with a rounded or arched shape instead of "
                    "straight sloping planes."
                ),
            },
            "Complex regular": {
                "path": "help_img/complex_regular.png",
                "description": (
                    "Roof formed by several repeated or organized roof "
                    "planes, with a regular and systematic geometry."
                ),
            },
            "Complex irregular": {
                "path": "help_img/complex_irregular.png",
                "description": (
                    "Roof formed by multiple roof planes with an "
                    "irregular, asymmetric, or non-repetitive geometry."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(700 * self.sf_factor),
            w_size_height=int(900 * self.sf_factor),
            w_title="Roof Shape - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_roof_material(self):
        """Open the roof-material visual help dialog."""
        description = (
            "Use these examples as a visual reference to identify the dominant roof "
            "covering material."
            "When multiple materials are visible, select the material that occupies "
            "the"
            "largest roof area."
        )

        # Paths to your example images for each roof material
        self.images = {
            "Concrete": {
                "path": "help_img/concrete.jpg",
                "description": (
                    "Plain concrete slab, commonly used in high-rise buildings."
                ),
            },
            "Clay or concrete tile": {
                "path": "help_img/clay_tile.jpg",
                "description": (
                    "Roof covering made of clay or concrete tiles, "
                    "commonly used in pitched roofs."
                ),
            },
            "Metal or asbestos sheets": {
                "path": "help_img/asbesto.png",
                "description": (
                    "Lightweight sheet roofing, often used in "
                    "industrial, rural, or older buildings."
                ),
            },
            "Wooden and asphalt shingles": {
                "path": "help_img/asphalt_shingles.jpg",
                "description": (
                    "Small overlapping roof units made of wood or "
                    "asphalt, commonly used in residential buildings."
                ),
            },
            "Slate": {
                "path": "help_img/Slate.png",
                "description": (
                    "Natural stone roofing material, usually "
                    "dark-colored and arranged in thin overlapping "
                    "pieces."
                ),
            },
            "Solar panelled roofs": {
                "path": "help_img/solar_panel.png",
                "description": (
                    "Roofs where solar panels are the dominant visible "
                    "element or cover a large portion of the roof."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(700 * self.sf_factor),
            w_size_height=int(750 * self.sf_factor),
            w_title="Roof Material - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Help button ################
    def help_irregularity(self):
        """Open the vertical-irregularity visual help dialog."""
        description = (
            "Use these examples as a visual reference to identify common vertical "
            "irregularities."
            "Select the option that best represents the irregularity observed in the "
            "building."
        )

        # Paths to your example images for each vertical irregularity type
        self.images = {
            "Soft story": {
                "path": "help_img/sos.jpg",
                "description": (
                    "A story, usually the ground floor, with significantly lower "
                    "stiffness or strength"
                    "than the stories above, often due to open parking areas, large "
                    "openings, or fewer walls."
                ),
            },
            "Short column": {
                "path": "help_img/SHC.png",
                "description": (
                    "A column whose effective height is reduced by partial-height "
                    "walls"
                    "or openings,"
                    "making it more vulnerable to concentrated seismic forces."
                ),
            },
            "Pounding": {
                "path": "help_img/POP.jpg",
                "description": (
                    "A condition where adjacent buildings are very close to each "
                    "other,"
                    "allowing them"
                    "to collide during lateral movement such as earthquake shaking."
                ),
            },
            "Setback": {
                "path": "help_img/set.jpg",
                "description": (
                    "A sudden reduction in the building plan area or width at upper "
                    "stories, creating"
                    "a step-like vertical profile."
                ),
            },
            "Change in vertical": {
                "path": "help_img/cvh.jpg",
                "description": (
                    "A noticeable change in the vertical structural configuration, "
                    "such"
                    "as changes in"
                    "wall alignment, column layout, stiffness, or mass distribution "
                    "along the height."
                ),
            },
        }

        help_window = HelpDialog(
            self.images,
            description=description,
            w_size_width=int(720 * self.sf_factor),
            w_size_height=int(850 * self.sf_factor),
            w_title="Vertical Irregularity - visual example",
            img_width=int(180 * self.sf_factor),
            img_height=int(180 * self.sf_factor),
            parent=self.ui,
            main_window=self.ui,
        )

        help_window.exec_()

    ############ Vulnerability/Fragility plot ################
    def vulnerability_curve(self):
        """Open the vulnerability-curve dialog for the current building."""
        # Conditional to avoid executing the method if there is no project folder
        if self.ui.output_folder_value == "-":
            QMessageBox.warning(
                self.ui,
                "File Error",
                "This option is only available once the building image is displayed.",
            )
        else:
            if (
                self.ui.material_cb_1.currentData() is None
                or self.ui.llrs_cb_1.currentData() is None
                or self.ui.n_stories_value_1.currentData() is None
                or self.ui.occup_cb_1.currentData() is None
            ):
                QMessageBox.warning(
                    self.ui,
                    "Feature required",
                    "This option becomes available after you have classified at least "
                    "the material, LLRS, occupancy, and number of stories.",
                )
            else:
                vul_plot = VulnerabilityDialog(parent=self.ui, main_window=self.ui)
                vul_plot.filter_comboboxes_main()
                vul_plot.exec_()
