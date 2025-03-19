# GUI pyqt5 libraries
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

# Libries for shape and geopackage creation
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import Point
from geopy.geocoders import Nominatim

# Utilities libraries
import pandas as pd
import os

# *.py scripts with complex methods
from gui_gis import GUI_geofiles
from extrapolation_options import extrapolation_options_window

class PolygonSettingWindow(QDialog):
    def __init__(self, parent=None, main_window=None, gui_methods=None):
        super().__init__(parent)
        self.main_window = main_window  # Reference to the main window (GUIInterface)
        self.gui_methods = gui_methods  # Reference to GUIMethods
        
        # Window size
        self.setWindowTitle("Setting Polygon Coordinates")
        self.resize(612, 846)

        # Main widget
        self.coord_frame = QtWidgets.QWidget(self)
        self.coord_frame.setObjectName("coord_frame")
        
        # Title of the window
        self.w_tittle = QtWidgets.QLabel(self.coord_frame)
        self.w_tittle.setGeometry(QtCore.QRect(170, 0, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.w_tittle.setFont(font)
        self.w_tittle.setObjectName("w_tittle")
        self.w_tittle.setText("Setting Polygon Coordinates")  # Title
        
        # Button for uploading the CSV file with coordinates
        self.csv_button = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button.setGeometry(QtCore.QRect(20, 305, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.csv_button.setFont(font)
        self.csv_button.setObjectName("csv_button")
        self.csv_button.setText("Upload polygon coordinates")  # Button text
        self.csv_button.clicked.connect(self.check_data_default)
        
        # Input fields for the coordinates of points (latitude and longitude)
        self.coord_1 = QtWidgets.QLineEdit(self.coord_frame)
        self.coord_1.setGeometry(QtCore.QRect(160, 116, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.coord_1.setFont(font)
        self.coord_1.setObjectName("coord_1")
        self.coord_1.setText("(lat1 , lon1)")  # Placeholder text
        
        self.coord_2 = QtWidgets.QLineEdit(self.coord_frame)
        self.coord_2.setGeometry(QtCore.QRect(160, 155, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.coord_2.setFont(font)
        self.coord_2.setObjectName("coord_2")
        self.coord_2.setText("(lat2 , lon2)")  # Placeholder text
        
        # Label for "First Point"
        self.point_1 = QtWidgets.QLabel(self.coord_frame)
        self.point_1.setGeometry(QtCore.QRect(20, 120, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.point_1.setFont(font)
        self.point_1.setObjectName("point_1")
        self.point_1.setText("First Point:")  # Text label
        
        # Image placeholder for a squared image (if needed)
        self.img_squared = QtWidgets.QLabel(self.coord_frame)
        self.img_squared.setGeometry(QtCore.QRect(420, 120, 151, 131))
        self.img_squared.setObjectName("img_squared")
        self.img_squared.setPixmap(QtGui.QPixmap("squared_coord.png"))
        self.img_squared.setScaledContents(True)
        
        # Button for saving the input and continuing
        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(320, 800, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.setText("Save and continue")  # Button text
        self.save_button.clicked.connect(self.building_sample)
        
        # Label for "Second Point"
        self.point_2 = QtWidgets.QLabel(self.coord_frame)
        self.point_2.setGeometry(QtCore.QRect(20, 155, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.point_2.setFont(font)
        self.point_2.setObjectName("point_2")
        self.point_2.setText("Second Point:")  # Label text
        
        # Label showing the CSV file path description
        self.label_csv_file = QtWidgets.QLabel(self.coord_frame)
        self.label_csv_file.setGeometry(QtCore.QRect(20, 275, 531, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_csv_file.setFont(font)
        self.label_csv_file.setObjectName("label_csv_file")
        self.label_csv_file.setText("Define a polygon with more than two points using a *.csv file")  # Instruction text
        
        # Backgrounds
        self.backg_1 = QtWidgets.QLabel(self.coord_frame)
        self.backg_1.setGeometry(QtCore.QRect(10, 80, 591, 261))
        self.backg_1.setStyleSheet("background-color: rgb(209, 255, 165);")
        self.backg_1.setText("")
        self.backg_1.setObjectName("backg_1")
        
        self.backg_2 = QtWidgets.QLabel(self.coord_frame)
        self.backg_2.setGeometry(QtCore.QRect(10, 40, 591, 31))
        # self.backg_2.setStyleSheet("background-color: rgb(255, 206, 253);")
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        
        self.backg_3 = QtWidgets.QLabel(self.coord_frame)
        self.backg_3.setGeometry(QtCore.QRect(10, 350, 591, 121))
        self.backg_3.setStyleSheet("background-color: rgb(255, 224, 185);")
        self.backg_3.setText("")
        self.backg_3.setObjectName("backg_3")
        
        self.backg_4 = QtWidgets.QLabel(self.coord_frame)
        self.backg_4.setGeometry(QtCore.QRect(10, 480, 591, 151))
        self.backg_4.setStyleSheet("background-color: rgb(157, 218, 255);")
        self.backg_4.setText("")
        self.backg_4.setObjectName("backg_4")
        
        self.backg_5 = QtWidgets.QLabel(self.coord_frame)
        self.backg_5.setGeometry(QtCore.QRect(10, 640, 591, 151))
        self.backg_5.setStyleSheet("background-color: rgb(254, 255, 160);")
        self.backg_5.setText("")
        self.backg_5.setObjectName("backg_5")
        
        # Label for displaying the CSV file name
        self.label = QtWidgets.QLabel(self.coord_frame)
        self.label.setGeometry(QtCore.QRect(270, 310, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("filename.csv")  # Placeholder text for CSV filename
        
        # Label to show the selected method for inspection
        self.method_value = QtWidgets.QLabel(self.coord_frame)
        self.method_value.setGeometry(QtCore.QRect(280, 45, 301, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.method_value.setFont(font)
        self.method_value.setObjectName("method_value")
               
        if self.main_window.insp_method == 0:
            self.method_value.setText("Polygon coordinates")  # Default method display
        elif self.main_window.insp_method == 1:
            self.method_value.setText("Specific locations")  # Default method display
        elif self.main_window.insp_method == 2:
            self.method_value.setText("Local building images folder")  # Default method display
        elif self.main_window.insp_method == 3:
            self.method_value.setText("Neighbors extrapolation")  # Default method display
            
        # Label for displaying the selected method description
        self.method_label = QtWidgets.QLabel(self.coord_frame)
        self.method_label.setGeometry(QtCore.QRect(20, 40, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.method_label.setFont(font)
        self.method_label.setObjectName("method_label")
        self.method_label.setText("Inspection method selected:")  # Text label
        
        # Input fields and labels for "Default" method
        self.default_label = QtWidgets.QLabel(self.coord_frame)
        self.default_label.setGeometry(QtCore.QRect(210, 80, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.default_label.setFont(font)
        self.default_label.setObjectName("default_label")
        self.default_label.setText("Polygon Method Input")  # Default input section title
        
        # Labels and inputs for building count and sample size
        self.n_building_default = QtWidgets.QLabel(self.coord_frame)
        self.n_building_default.setGeometry(QtCore.QRect(20, 200, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.n_building_default.setFont(font)
        self.n_building_default.setObjectName("n_building_default")
        self.n_building_default.setText("N° buildings:")  # Building count label
        
        self.sample_default = QtWidgets.QLabel(self.coord_frame)
        self.sample_default.setGeometry(QtCore.QRect(20, 230, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.sample_default.setFont(font)
        self.sample_default.setObjectName("sample_default")
        self.sample_default.setText("Sample size:")  # Sample size label
        
        self.building_value_default = QtWidgets.QLabel(self.coord_frame)
        self.building_value_default.setGeometry(QtCore.QRect(160, 200, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10) 
        self.building_value_default.setFont(font)
        self.building_value_default.setObjectName("building_value_default")
        self.building_value_default.setText("-")  # Default building count
        
        self.sample_size_default = QtWidgets.QLineEdit(self.coord_frame)
        self.sample_size_default.setGeometry(QtCore.QRect(160, 230, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sample_size_default.setFont(font)
        self.sample_size_default.setObjectName("sample_size_default")
        self.sample_size_default.setText("10")  # Default sample size
        
        # Load button for determine building population
        self.load_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_button.setGeometry(QtCore.QRect(110, 800, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.load_button.setFont(font)
        self.load_button.setObjectName("load_button")
        self.load_button.setText("Load data")  # Button text
        self.load_button.clicked.connect(self.save_coordinates)
        self.load_button.clicked.connect(self.building_polulation)
        
        # Input fields and labels for "Specific" method
        self.specific_label = QtWidgets.QLabel(self.coord_frame)
        self.specific_label.setGeometry(QtCore.QRect(160, 355, 301, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.specific_label.setFont(font)
        self.specific_label.setObjectName("specific_label")
        self.specific_label.setText("Specific Locations Method Input")  # Input section title
        
        self.specific_path = QtWidgets.QLabel(self.coord_frame)
        self.specific_path.setGeometry(QtCore.QRect(270, 440, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.specific_path.setFont(font)
        self.specific_path.setObjectName("specific_path")
        self.specific_path.setText("filename.csv")  # Placeholder text for CSV filename
        
        self.csv_button_specific = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_specific.setGeometry(QtCore.QRect(20, 435, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.csv_button_specific.setFont(font)
        self.csv_button_specific.setObjectName("csv_button_specific")
        self.csv_button_specific.setText("Upload building locations")  # Button text
        self.csv_button_specific.clicked.connect(self.check_data_specific)

        
        self.local_label = QtWidgets.QLabel(self.coord_frame)
        self.local_label.setGeometry(QtCore.QRect(190, 480, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.local_label.setFont(font)
        self.local_label.setObjectName("local_label")
        self.local_label.setText("Local Folder Images Method Input")  # Input section title
        
        self.csv_button_local = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_local.setGeometry(QtCore.QRect(20, 590, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.csv_button_local.setFont(font)
        self.csv_button_local.setObjectName("csv_button_local")
        self.csv_button_local.setText("Upload buildings information")  # Button text
        self.csv_button_local.clicked.connect(self.check_data_local)
        
        self.local_folder_path = QtWidgets.QLabel(self.coord_frame)
        self.local_folder_path.setGeometry(QtCore.QRect(270, 555, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.local_folder_path.setFont(font)
        self.local_folder_path.setObjectName("local_folder_path")
        self.local_folder_path.setText("---")  # Button text
        
        self.folder_local_button = QtWidgets.QPushButton(self.coord_frame)
        self.folder_local_button.setGeometry(QtCore.QRect(20, 550, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.folder_local_button.setFont(font)
        self.folder_local_button.setObjectName("folder_local_button")
        self.folder_local_button.setText("Select image folder")  # Button text
        self.folder_local_button.clicked.connect(self.check_data_local_folder)
        
        self.local_path = QtWidgets.QLabel(self.coord_frame)
        self.local_path.setGeometry(QtCore.QRect(270, 595, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.local_path.setFont(font)
        self.local_path.setObjectName("local_path")
        self.local_path.setText("filename.csv")  # Placeholder text for CSV filename
        
        self.output_label_specific = QtWidgets.QLabel(self.coord_frame)
        self.output_label_specific.setGeometry(QtCore.QRect(20, 390, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.output_label_specific.setFont(font)
        self.output_label_specific.setObjectName("output_label_specific")
        self.output_label_specific.setText("Output name:") 
        
        self.output_value_specific = QtWidgets.QLineEdit(self.coord_frame)
        self.output_value_specific.setGeometry(QtCore.QRect(160, 390, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.output_value_specific.setFont(font)
        self.output_value_specific.setObjectName("output_value_specific")
        self.output_value_specific.setText("Specific") 
        
        self.output_label_local = QtWidgets.QLabel(self.coord_frame)
        self.output_label_local.setGeometry(QtCore.QRect(20, 510, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.output_label_local.setFont(font)
        self.output_label_local.setObjectName("output_label_local")
        self.output_label_local.setText("Output name:")
        
        self.output_value_local = QtWidgets.QLineEdit(self.coord_frame)
        self.output_value_local.setGeometry(QtCore.QRect(160, 510, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.output_value_local.setFont(font)
        self.output_value_local.setObjectName("output_value_local")
        self.output_value_local.setText("Local")
        
        self.n_image_local_label = QtWidgets.QLabel(self.coord_frame)
        self.n_image_local_label.setGeometry(QtCore.QRect(310, 510, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.n_image_local_label.setFont(font)
        self.n_image_local_label.setObjectName("n_image_local_label")
        self.n_image_local_label.setText("N° images per location:")
        
        self.n_image_local_value = QtWidgets.QComboBox(self.coord_frame)
        self.n_image_local_value.setGeometry(QtCore.QRect(520, 511, 71, 31))
        self.n_image_local_value.setObjectName("n_image_local_value")
        font = QtGui.QFont()
        font.setPointSize(10)
        self.n_image_local_value.setFont(font)
        self.n_image_local_value.addItem("1", 1)
        self.n_image_local_value.addItem("2", 2)
        self.n_image_local_value.addItem("3", 3)
        
########################################################################################
########################################################################################

        self.extrapolation_label = QtWidgets.QLabel(self.coord_frame)
        self.extrapolation_label.setGeometry(QtCore.QRect(190, 640, 251, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.extrapolation_label.setFont(font)
        self.extrapolation_label.setObjectName("extrapolation_label")
        self.extrapolation_label.setText("Extrapolation Method Input")

        self.output_label_extra = QtWidgets.QLabel(self.coord_frame)
        self.output_label_extra.setGeometry(QtCore.QRect(20, 670, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.output_label_extra.setFont(font)
        self.output_label_extra.setObjectName("output_label_extra")
        self.output_label_extra.setText("Output name:")
        
        self.building_extra_button = QtWidgets.QPushButton(self.coord_frame)
        self.building_extra_button.setGeometry(QtCore.QRect(20, 710, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.building_extra_button.setFont(font)
        self.building_extra_button.setObjectName("building_extra_button")
        self.building_extra_button.setText("Upload building coordinates")
        self.building_extra_button.clicked.connect(self.check_data_extra_buildings)
        
        self.output_value_extra = QtWidgets.QLineEdit(self.coord_frame)
        self.output_value_extra.setGeometry(QtCore.QRect(160, 670, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.output_value_extra.setFont(font)
        self.output_value_extra.setObjectName("output_value_extra")
        self.output_value_extra.setText("Extrapolation")
        
        self.example_building_button = QtWidgets.QPushButton(self.coord_frame)
        self.example_building_button.setGeometry(QtCore.QRect(20, 750, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.example_building_button.setFont(font)
        self.example_building_button.setObjectName("example_building_button")
        self.example_building_button.setText("Upload neighbor buildings")
        self.example_building_button.clicked.connect(self.check_data_extra_neighbors)
        
        self.building_extra_path = QtWidgets.QLabel(self.coord_frame)
        self.building_extra_path.setGeometry(QtCore.QRect(270, 720, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.building_extra_path.setFont(font)
        self.building_extra_path.setObjectName("building_extra_path")
        self.building_extra_path.setText("filename.csv")
        
        self.example_building_path = QtWidgets.QLabel(self.coord_frame)
        self.example_building_path.setGeometry(QtCore.QRect(270, 750, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.example_building_path.setFont(font)
        self.example_building_path.setObjectName("example_building_path")
        self.example_building_path.setText("filename.csv")
        
        self.extra_option_button = QtWidgets.QPushButton(self.coord_frame)
        self.extra_option_button.setGeometry(QtCore.QRect(310, 670, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.extra_option_button.setFont(font)
        self.extra_option_button.setObjectName("extra_option_button")
        self.extra_option_button.setText("Extrapolation options")
        self.extra_option_button.clicked.connect(self.extrapolation_values)
        
        self.backg_3.raise_()
        self.backg_2.raise_()
        self.backg_1.raise_()
        self.w_tittle.raise_()
        self.csv_button.raise_()
        self.coord_1.raise_()
        self.coord_2.raise_()
        self.point_1.raise_()
        self.img_squared.raise_()
        self.save_button.raise_()
        self.point_2.raise_()
        self.label_csv_file.raise_()
        self.label.raise_()
        self.method_value.raise_()
        self.method_label.raise_()
        self.default_label.raise_()
        self.n_building_default.raise_()
        self.sample_default.raise_()
        self.building_value_default.raise_()
        self.sample_size_default.raise_()
        self.load_button.raise_()
        self.specific_label.raise_()
        self.backg_4.raise_()
        self.specific_path.raise_()
        self.csv_button_specific.raise_()
        self.local_label.raise_()
        self.csv_button_local.raise_()
        self.local_path.raise_()
        self.output_label_specific.raise_()
        self.output_value_specific.raise_()
        self.output_label_local.raise_()
        self.output_value_local.raise_()
        self.folder_local_button.raise_()
        self.local_folder_path.raise_()
        self.n_image_local_label.raise_()
        self.n_image_local_value.raise_()
        self.extrapolation_label.raise_()
        self.output_label_extra.raise_()
        self.building_extra_button.raise_()
        self.output_value_extra.raise_()
        self.example_building_button.raise_()
        self.building_extra_path.raise_()
        self.example_building_path.raise_()
        self.extra_option_button.raise_()
        
   ############ City Name by coordinates ################
    def get_city_name(self):
        """
        Get the city name for a given latitude and longitude using reverse geocoding.
        
        Args:
            lat (float): Latitude of the point.
            lon (float): Longitude of the point.
        
        Returns:
            str: The name of the city, including the country, or an error message.
        """

        if self.main_window.insp_method == 0  or self.main_window.insp_method == 1: 
            try:
                try:
                    coord1 = self.coord_1.text().strip()
                    lat, lon = map(float, coord1.strip("() ").split(",")) 
                except:
                    lat = self.df.iloc[0,1]
                    lon = self.df.iloc[0,2]
            except:
                QMessageBox.warning(self, "Input Error", "No file selected.")
            geolocator = Nominatim(user_agent="city_name_locator")
            location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
            
            if location and 'address' in location.raw:
                address = location.raw['address']
                self.city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                self.country = address.get('country', 'Unknown')
                self.city_name_manual = self.city+"_"+self.country
                self.main_window.city_value.setText(self.city) 
                self.main_window.country_value.setText(self.country) 
                return (self.city , self.country)
            
            return "City not found"
        
        elif self.main_window.insp_method == 2:
            
            lat = self.df.iloc[self.gui_methods.click_count, 1]
            lon = self.df.iloc[self.gui_methods.click_count, 2]
            
            geolocator = Nominatim(user_agent="city_name_locator")
            location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
            
            if location and 'address' in location.raw:
                address = location.raw['address']
                self.city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                self.country = address.get('country', 'Unknown')
                self.city_name_manual = self.city+"_"+self.country
                self.main_window.city_value.setText(self.city) 
                self.main_window.country_value.setText(self.country) 
                return (self.city , self.country)
        
        elif self.main_window.insp_method == 3:
            
            lat = self.df.iloc[self.gui_methods.click_count, 1]
            lon = self.df.iloc[self.gui_methods.click_count, 2]
            
            geolocator = Nominatim(user_agent="city_name_locator")
            location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
            
            if location and 'address' in location.raw:
                address = location.raw['address']
                self.city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                self.country = address.get('country', 'Unknown')
                self.city_name_manual = self.city+"_"+self.country
                self.main_window.city_value.setText(self.city) 
                self.main_window.country_value.setText(self.country) 
                return (self.city , self.country)
   
   ############ Create a squared base on coordinates ################
    def create_square_geopackage(self,output_file):
        """
        Create a square polygon from two coordinate points and save it as a GeoPackage.
    
        This method extracts latitude and longitude values from two input coordinates 
        and constructs a square polygon based on their minimum and maximum values. 
        The resulting polygon is stored in a GeoDataFrame and saved as a GeoPackage.
    
        Args:
            output_file (str): The file path where the GeoPackage will be saved.
    
        Effects:
            - Reads coordinate values from the UI fields (`coord_1`, `coord_2`).
            - Computes the bounding box from the two coordinate points.
            - Constructs a square polygon using the boundary coordinates.
            - Saves the polygon as a GeoPackage (`.gpkg`) using the `geopandas` library.
    
        Notes:
            - Coordinates should be provided as text in the format "(lat, lon)".
            - The coordinate reference system (CRS) is set to "EPSG:4326" (WGS 84).
            - The polygon follows a clockwise or counterclockwise order to ensure proper closure.
        """
        # Extract coordinates from the two points
        coord1 = self.coord_1.text().strip()
        coord2 = self.coord_2.text().strip()
        lat1, lon1 = map(float, coord1.strip("() ").split(","))
        lat2, lon2 = map(float, coord2.strip("() ").split(","))
        
        # Determine the boundaries
        min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
        min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
        
        # Define the square's vertices (clockwise or counterclockwise)
        square_coords = [
            (min_lon, min_lat),  # Bottom-left
            (min_lon, max_lat),  # Top-left
            (max_lon, max_lat),  # Top-right
            (max_lon, min_lat),  # Bottom-right
            (min_lon, min_lat)   # Close the polygon
        ]
        
        # Create a polygon
        square_polygon = Polygon(square_coords)
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': [square_polygon]}, crs="EPSG:4326")
        
        # Save as a GeoPackage
        gdf.to_file(output_file, driver="GPKG", layer="square_polygon")
        print(f"GeoPackage saved to {output_file}")
           
    ############ Save coordinates ################
    def save_coordinates(self):
        """
        Save building coordinates or process an uploaded CSV file to create a GeoPackage.
    
        This method saves user-entered coordinates or processes a CSV file containing 
        latitude and longitude values. It verifies that the necessary inputs are provided 
        and creates a GeoPackage (`.gpkg`) file with either a boundary polygon or individual 
        point geometries, depending on the selected inspection method.
    
        Args:
            None. The method relies on instance attributes such as `coord_1`, `coord_2`, 
            `label`, `df`, `main_window.insp_method`, and `output_value_specific`.
    
        Effects:
            - Extracts and validates user-entered coordinates.
            - Loads a CSV file and processes coordinate data if applicable.
            - Creates a square boundary polygon for method 0.
            - Creates a polygon from CSV coordinates for method 0.
            - Creates a GeoDataFrame with point geometries for method 1.
            - Updates the `file_name` or `file_name_local` attribute based on the method.
            - Displays messages for success, warnings, or errors.
    
        Notes:
            - The method supports three inspection methods:
                - `insp_method == 0`: Uses manual input or a CSV file to create a boundary polygon.
                - `insp_method == 1`: Loads a CSV and saves individual points as a GeoPackage.
                - `insp_method == 2`: Sets the local file name and validates input.
            - The coordinate reference system (CRS) is set to "EPSG:4326" (WGS 84).
            - If an existing GeoPackage file is found, it is deleted to prevent duplicate layers.
        """

        if self.main_window.insp_method == 0: 
            """Save the entered coordinates or handle the uploaded CSV file."""
            coord1 = self.coord_1.text().strip()
            coord2 = self.coord_2.text().strip()
            
            try:
                self.get_city_name()
                QMessageBox.information(self, "Success", "Done! Please click save and continue button")
            except:
                QMessageBox.warning(self, "Error", "Please check that all the required input files are completed appropriately.") 
            
            if self.main_window:
                try:
                    # Access the main window's variable
                    folder_project = self.main_window.output_folder_value.text()
                except AttributeError as e:
                    QMessageBox.warning(self, "Error", f"Failed to access main window: {e}") 
                    
            output_gpkg = folder_project+"/"+self.city_name_manual+"_boundary.gpkg"
        
            # Check if the GeoPackage file already exists
            if os.path.exists(output_gpkg):
                os.remove(output_gpkg)  # Delete the file to ensure only one layer is created
        
            if self.label.text() == "filename.csv":
                if not coord1 or not coord2:
                    QMessageBox.warning(self, "Input Error", "Please enter both coordinates.")
                    return
                
                # Validate coordinate format
                try:
                    lat1, lon1 = map(float, coord1.strip("() ").split(","))
                    lat2, lon2 = map(float, coord2.strip("() ").split(","))
                    QMessageBox.information(self, "Success", f"Coordinates saved:\nPoint 1: ({lat1}, {lon1})\nPoint 2: ({lat2}, {lon2})")
                   
                    output_file = folder_project+"/"+self.city_name_manual+"_boundary.gpkg"
                    self.create_square_geopackage(output_file)
    
                except ValueError:
                    QMessageBox.warning(self, "Input Error", "Invalid coordinate format. Use (lat, lon).")
            else:
                # Load the CSV file
                df = self.df
                lat_col = "latitude"
                lon_col = "longitude"
                # Check for required columns
                if lat_col not in df.columns or lon_col not in df.columns:
                    QMessageBox.warning(self, "Input Error", "The CSV file must have the correct format. Please refer to the user guidelines.")
        
                # Extract coordinates and create a polygon
                coordinates = list(zip(df[lon_col], df[lat_col]))  # Create tuples of (lon, lat)
                polygon = Polygon(coordinates)
        
                # Create a GeoDataFrame
                gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:4326")
                
                output_gpkg = folder_project+"/"+self.city_name_manual+"_boundary.gpkg"
                # Save as a GeoPackage
                gdf.to_file(output_gpkg, driver="GPKG", layer="polygon_layer")
                print(f"GeoPackage saved to {output_gpkg}")
                
        elif self.main_window.insp_method == 1: 
            
            """Save the entered coordinates or handle the uploaded CSV file."""
            coord1 = self.coord_1.text().strip()
            coord2 = self.coord_2.text().strip()
                 
            try:
                self.get_city_name()
                QMessageBox.information(self, "Success", "Done! Please click save and continue button")
            except:
                QMessageBox.warning(self, "Error", "Please check that all the required input files are completed appropriately.") 
            
            if self.main_window:
                try:
                    # Access the main window's variable
                    folder_project = self.main_window.output_folder_value.text()
                except AttributeError as e:
                    QMessageBox.warning(self, "Error", f"Failed to access main window: {e}") 
                    
            self.main_window.file_name = self.output_value_specific.text()
            output_gpkg = folder_project+"/"+self.main_window.file_name+".gpkg"
        
            # Check if the GeoPackage file already exists
            if os.path.exists(output_gpkg):
                os.remove(output_gpkg)  # Delete the file to ensure only one layer is created
            
            # Load the CSV file
            df = self.df

            # Create geometries for the points using latitude and longitude
            geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]

            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry)

            # Set the coordinate reference system (CRS) to WGS84 (latitude/longitude)
            gdf.set_crs('EPSG:4326', inplace=True)

            # Save the GeoDataFrame to a file, if needed (e.g., to GeoPackage or Shapefile)
            gdf.to_file(output_gpkg, driver='GPKG')  # This saves as GeoPackage
        
        ##### --------- Local Method ---------- #########
        elif self.main_window.insp_method == 2: 
            self.main_window.file_name_local = self.output_value_local.text()
            try:
                self.get_city_name()
                QMessageBox.information(self, "Success", "Done! Please click save and continue button")
            except:
                QMessageBox.warning(self, "Error", "Please check that all the required input files are completed appropriately.")
        
        ##### --------- Extrapolation Method ---------- #########
        elif self.main_window.insp_method == 3: 
            QMessageBox.information(self, "Success", "Done! Please click save and continue button")            
            
    ############ Folder Selection ################
    def select_folder(self):
        """Open a folder selection dialog and display the selected folder in a text output."""
        self.main_window.folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        folder_display = self.main_window.folder_path.rsplit("/", 1)[-1]
        if self.main_window.folder_path:  # If a folder is selected
            self.local_folder_path.setText(folder_display)
           
            
    ############ Polygon method selection ################       
    def check_data_default (self):
        """Check if polygon method was selected"""
        if self.main_window.insp_method !=0:
            self.check_default = False
            QMessageBox.warning(self, "Input Error", "This is not the inspection method selected")
        else:
            self.check_default = True
            self.upload_csv(self.label)
            
    
    ############ Specific method selection ################ 
    def check_data_specific (self):
        """Check if specific method was selected"""
        if self.main_window.insp_method !=1:
            self.check_specific = False
            QMessageBox.warning(self, "Input Error", "This is not the inspection method selected")
        else:
            self.check_specific = True
            self.upload_csv(self.specific_path)
      
            
    ############ Local method selection ################         
    def check_data_local (self):
        """Check if Local method was selected"""
        if self.main_window.insp_method !=2:
            self.check_local = False
            QMessageBox.warning(self, "Input Error", "This is not the inspection method selected")
        else:
            self.check_local = True
            self.upload_csv(self.local_path)
            
    def check_data_local_folder (self):
        """Check if Local method was selected"""
        if self.main_window.insp_method !=2:
            self.check_local = False
            QMessageBox.warning(self, "Input Error", "This is not the inspection method selected")
        else:
            self.check_local = True
            self.select_folder()
            
    ############ Extrapolation method selection ################         
    def check_data_extra_buildings (self):
        """Check if Local method was selected"""
        if self.main_window.insp_method !=3:
            self.extrapolation_check = False
            QMessageBox.warning(self, "Input Error", "This is not the inspection method selected")
        else:
            self.extrapolation_check = True
            self.upload_csv(self.building_extra_path)
            
    def check_data_extra_neighbors (self):
        """Check if Local method was selected"""
        if self.main_window.insp_method !=3:
            self.extrapolation_check = False
            QMessageBox.warning(self, "Input Error", "This is not the inspection method selected")
        else:
            self.extrapolation_check = True
            self.upload_csv(self.example_building_path)
            
        
    ############ Building population ################  
    def building_polulation(self):
        """
        Populate building data based on the selected inspection method.
    
        This method retrieves building data by either extracting footprints from OpenStreetMap 
        (OSM) or setting a population flag depending on the inspection method. If the user 
        selects a local inspection method, it ensures that the number of images does not 
        exceed three.
    
        Args:
            None. The method relies on instance attributes such as `main_window.insp_method`, 
            `building_value_default`, and `n_image_local_value`.
    
        Effects:
            - If `insp_method == 0` (polygon method):
                - Retrieves the administrative boundary.
                - Downloads building footprints from OSM.
                - Sets the building population value in the UI.
            - If `insp_method == 1` (specific method):
                - Sets `self.population = True` (indicating a specific dataset is used).
            - If `insp_method == 2` (local method):
                - Sets `self.population = False`.
                - Displays a warning if more than three images are selected.
    
        Notes:
            - The method relies on `GUI_geofiles.get_administrative_boundary()` and 
              `GUI_geofiles.download_building_footprints()` for data retrieval.
            - The UI prevents selecting more than three images for local inspections.
        """
        if self.main_window.insp_method == 0:
            # Polygon
            # Get polygon based on coordinates
            GUI_geofiles.get_administrative_boundary(self.main_window)
            # Get building footprint from OSM
            self.population = GUI_geofiles.download_building_footprints(self.main_window)
            # Get building population 
            self.building_value_default.setText(str(self.population)) 
        elif self.main_window.insp_method == 1:
            # Specific
            self.population = True
        elif self.main_window.insp_method == 2:
            # Local folder
            self.population = False
            if int(self.n_image_local_value.currentText()) > 3:
                QMessageBox.warning(self, "Input Error", "Please select a number of images equal to or less than 3.")
        elif self.main_window.insp_method == 3:
            # Extrapoaltion folder
            self.population = False
            self.main_window.extrapolation_name = self.output_value_extra.text()
            
      
    def building_sample(self):
        """
        Generate a building sample based on the selected inspection method.
    
        This method processes building data according to the chosen inspection method. 
        It either creates a centroid layer for specific/local methods or extracts a 
        random subset of buildings for the polygon method.
    
        Args:
            None. The method relies on instance attributes such as `population`, 
            `main_window`, and `sample_size_default`.
    
        Effects:
            - If `self.population == True` (specific method):
                - Calls `GUI_geofiles.create_centroid_layer()` to generate building centroids.
                - Closes the dialog (`accept()`).
            - If `self.population == False` (local method):
                - Simply closes the dialog (`accept()`).
            - If `self.population is not None` (polygon method):
                - Extracts a random subset of buildings using `extract_random_subset()`.
                - Creates a centroid layer from the selected subset.
                - Closes the dialog (`accept()`).
    
        Notes:
            - The method ensures the appropriate processing steps are taken depending 
              on the inspection method.
            - The `GUI_geofiles` module is responsible for performing geospatial operations.
        """
        # Checking is the inspection mode correspond to specific
        if self.population == True:
            GUI_geofiles.create_centroid_layer(self.main_window)   
            self.accept()
        elif self.population == False: # Local method
            self.accept()
        elif self.population != None:  # Polygon method
            # Extract a random subset 
            GUI_geofiles.extract_random_subset(self.main_window, self.sample_size_default.text())
            # Create a point layer and extract the coordinates of a subset of buildings
            GUI_geofiles.create_centroid_layer(self.main_window)   
            self.accept()
 
        
    def upload_csv(self, label):
        """
        Open a file dialog to upload a CSV file and process it.
        
        This method allows the user to select a CSV file using a file dialog. If a file 
        is selected, it attempts to read it into a pandas DataFrame and updates the UI 
        label with the file name. If an error occurs during reading, an appropriate 
        message is displayed.
        
        Args:
            label (QLabel): The UI label to update with the uploaded file name.
        
        Effects:
            - Opens a file dialog for CSV file selection.
            - Stores the selected file path in `self.main_window.file_local_csv`.
            - Loads the CSV data into `self.df` if the file is valid.
      
        Notes:
            - Only CSV files are explicitly filtered in the file dialog.
        """
        
        options = QFileDialog.Options()
        # File path
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        # Send path to UI
        self.main_window.file_local_csv = file_path
        
        try:
            if label == self.building_extra_path:
                self.gui_methods.building_extra_path = file_path
            elif label == self.example_building_path:
                self.gui_methods.example_building_path = file_path
        except:
            pass
            
        if file_path:
            try:
                # Upload csv with building coordinates
                self.df = pd.read_csv(file_path)
                file_path = file_path.rsplit("/", 1)[-1]
                label.setText(file_path)
            except FileNotFoundError:
                label.setText("File not found.")
            except pd.errors.ParserError:
                label.setText("Error parsing CSV file. Check the format.")
            except Exception as e: #catch other exceptions
                label.setText(f"An error occurred: {e}")
        else:
            label.setText("No file selected.")
            QMessageBox.warning(self, "Input Error", "No file selected.")
            
            
    def extrapolation_values(self):
        if self.example_building_path.text() == "filename.csv":
            QMessageBox.warning(self, "Input Error", "Please select neighbor building file information.")
        else:
            n_neigh = pd.read_csv(self.gui_methods.example_building_path)
            dialog = extrapolation_options_window(parent=self, n_neighbors=n_neigh)
            dialog.building_value_op1.setText(str(n_neigh.shape[0]))
            
            dialog.n_building_op2_button.clicked.connect(lambda: 
                                                         dialog.distance_neighbor(self.gui_methods.building_extra_path, 
                                                                                  self.gui_methods.example_building_path))
            
            if dialog.exec_() == QDialog.Accepted:
                self.main_window.neighbor_method = dialog.neighbor_method
                if dialog.neighbor_method == 1:
                    self.main_window.n_neighbor = dialog.selected_value_op1.text()
                elif dialog.neighbor_method == 2:
                    self.main_window.distance = dialog.distance_value.text()
                    self.main_window.n_neighbor = dialog.building_value_op2.text()
                    

                    
                
        
            