from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
from geopy.geocoders import Nominatim
import ctypes

from methods.gui_gis import GUI_geofiles

class PolygonSetting(QtWidgets.QDialog):
    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method

        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / 1920 
        sf_y = screen_height / 1080 
        
        # Reference DPI for 100% scaling
        LOGPIXELSX = 88
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        scale =  1.25/(dpi / 96)  # 96 DPI is 100%

        # Scale the GUI based on resolution
        sf_x_font = sf_x * scale

        self.setWindowTitle("Polygon Method Input")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(618 * sf_x), int(608 * sf_y))

        self.coord_frame = QtWidgets.QWidget(self)

        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x_font))
        self.backg_1 = QtWidgets.QLabel(self.coord_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(601 * sf_x), int(551 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(255, 224, 185);")

        self.polygon_label = QtWidgets.QLabel(self.coord_frame)
        self.polygon_label.setGeometry(QtCore.QRect(int(210 * sf_x), int(20 * sf_y), int(221 * sf_x), int(21 * sf_y)))
        title_font = QtGui.QFont()
        title_font.setPointSize(int(10 * sf_x_font))
        title_font.setBold(True)
        title_font.setItalic(True)
        title_font.setUnderline(True)
        self.polygon_label.setFont(title_font)
        self.polygon_label.setText("Polygon Method Input")

        self.output_label_polygon = QtWidgets.QLabel(self.coord_frame)
        self.output_label_polygon.setGeometry(QtCore.QRect(int(20 * sf_x), int(50 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        label_font = QtGui.QFont()
        label_font.setPointSize(int(10 * sf_x_font))
        label_font.setBold(True)
        self.output_label_polygon.setFont(label_font)
        self.output_label_polygon.setText("Output name:")

        self.output_polygon = QtWidgets.QLineEdit(self.coord_frame)
        self.output_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(50 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.output_polygon.setFont(font)
        self.output_polygon.setText("polygon_building")

        self.path_out_folder_bt = QtWidgets.QPushButton(self.coord_frame)
        self.path_out_folder_bt.setGeometry(QtCore.QRect(int(20 * sf_x), int(95 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.path_out_folder_bt.setFont(font)
        self.path_out_folder_bt.setText("Select output folder")
        self.path_out_folder_bt.clicked.connect(self.select_output_folder)

        self.output_folder_value = QtWidgets.QLabel(self.coord_frame)
        self.output_folder_value.setGeometry(QtCore.QRect(int(270 * sf_x), int(100 * sf_y), int(331 * sf_x), int(21 * sf_y)))
        self.output_folder_value.setFont(font)
        self.output_folder_value.setText("path/where/you/want/to/save/your/results")

        self.n_building = QtWidgets.QLabel(self.coord_frame)
        self.n_building.setGeometry(QtCore.QRect(int(20 * sf_x), int(230 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        self.n_building.setFont(label_font)
        self.n_building.setText("N° footprints:")

        self.building_value_polygon = QtWidgets.QLabel(self.coord_frame)
        self.building_value_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(230 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        self.building_value_polygon.setFont(font)
        self.building_value_polygon.setText("0000")

        self.sample_size = QtWidgets.QLabel(self.coord_frame)
        self.sample_size.setGeometry(QtCore.QRect(int(20 * sf_x), int(270 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        self.sample_size.setFont(label_font)
        self.sample_size.setText("Sample size:")

        self.sample_size_polygon = QtWidgets.QLineEdit(self.coord_frame)
        self.sample_size_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(270 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        self.sample_size_polygon.setFont(font)
        self.sample_size_polygon.setText("10")

        self.csv_button_polygon = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_polygon.setGeometry(QtCore.QRect(int(20 * sf_x), int(140 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.csv_button_polygon.setFont(font)
        self.csv_button_polygon.setText("Upload file with coordinates")
        self.csv_button_polygon.clicked.connect(self.upload_csv)

        self.polygon_path = QtWidgets.QLabel(self.coord_frame)
        self.polygon_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(140 * sf_y), int(291 * sf_x), int(31 * sf_y)))
        self.polygon_path.setFont(font)
        self.polygon_path.setText("filename.csv")

        self.load_data_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_data_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(190 * sf_y), int(271 * sf_x), int(31 * sf_y)))
        self.load_data_button.setFont(label_font)
        self.load_data_button.setText("Get footprints available")
        self.load_data_button.clicked.connect(self.save_coordinates)
        self.load_data_button.clicked.connect(self.building_polulation)

        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(220 * sf_x), int(570 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(label_font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.building_sample)

        # self.collection_mode = QtWidgets.QComboBox(self.coord_frame)
        # self.collection_mode.setGeometry(QtCore.QRect(int(250 * sf_x), int(310 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x_font))
        # self.collection_mode.setFont(font)
        # self.collection_mode.setObjectName("collection_mode")
        # self.collection_mode.addItem("Manual")
        # self.collection_mode.addItem("AI Powered")
        
        # self.feature_collection_label = QtWidgets.QLabel(self.coord_frame)
        # self.feature_collection_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(310 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x_font))
        # font.setBold(True)
        # font.setWeight(75)
        # self.feature_collection_label.setFont(font)
        # self.feature_collection_label.setObjectName("feature_collection_label")
        # self.feature_collection_label.setText("Feature collection mode:")
        
        self.tableWidget = QtWidgets.QTableWidget(self.coord_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(20 * sf_x), int(350 * sf_y), int(581 * sf_x), int(192 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)


    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if self.method:
            self.method.output_folder_value = folder_path
        if folder_path:
            folder_display = os.path.basename(folder_path)
            self.output_folder_value.setText(folder_display)

    def upload_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.polygon_path.setText(os.path.basename(file_path))
                if self.method:
                    self.method.file_polygon_csv = file_path

                df = self.df
                required_columns = ['ID', 'latitude', 'longitude']
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    QMessageBox.warning(self, "Missing Columns", f"Required columns missing: {', '.join(missing)}")
                    return
                QMessageBox.information(self, "Success", "Done! Please click load data button.")
                self.preview_data()
            except Exception as e:
                self.polygon_path.setText(f"Error: {str(e)}")
                QMessageBox.warning(self, "Input Error", f"Error loading CSV:\n{e}")
        else:
            self.polygon_path.setText("No file selected.")
            QMessageBox.warning(self, "Input Error", "No file selected.")
            
    def preview_data(self):
        if hasattr(self, 'df') and not self.df.empty:
            preview_df = self.df.head(10)  # Only show first 10 rows
    
            self.tableWidget.clear()
            self.tableWidget.setRowCount(len(preview_df))
            self.tableWidget.setColumnCount(len(preview_df.columns))
            self.tableWidget.setHorizontalHeaderLabels(preview_df.columns)
    
            for row in range(len(preview_df)):
                for column in range(len(preview_df.columns)):
                    value = str(preview_df.iloc[row, column])
                    item = QtWidgets.QTableWidgetItem(value)
                    self.tableWidget.setItem(row, column, item)
    
            self.tableWidget.resizeColumnsToContents()
        else:
            QMessageBox.warning(self, "No Data", "No data available to preview. Please upload a valid CSV first.")

    def save_coordinates(self):
        try:
            lat = self.df.iloc[0,1]
            lon = self.df.iloc[0,2]
            geolocator = Nominatim(user_agent="city_name_locator")
            location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
            if location and 'address' in location.raw:
                address = location.raw['address']
                self.city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                self.country = address.get('country', 'Unknown')
                self.method.city = self.city
                self.method.country = self.country
                self.city_name_manual = self.city + "_" + self.country

            output_gpkg = os.path.join(self.method.output_folder_value, f"{self.city_name_manual}_boundary.gpkg")
            self.boundary_path = output_gpkg
            if os.path.exists(output_gpkg):
                os.remove(output_gpkg)

            df = self.df
            if "latitude" not in df.columns or "longitude" not in df.columns:
                QMessageBox.warning(self, "Input Error", "The CSV file must have 'latitude' and 'longitude' columns.")
                return

            coordinates = list(zip(df["longitude"], df["latitude"]))
            polygon = Polygon(coordinates)
            gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:4326")
            gdf.to_file(output_gpkg, driver="GPKG", layer="polygon_layer")

            QMessageBox.information(
                self,
                "Success",
                f"GeoPackage successfully saved to:\n{output_gpkg}\n\n\u26a0\ufe0f Please verify the sample size before continuing.\n"
                "The sample size should be smaller than the total number of buildings."
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not generate GPKG:\n{str(e)}")

    def building_polulation(self):
        self.population = GUI_geofiles.download_building_footprints(self)
        self.building_value_polygon.setText(str(self.population))

    def building_sample(self):
        GUI_geofiles.extract_random_subset(self, self.sample_size_polygon.text())
        GUI_geofiles.create_centroid_layer(self)
        self.method.output_polygon = self.output_polygon
        self.accept()
