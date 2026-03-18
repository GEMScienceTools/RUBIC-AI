from geopy.distance import geodesic
from collections import defaultdict
import sys
import numpy as np 
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd

from methods.utilities import select_output_folder, upload_csv

class knn_options_window(QtWidgets.QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.method = parent

        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        #
        DESIGN_WIDTH = 1920
        DESIGN_HEIGHT = 1080
        DESIGN_DPI = 96 * 1.25  # 125% Windows baseline -> 120 DPI
        
        # Scale the GUI based on resolution
        sf_x = screen_width / DESIGN_WIDTH
        sf_y = screen_height / DESIGN_HEIGHT
        sf_factor = np.sqrt(sf_x * sf_y)

        # DPI-based scale
        # Get a reliable DPI value
        if sys.platform.startswith("win"):
            # Windows: use ctypes to get real DPI
            import ctypes
            LOGPIXELSX = 88
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
            ctypes.windll.user32.ReleaseDC(0, hdc)
        else:
            # macOS / Linux: start with logical DPI
            dpi = screen.logicalDotsPerInch()
            # If logical DPI looks weird, fallback to physical
            if dpi < 60 or dpi > 200:
                dpi = screen.physicalDotsPerInch()

        # Normalize to your design environment (Windows @ 125% = 120 DPI)
        # If dpi == 120 => scale_dpi = 1 (your original machine)
        scale_dpi = DESIGN_DPI / dpi
        
        # For geometry: mainly resolution-based
        sf_x = sf_factor
        sf_y = sf_factor
        # Scale the GUI based on resolution
        sf_font = sf_factor * scale_dpi

        self.setObjectName("DataSetting")
        self.resize(int(1210 * sf_x), int(600 * sf_y))
        self.setWindowTitle("Setting input files")

        # === UI Elements Start ===
        self.data_frame = QtWidgets.QWidget(self)
        self.data_frame.setObjectName("data_frame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.data_frame)

        self.w_title = QtWidgets.QLabel("Setting input files", self.data_frame)
        self.w_title.setGeometry(QtCore.QRect(int(510 * sf_x), int(0), int(191 * sf_x), int(41 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(12 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.w_title.setFont(font)

        self.save_button = QtWidgets.QPushButton(self.data_frame)
        self.save_button.setGeometry(QtCore.QRect(int(510 * sf_x), int(530 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self.select_method)
        
        self.backg_1 = QtWidgets.QLabel(self.data_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(39 * sf_y), int(591 * sf_x), int(271 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(209, 255, 165);")
        self.backg_1.setText("")
        self.backg_1.setObjectName("backg_1")
        
        self.unclassfied_path = QtWidgets.QLabel(self.data_frame)
        self.unclassfied_path.setGeometry(QtCore.QRect(int(280 * sf_x), int(220 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.unclassfied_path.setFont(font)
        self.unclassfied_path.setObjectName("unclassfied_path")
        
        self.unclassified_button = QtWidgets.QPushButton(self.data_frame)
        self.unclassified_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(215 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.unclassified_button.setFont(font)
        self.unclassified_button.setObjectName("unclassified_button")
        self.unclassified_button.clicked.connect(self.data_extrapolation)
        
        self.output_label_manual = QtWidgets.QLabel(self.data_frame)
        self.output_label_manual.setGeometry(QtCore.QRect(int(30 * sf_x), int(90 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_manual.setFont(font)
        self.output_label_manual.setObjectName("output_label_manual")
        
        self.output_manual_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_manual_value.setGeometry(QtCore.QRect(int(170 * sf_x), int(90 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.output_manual_value.setFont(font)
        self.output_manual_value.setObjectName("output_manual_value")
        
        self.manual_info_path = QtWidgets.QLabel(self.data_frame)
        self.manual_info_path.setGeometry(QtCore.QRect(int(280 * sf_x), int(180 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.manual_info_path.setFont(font)
        self.manual_info_path.setObjectName("manual_info_path")
        
        self.b_info_button = QtWidgets.QPushButton(self.data_frame)
        self.b_info_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(175 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.b_info_button.setFont(font)
        self.b_info_button.setObjectName("b_info_button")
        self.b_info_button.clicked.connect(self.data_existing)
        
        self.manual_op = QtWidgets.QCheckBox(self.data_frame)
        self.manual_op.setGeometry(QtCore.QRect(int(30 * sf_x), int(50 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.manual_op.setFont(font)
        self.manual_op.setObjectName("manual_op")
        
        self.backg_2 = QtWidgets.QLabel(self.data_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(610 * sf_x), int(40 * sf_y), int(591 * sf_x), int(271 * sf_y)))
        self.backg_2.setStyleSheet("background-color: rgb(255, 233, 167);")
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        
        self.dl_op = QtWidgets.QCheckBox(self.data_frame)
        self.dl_op.setGeometry(QtCore.QRect(int(640 * sf_x), int(45 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.dl_op.setFont(font)
        self.dl_op.setObjectName("dl_op")
        
        self.output_label_dl = QtWidgets.QLabel(self.data_frame)
        self.output_label_dl.setGeometry(QtCore.QRect(int(640 * sf_x), int(85 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_dl.setFont(font)
        self.output_label_dl.setObjectName("output_label_dl")
        
        self.output_dl_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_dl_value.setGeometry(QtCore.QRect(int(780 * sf_x), int(85 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.output_dl_value.setFont(font)
        self.output_dl_value.setObjectName("output_dl_value")
        
        self.unclassfied_dl_path = QtWidgets.QLabel(self.data_frame)
        self.unclassfied_dl_path.setGeometry(QtCore.QRect(int(890 * sf_x), int(210 * sf_y), int(291 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.unclassfied_dl_path.setFont(font)
        self.unclassfied_dl_path.setObjectName("unclassfied_dl_path")
        
        self.unclassified_dl_button = QtWidgets.QPushButton(self.data_frame)
        self.unclassified_dl_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(210 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.unclassified_dl_button.setFont(font)
        self.unclassified_dl_button.setObjectName("unclassified_dl_button")
        self.unclassified_dl_button.clicked.connect(self.data_extrapolation_dl)
        
        # KNN Coordinate Button
        self.coord_knn_button = QtWidgets.QPushButton(self.data_frame)
        self.coord_knn_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(170 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.coord_knn_button.setFont(font)
        self.coord_knn_button.setObjectName("coord_knn_button")
        self.coord_knn_button.clicked.connect(self.data_existing_dl)
                                           
        # KNN Coordinate Value Label
        self.coord_value_knn = QtWidgets.QLabel(self.data_frame)
        self.coord_value_knn.setGeometry(QtCore.QRect(int(890 * sf_x), int(170 * sf_y), int(291 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.coord_value_knn.setFont(font)
        self.coord_value_knn.setObjectName("coord_value_knn")
        
        self.tableWidget = QtWidgets.QTableWidget(self.data_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(10 * sf_x), int(320 * sf_y), int(1170 * sf_x), int(201 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        
        # K Value Label (Manual)
        self.k_value_label = QtWidgets.QLabel(self.data_frame)
        self.k_value_label.setGeometry(QtCore.QRect(int(30 * sf_x), int(259 * sf_y), int(81 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.k_value_label.setFont(font)
        self.k_value_label.setObjectName("k_value_label")
        
        # K Value SpinBox (Manual)
        self.k_value_manual = QtWidgets.QSpinBox(self.data_frame)
        self.k_value_manual.setGeometry(QtCore.QRect(int(110 * sf_x), int(260 * sf_y), int(51 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.k_value_manual.setFont(font)
        self.k_value_manual.setProperty("value", 10)
        self.k_value_manual.setObjectName("k_value_manual")
        
        # K Value Label (Deep Learning)
        self.k_value_label_dl = QtWidgets.QLabel(self.data_frame)
        self.k_value_label_dl.setGeometry(QtCore.QRect(int(640 * sf_x), int(260 * sf_y), int(81 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.k_value_label_dl.setFont(font)
        self.k_value_label_dl.setObjectName("k_value_label_dl")
        
        # K Value SpinBox (Deep Learning)
        self.k_value_dl = QtWidgets.QSpinBox(self.data_frame)
        self.k_value_dl.setGeometry(QtCore.QRect(int(720 * sf_x), int(260 * sf_y), int(51 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.k_value_dl.setFont(font)
        self.k_value_dl.setProperty("value", 10)
        self.k_value_dl.setObjectName("k_value_dl")
        
        # Saved Path Label (Manual)
        self.saved_path_manual = QtWidgets.QLabel(self.data_frame)
        self.saved_path_manual.setGeometry(QtCore.QRect(int(280 * sf_x), int(140 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.saved_path_manual.setFont(font)
        self.saved_path_manual.setObjectName("saved_path_manual")
        
        # Output Path Button (Manual)
        self.output_path_button = QtWidgets.QPushButton(self.data_frame)
        self.output_path_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(135 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.output_path_button.setFont(font)
        self.output_path_button.setObjectName("output_path_button")
        self.output_path_button.clicked.connect(self._on_select_output_folder)
        
          
        # Saved Path Label (DL)
        self.saved_path_dl = QtWidgets.QLabel(self.data_frame)
        self.saved_path_dl.setGeometry(QtCore.QRect(int(890 * sf_x), int(130 * sf_y), int(291 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.saved_path_dl.setFont(font)
        self.saved_path_dl.setObjectName("saved_path_dl")
        
        # Output Path Button (DL)
        self.output_path_dl_button = QtWidgets.QPushButton(self.data_frame)
        self.output_path_dl_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(130 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.output_path_dl_button.setFont(font)
        self.output_path_dl_button.setObjectName("output_path_dl_button")
        self.output_path_dl_button.clicked.connect(self._on_select_output_folder)
        
        # Set default values manually
        self.output_manual_value.setText("KNN_manual")
        self.output_dl_value.setText("KNN_dl")
        self.manual_op.setText("Upload data manually")
        self.dl_op.setText("Deep learning model")
        self.save_button.setText("Save and continue")
        self.unclassified_button.setText("Unclassified building coords")
        self.unclassfied_path.setText("filename.csv")
        self.output_label_manual.setText( "Output name:")
        self.b_info_button.setText("Buildings with information")
        self.manual_op.setText("Upload data manually")
        self.manual_info_path.setText("filename.csv")       
        self.output_label_dl.setText("Output name:")
        self.unclassfied_dl_path.setText("filename.csv")
        self.unclassified_dl_button.setText("Unclassified building coords")
        self.coord_knn_button.setText("Upload building coordinates")
        self.coord_value_knn.setText("filename.csv")
        self.k_value_label.setText("K value:")
        self.k_value_label_dl.setText("K value:")
        self.saved_path_dl.setText("path/where/save/the/results")
        self.output_path_dl_button.setText("Select output folder")
        self.output_path_button.setText("Select output folder")
        self.saved_path_manual.setText("path/where/save/the/results")
           
        
        # ==============================================================
        # KNN method functions
        # ==============================================================
        
    def _on_select_output_folder(self):
        """
        Opens a folder selection dialog, stores the selected output folder path, and displays 
        the folder name in the interface.
        """
        self.folder_path, self.display_folder = select_output_folder(self)
        if self.manual_op.isChecked():
            self.saved_path_manual.setText(self.display_folder)
        elif self.dl_op.isChecked():
            self.saved_path_dl.setText(self.display_folder)
            
    def data_existing(self):
        """
        Loads the reference dataset from the selected manual input file and stores its path.
        """
        self.sf_font = self.method.sf_font
        self.label_path = self.manual_info_path
        self.info_existing = upload_csv(self)
        
    def data_extrapolation(self):
        """
        Loads the reference dataset from the selected manual input file and stores its path.
        """
        self.sf_font = self.method.sf_font
        self.label_path = self.unclassfied_path
        self.info_pending = upload_csv(self)
        
    def data_existing_dl(self):
        """
        Loads the reference dataset from the selected manual input file and stores its path.
        """
        self.sf_font = self.method.sf_font
        self.label_path = self.coord_value_knn
        self.info_existing = upload_csv(self)
        
    def data_extrapolation_dl(self):
        """
        Loads the reference dataset from the selected manual input file and stores its path.
        """
        self.sf_font = self.method.sf_font
        self.label_path = self.unclassfied_dl_path
        self.info_pending = upload_csv(self)
        
        
    def select_method(self):
        """
        Validates the selected method, ensures that only one option is chosen, and saves the 
        corresponding input settings before continuing.
        """
        # Check how many checkboxes are checked
        checked_count = sum([self.manual_op.isChecked(), 
                              self.dl_op.isChecked()])
        # Check proper setting
        if checked_count > 1:
            QtWidgets.QMessageBox.warning(self, "Selection Warning", "You can only select one method at a time")
        elif checked_count == 0:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please select one method")
        else:
            if self.manual_op.isChecked():
                try:
                    self.info_existing
                    self.info_pending
                    self.accept()
                except:
                    QtWidgets.QMessageBox.warning(self, "Input Error", "There are missing the inputs files")
            elif self.dl_op.isChecked():
                #######===========  Input parameters =========###########
                try:
                    self.coord_reference = self.info_existing
                    self.coord_reference_building_feature_path = self.output_dl_value.text()+"_reference_results.csv"
                    self. data_extrapolation = self.info_pending
                    self.knn_dl_saved_path = self.output_dl_value.text()+".csv"
                    self.accept()
                except:
                    QtWidgets.QMessageBox.warning(self, "Input Error", "There are missing the inputs files")
 
            
#####################################################################################################    
############## --------- KNN function for calling gui_methods.py ---------------#####################
##################################################################################################### 


# Function to calculate Geodesic distance (in km)
def geodesic_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the geodesic distance in kilometers between two geographic coordinates.
    """
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).km


# Function to find 3 nearest neighbors using geodesic distance
def find_nearest_neighbors_geodesic(input_row, info_df, n_neighbors):
    """
    Finds the nearest neighbors to an input location using geodesic distance and returns 
    their data along with the computed distances in kilometers.
    """
    # Apply geodesic distance for each row in reference dataframe
    distances = info_df.apply(
        lambda row: geodesic_distance(
            input_row['latitude'],
            input_row['longitude'],
            row['latitude'],
            row['longitude']
        ), axis=1)

    nearest_indices = distances.nsmallest(n_neighbors).index
    
    # Extract neighbor data
    neighbor_data = info_df.loc[nearest_indices].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Add distance column to neighbor data
    neighbor_data['distance_km'] = distances.loc[nearest_indices].values
    return neighbor_data


# Function to compute taxonomy probabilities using inverse-distance weighted soft voting
def compute_taxonomy_distribution_full_structure(nearest_neighbors, input_row):
    """
    Computes a probability distribution of taxonomies from the nearest neighbors using 
    inverse-distance weighted voting and returns the results with the corresponding 
    building attributes.
    """
    class_weights = defaultdict(float)

    # Assign weights to each neighbor based on the chosen kernel
    for _, row in nearest_neighbors.iterrows():
        dist = row['distance_km']
        label = row['taxonomy']
        
        # Compute weight based on kernel
        weight = 1 / (dist + 1e-6)  # Avoid division by zero
        class_weights[label] += weight

    # Normalize weights to create a probability distribution
    total_weight = sum(class_weights.values())
    probs = {label: weight / total_weight for label, weight in class_weights.items()}

    # Prepare output rows
    distribution_rows = []
    for taxonomy, prob in probs.items():
        # Take a representative row (first one with the taxonomy)
        taxonomy_row = nearest_neighbors[nearest_neighbors['taxonomy'] == taxonomy].iloc[0]
        
        distribution_rows.append({
            'id': input_row['id'],
            'latitude': input_row['latitude'],
            'longitude': input_row['longitude'],
            'country': taxonomy_row['country'],
            'city': taxonomy_row['city'],
            'material': taxonomy_row['material'],
            'llrs': taxonomy_row['llrs'],
            'code_level': taxonomy_row['code_level'],
            'n_stories': taxonomy_row['n_stories'],
            'occupancy': taxonomy_row['occupancy'],
            'block_position': taxonomy_row['block_position'],
            'roof_shape': taxonomy_row['roof_shape'],
            'roof_material': taxonomy_row['roof_material'],
            'taxonomy': taxonomy,
            'probability': prob
        })

    return distribution_rows


def extrapolation_existing_reference(data_existing , data_extrapolation, saved_path, n_neigh):
    """
    Applies nearest-neighbor extrapolation to each input record, computes the taxonomy 
    probability distribution based on nearby reference data, and saves the results to a CSV file.
    """
    final_distribution_list_full = []   
    # Iterate over each building with no image
    for idx, input_row in data_extrapolation.iterrows():
        # Find 3 nearest neighbors using geodesic distance
        nearest_neighbors_value = find_nearest_neighbors_geodesic(input_row, data_existing, n_neigh)
        # Compute taxonomy-based distributions with full structure
        distribution_rows = compute_taxonomy_distribution_full_structure(nearest_neighbors_value, input_row)
        
        # Append to final result
        final_distribution_list_full.extend(distribution_rows)
    
    # Convert final list to DataFrame
    final_distribution_df_full = pd.DataFrame(final_distribution_list_full)
    # Export to CSV
    final_distribution_df_full.to_csv(saved_path, index=False)
  