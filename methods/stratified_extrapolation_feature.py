import os
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

import pandas as pd
import numpy as np
from methods.get_building_orientation import get_street_view_image
import requests
     
from dl_stratified import predict_llrs_img, predict_material_img, predict_code_img, predict_roof_shape_img
from dl_stratified import predict_occupancy_img, predict_block_position_img, predict_n_stories_img, predict_roof_material_img
####################################################

class stratified_extrapolation(QtWidgets.QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window

        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080

        self.setObjectName("DataSetting")
        self.resize(int(1210 * sf_x), int(750 * sf_y))
        self.setWindowTitle("Setting input files")
        
        # === UI Elements Start ===
        self.data_frame = QtWidgets.QWidget(self)
        self.data_frame.setObjectName("data_frame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.data_frame)
        
        # Title Label
        self.w_title = QtWidgets.QLabel(self.data_frame)
        self.w_title.setGeometry(QtCore.QRect(int(510 * sf_x), int(0 * sf_y), int(191 * sf_x), int(41 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(12 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.w_title.setFont(font)
        self.w_title.setObjectName("w_title")
        
        # Save Button
        self.save_button = QtWidgets.QPushButton(self.data_frame)
        self.save_button.setGeometry(QtCore.QRect(int(510 * sf_x), int(690 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self.select_method)
        
        # Background Label
        self.backg_1 = QtWidgets.QLabel(self.data_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(39 * sf_y), int(591 * sf_x), int(421 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(209, 255, 165);")
        self.backg_1.setText("")
        self.backg_1.setObjectName("backg_1")
        
        # Population Button
        self.population_button = QtWidgets.QPushButton(self.data_frame)
        self.population_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(415 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.population_button.setFont(font)
        self.population_button.setObjectName("population_button")
        
        # Population Data Path Label
        self.population_data_path = QtWidgets.QLabel(self.data_frame)
        self.population_data_path.setGeometry(QtCore.QRect(int(280 * sf_x), int(420 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.population_data_path.setFont(font)
        self.population_data_path.setObjectName("population_data_path")
        
        # Output Label for Existing
        self.output_label_existing = QtWidgets.QLabel(self.data_frame)
        self.output_label_existing.setGeometry(QtCore.QRect(int(30 * sf_x), int(90 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_existing.setFont(font)
        self.output_label_existing.setObjectName("output_label_existing")
        
        # Output Value LineEdit for Existing
        self.output_existing_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_existing_value.setGeometry(QtCore.QRect(int(170 * sf_x), int(90 * sf_y), int(331 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.output_existing_value.setFont(font)
        self.output_existing_value.setObjectName("output_existing_value")
        
        # Distribution Button
        self.distribution_button = QtWidgets.QPushButton(self.data_frame)
        self.distribution_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(260 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.distribution_button.setFont(font)
        self.distribution_button.setObjectName("distribution_button")
        
        # Checkbox for Existing
        self.existing_check = QtWidgets.QCheckBox(self.data_frame)
        self.existing_check.setGeometry(QtCore.QRect(int(30 * sf_x), int(50 * sf_y), int(261 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.existing_check.setFont(font)
        self.existing_check.setObjectName("existing_check")

        
        # Distribution Path Label
        self.distrib_path = QtWidgets.QLabel(self.data_frame)
        self.distrib_path.setGeometry(QtCore.QRect(int(280 * sf_x), int(265 * sf_y), int(151 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.distrib_path.setFont(font)
        self.distrib_path.setObjectName("distrib_path")
        
        # Output Path Button (existing)
        self.output_path_button = QtWidgets.QPushButton(self.data_frame)
        self.output_path_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(135 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.output_path_button.setFont(font)
        self.output_path_button.setObjectName("output_path_button")
        
        # Saved Path (existing)
        self.saved_path_existing = QtWidgets.QLabel(self.data_frame)
        self.saved_path_existing.setGeometry(QtCore.QRect(int(280 * sf_x), int(140 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.saved_path_existing.setFont(font)
        self.saved_path_existing.setObjectName("saved_path_existing")
        
        # Table Widget
        self.tableWidget = QtWidgets.QTableWidget(self.data_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(10 * sf_x), int(470 * sf_y), int(1170 * sf_x), int(211 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        
        # Checkbox - New
        self.new_check = QtWidgets.QCheckBox(self.data_frame)
        self.new_check.setGeometry(QtCore.QRect(int(640 * sf_x), int(45 * sf_y), int(301 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.new_check.setFont(font)
        self.new_check.setObjectName("new_check")

        # Output Label - New
        self.output_label_new = QtWidgets.QLabel(self.data_frame)
        self.output_label_new.setGeometry(QtCore.QRect(int(640 * sf_x), int(85 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_new.setFont(font)
        self.output_label_new.setObjectName("output_label_new")
        
        # Output Value - New
        self.output_new_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_new_value.setGeometry(QtCore.QRect(int(780 * sf_x), int(85 * sf_y), int(331 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.output_new_value.setFont(font)
        self.output_new_value.setObjectName("output_new_value")
        
        # Background 2
        self.backg_2 = QtWidgets.QLabel(self.data_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(610 * sf_x), int(40 * sf_y), int(591 * sf_x), int(421 * sf_y)))
        self.backg_2.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        self.backg_2.lower()
        
        # Population Button - New
        self.population_new_button = QtWidgets.QPushButton(self.data_frame)
        self.population_new_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(165 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.population_new_button.setFont(font)
        self.population_new_button.setObjectName("population_new_button")
        self.population_new_button.clicked.connect(self.data_population)
        
        # Population Path - New
        self.population_new_path = QtWidgets.QLabel(self.data_frame)
        self.population_new_path.setGeometry(QtCore.QRect(int(890 * sf_x), int(170 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.population_new_path.setFont(font)
        self.population_new_path.setObjectName("population_new_path")
        
        # Saved Path - New
        self.saved_path_new = QtWidgets.QLabel(self.data_frame)
        self.saved_path_new.setGeometry(QtCore.QRect(int(890 * sf_x), int(130 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.saved_path_new.setFont(font)
        self.saved_path_new.setObjectName("saved_path_new")
        
        # Output Path Button - New
        self.output_path_new_button = QtWidgets.QPushButton(self.data_frame)
        self.output_path_new_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(125 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.output_path_new_button.setFont(font)
        self.output_path_new_button.setObjectName("output_path_new_button")
        self.output_path_new_button.clicked.connect(self.select_output_folder_new)
        
        # Validation Label
        self.valid_label = QtWidgets.QLabel(self.data_frame)
        self.valid_label.setGeometry(QtCore.QRect(int(40 * sf_x), int(220 * sf_y), int(431 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.valid_label.setFont(font)
        self.valid_label.setObjectName("valid_label")
        
        # Validation ComboBox
        self.valid_value = QtWidgets.QComboBox(self.data_frame)
        self.valid_value.setGeometry(QtCore.QRect(int(480 * sf_x), int(220 * sf_y), int(91 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.valid_value.setFont(font)
        self.valid_value.setObjectName("valid_value")
        self.valid_value.addItem("")
        self.valid_value.addItem("")
        
        # Extrapolation Mode Label
        self.extra_label = QtWidgets.QLabel(self.data_frame)
        self.extra_label.setGeometry(QtCore.QRect(int(40 * sf_x), int(180 * sf_y), int(181 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.extra_label.setFont(font)
        self.extra_label.setObjectName("extra_label")
        
        # Extrapolation Mode ComboBox
        self.extrap_mode = QtWidgets.QComboBox(self.data_frame)
        self.extrap_mode.setGeometry(QtCore.QRect(int(230 * sf_x), int(180 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.extrap_mode.setFont(font)
        self.extrap_mode.setObjectName("extrap_mode")
        self.extrap_mode.addItem("")
        self.extrap_mode.addItem("")
        
        self.ask_1 = QtWidgets.QLabel(self.data_frame)
        self.ask_1.setGeometry(QtCore.QRect(int(40 * sf_x), int(300 * sf_y), int(231 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.ask_1.setFont(font)
        self.ask_1.setObjectName("ask_1")
        
        self.ask_2 = QtWidgets.QLabel(self.data_frame)
        self.ask_2.setGeometry(QtCore.QRect(int(270 * sf_x), int(300 * sf_y), int(41 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.ask_2.setFont(font)
        self.ask_2.setObjectName("ask_2")
        
        self.ask_3 = QtWidgets.QLabel(self.data_frame)
        self.ask_3.setGeometry(QtCore.QRect(int(300 * sf_x), int(300 * sf_y), int(41 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.ask_3.setFont(font)
        self.ask_3.setObjectName("ask_3")
        
        self.manage_dist = QtWidgets.QLabel(self.data_frame)
        self.manage_dist.setGeometry(QtCore.QRect(int(40 * sf_x), int(330 * sf_y), int(321 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.manage_dist.setFont(font)
        self.manage_dist.setObjectName("manage_dist")
        
        self.manage_value = QtWidgets.QComboBox(self.data_frame)
        self.manage_value.setGeometry(QtCore.QRect(int(350 * sf_x), int(331 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.manage_value.setFont(font)
        self.manage_value.setObjectName("manage_value")
        self.manage_value.addItem("")
        self.manage_value.addItem("")
        
        self.distance_label = QtWidgets.QLabel(self.data_frame)
        self.distance_label.setGeometry(QtCore.QRect(int(40 * sf_x), int(370 * sf_y), int(161 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.distance_label.setFont(font)
        self.distance_label.setObjectName("distance_label")
        
        self.distance_value = QtWidgets.QSpinBox(self.data_frame)
        self.distance_value.setGeometry(QtCore.QRect(int(200 * sf_x), int(370 * sf_y), int(91 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.distance_value.setFont(font)
        self.distance_value.setObjectName("distance_value")
        self.distance_value.setMaximum(9999)
        
        # --- Features of Interest filter (Excel-like) ---
        self.features_label = QtWidgets.QLabel(self.data_frame)
        self.features_label.setGeometry(QtCore.QRect(int(640 * sf_x), int(210 * sf_y), int(181 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        self.features_label.setFont(font)
        self.features_label.setText("Features of interest:")
        
        self.features_btn = CheckFilterButton(values=[""], parent=self.data_frame, text="Feature strata")
        self.features_btn.setGeometry(QtCore.QRect(int(830 * sf_x), int(210 * sf_y), int(161 * sf_x), int(31 * sf_y)))
        
        self.features_selected = QtWidgets.QLabel(self.data_frame)
        self.features_selected.setGeometry(QtCore.QRect(int(640 * sf_x), int(250 * sf_y), int(540 * sf_x), int(31 * sf_y)))
        self.features_selected.setText("Selected: ----")
        
        self.features_btn.selectionChanged.connect(self.on_features_changed)
        self.feature_strata = []   # initialize

        self.w_title.setText("Setting input files")
        self.save_button.setText("Save and continue")
        self.population_button.setText("Upload population data")
        self.population_data_path.setText("filename.csv")
        self.output_label_existing.setText("Output name:")
        self.output_existing_value.setText("existing_strat")
        self.distribution_button.setText("Upload feature distribution")
        self.existing_check.setText("Use existing distribution")
        self.distrib_path.setText("filename.csv")
        self.output_path_button.setText("Select output folder")
        self.saved_path_existing.setText("path/where/save/the/results")
        self.new_check.setText("There is no existing information")
        self.output_label_new.setText("Output name:")
        self.output_new_value.setText("new_strat")
        self.population_new_button.setText("Upload population data")
        self.population_new_path.setText("filename.csv")
        self.saved_path_new.setText("path/where/save/the/results")
        self.output_path_new_button.setText("Select output folder")
        self.valid_label.setText("Is the distribution valid for the entire population?")
        self.valid_value.setItemText(0, "YES")
        self.valid_value.setItemText(1, "NO")
        self.extra_label.setText("Extrapolation mode:")
        self.extrap_mode.setItemText(0, "Deep learning models")
        self.extrap_mode.setItemText(1, "Manually")
        self.ask_1.setText("(Only if the previous answer is")
        self.ask_2.setText("NO")
        self.ask_3.setText(")")
        self.manage_dist.setText("How to manage partial distribution:")
        self.manage_value.setItemText(0, "Keep the partial distribution")
        self.manage_value.setItemText(1, "Keep and mix until \"x\" distance")
        self.distance_label.setText("Mix distance [m]:")       
        
    def select_method(self):
        # Check how many checkboxes are checked
        checked_count = sum([self.existing_check.isChecked(), 
                              self.new_check.isChecked()])
        
        if checked_count > 1:
            # Show a warning if more than one checkbox is checked
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("You can only select one method at a time.")
            msg.setWindowTitle("Selection Warning")
            msg.exec_()
        elif checked_count == 0:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please select one method")
        else:
            if self.new_check.isChecked():
                self.stratified_manually()
                self.accept()
            elif self.existing_check.isChecked():
                self.stratified_dl()
                self.accept()
                
    def preview_data(self, database):
        if hasattr(self, 'df') and not self.df.empty:
            preview_df = database.head(10)  # Only show first 10 rows

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
            QtWidgets.QMessageBox.warning(self, "No Data", "No data available to preview. Please upload a valid CSV first.")

    def upload_csv(self, label):
        
        options = QtWidgets.QFileDialog.Options()
        # File path
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        # Send path to UI
        self.file_local_csv = file_path
            
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
            QtWidgets.QMessageBox.warning(self, "Input Error", "No file selected.")

        return self.df
    
    ############ Folder Selection ################
    def select_output_folder_existing(self):
        """Open a folder selection dialog and display the selected folder in a text output."""
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
        if self.folder_path:  # If a folder is selected
            folder_display = os.path.basename(self.folder_path)    
            self.saved_path_manual.setText(folder_display)
    
    def select_output_folder_new(self):
        """Open a folder selection dialog and display the selected folder in a text output."""
        self.folder_path_new = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
        if self.folder_path_new:  # If a folder is selected
            folder_display = os.path.basename(self.folder_path_new)    
            self.saved_path_new.setText(folder_display)
    
    def data_population(self):
        self.data_population = self.upload_csv(self.population_new_path)
        self.preview_data(self.data_population)
        # unique_vals = self.data_population.columns.tolist()
        unique_vals = ["material", "llrs", "code_level","n_stories","occupancy","block_position",
                       "epoch_construction","roof_shape", "roof_material"]
        self.features_btn.set_values(unique_vals)
 
    def on_features_changed(self, vals):
        self.feature_strata = list(vals)  # keep a copy
        self.features_selected.setText(
            "Selected: " + (", ".join(vals) if vals else "(none)")
        )

    def stratified_manually(self):
        building_data = self.data_population
        # ========== Run sampling for each feature ==========
        analysis_features = self.feature_strata
        
        for feature in analysis_features:
            print(" ========== " + feature + " ===========")
            final_sample, class_dist, final_size = iterative_distribution_stability_manual(
                data=building_data,
                id_feature=feature,
                id_column='id',
                initial_fraction=20/100,
                step_fraction= 10/100,
                max_fraction=0.80,
                max_iterations=10,
                stability_threshold=0.05
            )
            final_sample.to_csv(f"{self.folder_path_new}/stratified_{feature}.csv", index=False)
            print("")
            
    def stratified_dl(self):
        building_data = self.data_population
        # ========== Run sampling for each feature ==========
        analysis_features = self.feature_strata
        extra_mode = 1
        sample_size_def = []
        for aux in analysis_features:
            print(" ========== " + aux + " ===========")
            final_sample, class_dist, final_size = iterative_label_discovery_cached_fractional(
                data=building_data,
                labeling_function=lambda x: labeling_function(x, aux, extra_mode),
                id_column='id',
                id_feature=aux,
                initial_fraction=0.10,
                step_fraction=5/100,
                max_fraction=0.20,
                max_iterations=1,
                stability_threshold=0.05
            )
            final_sample.to_csv(f"{self.folder_path_new}/stratified_dl_{aux}.csv", index=False)
            print("")
            
        print("Sample size definitive: ", np.max(sample_size_def))
            
       
# ========== Iterative sampling using existing labels ==========
def iterative_distribution_stability_manual(
    id_feature,
    data: pd.DataFrame,
    id_column,
    initial_fraction,
    step_fraction,
    max_fraction,
    stability_threshold,
    max_iterations,
    random_state: int = 42
):
    """
    Iteratively samples data using existing class labels until label distribution stabilizes.

    Args:
        id_feature (str): Column name with class labels (e.g., 'LLRS', 'Taxonomy').
        data (pd.DataFrame): Dataset containing existing labels.
        id_column (str): Column used as unique identifier.
        initial_fraction (float): Starting fraction of dataset.
        step_fraction (float): Step increase per iteration.
        max_fraction (float): Maximum sample fraction.
        stability_threshold (float): Max change in distribution to stop iterations.
        max_iterations (int): Maximum number of iterations.
        random_state (int): Random seed.

    Returns:
        DataFrame: Final sampled data.
        dict: Final class distribution.
        int: Final sample size.
    """

    population_size = len(data)
    all_sampled = pd.DataFrame(columns=data.columns)
    previous_dist = None
    iteration = 0

    # Shuffle dataset
    np.random.seed(random_state)
    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    while iteration < max_iterations:
        current_fraction = min(initial_fraction + step_fraction * iteration, max_fraction)
        target_size = int(population_size * current_fraction)

        # Select next sample
        remaining = shuffled_data[~shuffled_data[id_column].isin(all_sampled[id_column])]
        next_sample = remaining.head(target_size - len(all_sampled))

        if next_sample.empty:
            break

        all_sampled = pd.concat([all_sampled, next_sample], ignore_index=True)

        # Compute current distribution
        current_counts = all_sampled[id_feature].value_counts(normalize=True).sort_index()
        current_dist = current_counts.to_dict()

        # Check stabilization
        if previous_dist is not None:
            all_keys = set(previous_dist) | set(current_dist)
            max_change = max(abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys)
            print(f"Iteration {iteration+1}: Sample size = {len(all_sampled)}, Max Δ = {max_change:.4f}")

            if max_change < stability_threshold:
                print("✅ Class proportions stabilized.")
                return all_sampled, current_dist, len(all_sampled)

        previous_dist = current_dist
        iteration += 1

    print("⚠️ Reached max iterations or sample limit without convergence.")
    return all_sampled, current_dist, len(all_sampled)


#########################################################################
#########################################################################
#########################################################################
#########################################################################

# ========== Main Iterative Sampling Function ==========
def iterative_label_discovery_cached_fractional( self,
    id_feature,
    data: pd.DataFrame,
    labeling_function,
    id_column: str = 'id',
    initial_fraction: float = 0.10,
    step_fraction: float = 0.05,
    max_fraction: float = 1.00,
    stability_threshold: float = 0.05,
    max_iterations: int = 20,
    random_state: int = 42
):
    """
    Iteratively samples and labels data using a labeling function until label distribution stabilizes.

    Args:
        data (pd.DataFrame): Input dataset with unique IDs.
        labeling_function (callable): Function to assign labels based on the ID.
        id_column (str): Column name with unique IDs (default: 'ID').
        initial_fraction (float): Initial fraction of data to label.
        step_fraction (float): Additional fraction added each iteration.
        max_fraction (float): Maximum fraction of data to label.
        stability_threshold (float): Maximum allowed change in class distribution for convergence.
        max_iterations (int): Maximum number of iterations.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (DataFrame of labeled samples, label distribution as dict, final sample size)
    """
      
    if check_street_view(self.lat_dl, self.lon_dl) == True:
        population_size = len(data)
        all_labeled = pd.DataFrame(columns=[id_column, id_feature])  # Initialize labeled dataset
        previous_dist = None  # Store label distribution from previous iteration
        iteration = 0  # Iteration counter
    
        # Shuffle the dataset for randomized sampling
        np.random.seed(random_state)
        shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
        # === Iterative sampling loop ===
        while iteration < max_iterations:
            # Calculate current target sample size
            current_fraction = min(initial_fraction + step_fraction * iteration, max_fraction)
            target_size = min(int(population_size * current_fraction), population_size)
    
            # Filter out already labeled IDs and select next batch
            already_labeled_ids = set(all_labeled[id_column])
            next_sample = shuffled_data[~shuffled_data[id_column].isin(already_labeled_ids)].head(target_size - len(all_labeled))
    
            if next_sample.empty:
                break  # Stop if no more samples to process
    
            # Apply labeling function to new samples
            next_sample[id_feature] = next_sample[id_column].apply(labeling_function)
            all_labeled = pd.concat([all_labeled, next_sample], ignore_index=True)
    
            # Calculate class distribution
            current_counts = all_labeled[id_feature].value_counts(normalize=True).sort_index()
            current_dist = current_counts.to_dict()
    
            # Check for stabilization in label distribution
            if previous_dist is not None:
                all_keys = set(previous_dist) | set(current_dist)
                print("Previous dist: ")
                print(previous_dist)
                max_change = max(abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys)
                print(f"Iteration {iteration+1}: Sample size = {len(all_labeled)}, Max Δ = {max_change:.4f}")
    
                if max_change < stability_threshold:
                    print("✅ Class proportions stabilized.")
                    return all_labeled, current_dist, len(all_labeled)
    
            previous_dist = current_dist
            iteration += 1
    
        print("⚠️ Reached max iterations or sample limit without convergence.")
        return all_labeled, current_dist, len(all_labeled)

####################################################
####################################################
####################################################

# =========== Data to change ============
############ Checks if there is GSV availability ################  
def check_street_view(lat,lon):
    # Input parameters
    with open("methods/gsv_api_key.txt", "r") as f:
        api_key = f.read().strip()

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


# ========== Labeling Function ==========
def labeling_function(self, image_id, id_feature, extra_mode):
    
    """
    Applies a prediction model to a building image given its ID.

    Args:
        image_id (str): Unique identifier for the image.

    Returns:
        str: Predicted LLRS Material class for the building.
    """
    if extra_mode == 0:
        lat_row = self.data_population.loc[self.data_population["id"] == image_id, "latitude"]
        lon_row = self.data_population.loc[self.data_population["id"] == image_id, "longitude"]
        self.lat_dl = lat_row.iloc[0]
        self.lon_dl = lon_row.iloc[0]
        # Building coordinates
        location = (self.lat_dl,self.lon_dl)
        # angles for taking the images
        angle = 0
        # Input parameters
        with open("methods/gsv_api_key.txt", "r") as f:
            api_key = f.read().strip()
        
        #Image from GSV
        image_path = get_street_view_image(location, api_key, angle) 
    else:
        # Local image path
        image_path = f"C:/Users/User/Documents/GitHub/RUBIC-AI/demos/local_images/images_ex1/{image_id}"
    
    # Deep learning models
    if id_feature == "LLRS":
        return predict_llrs_img(image_path)
    elif id_feature == "LLRS Material":
        return predict_material_img(image_path, extra_mode)
    elif id_feature == "Number of Stories":
        return predict_n_stories_img(image_path)
    elif id_feature == "Occupancy":
        return predict_occupancy_img(image_path)
    elif id_feature == "Code Level":
        return predict_code_img(image_path)
    elif id_feature == "Block Position":
        return predict_block_position_img(image_path)
    elif id_feature == "Roof Shape":
        return predict_roof_shape_img(image_path)
    elif id_feature == "Roof Material":
        return predict_roof_material_img(image_path)
    else:
        return None


























#########################################################################
#########################################################################
#########################################################################
#########################################################################

class CheckFilterPopup(QtWidgets.QWidget):
    selectionChanged = QtCore.pyqtSignal(list)

    def __init__(self, values=None, parent=None):
        super().__init__(parent, flags=QtCore.Qt.Popup)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint, True)
        self.resize(220, 280)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(6, 6, 6, 6)

        self.search = QtWidgets.QLineEdit(self)
        self.search.setPlaceholderText("Search")
        self.search.setClearButtonEnabled(True)
        vbox.addWidget(self.search)

        self.selectAll = QtWidgets.QCheckBox("(Select All)", self)
        self.selectAll.setTristate(True)
        vbox.addWidget(self.selectAll)

        self.list = QtWidgets.QListWidget(self)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        vbox.addWidget(self.list, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                          QtWidgets.QDialogButtonBox.Cancel, self)
        vbox.addWidget(btns)

        self._all_values = []
        if values:
            self.set_values(values)

        self.search.textChanged.connect(self._apply_filter)
        self.selectAll.stateChanged.connect(self._toggle_all)
        self.list.itemChanged.connect(self._update_select_all_state)
        btns.accepted.connect(self._emit_and_close)
        btns.rejected.connect(self.close)

    def set_values(self, values):
        uniq = sorted({str(v) for v in values if v is not None})
        self._all_values = uniq
        self._rebuild_list(uniq)

    def selected_values(self):
        vals = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == QtCore.Qt.Checked and not it.isHidden():
                vals.append(it.text())
        return vals

    # internals
    def _rebuild_list(self, vals):
        self.list.blockSignals(True)
        self.list.clear()
        for txt in vals:
            it = QtWidgets.QListWidgetItem(txt)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._update_select_all_state()

    def _apply_filter(self, text):
        text = text.strip().lower()
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(text not in it.text().lower())
        self.list.blockSignals(False)
        self._update_select_all_state()

    def _visible_items(self):
        return [self.list.item(i) for i in range(self.list.count()) if not self.list.item(i).isHidden()]

    def _toggle_all(self, state):
        if state == QtCore.Qt.PartiallyChecked:
            return
        target = QtCore.Qt.Checked if state == QtCore.Qt.Checked else QtCore.Qt.Unchecked
        self.list.blockSignals(True)
        for it in self._visible_items():
            it.setCheckState(target)
        self.list.blockSignals(False)
        self._update_select_all_state()

    def _update_select_all_state(self):
        vis = self._visible_items()
        if not vis:
            self.selectAll.setCheckState(QtCore.Qt.Unchecked)
            return
        checks = [it.checkState() == QtCore.Qt.Checked for it in vis]
        if all(checks):
            self.selectAll.setCheckState(QtCore.Qt.Checked)
        elif any(checks):
            self.selectAll.setCheckState(QtCore.Qt.PartiallyChecked)
        else:
            self.selectAll.setCheckState(QtCore.Qt.Unchecked)

    def _emit_and_close(self):
        self.selectionChanged.emit(self.selected_values())
        self.close()


class CheckFilterButton(QtWidgets.QToolButton):
    selectionChanged = QtCore.pyqtSignal(list)

    def __init__(self, values=None, parent=None, text="Feature strata"):
        super().__init__(parent)
        self.setText(text)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setArrowType(QtCore.Qt.DownArrow)
        self._popup = CheckFilterPopup(values, self)
        self.clicked.connect(self._show_popup)
        self._popup.selectionChanged.connect(self.selectionChanged)

    def set_values(self, values):
        self._popup.set_values(values)

    def selected_values(self):
        return self._popup.selected_values()

    def _show_popup(self):
        pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        self._popup.move(pos)
        self._popup.show()
