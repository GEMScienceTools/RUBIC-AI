from geopy.distance import geodesic
from collections import defaultdict
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from methods.dl_extrapolation import create_database, dl_models, inspection_database, extrapolation_existing_reference

class data_options_window(QtWidgets.QDialog):
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
        self.resize(int(1210 * sf_x), int(570 * sf_y))
        self.setWindowTitle("Setting input files")

        # === UI Elements Start ===
        self.data_frame = QtWidgets.QWidget(self)
        self.data_frame.setObjectName("data_frame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.data_frame)

        self.w_title = QtWidgets.QLabel("Setting input files", self.data_frame)
        self.w_title.setGeometry(QtCore.QRect(int(510 * sf_x), int(0), int(191 * sf_x), int(41 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(12 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.w_title.setFont(font)

        self.save_button = QtWidgets.QPushButton(self.data_frame)
        self.save_button.setGeometry(QtCore.QRect(int(510 * sf_x), int(510 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self.select_method)
        
        self.backg_1 = QtWidgets.QLabel(self.data_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(39 * sf_y), int(591 * sf_x), int(221 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(209, 255, 165);")
        self.backg_1.setText("")
        self.backg_1.setObjectName("backg_1")
        
        self.unclassfied_path = QtWidgets.QLabel(self.data_frame)
        self.unclassfied_path.setGeometry(QtCore.QRect(int(280 * sf_x), int(180 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.unclassfied_path.setFont(font)
        self.unclassfied_path.setObjectName("unclassfied_path")
        
        self.unclassified_button = QtWidgets.QPushButton(self.data_frame)
        self.unclassified_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(175 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.unclassified_button.setFont(font)
        self.unclassified_button.setObjectName("unclassified_button")
        self.unclassified_button.clicked.connect(self.data_extrapolation)
        
        self.output_label_manual = QtWidgets.QLabel(self.data_frame)
        self.output_label_manual.setGeometry(QtCore.QRect(int(30 * sf_x), int(90 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_manual.setFont(font)
        self.output_label_manual.setObjectName("output_label_manual")
        
        self.output_manual_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_manual_value.setGeometry(QtCore.QRect(int(170 * sf_x), int(90 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.output_manual_value.setFont(font)
        self.output_manual_value.setObjectName("output_manual_value")
        
        self.manual_info_path = QtWidgets.QLabel(self.data_frame)
        self.manual_info_path.setGeometry(QtCore.QRect(int(280 * sf_x), int(140 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.manual_info_path.setFont(font)
        self.manual_info_path.setObjectName("manual_info_path")
        
        self.b_info_button = QtWidgets.QPushButton(self.data_frame)
        self.b_info_button.setGeometry(QtCore.QRect(int(30 * sf_x), int(135 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.b_info_button.setFont(font)
        self.b_info_button.setObjectName("b_info_button")
        self.b_info_button.clicked.connect(self.data_existing)
        
        self.manual_op = QtWidgets.QCheckBox(self.data_frame)
        self.manual_op.setGeometry(QtCore.QRect(int(30 * sf_x), int(50 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.manual_op.setFont(font)
        self.manual_op.setObjectName("manual_op")
        
        self.backg_2 = QtWidgets.QLabel(self.data_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(610 * sf_x), int(40 * sf_y), int(591 * sf_x), int(221 * sf_y)))
        self.backg_2.setStyleSheet("background-color: rgb(255, 233, 167);")
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        
        self.dl_op = QtWidgets.QCheckBox(self.data_frame)
        self.dl_op.setGeometry(QtCore.QRect(int(640 * sf_x), int(45 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.dl_op.setFont(font)
        self.dl_op.setObjectName("dl_op")
        
        self.folder_img_path = QtWidgets.QLabel(self.data_frame)
        self.folder_img_path.setGeometry(QtCore.QRect(int(890 * sf_x), int(175 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.folder_img_path.setFont(font)
        self.folder_img_path.setObjectName("folder_img_path")
        
        self.folder_local_button = QtWidgets.QPushButton(self.data_frame)
        self.folder_local_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(170 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.folder_local_button.setFont(font)
        self.folder_local_button.setObjectName("folder_local_button")
        self.folder_local_button.clicked.connect(self.select_output_folder)
        
        self.output_label_dl = QtWidgets.QLabel(self.data_frame)
        self.output_label_dl.setGeometry(QtCore.QRect(int(640 * sf_x), int(85 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_dl.setFont(font)
        self.output_label_dl.setObjectName("output_label_dl")
        
        self.output_dl_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_dl_value.setGeometry(QtCore.QRect(int(780 * sf_x), int(85 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.output_dl_value.setFont(font)
        self.output_dl_value.setObjectName("output_dl_value")
        
        self.unclassfied_dl_path = QtWidgets.QLabel(self.data_frame)
        self.unclassfied_dl_path.setGeometry(QtCore.QRect(int(890 * sf_x), int(220 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.unclassfied_dl_path.setFont(font)
        self.unclassfied_dl_path.setObjectName("unclassfied_dl_path")
        
        self.unclassified_dl_button = QtWidgets.QPushButton(self.data_frame)
        self.unclassified_dl_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(215 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.unclassified_dl_button.setFont(font)
        self.unclassified_dl_button.setObjectName("unclassified_dl_button")
        self.unclassified_dl_button.clicked.connect(self.data_extrapolation_dl)
        
        # KNN Coordinate Button
        self.coord_knn_button = QtWidgets.QPushButton(self.data_frame)
        self.coord_knn_button.setGeometry(QtCore.QRect(int(640 * sf_x), int(125 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(50)
        self.coord_knn_button.setFont(font)
        self.coord_knn_button.setObjectName("coord_knn_button")
        self.coord_knn_button.clicked.connect(self.data_existing_dl)
                                           
        # KNN Coordinate Value Label
        self.coord_value_knn = QtWidgets.QLabel(self.data_frame)
        self.coord_value_knn.setGeometry(QtCore.QRect(int(890 * sf_x), int(130 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.coord_value_knn.setFont(font)
        self.coord_value_knn.setObjectName("coord_value_knn")
        
        self.tableWidget = QtWidgets.QTableWidget(self.data_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(10 * sf_x), int(270 * sf_y), int(1170 * sf_x), int(231 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        
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
        self.folder_img_path.setText("------------")
        self.folder_local_button.setText( "Image folder")
        self.output_label_dl.setText("Output name:")
        self.unclassfied_dl_path.setText("filename.csv")
        self.unclassified_dl_button.setText("Unclassified building coords")
        self.coord_knn_button.setText("Upload building coordinates")
        self.coord_value_knn.setText("filename.csv")
                                     
    def select_method(self):
        # Check how many checkboxes are checked
        checked_count = sum([self.manual_op.isChecked(), 
                              self.dl_op.isChecked()])
        
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
            if self.manual_op.isChecked():
                try:
                    self.info_existing
                    self.info_pending
                    self.accept()
                except:
                    QtWidgets.QMessageBox.warning(self, "Input Error", "There are missing the inputs files")
            elif self.dl_op.isChecked():
                #######===========  Input parameters =========###########
                self.coord_reference = self.info_existing
                self.coord_reference_building_feature_path = "demos/extrapolation/"+self.output_dl_value.text()+"_reference_results.csv"
                self. data_extrapolation = self.info_pending
                self.knn_dl_saved_path = "demos/extrapolation/extrapolation_"+self.output_dl_value.text()+".csv"
                

                self.accept()
                
    def data_existing(self):
        self.info_existing = self.upload_csv(self.manual_info_path)
        self.preview_data(self.info_existing)
        
    def data_extrapolation(self):
        self.info_pending = self.upload_csv(self.unclassfied_path)
        self.preview_data(self.info_pending)
        
    def data_existing_dl(self):
        self.info_existing = self.upload_csv(self.coord_value_knn)
        self.preview_data(self.info_existing)
        
    def data_extrapolation_dl(self):
        self.info_pending = self.upload_csv(self.unclassfied_dl_path)
        self.preview_data(self.info_pending)
        
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
    def select_output_folder(self):
        """Open a folder selection dialog and display the selected folder in a text output."""
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder_path:  # If a folder is selected
            folder_display = os.path.basename(folder_path)    
            self.folder_img_path.setText(folder_display)
    

# Function to calculate Geodesic distance (in km)
def geodesic_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).km

# Function to find 3 nearest neighbors using geodesic distance
def find_nearest_neighbors_geodesic(input_row, info_df, n_neighbors=10):
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
    Computes taxonomy probabilities using weighted soft voting based on geodesic distance.

    Parameters:
    - nearest_neighbors: DataFrame containing the neighbors with 'Taxonomy' and 'distance_km' columns.
    - input_row: The row of the input point.
    - kernel: Kernel type ('inverse' or 'gaussian').
    - bandwidth: Bandwidth for the Gaussian kernel.

    Returns:
    - List of dictionaries, each representing a taxonomy and its probability, along with extra metadata.
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
            'taxonomy': taxonomy,
            'probability': prob
        })

    return distribution_rows


