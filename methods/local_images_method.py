from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import os
import sys 
import numpy as np

class LocalImageSetting(QtWidgets.QDialog):
    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method  # Reference to GUIMethods
        
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
        self.sf_font = sf_font
        
        self.setWindowTitle("Local Images Method Input")
        self.resize(int(611 * sf_x), int(512 * sf_y))

        # Main Frame
        self.coord_frame = QtWidgets.QWidget(self)

        # Background
        self.backg_4 = QtWidgets.QLabel(self.coord_frame)
        self.backg_4.setGeometry(QtCore.QRect(int(10 * sf_x), int(9 * sf_y), int(591 * sf_x), int(451 * sf_y)))
        self.backg_4.setStyleSheet("background-color: rgb(255, 253, 187);")

        # Upload CSV
        self.csv_button_local = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_local.setGeometry(QtCore.QRect(int(20 * sf_x), int(165 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setBold(False)
        self.csv_button_local.setFont(font)
        self.csv_button_local.setText("Upload building information")       
        self.csv_button_local.clicked.connect(self.upload_csv)
        
        self.local_path = QtWidgets.QLabel(self.coord_frame)
        self.local_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(170 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        self.local_path.setFont(font)
        self.local_path.setText("filename.csv")
        
        # Output name
        self.output_label_local = QtWidgets.QLabel(self.coord_frame)
        self.output_label_local.setGeometry(QtCore.QRect(int(20 * sf_x), int(39 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font.setBold(True)
        self.output_label_local.setFont(font)
        self.output_label_local.setText("Output name:")

        self.output_local = QtWidgets.QLineEdit(self.coord_frame)
        self.output_local.setGeometry(QtCore.QRect(int(160 * sf_x), int(39 * sf_y), int(270 * sf_x), int(31 * sf_y)))
        font.setBold(False)
        self.output_local.setFont(font)
        self.output_local.setText("Local_images")
        
        # Output folder selection
        self.path_out_folder_bt = QtWidgets.QPushButton(self.coord_frame)
        self.path_out_folder_bt.setGeometry(QtCore.QRect(int(20 * sf_x), int(80 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.path_out_folder_bt.setFont(font)
        self.path_out_folder_bt.setText("Select output folder")
        self.path_out_folder_bt.clicked.connect(self.select_output_folder)
        
        self.output_folder_value = QtWidgets.QLabel(self.coord_frame)
        self.output_folder_value.setGeometry(QtCore.QRect(int(270 * sf_x), int(82 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        self.output_folder_value.setFont(font)
        self.output_folder_value.setText("path/where/will/save/your/results")

        # Image folder selection
        self.folder_local_button = QtWidgets.QPushButton(self.coord_frame)
        self.folder_local_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(125 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.folder_local_button.setFont(font)
        self.folder_local_button.setText("Select image folder")
        self.folder_local_button.clicked.connect(self.select_folder)
        
        self.local_folder_path = QtWidgets.QLabel(self.coord_frame)
        self.local_folder_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(130 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        self.local_folder_path.setFont(font)
        self.local_folder_path.setText("---")

        # Buttons
        font.setBold(True)
        self.load_data_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_data_button.setGeometry(QtCore.QRect(int(100 * sf_x), int(470 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.load_data_button.setFont(font)
        self.load_data_button.setText("Load data")
        self.load_data_button.clicked.connect(self.save_coordinates)
        
        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(320 * sf_x), int(470 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.building_sample)
        
        # Title
        self.local_label = QtWidgets.QLabel(self.coord_frame)
        self.local_label.setGeometry(QtCore.QRect(int(190 * sf_x), int(9 * sf_y), int(251 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        self.local_label.setFont(font)
        self.local_label.setText("Local Images Method Input")

        """ GEM icon GUI elements """
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        
        # Feature Collection Label
        self.feature_collection_label = QtWidgets.QLabel(self.coord_frame)
        self.feature_collection_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(210 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.feature_collection_label.setFont(font)
        self.feature_collection_label.setObjectName("feature_collection_label")
        self.feature_collection_label.setText("Feature collection mode:")
        
        # ComboBox for Collection Mode
        self.collection_mode = QtWidgets.QComboBox(self.coord_frame)
        self.collection_mode.setGeometry(QtCore.QRect(int(250 * sf_x), int(210 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.collection_mode.setFont(font)
        self.collection_mode.setObjectName("collection_mode")
        self.collection_mode.addItem("Manual")
        self.collection_mode.addItem("AI Powered")
        
        # TableWidget
        self.tableWidget = QtWidgets.QTableWidget(self.coord_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(20 * sf_x), int(250 * sf_y), int(571 * sf_x), int(192 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
         
        # Raise (stack order)
        self.backg_4.raise_()
        self.local_label.raise_()
        self.csv_button_local.raise_()
        self.local_path.raise_()
        self.output_label_local.raise_()
        self.output_local.raise_()
        self.folder_local_button.raise_()
        self.local_folder_path.raise_()
        self.load_data_button.raise_()
        self.save_button.raise_()
        self.path_out_folder_bt.raise_()
        self.output_folder_value.raise_()
        self.tableWidget.raise_()
        self.collection_mode.raise_()
        self.feature_collection_label.raise_()
        
    ############ Save coordinates ################
    def save_coordinates(self):
        try:
            # Load the uploaded file
            df = pd.read_csv(self.file_local_csv)  # Make sure self.path_csv_input stores the uploaded file path
            # Define the required columns
            required_columns = ['id', 'latitude', 'longitude']
    
            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                QMessageBox.warning(self, "Invalid Format",
                                    f"The uploaded file is missing the following required columns:\n{', '.join(missing_columns)}")
                return
    
            # If format is valid, continue with your function
            QMessageBox.information(self, "Success", "Done! Please click the save and continue button.")
            
            self.population = False
            
        except Exception as e:
            QMessageBox.warning(self, "Error",
                                f"An error occurred. Please check that all required input files are correctly formatted.\n\nDetails: {str(e)}")
            
            
    def building_sample(self):
        # Checking is the inspection mode correspond to specific
        try:
            if self.population == False:
                self.method.local_output_name = self.output_local
                self.mode_use()
                self.accept()
        except:
            QMessageBox.warning(self, "Input Error", "First Please upload the data using the **Load Data button**")
            
    # ############ Folder Selection ################           
    def select_folder(self):
        """Open a folder selection dialog, display selected folder, and check for valid image files."""
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
    
        if folder_path:
            self.method.folder_path = folder_path
            folder_display = os.path.basename(folder_path)
            self.local_folder_path.setText(folder_display)
    
            # List files in folder
            files = os.listdir(folder_path)
            valid_ext = ('.jpg', '.jpeg', '.png')
            supported_imgs = [f for f in files if f.lower().endswith(valid_ext)]
            unsupported_imgs = [f for f in files if f.lower().endswith('.heic')]
    
            if not supported_imgs:
                QMessageBox.warning(self, "No Valid Images",
                                    "The selected folder does not contain any valid image files (.jpg, .jpeg, .png).")
            elif unsupported_imgs:
                QMessageBox.information(self, "Unsupported Formats Found",
                                        f"The folder contains unsupported image formats like HEIC:\n\n" +
                                        "\n".join(unsupported_imgs))
            
    ############ Folder Selection ################
    def select_output_folder(self):
        """Open a folder selection dialog and display the selected folder in a text output."""
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        self.method.output_folder_value = folder_path
        if folder_path:  # If a folder is selected
            folder_display = os.path.basename(folder_path)    
            self.output_folder_value.setText(folder_display)
            
    def upload_csv(self):      
        options = QFileDialog.Options()
        # File path
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        # Send path to UI
        self.file_local_csv = file_path
        self.method.file_local_csv = file_path        
        if file_path:
            try:
                # Upload csv with building coordinates
                self.df = pd.read_csv(file_path)
                file_path = file_path.rsplit("/", 1)[-1]
                self.local_path.setText(file_path)
                self.preview_data()
            except FileNotFoundError:
                self.local_path.setText("File not found.")
            except pd.errors.ParserError:
                self.local_path.setText("Error parsing CSV file. Check the format.")
            except Exception as e: #catch other exceptions
                self.local_path.setText(f"An error occurred: {e}")
        else:
            self.local_path.setText("No file selected.")
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
                    # ---- Set font size ----
                    font = item.font()
                    font.setPointSize(int(10 * self.sf_font))  # change to any size
                    item.setFont(font)
                    self.tableWidget.setItem(row, column, item)
                    header = self.tableWidget.horizontalHeader()
                    font = header.font()
                    font.setPointSize(int(10 * self.sf_font))
                    font.setBold(True)  # optional
                    header.setFont(font)
                    vheader = self.tableWidget.verticalHeader()
                    vfont = vheader.font()
                    vfont.setPointSize(int(10 * self.sf_font))
                    vheader.setFont(vfont)
    
            self.tableWidget.resizeColumnsToContents()
        else:
            QMessageBox.warning(self, "No Data", "No data available to preview. Please upload a valid CSV first.")

    def mode_use(self):
        if self.collection_mode.currentText() == "Manual":
            self.ai_value = False 
        elif self.collection_mode.currentText() == "AI Powered":
            self.ai_value = True