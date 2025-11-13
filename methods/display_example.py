from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np

class PolygonSettingWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

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
        
        # Set window size dynamically
        self.setWindowTitle("Setting Polygon Coordinates")
        self.resize(int(612 * sf_x), int(846 * sf_y))
        
        # Main widget
        self.coord_frame = QtWidgets.QWidget(self)
        self.coord_frame.setObjectName("coord_frame")
        
        # Title of the window
        self.w_tittle = QtWidgets.QLabel(self.coord_frame)
        self.w_tittle.setGeometry(QtCore.QRect(int(170* sf_x), int(0* sf_y), int(301* sf_x), int(41* sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(12*sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.w_tittle.setFont(font)
        self.w_tittle.setObjectName("w_tittle")
        self.w_tittle.setText("Setting Polygon Coordinates")  # Title
        
        # Button for uploading the CSV file with coordinates
        self.csv_button = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(305 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.csv_button.setFont(font)
        self.csv_button.setObjectName("csv_button")
        self.csv_button.setText("Upload polygon coordinates")  # Button text
        
       # Input fields for the coordinates of points (latitude and longitude) 
        self.coord_1 = QtWidgets.QLineEdit(self.coord_frame)
        self.coord_1.setGeometry(QtCore.QRect(int(160 * sf_x), int(116 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.coord_1.setFont(font)
        self.coord_1.setObjectName("coord_1")
        self.coord_1.setText("(lat1 , lon1)")  # Placeholder text
        
        self.coord_2 = QtWidgets.QLineEdit(self.coord_frame)
        self.coord_2.setGeometry(QtCore.QRect(int(160 * sf_x), int(155 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.coord_2.setFont(font)
        self.coord_2.setObjectName("coord_2")
        self.coord_2.setText("(lat2 , lon2)")  # Placeholder text
        
        # Label for "First Point"
        self.point_1 = QtWidgets.QLabel(self.coord_frame)
        self.point_1.setGeometry(QtCore.QRect(int(20 * sf_x), int(120 * sf_y), int(101 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.point_1.setFont(font)
        self.point_1.setObjectName("point_1")
        self.point_1.setText("First Point:")  # Text label
        
        # Image placeholder for a squared image (if needed)
        self.img_squared = QtWidgets.QLabel(self.coord_frame)
        self.img_squared.setGeometry(QtCore.QRect(int(420 * sf_x), int(120 * sf_y), int(151 * sf_x), int(131 * sf_y)))
        self.img_squared.setObjectName("img_squared")
        self.img_squared.setPixmap(QtGui.QPixmap("squared_coord.png"))
        self.img_squared.setScaledContents(True)
        
        # Button for saving the input and continuing
        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(320 * sf_x), int(800 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.setText("Save and continue")  # Button text
        
        # Label for "Second Point"
        self.point_2 = QtWidgets.QLabel(self.coord_frame)
        self.point_2.setGeometry(QtCore.QRect(int(20 * sf_x), int(155 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.point_2.setFont(font)
        self.point_2.setObjectName("point_2")
        self.point_2.setText("Second Point:")  # Label text
        
        # Label showing the CSV file path description
        self.label_csv_file = QtWidgets.QLabel(self.coord_frame)
        self.label_csv_file.setGeometry(QtCore.QRect(int(20 * sf_x), int(275 * sf_y), int(531 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.label_csv_file.setFont(font)
        self.label_csv_file.setObjectName("label_csv_file")
        self.label_csv_file.setText("Define a polygon with more than two points using a *.csv file")  # Instruction text
        
        # Backgrounds
        self.backg_1 = QtWidgets.QLabel(self.coord_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(80 * sf_y), int(591 * sf_x), int(261 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(209, 255, 165);")
        self.backg_1.setText("")
        self.backg_1.setObjectName("backg_1")
        
        self.backg_2 = QtWidgets.QLabel(self.coord_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(40 * sf_y), int(591 * sf_x), int(31 * sf_y)))
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        
        self.backg_3 = QtWidgets.QLabel(self.coord_frame)
        self.backg_3.setGeometry(QtCore.QRect(int(10 * sf_x), int(350 * sf_y), int(591 * sf_x), int(121 * sf_y)))
        self.backg_3.setStyleSheet("background-color: rgb(255, 224, 185);")
        self.backg_3.setText("")
        self.backg_3.setObjectName("backg_3")
        
        self.backg_4 = QtWidgets.QLabel(self.coord_frame)
        self.backg_4.setGeometry(QtCore.QRect(int(10 * sf_x), int(480 * sf_y), int(591 * sf_x), int(151 * sf_y)))
        self.backg_4.setStyleSheet("background-color: rgb(157, 218, 255);")
        self.backg_4.setText("")
        self.backg_4.setObjectName("backg_4")
        
        self.backg_5 = QtWidgets.QLabel(self.coord_frame)
        self.backg_5.setGeometry(QtCore.QRect(int(10 * sf_x), int(640 * sf_y), int(591 * sf_x), int(151 * sf_y)))
        self.backg_5.setStyleSheet("background-color: rgb(254, 255, 160);")
        self.backg_5.setText("")
        self.backg_5.setObjectName("backg_5")
        
        # Label for displaying the CSV file name
        self.label = QtWidgets.QLabel(self.coord_frame)
        self.label.setGeometry(QtCore.QRect(int(270 * sf_x), int(310 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("filename.csv")  # Placeholder text for CSV filename
        
        # Label to show the selected method for inspection
        self.method_value = QtWidgets.QLabel(self.coord_frame)
        self.method_value.setGeometry(QtCore.QRect(int(280 * sf_x), int(45 * sf_y), int(301 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.method_value.setFont(font)
        self.method_value.setObjectName("method_value")
        
        # Label for displaying the selected method description
        self.method_label = QtWidgets.QLabel(self.coord_frame)
        self.method_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(40 * sf_y), int(251 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.method_label.setFont(font)
        self.method_label.setObjectName("method_label")
        self.method_label.setText("Inspection method selected:")  # Text label
        
        # Input fields and labels for "Default" method
        self.default_label = QtWidgets.QLabel(self.coord_frame)
        self.default_label.setGeometry(QtCore.QRect(int(210 * sf_x), int(80 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
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
        self.n_building_default.setGeometry(QtCore.QRect(int(20 * sf_x), int(200 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.n_building_default.setFont(font)
        self.n_building_default.setObjectName("n_building_default")
        self.n_building_default.setText("N° buildings:")  # Building count label
        
        self.sample_default = QtWidgets.QLabel(self.coord_frame)
        self.sample_default.setGeometry(QtCore.QRect(int(20 * sf_x), int(230 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.sample_default.setFont(font)
        self.sample_default.setObjectName("sample_default")
        self.sample_default.setText("Sample size:")  # Sample size label
        
        self.building_value_default = QtWidgets.QLabel(self.coord_frame)
        self.building_value_default.setGeometry(QtCore.QRect(int(160 * sf_x), int(200 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.building_value_default.setFont(font)
        self.building_value_default.setObjectName("building_value_default")
        self.building_value_default.setText("-")  # Default building count
        
        self.sample_size_default = QtWidgets.QLineEdit(self.coord_frame)
        self.sample_size_default.setGeometry(QtCore.QRect(int(160 * sf_x), int(230 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.sample_size_default.setFont(font)
        self.sample_size_default.setObjectName("sample_size_default")
        self.sample_size_default.setText("10")  # Default sample size
        
        # Load button for determine building population
        self.load_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_button.setGeometry(QtCore.QRect(int(110 * sf_x), int(800 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.load_button.setFont(font)
        self.load_button.setObjectName("load_button")
        self.load_button.setText("Load data")  # Button text
        
        # Input fields and labels for "Specific" method
        self.specific_label = QtWidgets.QLabel(self.coord_frame)
        self.specific_label.setGeometry(QtCore.QRect(int(160 * sf_x), int(355 * sf_y), int(301 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.specific_label.setFont(font)
        self.specific_label.setObjectName("specific_label")
        self.specific_label.setText("Specific Locations Method Input")  # Input section title
        
        self.specific_path = QtWidgets.QLabel(self.coord_frame)
        self.specific_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(440 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.specific_path.setFont(font)
        self.specific_path.setObjectName("specific_path")
        self.specific_path.setText("filename.csv")  # Placeholder text for CSV filename
        
        self.csv_button_specific = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_specific.setGeometry(QtCore.QRect(int(20 * sf_x), int(435 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.csv_button_specific.setFont(font)
        self.csv_button_specific.setObjectName("csv_button_specific")
        self.csv_button_specific.setText("Upload building locations")  # Button text
        
        self.local_label = QtWidgets.QLabel(self.coord_frame)
        self.local_label.setGeometry(QtCore.QRect(int(190 * sf_x), int(480 * sf_y), int(251 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.local_label.setFont(font)
        self.local_label.setObjectName("local_label")
        self.local_label.setText("Local Folder Images Method Input")  # Input section title
        
        self.csv_button_local = QtWidgets.QPushButton(self.coord_frame)
        self.csv_button_local.setGeometry(QtCore.QRect(int(20 * sf_x), int(590 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.csv_button_local.setFont(font)
        self.csv_button_local.setObjectName("csv_button_local")
        self.csv_button_local.setText("Upload buildings information")  # Button text
        
        self.local_folder_path = QtWidgets.QLabel(self.coord_frame)
        self.local_folder_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(555 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.local_folder_path.setFont(font)
        self.local_folder_path.setObjectName("local_folder_path")
        self.local_folder_path.setText("---")  # Button text
        
        self.folder_local_button = QtWidgets.QPushButton(self.coord_frame)
        self.folder_local_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(550 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.folder_local_button.setFont(font)
        self.folder_local_button.setObjectName("folder_local_button")
        self.folder_local_button.setText("Select image folder")  # Button text
        
        self.local_path = QtWidgets.QLabel(self.coord_frame)
        self.local_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(595 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.local_path.setFont(font)
        self.local_path.setObjectName("local_path")
        self.local_path.setText("filename.csv")  # Placeholder text for CSV filename
        
        self.output_label_specific = QtWidgets.QLabel(self.coord_frame)
        self.output_label_specific.setGeometry(QtCore.QRect(int(20 * sf_x), int(390 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_specific.setFont(font)
        self.output_label_specific.setObjectName("output_label_specific")
        self.output_label_specific.setText("Output name:")
        
        self.output_value_specific = QtWidgets.QLineEdit(self.coord_frame)
        self.output_value_specific.setGeometry(QtCore.QRect(int(160 * sf_x), int(390 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.output_value_specific.setFont(font)
        self.output_value_specific.setObjectName("output_value_specific")
        self.output_value_specific.setText("Specific")
        
        self.output_label_local = QtWidgets.QLabel(self.coord_frame)
        self.output_label_local.setGeometry(QtCore.QRect(int(20 * sf_x), int(510 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_local.setFont(font)
        self.output_label_local.setObjectName("output_label_local")
        self.output_label_local.setText("Output name:")
        
        self.output_value_local = QtWidgets.QLineEdit(self.coord_frame)
        self.output_value_local.setGeometry(QtCore.QRect(int(160 * sf_x), int(510 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.output_value_local.setFont(font)
        self.output_value_local.setObjectName("output_value_local")
        self.output_value_local.setText("Local")
        
        self.n_image_local_label = QtWidgets.QLabel(self.coord_frame)
        self.n_image_local_label.setGeometry(QtCore.QRect(int(310 * sf_x), int(510 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.n_image_local_label.setFont(font)
        self.n_image_local_label.setObjectName("n_image_local_label")
        self.n_image_local_label.setText("N° images per location:")
        
        self.n_image_local_value = QtWidgets.QComboBox(self.coord_frame)
        self.n_image_local_value.setGeometry(QtCore.QRect(int(520 * sf_x), int(511 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        self.n_image_local_value.setObjectName("n_image_local_value")
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.n_image_local_value.setFont(font)
        self.n_image_local_value.addItem("1", 1)
        self.n_image_local_value.addItem("2", 2)
        self.n_image_local_value.addItem("3", 3)
        
        self.extrapolation_label = QtWidgets.QLabel(self.coord_frame)
        self.extrapolation_label.setGeometry(QtCore.QRect(int(190 * sf_x), int(640 * sf_y), int(251 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.extrapolation_label.setFont(font)
        self.extrapolation_label.setObjectName("extrapolation_label")
        self.extrapolation_label.setText("Extrapolation Method Input")
        
        self.output_label_extra = QtWidgets.QLabel(self.coord_frame)
        self.output_label_extra.setGeometry(QtCore.QRect(int(20 * sf_x), int(670 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_extra.setFont(font)
        self.output_label_extra.setObjectName("output_label_extra")
        self.output_label_extra.setText("Output name:")
        
        self.building_extra_button = QtWidgets.QPushButton(self.coord_frame)
        self.building_extra_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(710 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.building_extra_button.setFont(font)
        self.building_extra_button.setObjectName("building_extra_button")
        self.building_extra_button.setText("Upload building coordinates")
        
        self.output_value_extra = QtWidgets.QLineEdit(self.coord_frame)
        self.output_value_extra.setGeometry(QtCore.QRect(int(160 * sf_x), int(670 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.output_value_extra.setFont(font)
        self.output_value_extra.setObjectName("output_value_extra")
        self.output_value_extra.setText("Extrapolation")
        
        self.example_building_button = QtWidgets.QPushButton(self.coord_frame)
        self.example_building_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(750 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.example_building_button.setFont(font)
        self.example_building_button.setObjectName("example_building_button")
        self.example_building_button.setText("Upload neighbor buildings")
        
        self.building_extra_path = QtWidgets.QLabel(self.coord_frame)
        self.building_extra_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(720 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.building_extra_path.setFont(font)
        self.building_extra_path.setObjectName("building_extra_path")
        self.building_extra_path.setText("filename.csv")
        
        self.example_building_path = QtWidgets.QLabel(self.coord_frame)
        self.example_building_path.setGeometry(QtCore.QRect(int(270 * sf_x), int(750 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.example_building_path.setFont(font)
        self.example_building_path.setObjectName("example_building_path")
        self.example_building_path.setText("filename.csv")
        
        self.extra_option_button = QtWidgets.QPushButton(self.coord_frame)
        self.extra_option_button.setGeometry(QtCore.QRect(int(310 * sf_x), int(670 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.extra_option_button.setFont(font)
        self.extra_option_button.setObjectName("extra_option_button")
        self.extra_option_button.setText("Extrapolation options")

        
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
        self.output_label_extra.raise_()
        self.building_extra_button.raise_()
        self.output_value_extra.raise_()
        self.example_building_button.raise_()
        self.building_extra_path.raise_()
        self.example_building_path.raise_()
        self.extra_option_button.raise_()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = PolygonSettingWindow()
    window.show()
    sys.exit(app.exec_())   

