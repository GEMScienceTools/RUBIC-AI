"""Define the main RUBIC-AI graphical user interface."""

import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from methods.gui_gis import GUI_geofiles
from methods.gui_methods import GUIMethods
from methods.method_window import InspectionSetting  # import the method window

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 120.0
LOG_PIXELS_X = 88


# Main Class
class GUIInterface(QtWidgets.QMainWindow):
    """Provide the main window for building inventory classification."""

    def __init__(self):
        super().__init__()
        self.setup_ui(self)
        self.methods = GUIMethods(self)  # Pass the UI to the methods class
        self.geo_qgis = GUI_geofiles(self)  # Pass the UI to the methods class

        # Automatically launch the InspectionSetting pop-up window
        self.method_dialog = InspectionSetting()  # Pass main window reference if needed
        self.method_dialog.exec_()  # This will show the method window as a modal dialog
        self.insp_method = self.method_dialog.insp_method

        # Polygon method
        if self.insp_method == 0:
            self.output_folder_value = self.method_dialog.output_folder_value
            self.file_name = self.method_dialog.output_polygon.text()
            self.ai_value = self.method_dialog.ai_value
            self.img_source = 1

        # Specific coordinates
        elif self.insp_method == 1:
            self.output_folder_value = self.method_dialog.output_folder_value
            self.file_name = self.method_dialog.specific_output_name.text()
            self.ai_value = self.method_dialog.ai_value
            self.img_source = self.method_dialog.img_source

        # Local images
        elif self.insp_method == 2:
            self.data_method = self.method_dialog.data_local
            self.folder_path = self.method_dialog.folder_path
            self.file_local_csv = self.method_dialog.file_local_csv
            self.output_folder_value = self.method_dialog.output_folder_value
            self.file_name_local = self.method_dialog.local_output_name
            self.ai_value = self.method_dialog.ai_value

        # Extrapolation
        elif self.insp_method == 3:
            # Variable which define the extrapolation mode
            self.extrapolation_mode = self.method_dialog.extrapolation_mode

            if self.extrapolation_mode == 2:
                # KNN method
                self.building_extra_path = self.method_dialog.info_pending
                self.example_building_path = self.method_dialog.info_existing
                self.extrapolation_name = self.method_dialog.extrapolation_name
                self.coord_reference = self.method_dialog.coord_reference
                self.use_coord_reference = self.method_dialog.use_coord_reference
                self.output_path = self.method_dialog.output_path
                self.k_value = self.method_dialog.k_value
                if self.use_coord_reference is True:
                    self.knn_dl_saved_path = self.method_dialog.knn_dl_saved_path
                    self.coord_reference_building_feature_path = (
                        self.method_dialog.coord_reference_building_feature_path
                    )
            else:
                # Stratified method
                self.data_population = self.method_dialog.data_population
                self.folder_path_new = self.method_dialog.folder_path_new
                self.initial_fraction = self.method_dialog.initial_fraction
                self.step_fraction = self.method_dialog.step_fraction
                self.max_fraction = self.method_dialog.max_fraction
                self.max_iterations = self.method_dialog.max_iterations
                self.stability_threshold = self.method_dialog.stability_threshold
                self.feature_strata = self.method_dialog.feature_strata

        QtWidgets.QMessageBox.information(
            self,
            "Success",
            "✅ Setup complete!\n\n"
            "Please click **OK** to proceed to the main classification window.\n"
            "Once there, click the **Next Building** button in the purple "
            "panel to begin the classification process.",
        )

        # Create the building-information CSV with OSM IDs and coordinates
        self.next_button.clicked.connect(self.methods.create_database)

        # Create the building-information CSV with OSM IDs and coordinates
        self.next_button.clicked.connect(self.methods.load_existing_insp)

        # Counts the ID of the images
        self.next_button.clicked.connect(self.methods.count_clicks_next)
        self.previous_button.clicked.connect(self.methods.count_clicks_previous)

        # Sets lat and lot values
        self.next_button.clicked.connect(self.methods.get_city_name)
        self.previous_button.clicked.connect(self.methods.get_city_name)

        # Set epoch construction values
        self.next_button.clicked.connect(self.methods.epoch_construction)

        # Gets the image from Google Street View
        self.next_button.clicked.connect(self.methods.fetch_three_step_views)
        self.previous_button.clicked.connect(self.methods.fetch_three_step_views)

        # Excutes a object detector model and trim the isolated building
        self.next_button.clicked.connect(self.methods.object_detector_building)
        self.previous_button.clicked.connect(self.methods.object_detector_building)

        # Create a manual bounding box
        self.bounding_box_1.clicked.connect(self.methods.bounding_box_frame_left)
        self.bounding_box_2.clicked.connect(self.methods.bounding_box_frame_central)
        self.bounding_box_3.clicked.connect(self.methods.bounding_box_frame_right)
        self.bounding_box_1.clicked.connect(self.methods.bounding_box)
        self.bounding_box_2.clicked.connect(self.methods.bounding_box)
        self.bounding_box_3.clicked.connect(self.methods.bounding_box)

        # GSV image angle setting
        self.set_angle_1.clicked.connect(self.methods.img_angle_left)
        self.set_angle_2.clicked.connect(self.methods.img_angle_central)
        self.set_angle_3.clicked.connect(self.methods.img_angle_right)

        # Save the values of the inspections
        self.save_data_button.clicked.connect(self.methods.save_database)

        # Save the values of the inspections
        self.vulnerability_curve_button.clicked.connect(
            self.methods.vulnerability_curve
        )

        # Clean the values of the previous inspections
        self.next_button.clicked.connect(self.methods.clean_database)
        self.previous_button.clicked.connect(self.methods.clean_database)

        # Loads and excute a deep learning model for LLRS feature prediction
        self.next_button.clicked.connect(self.methods.material_prediction)
        self.next_button.clicked.connect(self.methods.llrs_prediction)
        self.next_button.clicked.connect(self.methods.n_stories_prediction)
        self.next_button.clicked.connect(self.methods.occupancy_prediction)
        self.next_button.clicked.connect(self.methods.roof_shape_prediction)
        self.next_button.clicked.connect(self.methods.roof_material_prediction)
        self.next_button.clicked.connect(self.methods.code_level_prediction)
        self.next_button.clicked.connect(self.methods.block_position_prediction)

        # Search existing inspections
        self.search_img_button.clicked.connect(self.methods.search_inspection)

        # Extrapolation method
        self.next_button.clicked.connect(self.methods.neighbor_extrapolation)

        # Extrapolation method
        self.bloc_pos_help.clicked.connect(self.methods.help_block_position)
        self.roof_shape_help.clicked.connect(self.methods.help_roof_shape)
        self.roof_material_help.clicked.connect(self.methods.help_roof_material)
        self.irregularity_help.clicked.connect(self.methods.help_irregularity)
        self.llrs_material_help.clicked.connect(self.methods.help_llrs_material)
        self.llrs_help.clicked.connect(self.methods.help_llrs)
        self.occupancy_help.clicked.connect(self.methods.help_occupancy)
        self.code_help.clicked.connect(self.methods.help_code)
        self.n_bay_help.clicked.connect(self.methods.help_n_bay)
        self.img_quality_help.clicked.connect(self.methods.help_image_quality)

    # Method which close the GUI in the console
    def closeEvent(self, event):  # noqa: N802
        """Handle the default close button (X) event with a confirmation dialog."""
        # Create the QMessageBox dialog
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Warning)
        message_box.setWindowTitle("Confirmation")
        message_box.setText("Have you saved your results?")
        message_box.setInformativeText("Do you want to close the application?")

        # Add buttons for Yes and No
        yes_button = message_box.addButton("Yes", QtWidgets.QMessageBox.YesRole)
        message_box.addButton("No", QtWidgets.QMessageBox.NoRole)

        # Execute the QMessageBox and wait for user response
        message_box.exec()

        # Handle the response
        if message_box.clickedButton() == yes_button:
            # User confirmed, close the application
            print("GUI is closing...")
            QtWidgets.QApplication.quit()
            # Terminate the GUI and allow the script to continue.
            print("------------- Thank you ------------")
        else:
            # User canceled, do not close
            print("User canceled the close operation.")
            event.ignore()  # Prevent the GUI from closing

    # Definition of the different pluggin, button, among others (objects) of the GUI
    def setup_ui(self, window):
        """Get screen resolution to adapt to different screen sizes."""
        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        #
        # Scale the GUI based on resolution
        sf_x = screen_width / DESIGN_WIDTH
        sf_y = screen_height / DESIGN_HEIGHT
        sf_factor = np.sqrt(sf_x * sf_y)

        # DPI-based scale
        # Get a reliable DPI value
        if sys.platform.startswith("win"):
            # Windows: use ctypes to get real DPI
            import ctypes

            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOG_PIXELS_X)
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

        """Set up the user interface components."""
        # Configure main window properties
        window.setObjectName("GUIInterface")
        window.resize(
            int(1570 * sf_x), int(740 * sf_y)
        )  # Size of the GUI (Display resolution)

        # Create a central widget where other widgets will be added
        self.centralwidget = QtWidgets.QWidget(window)
        self.centralwidget.setObjectName("centralwidget")

        """ GUI Title elements """
        # Title of the GUI
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(
            QtCore.QRect(
                int(500 * sf_x), int(0 * sf_y), int(571 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(16 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")

        """ Coordinates data elements """
        # Label of latitude
        self.lat_label = QtWidgets.QLabel(self.centralwidget)
        self.lat_label.setGeometry(
            QtCore.QRect(
                int(600 * sf_x), int(50 * sf_y), int(111 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.lat_label.setFont(font)
        self.lat_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter
        )
        self.lat_label.setObjectName("lat_label")

        # Latitude value
        self.lat_value = QtWidgets.QLabel(self.centralwidget)
        self.lat_value.setGeometry(
            QtCore.QRect(
                int(720 * sf_x), int(50 * sf_y), int(171 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.lat_value.setFont(font)
        self.lat_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lat_value.setFrameShape(QtWidgets.QFrame.Box)
        self.lat_value.setObjectName("lat_value")

        # Label of longitude
        self.lon_label = QtWidgets.QLabel(self.centralwidget)
        self.lon_label.setGeometry(
            QtCore.QRect(
                int(880 * sf_x), int(50 * sf_y), int(111 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.lon_label.setFont(font)
        self.lon_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter
        )
        self.lon_label.setObjectName("lon_label")

        # Longitude value
        self.lon_value = QtWidgets.QLabel(self.centralwidget)
        self.lon_value.setGeometry(
            QtCore.QRect(
                int(1000 * sf_x), int(50 * sf_y), int(171 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.lon_value.setFont(font)
        self.lon_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lon_value.setFrameShape(QtWidgets.QFrame.Box)
        self.lon_value.setObjectName("lon_value")

        """ Country and city elements """
        # Label for the country
        self.country_label_input = QtWidgets.QLabel(self.centralwidget)
        self.country_label_input.setGeometry(
            QtCore.QRect(
                int(260 * sf_x), int(50 * sf_y), int(81 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(11 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.country_label_input.setFont(font)
        self.country_label_input.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.country_label_input.setObjectName("country_label_input")

        # Country value
        self.country_value = QtWidgets.QLabel(self.centralwidget)
        self.country_value.setGeometry(
            QtCore.QRect(
                int(350 * sf_x), int(50 * sf_y), int(161 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.country_value.setFont(font)
        self.country_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.country_value.setFrameShape(QtWidgets.QFrame.Box)
        self.country_value.setObjectName("country_value")

        # Label for the city
        self.city_label = QtWidgets.QLabel(self.centralwidget)
        self.city_label.setGeometry(
            QtCore.QRect(int(20 * sf_x), int(50 * sf_y), int(71 * sf_x), int(31 * sf_y))
        )
        font = QtGui.QFont()
        font.setPointSize(int(11 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.city_label.setFont(font)
        self.city_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.city_label.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.city_label.setObjectName("city_label")

        # City value
        self.city_value = QtWidgets.QLabel(self.centralwidget)
        self.city_value.setGeometry(
            QtCore.QRect(
                int(80 * sf_x), int(50 * sf_y), int(161 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.city_value.setFont(font)
        self.city_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.city_value.setFrameShape(QtWidgets.QFrame.Box)
        self.city_value.setObjectName("city_value")

        # Colored frame area
        self.frame_location = QtWidgets.QFrame(self.centralwidget)
        self.frame_location.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(40 * sf_y), int(1171 * sf_x), int(51 * sf_y)
            )
        )
        self.frame_location.setStyleSheet("background-color: rgb(254, 255, 174);")
        self.frame_location.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_location.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_location.setObjectName("frame_location")

        """ Button for next and previous building image """
        # Button to get the next building images
        self.next_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_button.setGeometry(
            QtCore.QRect(
                int(210 * sf_x), int(670 * sf_y), int(151 * sf_x), int(41 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.next_button.setFont(font)
        self.next_button.setObjectName("next_button")

        # Button to get the previous building images
        self.previous_button = QtWidgets.QPushButton(self.centralwidget)
        self.previous_button.setGeometry(
            QtCore.QRect(
                int(20 * sf_x), int(670 * sf_y), int(171 * sf_x), int(41 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.previous_button.setFont(font)
        self.previous_button.setObjectName("previous_button")

        # Colored area for next and previous button
        self.img_area = QtWidgets.QLabel(self.centralwidget)
        self.img_area.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(660 * sf_y), int(371 * sf_x), int(61 * sf_y)
            )
        )
        self.img_area.setStyleSheet("background-color: rgb(209, 170, 255);")
        self.img_area.setText("")
        self.img_area.setObjectName("img_area")
        self.img_area.raise_()

        """ Left Building images elements """
        # Left image color frame
        self.frame_left_img = QtWidgets.QFrame(self.centralwidget)
        self.frame_left_img.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(100 * sf_y), int(511 * sf_x), int(361 * sf_y)
            )
        )
        self.frame_left_img.setStyleSheet("background-color: rgb(102, 220, 255);")
        self.frame_left_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_left_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left_img.setObjectName("frame_left_img")

        # Image ID label for left image
        self.img_id_label_1 = QtWidgets.QLabel(self.frame_left_img)
        self.img_id_label_1.setGeometry(
            QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(91 * sf_x), int(21 * sf_y))
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.img_id_label_1.setFont(font)
        self.img_id_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.img_id_label_1.setObjectName("img_id_label_1")

        # Left image widget for best performance
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.frame_left_img)
        self.verticalLayoutWidget_5.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(40 * sf_y), int(491 * sf_x), int(311 * sf_y)
            )
        )
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")

        # Edge color left image
        self.verticalLayoutWidget_5.setStyleSheet("""
            QWidget#verticalLayoutWidget_5 {
                border: 4px solid lightgreen;
                border-radius: 6px;
            }
        """)

        # Left image bounding box
        self.left_gsv_bb = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.left_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.left_gsv_bb.setObjectName("left_gsv_bb")

        # Left building image
        self.left_gsv_img = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.left_gsv_img.setText("")
        self.left_gsv_img.setObjectName("left_gsv_img")
        self.left_gsv_bb.addWidget(self.left_gsv_img)

        # Image ID for left frame
        self.img_id_value_1 = QtWidgets.QLabel(self.frame_left_img)
        self.img_id_value_1.setGeometry(
            QtCore.QRect(
                int(110 * sf_x), int(10 * sf_y), int(61 * sf_x), int(21 * sf_y)
            )
        )
        self.img_id_value_1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.img_id_value_1.setObjectName("img_id_value_1")

        # Bounding box for image left frame
        self.bounding_box_1 = QtWidgets.QPushButton(self.frame_left_img)
        self.bounding_box_1.setGeometry(
            QtCore.QRect(
                int(380 * sf_x), int(10 * sf_y), int(121 * sf_x), int(21 * sf_y)
            )
        )
        self.bounding_box_1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.bounding_box_1.setObjectName("bounding_box_1")
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.bounding_box_1.setFont(font)
        self.bounding_box_1.setObjectName("bounding_box_1")

        # Year label for left image
        self.year_label_1 = QtWidgets.QLabel(self.frame_left_img)
        self.year_label_1.setGeometry(
            QtCore.QRect(
                int(190 * sf_x), int(10 * sf_y), int(21 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.year_label_1.setFont(font)
        self.year_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.year_label_1.setObjectName("year_label_1")
        self.year_label_1.setText("Y:")

        # Year value for left image
        self.year_value_1 = QtWidgets.QLabel(self.frame_left_img)
        self.year_value_1.setGeometry(
            QtCore.QRect(
                int(220 * sf_x), int(10 * sf_y), int(51 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(75)
        self.year_value_1.setFont(font)
        self.year_value_1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.year_value_1.setFrameShape(QtWidgets.QFrame.Box)
        self.year_value_1.setObjectName("year_value_1")
        self.year_value_1.setAlignment(QtCore.Qt.AlignCenter)
        self.year_value_1.setText("----")

        # Angle label for left image
        self.set_angle_1 = QtWidgets.QPushButton(self.frame_left_img)
        self.set_angle_1.setGeometry(
            QtCore.QRect(
                int(280 * sf_x), int(10 * sf_y), int(81 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.set_angle_1.setFont(font)
        self.set_angle_1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.set_angle_1.setObjectName("set_angle_1")
        self.set_angle_1.setText("Set ∠")

        ################## Form for building feature ##########################

        # Material for left image label
        self.material_1 = QtWidgets.QLabel(self.centralwidget)
        self.material_1.setGeometry(
            QtCore.QRect(
                int(20 * sf_x), int(470 * sf_y), int(121 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.material_1.setFont(font)
        self.material_1.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.material_1.setObjectName("material_1")
        # Material Combobox elements
        self.material_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.material_cb_1.setGeometry(
            QtCore.QRect(
                int(170 * sf_x), int(470 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.material_cb_1.setFont(font)
        self.material_cb_1.setObjectName("material_cb_1")
        # Adding material options
        self.material_cb_1.addItem("Select Material")
        self.material_cb_1.addItem("Concrete", "CR")
        self.material_cb_1.addItem("Masonry - Confined", "MCF")
        self.material_cb_1.addItem("Masonry - Reinforced", "MR")
        self.material_cb_1.addItem("Masonry - Unreinforced", "MUR")
        self.material_cb_1.addItem("Steel", "S")
        self.material_cb_1.addItem(
            "Hybrid - Confined and Unreinforced masonry", "HYB(MCF;MUR)"
        )
        self.material_cb_1.addItem("Hybrid - Concrete and Steel", "HYB(CR;S)")
        self.material_cb_1.addItem("Informal materials", "INF")
        self.material_cb_1.addItem("Wood", "W")
        self.material_cb_1.addItem("Adobe", "ADO")
        self.material_cb_1.addItem("Different materials in the two directions", "MDD")
        self.material_cb_1.addItem("Different materials in height ", "MDV")
        self.material_cb_1.addItem("Other material", "MATO")
        # Set default index
        self.material_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.material_cb_1.view().setFixedWidth(int(380 * sf_x))

        # LLRS for left image label
        self.llrs_1 = QtWidgets.QLabel(self.centralwidget)
        self.llrs_1.setGeometry(
            QtCore.QRect(
                int(20 * sf_x), int(510 * sf_y), int(121 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.llrs_1.setFont(font)
        self.llrs_1.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.llrs_1.setObjectName("llrs_1")
        # LLRS Combobox elements
        self.llrs_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.llrs_cb_1.setGeometry(
            QtCore.QRect(
                int(170 * sf_x), int(510 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.llrs_cb_1.setFont(font)
        self.llrs_cb_1.setObjectName("llrs_cb_1")
        # Adding LLRS options
        self.llrs_cb_1.addItem("Select LLRS")
        self.llrs_cb_1.addItem("Dual System", "LDUAL")
        self.llrs_cb_1.addItem("Infilled Frames", "LFINF")
        self.llrs_cb_1.addItem("Moment Frames", "LFM")
        self.llrs_cb_1.addItem("Walls", "LWAL")
        self.llrs_cb_1.addItem("Braced Frames", "LFBR")
        self.llrs_cb_1.addItem("No lateral load-resisting system", "LN")
        self.llrs_cb_1.addItem("Post and beam", "LPB")
        self.llrs_cb_1.addItem("Flat slab/plate or waffle slab", "LFLS")
        self.llrs_cb_1.addItem("Different LLRS in the two directions", "LDD")
        self.llrs_cb_1.addItem("Different LLRS in height ", "LHV")
        self.llrs_cb_1.addItem("Other", "LO")
        # Set default index
        self.llrs_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.llrs_cb_1.view().setFixedWidth(int(350 * sf_x))

        # Number of stories for left image label
        self.n_stories_1 = QtWidgets.QLabel(self.centralwidget)
        self.n_stories_1.setGeometry(
            QtCore.QRect(
                int(20 * sf_x), int(550 * sf_y), int(121 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.n_stories_1.setFont(font)
        self.n_stories_1.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.n_stories_1.setObjectName("n_stories_1")
        # Number of stories value
        self.n_stories_value_1 = QtWidgets.QComboBox(self.centralwidget)
        self.n_stories_value_1.setGeometry(
            QtCore.QRect(
                int(170 * sf_x), int(550 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.n_stories_value_1.setFont(font)
        self.n_stories_value_1.setObjectName("n_stories_value_1")
        # Adding Number of Stories options
        self.n_stories_value_1.addItem("Select Number of Stories")
        self.n_stories_value_1.addItem("1", "1")
        self.n_stories_value_1.addItem("2", "2")
        self.n_stories_value_1.addItem("3", "3")
        self.n_stories_value_1.addItem("4", "4")
        self.n_stories_value_1.addItem("5", "5")
        self.n_stories_value_1.addItem("6-7", "6.0-7.0")
        self.n_stories_value_1.addItem("8-9", "8.0-9.0")
        self.n_stories_value_1.addItem("10-12", "10.0-12.0")
        self.n_stories_value_1.addItem("13+", "13+")
        # Set default index
        self.n_stories_value_1.setCurrentIndex(0)
        # Scale dropdown width
        self.n_stories_value_1.view().setFixedWidth(int(250 * sf_x))

        # Occupancy type for left image label
        self.occupancy = QtWidgets.QLabel(self.centralwidget)
        self.occupancy.setGeometry(
            QtCore.QRect(
                int(20 * sf_x), int(590 * sf_y), int(121 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.occupancy.setFont(font)
        self.occupancy.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.occupancy.setObjectName("occupancy")
        # Occupancy Combobox elements
        self.occup_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.occup_cb_1.setGeometry(
            QtCore.QRect(
                int(170 * sf_x), int(590 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.occup_cb_1.setFont(font)
        self.occup_cb_1.setObjectName("occup_cb_1")
        # Adding Occupancy options
        self.occup_cb_1.addItem("Select Occupancy")
        self.occup_cb_1.addItem("Commercial", "COM")
        self.occup_cb_1.addItem("Industrial", "IND")
        self.occup_cb_1.addItem("Mixed (Residential + Commercial)", "MIX(RES;COM)")
        self.occup_cb_1.addItem("Residential", "RES")
        self.occup_cb_1.addItem("Educational", "EDU")
        self.occup_cb_1.addItem("Government", "GOV")
        self.occup_cb_1.addItem("Healthcare", "HEA")
        self.occup_cb_1.addItem("Other", "OCO")
        # Set default index
        self.occup_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.occup_cb_1.view().setFixedWidth(int(300 * sf_x))

        # Roof Shape for left image label
        self.roof_shape_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.roof_shape_label_1.setGeometry(
            QtCore.QRect(
                int(490 * sf_x), int(470 * sf_y), int(131 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.roof_shape_label_1.setFont(font)
        self.roof_shape_label_1.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.roof_shape_label_1.setObjectName("roof_shape_label_1")
        self.roof_shape_label_1.setText("Roof Shape:")

        self.roof_shape_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.roof_shape_cb_1.setGeometry(
            QtCore.QRect(
                int(640 * sf_x), int(470 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.roof_shape_cb_1.setFont(font)
        self.roof_shape_cb_1.setObjectName("roof_shape_cb_1")
        self.roof_shape_cb_1.addItem("Select Roof Shape")
        self.roof_shape_cb_1.addItem("Flat", "RSH1")
        self.roof_shape_cb_1.addItem("Pitched with gable ends", "RSH2")
        self.roof_shape_cb_1.addItem("Pitched and hipped", "RSH3")
        self.roof_shape_cb_1.addItem("Monopitch", "RSH5")
        self.roof_shape_cb_1.addItem("Curved", "RSH7")
        self.roof_shape_cb_1.addItem("Pitched with dormers", "RSH4")
        self.roof_shape_cb_1.addItem("Sawtooth", "RSH6")
        self.roof_shape_cb_1.addItem("Complex regular", "RSH8")
        self.roof_shape_cb_1.addItem("Complex irregular", "RSH9")
        self.roof_shape_cb_1.addItem("Other", "RSHO")

        # Roof Material for left image label
        self.roof_material_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.roof_material_label_1.setGeometry(
            QtCore.QRect(
                int(490 * sf_x), int(510 * sf_y), int(131 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.roof_material_label_1.setFont(font)
        self.roof_material_label_1.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.roof_material_label_1.setObjectName("roof_material_label_1")
        self.roof_material_label_1.setText("Roof Material:")

        self.roof_material_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.roof_material_cb_1.setGeometry(
            QtCore.QRect(
                int(640 * sf_x), int(510 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.roof_material_cb_1.setFont(font)
        self.roof_material_cb_1.setObjectName("roof_material_cb_1")
        self.roof_material_cb_1.addItem("Select Roof Material")
        self.roof_material_cb_1.addItem("Concrete", "RMN")
        self.roof_material_cb_1.addItem("Clay or concrete tile", "RMT1")
        self.roof_material_cb_1.addItem("Metal or asbestos sheets", "RMT6")
        self.roof_material_cb_1.addItem("Wooden and asphalt shingles", "RMT7")
        self.roof_material_cb_1.addItem("Slate", "RMT4")
        self.roof_material_cb_1.addItem("Solar panelled roofs", "RMT10")
        self.roof_material_cb_1.addItem("Other", "RMTO")
        # Set default index
        self.roof_material_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.roof_material_cb_1.view().setFixedWidth(int(250 * sf_x))

        # Code level for left image label
        self.age_1 = QtWidgets.QLabel(self.centralwidget)
        self.age_1.setGeometry(
            QtCore.QRect(
                int(490 * sf_x), int(550 * sf_y), int(121 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.age_1.setFont(font)
        self.age_1.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.age_1.setObjectName("age_1")
        # Code level Combobox elements
        self.age_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.age_cb_1.setGeometry(
            QtCore.QRect(
                int(640 * sf_x), int(550 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.age_cb_1.setFont(font)
        self.age_cb_1.setObjectName("age_cb_1")
        # Adding Code Level options
        self.age_cb_1.addItem("Select Code Level")
        self.age_cb_1.addItem("High-Code", "CDH")
        self.age_cb_1.addItem("Moderate-code", "CDM")
        self.age_cb_1.addItem("Low-Code", "CDL")
        self.age_cb_1.addItem("No-Code", "CDN")
        # Set default index
        self.age_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.age_cb_1.view().setFixedWidth(int(250 * sf_x))

        # Block position for left image label
        self.block_position = QtWidgets.QLabel(self.centralwidget)
        self.block_position.setGeometry(
            QtCore.QRect(
                int(490 * sf_x), int(590 * sf_y), int(131 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.block_position.setFont(font)
        self.block_position.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.block_position.setObjectName("block_position")
        # Block position Combobox elements
        self.bck_pos_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.bck_pos_cb_1.setGeometry(
            QtCore.QRect(
                int(640 * sf_x), int(590 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.bck_pos_cb_1.setFont(font)
        self.bck_pos_cb_1.setObjectName("bck_pos_cb_1")
        # Adding Block Position options
        self.bck_pos_cb_1.addItem("Select Block Position")
        self.bck_pos_cb_1.addItem("Adjoining building(s) one side", "BP1")
        self.bck_pos_cb_1.addItem("Adjoining building(s) two side", "BP2")
        self.bck_pos_cb_1.addItem("Adjoining building(s) three side", "BP3")
        self.bck_pos_cb_1.addItem("Detached building", "BPD")
        # Set default index
        self.bck_pos_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.bck_pos_cb_1.view().setFixedWidth(int(250 * sf_x))

        # Irregularity
        self.irregularity = QtWidgets.QLabel(self.centralwidget)
        self.irregularity.setGeometry(
            QtCore.QRect(
                int(1010 * sf_x), int(470 * sf_y), int(201 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.irregularity.setFont(font)
        self.irregularity.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.irregularity.setObjectName("irregularity")
        self.irregularity.setText("Vertical Irregularity:")

        self.irregularity_cb = QtWidgets.QComboBox(self.centralwidget)
        self.irregularity_cb.setGeometry(
            QtCore.QRect(
                int(1220 * sf_x), int(470 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.irregularity_cb.setFont(font)
        self.irregularity_cb.setObjectName("irregularity_cb")
        self.irregularity_cb.addItem("Select Irregularity")
        self.irregularity_cb.addItem("Soft story", "SOS")
        self.irregularity_cb.addItem("Short column", "SHC")
        self.irregularity_cb.addItem("Pounding potential", "POP")
        self.irregularity_cb.addItem("Setback", "SET")
        self.irregularity_cb.addItem("Change in vertical structure", "CHV")
        self.irregularity_cb.addItem("Other vertical irregularity", "IRVO")

        self.n_bay_label = QtWidgets.QLabel(self.centralwidget)
        self.n_bay_label.setGeometry(
            QtCore.QRect(
                int(1010 * sf_x), int(510 * sf_y), int(201 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.n_bay_label.setFont(font)
        self.n_bay_label.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.n_bay_label.setObjectName("n_bay_label")

        self.n_bay_cb = QtWidgets.QComboBox(self.centralwidget)
        self.n_bay_cb.setGeometry(
            QtCore.QRect(
                int(1220 * sf_x), int(510 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.n_bay_cb.setFont(font)
        self.n_bay_cb.setObjectName("n_bay_cb")
        self.n_bay_cb.addItem("Select Number of Bays")
        self.n_bay_cb.addItem("1", "1")
        self.n_bay_cb.addItem("2", "2")
        self.n_bay_cb.addItem("3", "3")
        self.n_bay_cb.addItem("4", "4")
        self.n_bay_cb.addItem("5", "5")
        self.n_bay_cb.addItem("6", "6")
        self.n_bay_cb.addItem("7+", "7+")
        self.n_bay_cb.setCurrentIndex(0)
        self.n_bay_cb.view().setFixedWidth(int(250 * sf_x))

        # Epoch of construction for left image label
        self.epc_const_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.epc_const_label_1.setGeometry(
            QtCore.QRect(
                int(1010 * sf_x), int(550 * sf_y), int(201 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.epc_const_label_1.setFont(font)
        self.epc_const_label_1.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.epc_const_label_1.setObjectName("epc_const_label_1")
        self.epc_const_label_1.setText("Epoch of construction:")

        self.epc_const_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.epc_const_cb_1.setGeometry(
            QtCore.QRect(
                int(1220 * sf_x), int(550 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.epc_const_cb_1.setFont(font)
        self.epc_const_cb_1.setObjectName("epc_const_cb_1")
        self.epc_const_cb_1.addItem("Select Epoch of Construction")

        # Image quality for left image label
        self.img_quality = QtWidgets.QLabel(self.centralwidget)
        self.img_quality.setGeometry(
            QtCore.QRect(
                int(1010 * sf_x), int(590 * sf_y), int(131 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.img_quality.setFont(font)
        self.img_quality.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.img_quality.setObjectName("img_quality")

        # Image quality Combobox elements
        self.img_q_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.img_q_cb_1.setGeometry(
            QtCore.QRect(
                int(1220 * sf_x), int(590 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.img_q_cb_1.setFont(font)
        self.img_q_cb_1.setObjectName("img_q_cb_1")
        self.img_q_cb_1.addItem("Select Image quality")
        self.img_q_cb_1.addItem("Excellent", "Excellent")
        self.img_q_cb_1.addItem("Good", "Good")
        self.img_q_cb_1.addItem("Intermediate", "Intermediate")
        self.img_q_cb_1.addItem("Bad", "Bad")
        self.img_q_cb_1.setCurrentIndex(0)
        self.img_q_cb_1.view().setFixedWidth(int(250 * sf_x))

        """ Central Building images elements """
        # Central image color frame
        self.frame_central_img = QtWidgets.QFrame(self.centralwidget)
        self.frame_central_img.setGeometry(
            QtCore.QRect(
                int(530 * sf_x), int(100 * sf_y), int(511 * sf_x), int(361 * sf_y)
            )
        )
        self.frame_central_img.setStyleSheet("background-color: rgb(255, 170, 127)")
        self.frame_central_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_central_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_central_img.setObjectName("frame_central_img")
        # Elements for display central image or error message
        self.img_id_label_2 = QtWidgets.QLabel(self.frame_central_img)
        self.img_id_label_2.setGeometry(
            QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(91 * sf_x), int(21 * sf_y))
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.img_id_label_2.setFont(font)
        self.img_id_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.img_id_label_2.setObjectName("img_id_label_2")
        # Central image widget for best performance
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_central_img)
        self.verticalLayoutWidget.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(40 * sf_y), int(491 * sf_x), int(311 * sf_y)
            )
        )
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        # Thick green border
        self.verticalLayoutWidget.setStyleSheet("""
            QWidget#verticalLayoutWidget {
                border: 4px solid lightgreen;
                border-radius: 6px;
            }
        """)

        # Central image bounding box
        self.central_gsv_bb = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.central_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.central_gsv_bb.setObjectName("central_gsv_bb")
        # Central building image
        self.central_gsv_img = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.central_gsv_img.setText("")
        self.central_gsv_img.setObjectName("central_gsv_img")
        self.central_gsv_bb.addWidget(self.central_gsv_img)
        # Image ID for central frame
        self.img_id_value_2 = QtWidgets.QLabel(self.frame_central_img)
        self.img_id_value_2.setGeometry(
            QtCore.QRect(
                int(110 * sf_x), int(10 * sf_y), int(61 * sf_x), int(21 * sf_y)
            )
        )
        self.img_id_value_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.img_id_value_2.setObjectName("img_id_value_2")
        # Bounding box for image central frame
        self.bounding_box_2 = QtWidgets.QPushButton(self.frame_central_img)
        self.bounding_box_2.setGeometry(
            QtCore.QRect(
                int(380 * sf_x), int(10 * sf_y), int(121 * sf_x), int(21 * sf_y)
            )
        )
        self.bounding_box_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.bounding_box_2.setObjectName("bounding_box_2")
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.bounding_box_2.setFont(font)
        self.bounding_box_2.setObjectName("bounding_box_2")

        # Year label for central image
        self.year_label_2 = QtWidgets.QLabel(self.frame_central_img)
        self.year_label_2.setGeometry(
            QtCore.QRect(
                int(190 * sf_x), int(10 * sf_y), int(21 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.year_label_2.setFont(font)
        self.year_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.year_label_2.setObjectName("year_label_2")
        self.year_label_2.setText("Y:")

        # Year value for central image
        self.year_value_2 = QtWidgets.QLabel(self.frame_central_img)
        self.year_value_2.setGeometry(
            QtCore.QRect(
                int(220 * sf_x), int(10 * sf_y), int(51 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(75)
        self.year_value_2.setFont(font)
        self.year_value_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.year_value_2.setFrameShape(QtWidgets.QFrame.Box)
        self.year_value_2.setObjectName("year_value_2")
        self.year_value_2.setAlignment(QtCore.Qt.AlignCenter)
        self.year_value_2.setText("----")

        # Angle label central image
        self.set_angle_2 = QtWidgets.QPushButton(self.frame_central_img)
        self.set_angle_2.setGeometry(
            QtCore.QRect(
                int(280 * sf_x), int(10 * sf_y), int(81 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.set_angle_2.setFont(font)
        self.set_angle_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.set_angle_2.setObjectName("set_angle_2")
        self.set_angle_2.setText("Set ∠")

        """ Right Building images elements """
        # Right image color frame
        self.frame_right_img = QtWidgets.QFrame(self.centralwidget)
        self.frame_right_img.setGeometry(
            QtCore.QRect(
                int(1050 * sf_x), int(100 * sf_y), int(511 * sf_x), int(361 * sf_y)
            )
        )
        self.frame_right_img.setStyleSheet("background-color: rgb(183, 252, 172)")
        self.frame_right_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_right_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_right_img.setObjectName("frame_right_img")
        # Elements for display right image or error message
        self.img_id_label_3 = QtWidgets.QLabel(self.frame_right_img)
        self.img_id_label_3.setGeometry(
            QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(91 * sf_x), int(21 * sf_y))
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.img_id_label_3.setFont(font)
        self.img_id_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.img_id_label_3.setObjectName("img_id_label_3")
        # Right image widget for best performance
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_right_img)
        self.verticalLayoutWidget_2.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(39 * sf_y), int(491 * sf_x), int(311 * sf_y)
            )
        )
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")

        # Thick green border
        self.verticalLayoutWidget_2.setStyleSheet("""
            QWidget#verticalLayoutWidget_2 {
                border: 4px solid cyan;
                border-radius: 6px;
            }
        """)

        # Right image bounding box
        self.right_gsv_bb = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.right_gsv_bb.setContentsMargins(0, 0, 0, 0)
        self.right_gsv_bb.setObjectName("right_gsv_bb")
        # Right building image
        self.right_gsv_img = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.right_gsv_img.setText("")
        self.right_gsv_img.setObjectName("right_gsv_img")
        self.right_gsv_bb.addWidget(self.right_gsv_img)
        # Image ID for right frame
        self.img_id_value_3 = QtWidgets.QLabel(self.frame_right_img)
        self.img_id_value_3.setGeometry(
            QtCore.QRect(
                int(110 * sf_x), int(10 * sf_y), int(61 * sf_x), int(21 * sf_y)
            )
        )
        self.img_id_value_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.img_id_value_3.setObjectName("img_id_value_3")
        # Bounding box for image right frame
        self.bounding_box_3 = QtWidgets.QPushButton(self.frame_right_img)
        self.bounding_box_3.setGeometry(
            QtCore.QRect(
                int(380 * sf_x), int(10 * sf_y), int(121 * sf_x), int(21 * sf_y)
            )
        )
        self.bounding_box_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.bounding_box_3.setObjectName("bounding_box_3")
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.bounding_box_3.setFont(font)
        self.bounding_box_3.setObjectName("bounding_box_3")

        # Year label for right image
        self.year_label_3 = QtWidgets.QLabel(self.frame_right_img)
        self.year_label_3.setGeometry(
            QtCore.QRect(
                int(190 * sf_x), int(10 * sf_y), int(21 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.year_label_3.setFont(font)
        self.year_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.year_label_3.setObjectName("year_label_3")
        self.year_label_3.setText("Y:")

        # Year value for right image
        self.year_value_3 = QtWidgets.QLabel(self.frame_right_img)
        self.year_value_3.setGeometry(
            QtCore.QRect(
                int(220 * sf_x), int(10 * sf_y), int(51 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(75)
        self.year_value_3.setFont(font)
        self.year_value_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.year_value_3.setFrameShape(QtWidgets.QFrame.Box)
        self.year_value_3.setObjectName("year_value_3")
        self.year_value_3.setAlignment(QtCore.Qt.AlignCenter)
        self.year_value_3.setText("----")

        # Angle label for right image
        self.set_angle_3 = QtWidgets.QPushButton(self.frame_right_img)
        self.set_angle_3.setGeometry(
            QtCore.QRect(
                int(280 * sf_x), int(10 * sf_y), int(81 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.set_angle_3.setFont(font)
        self.set_angle_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.set_angle_3.setObjectName("set_angle_3")
        self.set_angle_3.setText("Set ∠")

        ###########################################################################################
        ###########################################################################################

        """ Search button elements """
        self.search_img_button = QtWidgets.QPushButton(self.centralwidget)
        self.search_img_button.setGeometry(
            QtCore.QRect(
                int(1400 * sf_x), int(650 * sf_y), int(151 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.search_img_button.setFont(font)
        self.search_img_button.setObjectName("search_img_button")

        self.search_img_value = QtWidgets.QLineEdit(self.centralwidget)
        self.search_img_value.setPlaceholderText("Enter image ID to search")
        self.search_img_value.setGeometry(
            QtCore.QRect(
                int(1250 * sf_x), int(650 * sf_y), int(141 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.search_img_value.setFont(font)
        self.search_img_value.setObjectName("search_img_value")

        self.search_img_label = QtWidgets.QLabel(self.centralwidget)
        self.search_img_label.setGeometry(
            QtCore.QRect(
                int(1130 * sf_x), int(650 * sf_y), int(101 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.search_img_label.setFont(font)
        self.search_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.search_img_label.setObjectName("search_img_label")

        """ GEM Logo elements """
        self.GEM_logo = QtWidgets.QLabel(self.centralwidget)
        self.GEM_logo.setGeometry(
            QtCore.QRect(
                int(1380 * sf_x), int(5 * sf_y), int(177 * sf_x), int(65 * sf_y)
            )
        )
        self.GEM_logo.setText("")
        self.GEM_logo.setPixmap(QtGui.QPixmap("help_img/GEM_Logo.png"))
        self.GEM_logo.setScaledContents(True)
        self.GEM_logo.setObjectName("GEM_logo")

        self.RUBIC_logo = QtWidgets.QLabel(self.centralwidget)
        self.RUBIC_logo.setGeometry(
            QtCore.QRect(
                int(1323 * sf_x), int(5 * sf_y), int(60 * sf_x), int(65 * sf_y)
            )
        )
        self.RUBIC_logo.setText("")
        self.RUBIC_logo.setPixmap(QtGui.QPixmap("help_img/RUBIC_logo.png"))
        self.RUBIC_logo.setScaledContents(True)
        self.RUBIC_logo.setObjectName("GEM_logo")

        """ GEM icon GUI elements """
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))

        """ Progress Bar elements """
        # Progress bar widget
        self.progress_bar_method = QtWidgets.QProgressBar(self.centralwidget)
        self.progress_bar_method.setGeometry(
            QtCore.QRect(
                int(620 * sf_x), int(680 * sf_y), int(161 * sf_x), int(23 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.progress_bar_method.setFont(font)
        self.progress_bar_method.setProperty("value", 0)
        self.progress_bar_method.setObjectName("progress_bar_method")

        # Progress bar label value
        self.method_progress = QtWidgets.QLabel(self.centralwidget)
        self.method_progress.setGeometry(
            QtCore.QRect(
                int(800 * sf_x), int(670 * sf_y), int(291 * sf_x), int(41 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.method_progress.setFont(font)
        self.method_progress.setObjectName("method_progress")

        """ AI Powered activation elements """
        # AI checkbox activation
        self.ai_check = QtWidgets.QCheckBox(self.centralwidget)
        self.ai_check.setGeometry(
            QtCore.QRect(
                int(400 * sf_x), int(680 * sf_y), int(131 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.ai_check.setFont(font)
        self.ai_check.setObjectName("ai_check")

        """ Save data button elements """
        # Save data button
        self.save_data_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_data_button.setGeometry(
            QtCore.QRect(
                int(1400 * sf_x), int(690 * sf_y), int(111 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.save_data_button.setStyleSheet("background-color: rgb(216, 216, 216);")
        self.save_data_button.setFont(font)
        self.save_data_button.setObjectName("save_data_button")

        """ Vulnerability curve button"""
        # Save data button
        self.vulnerability_curve_button = QtWidgets.QPushButton(self.centralwidget)
        self.vulnerability_curve_button.setGeometry(
            QtCore.QRect(
                int(1130 * sf_x), int(690 * sf_y), int(191 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.vulnerability_curve_button.setFont(font)
        self.vulnerability_curve_button.setObjectName("vulnerability_curve_button")
        self.vulnerability_curve_button.setText("Vulnerability curve")

        """ Saved frame"""
        self.saved_frame = QtWidgets.QLabel(self.centralwidget)
        self.saved_frame.setGeometry(
            QtCore.QRect(
                int(1120 * sf_x), int(640 * sf_y), int(441 * sf_x), int(91 * sf_y)
            )
        )
        self.saved_frame.setStyleSheet("background-color: rgb(255, 255, 170);")
        self.saved_frame.setText("")
        self.saved_frame.setObjectName("saved_frame")

        icon_size = QtCore.QSize(int(31 * sf_x), int(31 * sf_x))  # Icon is square
        self.bloc_pos_help = QtWidgets.QPushButton(self.centralwidget)

        # Load and scale the icon
        pixmap = QtGui.QPixmap("help_img/help_icon.png").scaled(
            icon_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        icon = QtGui.QIcon(pixmap)

        self.bloc_pos_help.setIcon(icon)
        self.bloc_pos_help.setIconSize(icon_size)

        # Remove borders, background, and relief
        self.bloc_pos_help.setFlat(True)
        self.bloc_pos_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05); /* optional light hover effect */
            }
        """)

        # Geometry
        self.bloc_pos_help.setGeometry(
            QtCore.QRect(
                int(900 * sf_x), int(590 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.bloc_pos_help.setObjectName("bloc_pos_help")

        # ===========================
        # Roof Shape Help Button
        # ===========================
        self.roof_shape_help = QtWidgets.QPushButton(self.centralwidget)
        self.roof_shape_help.setIcon(icon)
        self.roof_shape_help.setIconSize(icon_size)
        self.roof_shape_help.setGeometry(
            QtCore.QRect(
                int(900 * sf_x), int(470 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.roof_shape_help.setObjectName("roof_shape_help")

        # Flat, transparent style
        self.roof_shape_help.setFlat(True)
        self.roof_shape_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05); /* optional hover glow */
            }
        """)

        # ===========================
        # Roof Material Help Button
        # ===========================
        self.roof_material_help = QtWidgets.QPushButton(self.centralwidget)
        self.roof_material_help.setIcon(icon)
        self.roof_material_help.setIconSize(icon_size)
        self.roof_material_help.setGeometry(
            QtCore.QRect(
                int(900 * sf_x), int(510 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.roof_material_help.setObjectName("roof_material_help")

        # Flat, transparent style
        self.roof_material_help.setFlat(True)
        self.roof_material_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        # ===========================
        # Irregularity Help Button
        # ===========================
        self.irregularity_help = QtWidgets.QPushButton(self.centralwidget)
        self.irregularity_help.setIcon(icon)
        self.irregularity_help.setIconSize(icon_size)
        self.irregularity_help.setGeometry(
            QtCore.QRect(
                int(1480 * sf_x), int(470 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.irregularity_help.setObjectName("irregularity_help")

        # Flat, transparent style
        self.irregularity_help.setFlat(True)
        self.irregularity_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        # ===========================
        # LLRS Material Help Button
        # ===========================
        self.llrs_material_help = QtWidgets.QPushButton(self.centralwidget)
        self.llrs_material_help.setIcon(icon)
        self.llrs_material_help.setIconSize(icon_size)
        self.llrs_material_help.setGeometry(
            QtCore.QRect(
                int(420 * sf_x), int(470 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.llrs_material_help.setObjectName("llrs_material_help")

        # Flat, transparent style
        self.llrs_material_help.setFlat(True)
        self.llrs_material_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        # ===========================
        # LLRS Help Button
        # ===========================
        self.llrs_help = QtWidgets.QPushButton(self.centralwidget)
        self.llrs_help.setIcon(icon)
        self.llrs_help.setIconSize(icon_size)
        self.llrs_help.setGeometry(
            QtCore.QRect(
                int(420 * sf_x), int(510 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.llrs_help.setObjectName("llrs_help")

        # Flat, transparent style
        self.llrs_help.setFlat(True)
        self.llrs_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        # ===========================
        # Occupancy Help Button
        # ===========================
        self.occupancy_help = QtWidgets.QPushButton(self.centralwidget)
        self.occupancy_help.setIcon(icon)
        self.occupancy_help.setIconSize(icon_size)
        self.occupancy_help.setGeometry(
            QtCore.QRect(
                int(420 * sf_x), int(590 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.occupancy_help.setObjectName("occupancy_help")

        # Flat, transparent style
        self.occupancy_help.setFlat(True)
        self.occupancy_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        # ===========================
        # Code Level Help Button
        # ===========================
        self.code_help = QtWidgets.QPushButton(self.centralwidget)
        self.code_help.setIcon(icon)
        self.code_help.setIconSize(icon_size)
        self.code_help.setGeometry(
            QtCore.QRect(
                int(900 * sf_x), int(550 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.code_help.setObjectName("code_help")

        # Flat, transparent style
        self.code_help.setFlat(True)
        self.code_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        # ===========================
        # Number of bays Help Button
        # ===========================
        self.n_bay_help = QtWidgets.QPushButton(self.centralwidget)
        self.n_bay_help.setIcon(icon)
        self.n_bay_help.setIconSize(icon_size)
        self.n_bay_help.setGeometry(
            QtCore.QRect(
                int(1480 * sf_x), int(510 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.n_bay_help.setObjectName("n_bay_help")

        # Flat, transparent style
        self.n_bay_help.setFlat(True)
        self.n_bay_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        # ===========================
        # Image quality Help Button
        # ===========================
        self.img_quality_help = QtWidgets.QPushButton(self.centralwidget)
        self.img_quality_help.setIcon(icon)
        self.img_quality_help.setIconSize(icon_size)
        self.img_quality_help.setGeometry(
            QtCore.QRect(
                int(1480 * sf_x), int(590 * sf_y), int(31 * sf_x), int(31 * sf_y)
            )
        )
        self.img_quality_help.setObjectName("img_quality_help")

        # Flat, transparent style
        self.img_quality_help.setFlat(True)
        self.img_quality_help.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
            }
        """)

        ################################################################################
        ################################################################################
        """ Raise all elements """
        self.frame_location.raise_()
        self.saved_frame.raise_()
        self.country_label_input.raise_()
        self.city_label.raise_()
        self.title.raise_()
        self.lat_label.raise_()
        self.lon_label.raise_()
        self.previous_button.raise_()
        self.lat_value.raise_()
        self.lon_value.raise_()
        self.country_value.raise_()
        self.city_value.raise_()
        self.frame_central_img.raise_()
        self.frame_right_img.raise_()
        self.frame_left_img.raise_()
        self.material_1.raise_()
        self.llrs_1.raise_()
        self.age_1.raise_()
        self.n_stories_1.raise_()
        self.occupancy.raise_()
        self.block_position.raise_()
        self.next_button.raise_()
        self.material_cb_1.raise_()
        self.llrs_cb_1.raise_()
        self.age_cb_1.raise_()
        self.occup_cb_1.raise_()
        self.bck_pos_cb_1.raise_()
        self.img_q_cb_1.raise_()
        self.search_img_button.raise_()
        self.search_img_value.raise_()
        self.search_img_label.raise_()
        self.GEM_logo.raise_()
        self.RUBIC_logo.raise_()
        self.progress_bar_method.raise_()
        self.method_progress.raise_()
        self.ai_check.raise_()
        self.save_data_button.raise_()
        self.n_stories_value_1.raise_()
        self.bounding_box_1.raise_()
        self.bounding_box_2.raise_()
        self.bounding_box_3.raise_()
        self.roof_shape_cb_1.raise_()
        self.epc_const_cb_1.raise_()
        self.roof_shape_cb_1.raise_()
        self.epc_const_label_1.raise_()
        self.roof_shape_label_1.raise_()
        self.roof_material_label_1.raise_()
        self.roof_material_cb_1.raise_()
        self.bloc_pos_help.raise_()
        self.roof_shape_help.raise_()
        self.roof_material_help.raise_()
        self.irregularity_help.raise_()
        self.llrs_material_help.raise_()
        self.llrs_help.raise_()
        self.occupancy_help.raise_()
        self.code_help.raise_()
        self.n_bay_help.raise_()
        self.img_quality_help.raise_()
        self.vulnerability_curve_button.raise_()
        self.n_bay_label.raise_()
        self.n_bay_cb.raise_()

        window.setCentralWidget(self.centralwidget)

        self.retranslate_ui(window)
        QtCore.QMetaObject.connectSlotsByName(window)

    def retranslate_ui(self, window):
        """Set translatable text for the interface widgets."""
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(
            _translate("GUIInterface", "RUBIC-AI: Building Inventory Classifier")
        )
        self.country_label_input.setText(_translate("GUIInterface", "Country:"))
        self.city_label.setText(_translate("GUIInterface", "City:"))
        self.title.setText(
            _translate("GUIInterface", "RUBIC-AI: Building Inventory Classifier")
        )
        self.lat_label.setText(_translate("GUIInterface", "Latitude: "))
        self.lon_label.setText(_translate("GUIInterface", "Longitude:"))
        self.previous_button.setText(_translate("GUIInterface", "Previous Building"))
        self.lat_value.setText(_translate("GUIInterface", "-"))
        self.lon_value.setText(_translate("GUIInterface", "-"))
        self.country_value.setText(_translate("GUIInterface", "-"))
        self.city_value.setText(_translate("GUIInterface", "-"))
        self.img_id_label_2.setText(_translate("GUIInterface", "Image ID:"))
        self.img_id_value_2.setText(_translate("GUIInterface", "-"))
        self.img_id_label_3.setText(_translate("GUIInterface", "Image ID:"))
        self.img_id_value_3.setText(_translate("GUIInterface", "-"))
        self.img_id_label_1.setText(_translate("GUIInterface", "Image ID:"))
        self.img_id_value_1.setText(_translate("GUIInterface", "-"))
        self.img_quality.setText(_translate("GUIInterface", "Image Quality:"))
        self.material_1.setText(_translate("GUIInterface", "LLRS Material:"))
        self.llrs_1.setText(_translate("GUIInterface", "LLRS:"))
        self.age_1.setText(_translate("GUIInterface", "Code Level:"))
        self.n_stories_1.setText(_translate("GUIInterface", "N° of Stories:"))
        self.occupancy.setText(_translate("GUIInterface", "Occupancy:"))
        self.block_position.setText(_translate("GUIInterface", "Block Position:"))
        self.next_button.setText(_translate("GUIInterface", "Next Building"))
        self.material_cb_1.setItemText(0, _translate("GUIInterface", "Select Material"))
        self.llrs_cb_1.setItemText(0, _translate("GUIInterface", "Select LLRS"))
        self.age_cb_1.setItemText(0, _translate("GUIInterface", "Select Code Level"))
        self.occup_cb_1.setItemText(
            0, _translate("GUIInterface", "Select Occupancy Type")
        )
        self.bck_pos_cb_1.setItemText(
            0, _translate("GUIInterface", "Select Block Position")
        )
        self.img_q_cb_1.setItemText(
            0, _translate("GUIInterface", "Select Image Quality")
        )
        self.search_img_button.setText(_translate("GUIInterface", "Search Building"))
        self.search_img_label.setText(_translate("GUIInterface", "Image ID:"))
        self.method_progress.setText(_translate("GUIInterface", "-"))
        self.ai_check.setText(_translate("GUIInterface", "AI Powered"))
        self.save_data_button.setText(_translate("GUIInterface", "Save data"))
        self.bounding_box_1.setText(_translate("search_img_value", "Manual box"))
        self.bounding_box_2.setText(_translate("search_img_value", "Manual box"))
        self.bounding_box_3.setText(_translate("search_img_value", "Manual box"))
        self.n_bay_label.setText(_translate("search_img_value", "Number of bays:"))
        self.n_bay_cb.setItemText(
            0, _translate("search_img_value", "Select Number of Bays")
        )
