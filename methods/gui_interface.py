# import libraries
from PyQt5 import QtWidgets, QtCore, QtGui
from methods.gui_methods import GUIMethods
from methods.gui_gis import GUI_geofiles

from methods.method_window import InspectionSetting  # import the method window

# Main Class
class GUIInterface(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.methods = GUIMethods(self)  # Pass the UI to the methods class
        self.geo_qgis = GUI_geofiles(self)  # Pass the UI to the methods class
        
        # ✅ Automatically launch the InspectionSetting pop-up window
        self.method_dialog = InspectionSetting()  # Pass main window reference if needed
        self.method_dialog.exec_()  # This will show the method window as a modal dialog    
        self.insp_method = self.method_dialog.insp_method
        if self.insp_method == 0:
            self.output_folder_value = self.method_dialog.output_folder_value
            self.file_name = self.method_dialog.output_polygon.text()
            self.city = self.method_dialog.city
            self.country = self.method_dialog.country
            self.ai_value = self.method_dialog.ai_value
            
        elif self.insp_method == 1:
            self.output_folder_value = self.method_dialog.output_folder_value
            self.file_name = self.method_dialog.specific_output_name.text()
            self.ai_value = self.method_dialog.ai_value 
            
        elif self.insp_method == 2:
            self.data_method = self.method_dialog.data_local
            self.folder_path = self.method_dialog.folder_path
            self.file_local_csv = self.method_dialog.file_local_csv
            self.output_folder_value = self.method_dialog.output_folder_value
            self.file_name_local = self.method_dialog.local_output_name
            self.ai_value = self.method_dialog.ai_value 
            
        elif self.insp_method == 3:
            # Variable which define the extrapolation mode
            self.extrapolation_mode = self.method_dialog.extrapolation_mode
            
            if self.extrapolation_mode == 2:
                #KNN method
                self.building_extra_path = self.method_dialog.info_pending
                self.example_building_path = self.method_dialog.info_existing
                self.extrapolation_name = self.method_dialog.extrapolation_name
                self.coord_reference = self.method_dialog.coord_reference
                self.output_path = self.method_dialog.output_path
                self.k_value = self.method_dialog.k_value
                try:
                    self.knn_dl_saved_path = self.method_dialog.knn_dl_saved_path
                    self.coord_reference_building_feature_path = self.method_dialog.coord_reference_building_feature_path
                except:
                    pass
            else:
                #Stratified method
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
            "Once there, click the **Next Building** button in the purple panel to begin the classification process."
        )
        
        """ Methods applied with button and others elements """
        
        # Create a *.csv file "Country_City_building_info.csv" with its OSM ID and its coordinates
        self.next_button.clicked.connect(self.methods.create_database)
        
        # Create a *.csv file "Country_City_building_info.csv" with its OSM ID and its coordinates
        self.next_button.clicked.connect(self.methods.load_existing_insp)
                                         
        # Counts the ID of the images
        self.next_button.clicked.connect(self.methods.count_clicks_next)
        self.previous_button.clicked.connect(self.methods.count_clicks_previous)
        
        # Sets lat and lot values
        self.next_button.clicked.connect(self.methods.get_city_name)
        self.previous_button.clicked.connect(self.methods.get_city_name)
        
        # Set eóch construction values
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
        
        # Save the values of the inspections
        self.save_data_button.clicked.connect(self.methods.save_database)
        
        # Clean the values of the previous inspections
        self.next_button.clicked.connect(self.methods.clean_database)
        self.previous_button.clicked.connect(self.methods.clean_database)
        
        # Loads and excute a deep learning model for LLRS feature prediction
        self.next_button.clicked.connect(self.methods.material_prediction)
        self.next_button.clicked.connect(self.methods.llrs_prediction)
        self.next_button.clicked.connect(self.methods.code_level_prediction)
        self.next_button.clicked.connect(self.methods.n_stories_prediction)
        self.next_button.clicked.connect(self.methods.occupancy_prediction)
        self.next_button.clicked.connect(self.methods.block_position_prediction)
        self.next_button.clicked.connect(self.methods.roof_shape_prediction)
        self.next_button.clicked.connect(self.methods.roof_material_prediction)
        
        # Search existing inspections
        self.search_img_button.clicked.connect(self.methods.search_inspection)
        self.search_img_button.clicked.connect(self.methods.get_city_name)
        self.search_img_button.clicked.connect(self.methods.fetch_three_step_views)
        self.search_img_button.clicked.connect(self.methods.object_detector_building)
        self.search_img_button.clicked.connect(self.methods.clean_database)
        
        # Extrapolation method
        self.next_button.clicked.connect(self.methods.neighbor_extrapolation)
        
        # Extrapolation method
        self.bloc_pos_help.clicked.connect(self.methods.help_block_position)
        self.roof_shape_help.clicked.connect(self.methods.help_roof_shape)
        self.roof_material_help.clicked.connect(self.methods.help_roof_material)
        
        # self.bloc_pos_help_2.clicked.connect(self.methods.help_block_position)
        # self.roof_shape_help_2.clicked.connect(self.methods.help_roof_shape)
        # self.roof_material_help_2.clicked.connect(self.methods.help_roof_material)
        
        # self.bloc_pos_help_3.clicked.connect(self.methods.help_block_position)
        # self.roof_shape_help_3.clicked.connect(self.methods.help_roof_shape)
        # self.roof_material_help_3.clicked.connect(self.methods.help_roof_material)
        
    # Method which close the GUI in the console     
    def closeEvent(self, event):
        """Handle the default close button (X) event with a confirmation dialog."""
        # Create the QMessageBox dialog
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setWindowTitle("Confirmation")
        msgBox.setText("Have you saved your results?")
        msgBox.setInformativeText("Do you want to close the application?")
        
        # Add buttons for Yes and No
        yes_button = msgBox.addButton("Yes", QtWidgets.QMessageBox.YesRole)
        msgBox.addButton("No", QtWidgets.QMessageBox.NoRole)
    
        # Execute the QMessageBox and wait for user response
        msgBox.exec()
    
        # Handle the response
        if msgBox.clickedButton() == yes_button:
            # User confirmed, close the application
            print("GUI is closing...")
            QtWidgets.QApplication.quit()  # This will terminate the GUI and allow the script to continue
            print("------------- Thank you ------------")
        else:
            # User canceled, do not close
            print("User canceled the close operation.")
            event.ignore()  # Prevent the GUI from closing
            
    # Definition of the different pluggin, button, among others (objects) of the GUI
    def setupUi(self, GUIInterface):
        
        """Get screen resolution to adapt to different screen sizes"""
        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        try:
            import ctypes
            # Reference DPI for 100% scaling
            LOGPIXELSX = 88
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
            ctypes.windll.user32.ReleaseDC(0, hdc)
            scale =  int(1.25/(dpi / 96))  # 96 DPI is 100%
        except:
            scale = 1.25
            
        # Scale the GUI based on resolution
        sf_x = screen_width / 1920 * scale
        sf_y = screen_height / 1080 * scale

        """Set up the user interface components."""
        # Configure main window properties
        GUIInterface.setObjectName("GUIInterface")
        GUIInterface.resize(int(1570*sf_x), int(715*sf_y))  # Size of the GUI (Display resolution)
        
        # Create a central widget where other widgets will be added
        self.centralwidget = QtWidgets.QWidget(GUIInterface)
        self.centralwidget.setObjectName("centralwidget")
        
        """ GUI Title elements """
        # Title of the GUI
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(int(500 * sf_x), int(0 * sf_y), int(571 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(16 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        
        """ Coordinates data elements """
        # Label of latitude 
        self.lat_label = QtWidgets.QLabel(self.centralwidget)
        self.lat_label.setGeometry(QtCore.QRect(int(600 * sf_x), int(50 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.lat_label.setFont(font)
        self.lat_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.lat_label.setObjectName("lat_label")
        
        # Latitude value
        self.lat_value = QtWidgets.QLabel(self.centralwidget)
        self.lat_value.setGeometry(QtCore.QRect(int(720 * sf_x), int(50 * sf_y), int(171 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.lat_value.setFont(font)
        self.lat_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lat_value.setFrameShape(QtWidgets.QFrame.Box)
        self.lat_value.setObjectName("lat_value")
        
        # Label of longitude
        self.lon_label = QtWidgets.QLabel(self.centralwidget)
        self.lon_label.setGeometry(QtCore.QRect(int(880 * sf_x), int(50 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.lon_label.setFont(font)
        self.lon_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.lon_label.setObjectName("lon_label")
        
        # Longitude value
        self.lon_value = QtWidgets.QLabel(self.centralwidget)
        self.lon_value.setGeometry(QtCore.QRect(int(1000 * sf_x), int(50 * sf_y), int(171 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.lon_value.setFont(font)
        self.lon_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lon_value.setFrameShape(QtWidgets.QFrame.Box)
        self.lon_value.setObjectName("lon_value")
        
        """ Country and city elements """
        # Label for the country 
        self.country_label_input = QtWidgets.QLabel(self.centralwidget)
        self.country_label_input.setGeometry(QtCore.QRect(int(260 * sf_x), int(50 * sf_y), int(81 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(11 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.country_label_input.setFont(font)
        self.country_label_input.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.country_label_input.setObjectName("country_label_input")
        
        # Country value
        self.country_value = QtWidgets.QLabel(self.centralwidget)
        self.country_value.setGeometry(QtCore.QRect(int(350 * sf_x), int(50 * sf_y), int(161 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.country_value.setFont(font)
        self.country_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.country_value.setFrameShape(QtWidgets.QFrame.Box)
        self.country_value.setObjectName("country_value")
        
        # Label for the city
        self.city_label = QtWidgets.QLabel(self.centralwidget)
        self.city_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(50 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(11 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.city_label.setFont(font)
        self.city_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.city_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.city_label.setObjectName("city_label")
        
        # City value
        self.city_value = QtWidgets.QLabel(self.centralwidget)
        self.city_value.setGeometry(QtCore.QRect(int(80 * sf_x), int(50 * sf_y), int(161 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.city_value.setFont(font)
        self.city_value.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.city_value.setFrameShape(QtWidgets.QFrame.Box)
        self.city_value.setObjectName("city_value")
        
        # Colored frame area
        self.frame_location = QtWidgets.QFrame(self.centralwidget)
        self.frame_location.setGeometry(QtCore.QRect(int(10 * sf_x), int(40 * sf_y), int(1171 * sf_x), int(51 * sf_y)))
        self.frame_location.setStyleSheet("background-color: rgb(254, 255, 174);")
        self.frame_location.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_location.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_location.setObjectName("frame_location")       
        
        """ Button for next and previous building image """
        # Button to get the next building images
        self.next_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_button.setGeometry(QtCore.QRect(int(210 * sf_x), int(650 * sf_y), int(151 * sf_x), int(41 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.next_button.setFont(font)
        self.next_button.setObjectName("next_button")
        
        # Button to get the previous building images
        self.previous_button = QtWidgets.QPushButton(self.centralwidget)
        self.previous_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(650 * sf_y), int(171 * sf_x), int(41 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.previous_button.setFont(font)
        self.previous_button.setObjectName("previous_button")
        
        # Colored area for next and previous button
        self.img_area = QtWidgets.QLabel(self.centralwidget)
        self.img_area.setGeometry(QtCore.QRect(int(10 * sf_x), int(640 * sf_y), int(371 * sf_x), int(61 * sf_y)))
        self.img_area.setStyleSheet("background-color: rgb(209, 170, 255);")
        self.img_area.setText("")
        self.img_area.setObjectName("img_area")
        self.img_area.raise_()
        
        
        """ Left Building images elements """
        # Left image color frame
        self.frame_left_img = QtWidgets.QFrame(self.centralwidget)
        self.frame_left_img.setGeometry(QtCore.QRect(int(10 * sf_x), int(100 * sf_y), int(511 * sf_x), int(361 * sf_y)))
        self.frame_left_img.setStyleSheet("background-color: rgb(102, 220, 255);")
        self.frame_left_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_left_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left_img.setObjectName("frame_left_img")
        
        # Image ID label for left image
        self.img_id_label_1 = QtWidgets.QLabel(self.frame_left_img)
        self.img_id_label_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(111 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.img_id_label_1.setFont(font)
        self.img_id_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.img_id_label_1.setObjectName("img_id_label_1")
        
        # Left image widget for best performance
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.frame_left_img)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(int(10 * sf_x), int(40 * sf_y), int(491 * sf_x), int(311 * sf_y)))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        
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
        self.img_id_value_1 = QtWidgets.QLineEdit(self.frame_left_img)
        self.img_id_value_1.setGeometry(QtCore.QRect(int(120 * sf_x), int(10 * sf_y), int(111 * sf_x), int(21 * sf_y)))
        self.img_id_value_1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.img_id_value_1.setObjectName("img_id_value_1")
        
        # Bounding box for image left frame
        self.bounding_box_1 = QtWidgets.QPushButton(self.centralwidget)
        self.bounding_box_1.setGeometry(QtCore.QRect(int(280 * sf_x), int(110 * sf_y), int(131 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.bounding_box_1.setFont(font)
        self.bounding_box_1.setObjectName("bounding_box_1")
        
        # Year label for left image
        self.year_label_1 = QtWidgets.QLabel(self.frame_left_img)
        self.year_label_1.setGeometry(QtCore.QRect(int(430 * sf_x), int(10 * sf_y), int(21 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.year_label_1.setFont(font)
        self.year_label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.year_label_1.setObjectName("year_label_1")
        self.year_label_1.setText("Y:")
        
        # Year value for left image
        self.year_value_1 = QtWidgets.QLabel(self.frame_left_img)
        self.year_value_1.setGeometry(QtCore.QRect(int(450 * sf_x), int(10 * sf_y), int(51 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(75)
        self.year_value_1.setFont(font)
        self.year_value_1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.year_value_1.setFrameShape(QtWidgets.QFrame.Box)
        self.year_value_1.setObjectName("year_value_1")
        self.year_value_1.setAlignment(QtCore.Qt.AlignCenter)
        self.year_value_1.setText("----")
        
        ################## Form for building feature ##########################
        
        # Material for left image label
        self.material_1 = QtWidgets.QLabel(self.centralwidget)
        self.material_1.setGeometry(QtCore.QRect(int(20 * sf_x), int(470 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.material_1.setFont(font)
        self.material_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.material_1.setObjectName("material_1")
        # Material Combobox elements
        self.material_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.material_cb_1.setGeometry(QtCore.QRect(int(170 * sf_x), int(470 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.material_cb_1.setFont(font)
        self.material_cb_1.setObjectName("material_cb_1")
        # Adding material options
        self.material_cb_1.addItem("Select Material")
        self.material_cb_1.addItem("Adobe", "ADO")
        self.material_cb_1.addItem("Concrete", "CR")
        self.material_cb_1.addItem("Masonry - Confined", "MCF")
        self.material_cb_1.addItem("Masonry - Reinforced", "MR")
        self.material_cb_1.addItem("Masonry - Unreinforced", "MUR")
        self.material_cb_1.addItem("Hybrid or composite (mixed) materials", "HYB")
        self.material_cb_1.addItem("Steel", "S")
        self.material_cb_1.addItem("Wood", "W")
        self.material_cb_1.addItem("Informal materials", "INF")
        self.material_cb_1.addItem("Different materials in the two directions", "MDD")
        self.material_cb_1.addItem("Different materials in height ", "MDV")
        self.material_cb_1.addItem("Other material", "MATO")
        # Set default index
        self.material_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.material_cb_1.view().setFixedWidth(int(350 * sf_x))

        
        # LLRS for left image label
        self.llrs_1 = QtWidgets.QLabel(self.centralwidget)
        self.llrs_1.setGeometry(QtCore.QRect(int(20 * sf_x), int(510 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.llrs_1.setFont(font)
        self.llrs_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.llrs_1.setObjectName("llrs_1")
        # LLRS Combobox elements
        self.llrs_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.llrs_cb_1.setGeometry(QtCore.QRect(int(170 * sf_x), int(510 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.llrs_cb_1.setFont(font)
        self.llrs_cb_1.setObjectName("llrs_cb_1")        
        # Adding LLRS options
        self.llrs_cb_1.addItem("Select LLRS")
        self.llrs_cb_1.addItem("Dual System", "LDUAL")
        self.llrs_cb_1.addItem("Infilled Frames", "LFINF")
        self.llrs_cb_1.addItem("Moment Frames", "LFM")
        self.llrs_cb_1.addItem("Walls", "LWAL")
        self.llrs_cb_1.addItem("Braced frame", "LFBR")
        self.llrs_cb_1.addItem("Post and beam", "LPB") 
        self.llrs_cb_1.addItem("Flat slab/plate or waffle slab", "LFLS") 
        self.llrs_cb_1.addItem("Different LLRS in the two directions", "LDD")
        self.llrs_cb_1.addItem("Different LLRS in height ", "LHV") 
        self.llrs_cb_1.addItem("No lateral load-resisting system", "LN") 
        self.llrs_cb_1.addItem("Other", "LO") 
        # Set default index
        self.llrs_cb_1.setCurrentIndex(0)       
        # Scale dropdown width
        self.llrs_cb_1.view().setFixedWidth(int(350 * sf_x))

        # Number of stories for left image label
        self.n_stories_1 = QtWidgets.QLabel(self.centralwidget)
        self.n_stories_1.setGeometry(QtCore.QRect(int(20 * sf_x), int(550 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.n_stories_1.setFont(font)
        self.n_stories_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.n_stories_1.setObjectName("n_stories_1")
        # Number of stories value
        self.n_stories_value_1 = QtWidgets.QComboBox(self.centralwidget)
        self.n_stories_value_1.setGeometry(QtCore.QRect(int(170 * sf_x), int(550 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
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
        self.occupancy.setGeometry(QtCore.QRect(int(20 * sf_x), int(590 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.occupancy.setFont(font)
        self.occupancy.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.occupancy.setObjectName("occupancy")
        # Occupancy Combobox elements
        self.occup_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.occup_cb_1.setGeometry(QtCore.QRect(int(170 * sf_x), int(590 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.occup_cb_1.setFont(font)
        self.occup_cb_1.setObjectName("occup_cb_1")
        # Adding Occupancy options
        self.occup_cb_1.addItem("Select Occupancy")
        self.occup_cb_1.addItem("Residential", "RES")
        self.occup_cb_1.addItem("Commercial", "COM")
        self.occup_cb_1.addItem("Mixed", "MIX")
        self.occup_cb_1.addItem("Educational", "EDU")
        self.occup_cb_1.addItem("Government", "GOV")
        self.occup_cb_1.addItem("Healthcare", "HEA")
        self.occup_cb_1.addItem("Industrial", "IND")
        self.occup_cb_1.addItem("Other", "OCO")
        # Set default index
        self.occup_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.occup_cb_1.view().setFixedWidth(int(250 * sf_x))
        
        
        # Roof Shape for left image label
        self.roof_shape_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.roof_shape_label_1.setGeometry(QtCore.QRect(int(490 * sf_x), int(470 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.roof_shape_label_1.setFont(font)
        self.roof_shape_label_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.roof_shape_label_1.setObjectName("roof_shape_label_1")
        self.roof_shape_label_1.setText("Roof Shape:")
        
        self.roof_shape_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.roof_shape_cb_1.setGeometry(QtCore.QRect(int(640 * sf_x), int(470 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.roof_shape_cb_1.setFont(font)
        self.roof_shape_cb_1.setObjectName("roof_shape_cb_1")
        self.roof_shape_cb_1.addItem("Select Roof Shape")
        self.roof_shape_cb_1.addItem("Flat", "RSH1")
        self.roof_shape_cb_1.addItem("Pitched with gable ends", "RSH2")
        self.roof_shape_cb_1.addItem("Pitched and hipped", "RSH3")
        self.roof_shape_cb_1.addItem("Pitched with dormers", "RSH4")
        self.roof_shape_cb_1.addItem("Monopitch", "RSH5")
        self.roof_shape_cb_1.addItem("Sawtooth", "RSH6")
        self.roof_shape_cb_1.addItem("Curved", "RSH7")
        self.roof_shape_cb_1.addItem("Complex regular", "RSH8")
        self.roof_shape_cb_1.addItem("Complex irregular", "RSH9")
        self.roof_shape_cb_1.addItem("Other", "RSHO")
        
        # Roof Material for left image label
        self.roof_material_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.roof_material_label_1.setGeometry(QtCore.QRect(int(490 * sf_x), int(510 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.roof_material_label_1.setFont(font)
        self.roof_material_label_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.roof_material_label_1.setObjectName("roof_material_label_1")
        self.roof_material_label_1.setText("Roof Material:")
        
        self.roof_material_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.roof_material_cb_1.setGeometry(QtCore.QRect(int(640 * sf_x), int(510 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
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
        self.age_1.setGeometry(QtCore.QRect(int(490 * sf_x), int(550 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.age_1.setFont(font)
        self.age_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.age_1.setObjectName("age_1")
        # Code level Combobox elements
        self.age_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.age_cb_1.setGeometry(QtCore.QRect(int(640 * sf_x), int(550 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
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
        self.block_position.setGeometry(QtCore.QRect(int(490 * sf_x), int(590 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.block_position.setFont(font)
        self.block_position.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.block_position.setObjectName("block_position")
        # Block position Combobox elements
        self.bck_pos_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.bck_pos_cb_1.setGeometry(QtCore.QRect(int(640 * sf_x), int(590 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.bck_pos_cb_1.setFont(font)
        self.bck_pos_cb_1.setObjectName("bck_pos_cb_1")
        # Adding Block Position options
        self.bck_pos_cb_1.addItem("Select Block Position")
        self.bck_pos_cb_1.addItem("Detached building", "BDP")
        self.bck_pos_cb_1.addItem("Adjoining building(s) one side", "BP1")
        self.bck_pos_cb_1.addItem("Adjoining building(s) two side", "BP2")
        self.bck_pos_cb_1.addItem("Adjoining building(s) three side", "BP3")
        # Set default index
        self.bck_pos_cb_1.setCurrentIndex(0)
        # Scale dropdown width
        self.bck_pos_cb_1.view().setFixedWidth(int(250 * sf_x))

        # Epoch of construction for left image label
        self.epc_const_label_1 = QtWidgets.QLabel(self.centralwidget)
        self.epc_const_label_1.setGeometry(QtCore.QRect(int(960 * sf_x), int(470 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.epc_const_label_1.setFont(font)
        self.epc_const_label_1.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.epc_const_label_1.setObjectName("epc_const_label_1")
        self.epc_const_label_1.setText("Epoch of construction:")
        
        self.epc_const_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.epc_const_cb_1.setGeometry(QtCore.QRect(int(1170 * sf_x), int(470 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.epc_const_cb_1.setFont(font)
        self.epc_const_cb_1.setObjectName("epc_const_cb_1")
        self.epc_const_cb_1.addItem("Select Epoch of Construction")
        
        
        # Image quality for left image label
        self.img_quality = QtWidgets.QLabel(self.centralwidget)
        self.img_quality.setGeometry(QtCore.QRect(int(960 * sf_x), int(510 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.img_quality.setFont(font)
        self.img_quality.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.img_quality.setObjectName("img_quality")
        
        # Image quality Combobox elements
        self.img_q_cb_1 = QtWidgets.QComboBox(self.centralwidget)
        self.img_q_cb_1.setGeometry(QtCore.QRect(int(1170 * sf_x), int(510 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
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
        self.frame_central_img.setGeometry(QtCore.QRect(int(530 * sf_x), int(100 * sf_y), int(511 * sf_x), int(361 * sf_y)))
        self.frame_central_img.setStyleSheet("background-color: rgb(255, 170, 127)")
        self.frame_central_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_central_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_central_img.setObjectName("frame_central_img")
        # Elements for display central image or error message
        self.img_id_label_2 = QtWidgets.QLabel(self.frame_central_img)
        self.img_id_label_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(131 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.img_id_label_2.setFont(font)
        self.img_id_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.img_id_label_2.setObjectName("img_id_label_2")
        # Central image widget for best performance
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_central_img)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(int(10 * sf_x), int(40 * sf_y), int(491 * sf_x), int(311 * sf_y)))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
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
        self.img_id_value_2 = QtWidgets.QLineEdit(self.frame_central_img)
        self.img_id_value_2.setGeometry(QtCore.QRect(int(130 * sf_x), int(10 * sf_y), int(111 * sf_x), int(21 * sf_y)))
        self.img_id_value_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.img_id_value_2.setObjectName("img_id_value_2")
        # Bounding box for image central frame
        self.bounding_box_2 = QtWidgets.QPushButton(self.centralwidget)
        self.bounding_box_2.setGeometry(QtCore.QRect(int(810 * sf_x), int(110 * sf_y), int(131 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.bounding_box_2.setFont(font)
        self.bounding_box_2.setObjectName("bounding_box_2")

        # Year label for central image
        self.year_label_2 = QtWidgets.QLabel(self.frame_central_img)
        self.year_label_2.setGeometry(QtCore.QRect(int(430 * sf_x), int(10 * sf_y), int(21 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.year_label_2.setFont(font)
        self.year_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.year_label_2.setObjectName("year_label_2")
        self.year_label_2.setText("Y:")
        
        # Year value for left image
        self.year_value_2 = QtWidgets.QLabel(self.frame_central_img)
        self.year_value_2.setGeometry(QtCore.QRect(int(450 * sf_x), int(10 * sf_y), int(51 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(75)
        self.year_value_2.setFont(font)
        self.year_value_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.year_value_2.setFrameShape(QtWidgets.QFrame.Box)
        self.year_value_2.setObjectName("year_value_2")
        self.year_value_2.setAlignment(QtCore.Qt.AlignCenter)
        self.year_value_2.setText("----")
        
        # ################## Form for building feature ##########################
        # # Material for central image label 
        # self.material_2 = QtWidgets.QLabel(self.centralwidget)
        # self.material_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(460 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.material_2.setFont(font)
        # self.material_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.material_2.setObjectName("material_2")
        # # Material Combobox elements
        # self.material_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.material_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(460 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.material_cb_2.setFont(font)
        # self.material_cb_2.setObjectName("material_cb_2")
        # # Adding material options
        # self.material_cb_2.addItem("Select Material")
        # self.material_cb_2.addItem("Adobe", "ADO")
        # self.material_cb_2.addItem("Concrete", "CR")
        # self.material_cb_2.addItem("Masonry - Confined", "MCF")
        # self.material_cb_2.addItem("Masonry - Reinforced", "MR")
        # self.material_cb_2.addItem("Masonry - Unreinforced", "MUR")
        # self.material_cb_2.addItem("Hybrid or composite (mixed) materials", "HYB")
        # self.material_cb_2.addItem("Steel", "S")
        # self.material_cb_2.addItem("Wood", "W")
        # self.material_cb_2.addItem("Informal materials", "INF")
        # self.material_cb_2.addItem("Different materials in the two directions", "MDD")
        # self.material_cb_2.addItem("Different materials in height ", "MDV")
        # self.material_cb_2.addItem("Other material", "MATO")
        # # Set default index
        # self.material_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.material_cb_2.view().setFixedWidth(int(350 * sf_x))

        
        # # LLRS Central image label
        # self.llrs_2 = QtWidgets.QLabel(self.centralwidget)
        # self.llrs_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(500 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.llrs_2.setFont(font)
        # self.llrs_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.llrs_2.setObjectName("llrs_2")
        # # LLRS Combobox elements
        # self.llrs_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.llrs_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(500 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.llrs_cb_2.setFont(font)
        # self.llrs_cb_2.setObjectName("llrs_cb_2")
        # # Adding LLRS options
        # self.llrs_cb_2.addItem("Select LLRS")
        # self.llrs_cb_2.addItem("Dual System", "LDUAL")
        # self.llrs_cb_2.addItem("Infilled Frames", "LFINF")
        # self.llrs_cb_2.addItem("Moment Frames", "LFM")
        # self.llrs_cb_2.addItem("Walls", "LWAL")
        # self.llrs_cb_2.addItem("Braced frame", "LFBR")
        # self.llrs_cb_2.addItem("Post and beam", "LPB") 
        # self.llrs_cb_2.addItem("Flat slab/plate or waffle slab", "LFLS") 
        # self.llrs_cb_2.addItem("Different LLRS in the two directions", "LDD")
        # self.llrs_cb_2.addItem("Different LLRS in height ", "LHV") 
        # self.llrs_cb_2.addItem("No lateral load-resisting system", "LN") 
        # self.llrs_cb_2.addItem("Other", "LO") 
        # # Set default index
        # self.llrs_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.llrs_cb_2.view().setFixedWidth(int(350 * sf_x))

        
        # # Code level for central image
        # self.age_2 = QtWidgets.QLabel(self.frame_central_img)
        # self.age_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(440 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.age_2.setFont(font)
        # self.age_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.age_2.setObjectName("age_2")
        # # Code level Combobox elements
        # self.age_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.age_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(540 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.age_cb_2.setFont(font)
        # self.age_cb_2.setObjectName("age_cb_2")
        # # Adding Code Level options
        # self.age_cb_2.addItem("Select Code Level")
        # self.age_cb_2.addItem("High-Code", "CDH")
        # self.age_cb_2.addItem("Moderate-code", "CDM")
        # self.age_cb_2.addItem("Low-Code", "CDL")
        # self.age_cb_2.addItem("No-Code", "CDN")
        # # Set default index
        # self.age_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.age_cb_2.view().setFixedWidth(int(250 * sf_x))

        # # Number of stories elements
        # self.n_stories_2 = QtWidgets.QLabel(self.centralwidget)
        # self.n_stories_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(580 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.n_stories_2.setFont(font)
        # self.n_stories_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.n_stories_2.setObjectName("n_stories_2")
        # # Number of stories value
        # self.n_stories_value_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.n_stories_value_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(580 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.n_stories_value_2.setFont(font)
        # self.n_stories_value_2.setObjectName("n_stories_value_2")
        # # Adding Number of Stories options
        # self.n_stories_value_2.addItem("Select Number of Stories")
        # self.n_stories_value_2.addItem("1", "1")
        # self.n_stories_value_2.addItem("2", "2")
        # self.n_stories_value_2.addItem("3", "3")
        # self.n_stories_value_2.addItem("4", "4")
        # self.n_stories_value_2.addItem("5", "5")
        # self.n_stories_value_2.addItem("6-7", "6.0-7.0")
        # self.n_stories_value_2.addItem("8-9", "8.0-9.0")
        # self.n_stories_value_2.addItem("10-12", "10.0-12.0")
        # self.n_stories_value_2.addItem("13+", "13+")
        # # Set default index
        # self.n_stories_value_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.n_stories_value_2.view().setFixedWidth(int(250 * sf_x))
        
        # # Occupancy type label
        # self.occupancy_2 = QtWidgets.QLabel(self.centralwidget)
        # self.occupancy_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(620 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.occupancy_2.setFont(font)
        # self.occupancy_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.occupancy_2.setObjectName("occupancy_2")
        # # Occupancy type Combobox elements
        # self.occup_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.occup_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(620 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.occup_cb_2.setFont(font)
        # self.occup_cb_2.setObjectName("occup_cb_2")
        # # Adding Occupancy options
        # self.occup_cb_2.addItem("Select Occupancy")
        # self.occup_cb_2.addItem("Residential", "RES")
        # self.occup_cb_2.addItem("Commercial", "COM")
        # self.occup_cb_2.addItem("Mixed", "MIX")
        # self.occup_cb_2.addItem("Educational", "EDU")
        # self.occup_cb_2.addItem("Government", "GOV")
        # self.occup_cb_2.addItem("Healthcare", "HEA")
        # self.occup_cb_2.addItem("Industrial", "IND")
        # self.occup_cb_2.addItem("Other", "OCO")
        # # Set default index
        # self.occup_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.occup_cb_2.view().setFixedWidth(int(250 * sf_x))

       
        # # Block position label
        # self.block_position_2 = QtWidgets.QLabel(self.centralwidget)
        # self.block_position_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(660 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.block_position_2.setFont(font)
        # self.block_position_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.block_position_2.setObjectName("block_position_2")
        # # Block Position Combobox elements
        # self.bck_pos_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.bck_pos_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(660 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.bck_pos_cb_2.setFont(font)
        # self.bck_pos_cb_2.setObjectName("bck_pos_cb_2")
        # # Adding Block Position options
        # self.bck_pos_cb_2.addItem("Select Block Position")
        # self.bck_pos_cb_2.addItem("Detached building", "BDP")
        # self.bck_pos_cb_2.addItem("Adjoining building(s) one side", "BP1")
        # self.bck_pos_cb_2.addItem("Adjoining building(s) two sides", "BP2")
        # self.bck_pos_cb_2.addItem("Adjoining building(s) three sides", "BP3")
        # # Set default index
        # self.bck_pos_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.bck_pos_cb_2.view().setFixedWidth(int(250 * sf_x))
        
        # # Epoch of construction for left image label
        # self.epc_const_label_2 = QtWidgets.QLabel(self.centralwidget)
        # self.epc_const_label_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(700 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.epc_const_label_2.setFont(font)
        # self.epc_const_label_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.epc_const_label_2.setObjectName("epc_const_label_2")
        # self.epc_const_label_2.setText("Epoch of const:")

        # self.epc_const_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.epc_const_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(700 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.epc_const_cb_2.setFont(font)
        # self.epc_const_cb_2.setObjectName("epc_const_cb_2")
        # self.epc_const_cb_2.addItem("Select Epoch of construction")
        
        # # Roof shape for central image
        # self.roof_shape_label_2 = QtWidgets.QLabel(self.centralwidget)
        # self.roof_shape_label_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(740 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.roof_shape_label_2.setFont(font)
        # self.roof_shape_label_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.roof_shape_label_2.setObjectName("roof_shape_label_2")
        # self.roof_shape_label_2.setText("Roof Shape:")
        
        # self.roof_shape_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.roof_shape_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(740 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.roof_shape_cb_2.setFont(font)
        # self.roof_shape_cb_2.setObjectName("roof_shape_cb_2")
        # self.roof_shape_cb_2.addItem("Select Roof Shape")
        # self.roof_shape_cb_2.addItem("Flat", "RSH1")
        # self.roof_shape_cb_2.addItem("Pitched with gable ends", "RSH2")
        # self.roof_shape_cb_2.addItem("Pitched and hipped", "RSH3")
        # self.roof_shape_cb_2.addItem("Pitched with dormers", "RSH4")
        # self.roof_shape_cb_2.addItem("Monopitch", "RSH5")
        # self.roof_shape_cb_2.addItem("Sawtooth", "RSH6")
        # self.roof_shape_cb_2.addItem("Curved", "RSH7")
        # self.roof_shape_cb_2.addItem("Complex regular", "RSH8")
        # self.roof_shape_cb_2.addItem("Complex irregular", "RSH9")
        # self.roof_shape_cb_2.addItem("Other", "RSHO")
        # # Set default index
        # self.roof_shape_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.roof_shape_cb_2.view().setFixedWidth(int(250 * sf_x))
        
        # # Roof Material for central image label
        # self.roof_material_label_2 = QtWidgets.QLabel(self.centralwidget)
        # self.roof_material_label_2.setGeometry(QtCore.QRect(int(540 * sf_x), int(780 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.roof_material_label_2.setFont(font)
        # self.roof_material_label_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.roof_material_label_2.setObjectName("roof_material_label_2")
        # self.roof_material_label_2.setText("Roof Material:")
        
        # self.roof_material_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.roof_material_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(780 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.roof_material_cb_2.setFont(font)
        # self.roof_material_cb_2.setObjectName("roof_material_cb_2")
        # self.roof_material_cb_2.addItem("Select Roof Material")
        # self.roof_material_cb_2.addItem("Concrete", "RMN")
        # self.roof_material_cb_2.addItem("Clay or concrete tile", "RMT1")
        # self.roof_material_cb_2.addItem("Metal or asbestos sheets", "RMT6")
        # self.roof_material_cb_2.addItem("Wooden and asphalt shingles", "RMT7")
        # self.roof_material_cb_2.addItem("Slate", "RMT4")
        # self.roof_material_cb_2.addItem("Solar panelled roofs", "RMT10")
        # self.roof_material_cb_2.addItem("Other", "RMTO")
        # # Set default index
        # self.roof_material_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.roof_material_cb_2.view().setFixedWidth(int(250 * sf_x))
        
        # self.img_quality_2 = QtWidgets.QLabel(self.frame_central_img)
        # self.img_quality_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(720 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.img_quality_2.setFont(font)
        # self.img_quality_2.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.img_quality_2.setObjectName("img_quality_2")
        
        # # Image quality Combobox elements
        # self.img_q_cb_2 = QtWidgets.QComboBox(self.centralwidget)
        # self.img_q_cb_2.setGeometry(QtCore.QRect(int(690 * sf_x), int(820 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.img_q_cb_2.setFont(font)
        # self.img_q_cb_2.setObjectName("img_q_cb_2")
        # # Adding Image Quality options
        # self.img_q_cb_2.addItem("Select Image quality")
        # self.img_q_cb_2.addItem("Excellent", "Excellent")
        # self.img_q_cb_2.addItem("Good", "Good")
        # self.img_q_cb_2.addItem("Intermediate", "Intermediate")
        # self.img_q_cb_2.addItem("Bad", "Bad")
        # # Set default index
        # self.img_q_cb_2.setCurrentIndex(0)
        # # Scale dropdown width
        # self.img_q_cb_2.view().setFixedWidth(int(250 * sf_x))
        
        """ Right Building images elements """
        # Right image color frame
        self.frame_right_img = QtWidgets.QFrame(self.centralwidget)
        self.frame_right_img.setGeometry(QtCore.QRect(int(1050 * sf_x), int(100 * sf_y), int(511 * sf_x), int(361 * sf_y)))
        self.frame_right_img.setStyleSheet("background-color: rgb(183, 252, 172)")
        self.frame_right_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_right_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_right_img.setObjectName("frame_right_img")
        # Elements for display right image or error message
        self.img_id_label_3 = QtWidgets.QLabel(self.frame_right_img)
        self.img_id_label_3.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(131 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.img_id_label_3.setFont(font)
        self.img_id_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.img_id_label_3.setObjectName("img_id_label_3")
        # Right image widget for best performance
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_right_img)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(39 * sf_y), int(491 * sf_x), int(311 * sf_y)))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
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
        self.img_id_value_3 = QtWidgets.QLineEdit(self.frame_right_img)
        self.img_id_value_3.setGeometry(QtCore.QRect(int(130 * sf_x), int(10 * sf_y), int(111 * sf_x), int(21 * sf_y)))
        self.img_id_value_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.img_id_value_3.setObjectName("img_id_value_3")
        # Bounding box for image right frame
        self.bounding_box_3 = QtWidgets.QPushButton(self.centralwidget)
        self.bounding_box_3.setGeometry(QtCore.QRect(int(1320 * sf_x), int(110 * sf_y), int(131 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.bounding_box_3.setFont(font)
        self.bounding_box_3.setObjectName("bounding_box_3")

        # Year label for central image
        self.year_label_3 = QtWidgets.QLabel(self.frame_right_img)
        self.year_label_3.setGeometry(QtCore.QRect(int(430 * sf_x), int(10 * sf_y), int(21 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.year_label_3.setFont(font)
        self.year_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.year_label_3.setObjectName("year_label_3")
        self.year_label_3.setText("Y:")
        
        # Year value for left image
        self.year_value_3 = QtWidgets.QLabel(self.frame_right_img)
        self.year_value_3.setGeometry(QtCore.QRect(int(450 * sf_x), int(10 * sf_y), int(51 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(False)
        font.setWeight(75)
        self.year_value_3.setFont(font)
        self.year_value_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.year_value_3.setFrameShape(QtWidgets.QFrame.Box)
        self.year_value_3.setObjectName("year_value_3")
        self.year_value_3.setAlignment(QtCore.Qt.AlignCenter)
        self.year_value_3.setText("----")
        
        # ################## Form for building feature ##########################
        
        # # Material for right image label
        # self.material_3 = QtWidgets.QLabel(self.centralwidget)
        # self.material_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(460 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.material_3.setFont(font)
        # self.material_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.material_3.setObjectName("material_3")
        # # Material right combobox elements
        # self.material_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.material_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(460 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.material_cb_3.setFont(font)
        # self.material_cb_3.setObjectName("material_cb_3")
        # # Adding material options
        # self.material_cb_3.addItem("Select Material")
        # self.material_cb_3.addItem("Adobe", "ADO")
        # self.material_cb_3.addItem("Concrete", "CR")
        # self.material_cb_3.addItem("Masonry - Confined", "MCF")
        # self.material_cb_3.addItem("Masonry - Reinforced", "MR")
        # self.material_cb_3.addItem("Masonry - Unreinforced", "MUR")
        # self.material_cb_3.addItem("Hybrid or composite (mixed) materials", "HYB")
        # self.material_cb_3.addItem("Steel", "S")
        # self.material_cb_3.addItem("Wood", "W")
        # self.material_cb_3.addItem("Informal materials", "INF")
        # self.material_cb_3.addItem("Different materials in the two directions", "MDD")
        # self.material_cb_3.addItem("Different materials in height ", "MDV")
        # self.material_cb_3.addItem("Other material", "MATO")
        # # Set default index
        # self.material_cb_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.material_cb_3.view().setFixedWidth(int(350 * sf_x))

        
        # # LLRS for right image label
        # self.llrs_3 = QtWidgets.QLabel(self.centralwidget)
        # self.llrs_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(500 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.llrs_3.setFont(font)
        # self.llrs_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.llrs_3.setObjectName("llrs_3")
        # # LLRS right combobox elements
        # self.llrs_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.llrs_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(500 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.llrs_cb_3.setFont(font)
        # self.llrs_cb_3.setObjectName("llrs_cb_3")
        # # Adding LLRS options
        # self.llrs_cb_3.addItem("Select LLRS")
        # self.llrs_cb_3.addItem("Dual System", "LDUAL")
        # self.llrs_cb_3.addItem("Infilled Frames", "LFINF")
        # self.llrs_cb_3.addItem("Moment Frames", "LFM")
        # self.llrs_cb_3.addItem("Walls", "LWAL")
        # self.llrs_cb_3.addItem("Braced frame", "LFBR")
        # self.llrs_cb_3.addItem("Post and beam", "LPB") 
        # self.llrs_cb_3.addItem("Flat slab/plate or waffle slab", "LFLS") 
        # self.llrs_cb_3.addItem("Different LLRS in the two directions", "LDD")
        # self.llrs_cb_3.addItem("Different LLRS in height ", "LHV") 
        # self.llrs_cb_3.addItem("No lateral load-resisting system", "LN") 
        # self.llrs_cb_3.addItem("Other", "LO")    
        # # Set default index
        # self.llrs_cb_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.llrs_cb_3.view().setFixedWidth(int(350 * sf_x))
        
        # # Code level for right image label
        # self.age_3 = QtWidgets.QLabel(self.centralwidget)
        # self.age_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(540 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.age_3.setFont(font)
        # self.age_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.age_3.setObjectName("age_3")
        # # Code level right combobox elements
        # self.age_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.age_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(540 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.age_cb_3.setFont(font)
        # self.age_cb_3.setObjectName("age_cb_3")
        # # Adding Code Level options
        # self.age_cb_3.addItem("Select Code Level")
        # self.age_cb_3.addItem("High-Code", "CDH")
        # self.age_cb_3.addItem("Moderate-code", "CDM")
        # self.age_cb_3.addItem("Low-Code", "CDL")
        # self.age_cb_3.addItem("No-Code", "CDN")
        # # Set default index
        # self.age_cb_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.age_cb_3.view().setFixedWidth(int(250 * sf_x))

        
        # # Number of stories for right image label
        # self.n_stories_3 = QtWidgets.QLabel(self.centralwidget)
        # self.n_stories_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(580 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.n_stories_3.setFont(font)
        # self.n_stories_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.n_stories_3.setObjectName("n_stories_3")
        # # Number of stories value
        # self.n_stories_value_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.n_stories_value_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(580 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.n_stories_value_3.setFont(font)
        # self.n_stories_value_3.setObjectName("n_stories_value_3")
        # # Adding Number of Stories options
        # self.n_stories_value_3.addItem("Select Number of Stories")
        # self.n_stories_value_3.addItem("1", "1")
        # self.n_stories_value_3.addItem("2", "2")
        # self.n_stories_value_3.addItem("3", "3")
        # self.n_stories_value_3.addItem("4", "4")
        # self.n_stories_value_3.addItem("5", "5")
        # self.n_stories_value_3.addItem("6-7", "6.0-7.0")
        # self.n_stories_value_3.addItem("8-9", "8.0-9.0")
        # self.n_stories_value_3.addItem("10-12", "10.0-12.0")
        # self.n_stories_value_3.addItem("13+", "13+")
        # # Set default index
        # self.n_stories_value_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.n_stories_value_3.view().setFixedWidth(int(250 * sf_x))

        
        # # Occupancy type for right image label
        # self.occupancy_3 = QtWidgets.QLabel(self.centralwidget)
        # self.occupancy_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(620 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.occupancy_3.setFont(font)
        # self.occupancy_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.occupancy_3.setObjectName("occupancy_3")
        # # Occupancy type right combobox elements
        # self.occup_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.occup_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(620 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.occup_cb_3.setFont(font)
        # self.occup_cb_3.setObjectName("occup_cb_3")
        # # Adding Occupancy options
        # self.occup_cb_3.addItem("Select Occupancy")
        # self.occup_cb_3.addItem("Residential", "RES")
        # self.occup_cb_3.addItem("Commercial", "COM")
        # self.occup_cb_3.addItem("Mixed", "MIX")
        # self.occup_cb_3.addItem("Educational", "EDU")
        # self.occup_cb_3.addItem("Government", "GOV")
        # self.occup_cb_3.addItem("Healthcare", "HEA")
        # self.occup_cb_3.addItem("Industrial", "IND")
        # self.occup_cb_3.addItem("Other", "OCO")
        # # Set default index
        # self.occup_cb_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.occup_cb_3.view().setFixedWidth(int(250 * sf_x))

                     
        # # Block position for right image label
        # self.block_position_3 = QtWidgets.QLabel(self.centralwidget)
        # self.block_position_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(660 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.block_position_3.setFont(font)
        # self.block_position_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.block_position_3.setObjectName("block_position_3")
        # # Block position right combobox elements
        # self.bck_pos_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.bck_pos_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(660 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.bck_pos_cb_3.setFont(font)
        # self.bck_pos_cb_3.setObjectName("bck_pos_cb_3")
        # # Adding Block Position options
        # self.bck_pos_cb_3.addItem("Select Block Position")
        # self.bck_pos_cb_3.addItem("Detached building", "BDP")
        # self.bck_pos_cb_3.addItem("Adjoining building(s) one side", "BP1")
        # self.bck_pos_cb_3.addItem("Adjoining building(s) two sides", "BP2")
        # self.bck_pos_cb_3.addItem("Adjoining building(s) three sides", "BP3")
        # # Set default index
        # self.bck_pos_cb_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.bck_pos_cb_3.view().setFixedWidth(int(250 * sf_x))
        
        # # Epoch of construction for right image label
        # self.epc_const_label_3 = QtWidgets.QLabel(self.centralwidget)
        # self.epc_const_label_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(700 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.epc_const_label_3.setFont(font)
        # self.epc_const_label_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.epc_const_label_3.setObjectName("epc_const_label_3")
        # self.epc_const_label_3.setText("Epoch of const:")
        
        # self.epc_const_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.epc_const_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(700 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.epc_const_cb_3.setFont(font)
        # self.epc_const_cb_3.setObjectName("epc_const_cb_3")
        # self.epc_const_cb_3.addItem("Select Epoch of construction")
        
        # # Roof shape for right image
        # self.roof_shape_label_3 = QtWidgets.QLabel(self.centralwidget)
        # self.roof_shape_label_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(740 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.roof_shape_label_3.setFont(font)
        # self.roof_shape_label_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.roof_shape_label_3.setObjectName("roof_shape_label_3")
        # self.roof_shape_label_3.setText("Roof Shape:")
        
        # self.roof_shape_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.roof_shape_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(740 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.roof_shape_cb_3.setFont(font)
        # self.roof_shape_cb_3.setObjectName("roof_shape_cb_3")
        # self.roof_shape_cb_3.addItem("Select Roof Shape")
        # self.roof_shape_cb_3.addItem("Flat", "RSH1")
        # self.roof_shape_cb_3.addItem("Pitched with gable ends", "RSH2")
        # self.roof_shape_cb_3.addItem("Pitched and hipped", "RSH3")
        # self.roof_shape_cb_3.addItem("Pitched with dormers", "RSH4")
        # self.roof_shape_cb_3.addItem("Monopitch", "RSH5")
        # self.roof_shape_cb_3.addItem("Sawtooth", "RSH6")
        # self.roof_shape_cb_3.addItem("Curved", "RSH7")
        # self.roof_shape_cb_3.addItem("Complex regular", "RSH8")
        # self.roof_shape_cb_3.addItem("Complex irregular", "RSH9")
        # self.roof_shape_cb_3.addItem("Other", "RSHO")
        
        # # Roof Material for left image label
        # self.roof_material_label_3 = QtWidgets.QLabel(self.centralwidget)
        # self.roof_material_label_3.setGeometry(QtCore.QRect(int(1060 * sf_x), int(780 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.roof_material_label_3.setFont(font)
        # self.roof_material_label_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.roof_material_label_3.setObjectName("roof_material_label_3")
        # self.roof_material_label_3.setText("Roof Material:")
        
        # self.roof_material_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.roof_material_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(780 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.roof_material_cb_3.setFont(font)
        # self.roof_material_cb_3.setObjectName("roof_material_cb_3")
        # self.roof_material_cb_3.addItem("Select Roof Material")
        # self.roof_material_cb_3.addItem("Concrete", "RMN")
        # self.roof_material_cb_3.addItem("Clay or concrete tile", "RMT1")
        # self.roof_material_cb_3.addItem("Metal or asbestos sheets", "RMT6")
        # self.roof_material_cb_3.addItem("Wooden and asphalt shingles", "RMT7")
        # self.roof_material_cb_3.addItem("Slate", "RMT4")
        # self.roof_material_cb_3.addItem("Solar panelled roofs", "RMT10")
        # self.roof_material_cb_3.addItem("Other", "RMTO")
        # # Set default index
        # self.roof_material_cb_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.roof_material_cb_3.view().setFixedWidth(int(250 * sf_x))
        
        # # Image quality for right image label
        # self.img_quality_3 = QtWidgets.QLabel(self.frame_right_img)
        # self.img_quality_3.setGeometry(QtCore.QRect(int(10 * sf_x), int(720 * sf_y), int(131 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # font.setBold(True)
        # font.setWeight(75)
        # self.img_quality_3.setFont(font)
        # self.img_quality_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.img_quality_3.setObjectName("img_quality_3")
        # # Image quality right combobox elements
        # self.img_q_cb_3 = QtWidgets.QComboBox(self.centralwidget)
        # self.img_q_cb_3.setGeometry(QtCore.QRect(int(1210 * sf_x), int(820 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        # font = QtGui.QFont()
        # font.setPointSize(int(10 * sf_x))
        # self.img_q_cb_3.setFont(font)
        # self.img_q_cb_3.setObjectName("img_q_cb_3")
        # # Adding Image Quality options
        # self.img_q_cb_3.addItem("Select Image quality")
        # self.img_q_cb_3.addItem("Excellent", "Excellent")
        # self.img_q_cb_3.addItem("Good", "Good")
        # self.img_q_cb_3.addItem("Intermediate", "Intermediate")
        # self.img_q_cb_3.addItem("Bad", "Bad")
        # # Set default index
        # self.img_q_cb_3.setCurrentIndex(0)
        # # Scale dropdown width
        # self.img_q_cb_3.view().setFixedWidth(int(250 * sf_x))


        
###########################################################################################        
###########################################################################################   
        
        """ Search button elements """
        self.search_img_button = QtWidgets.QPushButton(self.centralwidget)
        self.search_img_button.setGeometry(QtCore.QRect(int(1400 * sf_x), int(640 * sf_y), int(151 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.search_img_button.setFont(font)
        self.search_img_button.setObjectName("search_img_button")
        
        self.search_img_value = QtWidgets.QLineEdit(self.centralwidget)
        self.search_img_value.setPlaceholderText("Enter image ID to search")
        self.search_img_value.setGeometry(QtCore.QRect(int(1250 * sf_x), int(640 * sf_y), int(141 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.search_img_value.setFont(font)
        self.search_img_value.setObjectName("search_img_value")
        
        self.search_img_label = QtWidgets.QLabel(self.centralwidget)
        self.search_img_label.setGeometry(QtCore.QRect(int(1140 * sf_x), int(640 * sf_y), int(91 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.search_img_label.setFont(font)
        self.search_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.search_img_label.setObjectName("search_img_label")
        
        """ GEM Logo elements """
        self.GEM_logo = QtWidgets.QLabel(self.centralwidget)
        self.GEM_logo.setGeometry(QtCore.QRect(int(1380 * sf_x), int(5 * sf_y), int(177 * sf_x), int(65 * sf_y)))
        self.GEM_logo.setText("")
        self.GEM_logo.setPixmap(QtGui.QPixmap("help_img/GEM_Logo.png"))
        self.GEM_logo.setScaledContents(True)
        self.GEM_logo.setObjectName("GEM_logo")
        
        self.RUBIC_logo = QtWidgets.QLabel(self.centralwidget)
        self.RUBIC_logo.setGeometry(QtCore.QRect(int(1323 * sf_x), int(5 * sf_y), int(60 * sf_x), int(65 * sf_y)))
        self.RUBIC_logo.setText("")
        self.RUBIC_logo.setPixmap(QtGui.QPixmap("help_img/RUBIC_logo.png"))
        self.RUBIC_logo.setScaledContents(True)
        self.RUBIC_logo.setObjectName("GEM_logo")
        
        """ GEM icon GUI elements """
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        
        """ GEM icon GUI elements """
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        
        """ Progress Bar elements """
        # Progress bar widget
        self.progress_bar_method = QtWidgets.QProgressBar(self.centralwidget)
        self.progress_bar_method.setGeometry(QtCore.QRect(int(620 * sf_x), int(660 * sf_y), int(161 * sf_x), int(23 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.progress_bar_method.setFont(font)
        self.progress_bar_method.setProperty("value", 0)
        self.progress_bar_method.setObjectName("progress_bar_method")
        
        # Progress bar label value
        self.method_progress = QtWidgets.QLabel(self.centralwidget)
        self.method_progress.setGeometry(QtCore.QRect(int(800 * sf_x), int(650 * sf_y), int(291 * sf_x), int(41 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.method_progress.setFont(font)
        self.method_progress.setObjectName("method_progress")
        
        """ AI Powered activation elements """
        # AI checkbox activation
        self.ai_check = QtWidgets.QCheckBox(self.centralwidget)
        self.ai_check.setGeometry(QtCore.QRect(int(400 * sf_x), int(660 * sf_y), int(131 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.ai_check.setFont(font)
        self.ai_check.setObjectName("ai_check")
        
        """ Save data button elements """ 
        # Save data button
        self.save_data_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_data_button.setGeometry(QtCore.QRect(int(1130 * sf_x), int(670 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.save_data_button.setFont(font)
        self.save_data_button.setObjectName("save_data_button")
        
        # Left image frame buttons
        icon_size = QtCore.QSize(int(31 * sf_x), int(31 * sf_x))  # Icon is square
        self.bloc_pos_help = QtWidgets.QPushButton(self.frame_left_img)
        pixmap = QtGui.QPixmap("help_img/help_icon.png").scaled(icon_size)
        icon = QtGui.QIcon(pixmap)
        self.bloc_pos_help.setIcon(icon)
        self.bloc_pos_help.setIconSize(icon_size)
        self.bloc_pos_help.setGeometry(QtCore.QRect(int(420 * sf_x), int(560 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        self.bloc_pos_help.setObjectName("bloc_pos_help")
        
        self.roof_shape_help = QtWidgets.QPushButton(self.frame_left_img)
        self.roof_shape_help.setIcon(icon)
        self.roof_shape_help.setIconSize(icon_size)
        self.roof_shape_help.setGeometry(QtCore.QRect(int(420 * sf_x), int(640 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        self.roof_shape_help.setObjectName("roof_shape_help")
        
        self.roof_material_help = QtWidgets.QPushButton(self.frame_left_img)
        self.roof_material_help.setIcon(icon)
        self.roof_material_help.setIconSize(icon_size)
        self.roof_material_help.setGeometry(QtCore.QRect(int(420 * sf_x), int(680 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        self.roof_material_help.setObjectName("roof_material_help")
        
        # # Central image frame buttons
        # self.bloc_pos_help_2 = QtWidgets.QPushButton(self.frame_central_img)
        # self.bloc_pos_help_2.setIcon(icon)
        # self.bloc_pos_help_2.setIconSize(icon_size)
        # self.bloc_pos_help_2.setGeometry(QtCore.QRect(int(420 * sf_x), int(560 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        # self.bloc_pos_help_2.setObjectName("bloc_pos_help_2")
        
        # self.roof_shape_help_2 = QtWidgets.QPushButton(self.frame_central_img)
        # self.roof_shape_help_2.setIcon(icon)
        # self.roof_shape_help_2.setIconSize(icon_size)
        # self.roof_shape_help_2.setGeometry(QtCore.QRect(int(420 * sf_x), int(640 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        # self.roof_shape_help_2.setObjectName("roof_shape_help_2")
        
        # self.roof_material_help_2 = QtWidgets.QPushButton(self.frame_central_img)
        # self.roof_material_help_2.setIcon(icon)
        # self.roof_material_help_2.setIconSize(icon_size)
        # self.roof_material_help_2.setGeometry(QtCore.QRect(int(420 * sf_x), int(680 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        # self.roof_material_help_2.setObjectName("roof_material_help_2")
        
        # # Right image frame buttons
        # self.bloc_pos_help_3 = QtWidgets.QPushButton(self.frame_right_img)
        # self.bloc_pos_help_3.setIcon(icon)
        # self.bloc_pos_help_3.setIconSize(icon_size)
        # self.bloc_pos_help_3.setGeometry(QtCore.QRect(int(420 * sf_x), int(560 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        # self.bloc_pos_help_3.setObjectName("bloc_pos_help_3")
        
        # self.roof_shape_help_3 = QtWidgets.QPushButton(self.frame_right_img)
        # self.roof_shape_help_3.setIcon(icon)
        # self.roof_shape_help_3.setIconSize(icon_size)
        # self.roof_shape_help_3.setGeometry(QtCore.QRect(int(420 * sf_x), int(640 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        # self.roof_shape_help_3.setObjectName("roof_shape_help_3")
        
        # self.roof_material_help_3 = QtWidgets.QPushButton(self.frame_right_img)
        # self.roof_material_help_3.setIcon(icon)
        # self.roof_material_help_3.setIconSize(icon_size)
        # self.roof_material_help_3.setGeometry(QtCore.QRect(int(420 * sf_x), int(680 * sf_y), int(31 * sf_x), int(31 * sf_y)))
        # self.roof_material_help_3.setObjectName("roof_material_help_3")

        
        """ Raise all elements """
        self.frame_location.raise_()
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
        # self.material_cb_2.raise_()
        # self.llrs_cb_2.raise_()
        # self.img_q_cb_2.raise_()
        # self.age_cb_2.raise_()
        # self.bck_pos_cb_2.raise_()
        # self.n_stories_2.raise_()
        # self.occup_cb_2.raise_()
        # self.block_position_2.raise_()
        # self.material_2.raise_()
        # self.llrs_2.raise_()
        # self.occupancy_2.raise_()
        # self.material_cb_3.raise_()
        # self.llrs_cb_3.raise_()
        # self.img_q_cb_3.raise_()
        # self.age_cb_3.raise_()
        # self.bck_pos_cb_3.raise_()
        # self.n_stories_3.raise_()
        # self.occup_cb_3.raise_()
        # self.block_position_3.raise_()
        # self.material_3.raise_()
        # self.llrs_3.raise_()
        # self.age_3.raise_()
        # self.occupancy_3.raise_()
        self.progress_bar_method.raise_()
        self.method_progress.raise_()
        self.ai_check.raise_()
        self.save_data_button.raise_()
        self.n_stories_value_1.raise_()
        # self.n_stories_value_2.raise_()
        # self.n_stories_value_3.raise_()
        self.bounding_box_1.raise_()
        self.bounding_box_2.raise_()
        self.bounding_box_3.raise_()
        self.roof_shape_cb_1.raise_()
        self.epc_const_cb_1.raise_()
        self.roof_shape_cb_1.raise_()
        self.epc_const_label_1.raise_()
        self.roof_shape_label_1.raise_()
        # self.epc_const_cb_2.raise_()
        # self.roof_shape_cb_2.raise_()
        # self.epc_const_label_2.raise_()
        # self.roof_shape_label_2.raise_()
        # self.epc_const_label_3.raise_()
        # self.roof_shape_label_3.raise_()
        # self.epc_const_cb_3.raise_()
        # self.roof_shape_cb_3.raise_()
        self.roof_material_label_1.raise_()
        self.roof_material_cb_1.raise_()
        # self.roof_material_label_2.raise_()
        # self.roof_material_cb_2.raise_()
        # self.roof_material_label_3.raise_()
        # self.roof_material_cb_3.raise_()      
        self.bloc_pos_help.raise_()
        self.roof_shape_help.raise_()
        self.roof_material_help.raise_()  
        # self.bloc_pos_help_2.raise_()
        # self.roof_shape_help_2.raise_()
        # self.roof_material_help_2.raise_() 
        # self.bloc_pos_help_3.raise_()
        # self.roof_shape_help_3.raise_()
        # self.roof_material_help_3.raise_()
        
        GUIInterface.setCentralWidget(self.centralwidget)

        self.retranslateUi(GUIInterface)
        QtCore.QMetaObject.connectSlotsByName(GUIInterface)
        
    
        """ Method for translation """
    def retranslateUi(self, GUIInterface):
        _translate = QtCore.QCoreApplication.translate
        GUIInterface.setWindowTitle(_translate("GUIInterface", "RUBIC-AI: Building Inventory Classifier"))
        self.country_label_input.setText(_translate("GUIInterface", "Country:"))
        self.city_label.setText(_translate("GUIInterface", "City:"))
        self.title.setText(_translate("GUIInterface", "RUBIC-AI: Building Inventory Classifier"))
        self.lat_label.setText(_translate("GUIInterface", "Latitude: "))
        self.lon_label.setText(_translate("GUIInterface", "Longitude:"))
        self.previous_button.setText(_translate("GUIInterface", "Previous Building"))
        self.lat_value.setText(_translate("GUIInterface", "-"))
        self.lon_value.setText(_translate("GUIInterface", "-"))
        self.country_value.setText(_translate("GUIInterface", "-"))
        self.city_value.setText(_translate("GUIInterface", "-"))
        self.img_id_label_2.setText(_translate("GUIInterface", "Image ID:"))
        self.img_id_value_2.setText(_translate("GUIInterface", "-"))
        # self.img_quality_2.setText(_translate("GUIInterface", "Image Quality:"))
        # self.age_2.setText(_translate("GUIInterface", "Code Level:"))
        self.img_id_label_3.setText(_translate("GUIInterface", "Image ID:"))
        self.img_id_value_3.setText(_translate("GUIInterface", "-"))
        # self.img_quality_3.setText(_translate("GUIInterface", "Image Quality:"))
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
        self.occup_cb_1.setItemText(0, _translate("GUIInterface", "Select Occupancy Type"))
        self.bck_pos_cb_1.setItemText(0, _translate("GUIInterface", "Select Block Position"))
        self.img_q_cb_1.setItemText(0, _translate("GUIInterface", "Select Image Quality"))
        self.search_img_button.setText(_translate("GUIInterface", "Search Building"))
        self.search_img_label.setText(_translate("GUIInterface", "Image ID:"))
        # self.material_cb_2.setItemText(0, _translate("GUIInterface", "Select Material"))
        # self.llrs_cb_2.setItemText(0, _translate("GUIInterface", "Select LLRS"))
        # self.img_q_cb_2.setItemText(0, _translate("GUIInterface", "Select Image Quality"))
        # self.age_cb_2.setItemText(0, _translate("GUIInterface", "Select Code Level"))
        # self.bck_pos_cb_2.setItemText(0, _translate("GUIInterface", "Select Block Position"))
        # self.n_stories_2.setText(_translate("GUIInterface", "N° of Stories:"))
        # self.occup_cb_2.setItemText(0, _translate("GUIInterface", "Select Occupancy Type"))
        # self.block_position_2.setText(_translate("GUIInterface", "Block Position:"))
        # self.material_2.setText(_translate("GUIInterface", "LLRS Material:"))
        # self.llrs_2.setText(_translate("GUIInterface", "LLRS:"))
        # self.occupancy_2.setText(_translate("GUIInterface", "Occupancy:"))
        # self.material_cb_3.setItemText(0, _translate("GUIInterface", "Select Material"))
        # self.llrs_cb_3.setItemText(0, _translate("GUIInterface", "Select LLRS"))
        # self.img_q_cb_3.setItemText(0, _translate("GUIInterface", "Select Image Quality"))
        # self.age_cb_3.setItemText(0, _translate("GUIInterface", "Select Code Level"))
        # self.bck_pos_cb_3.setItemText(0, _translate("GUIInterface", "Select Block Position"))
        # self.n_stories_3.setText(_translate("GUIInterface", "N° of Stories:"))
        # self.occup_cb_3.setItemText(0, _translate("GUIInterface", "Select Occupancy Type"))
        # self.block_position_3.setText(_translate("GUIInterface", "Block Position:"))
        # self.material_3.setText(_translate("GUIInterface", "LLRS Material:"))
        # self.llrs_3.setText(_translate("GUIInterface", "LLRS:"))
        # self.age_3.setText(_translate("GUIInterface", "Code Level:"))
        # self.occupancy_3.setText(_translate("GUIInterface", "Occupancy:"))
        self.method_progress.setText(_translate("GUIInterface", "-"))
        self.ai_check.setText(_translate("GUIInterface", "AI Powered"))
        self.save_data_button.setText(_translate("GUIInterface", "Save data"))
        self.bounding_box_1.setText(_translate("search_img_value", "Manual box"))
        self.bounding_box_2.setText(_translate("search_img_value", "Manual box"))
        self.bounding_box_3.setText(_translate("search_img_value", "Manual box"))
        
        

        
