from PyQt5 import QtCore, QtGui, QtWidgets
from methods.neighbor_building_extrapolation_feature import find_nearest_neighbors_geodesic
import pandas as pd

class extrapolation_options_window(QtWidgets.QDialog):  # Inherit from QDialog
    def __init__(self, parent=None, n_neighbors=None):
        super().__init__(parent)
        self.n_neighbors = n_neighbors  # Data passed from main window
        self.neighbor_data = None
        self.setupUi(self)  # Call setupUi and pass self (QDialog instance)


    def setupUi(self, PolygonSetting):  # PolygonSetting is now self
    
        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080

        
        PolygonSetting.setObjectName("PolygonSetting")
        PolygonSetting.resize(int(400*sf_x), int(370*sf_y))
        self.coord_frame = QtWidgets.QWidget(PolygonSetting)
        self.coord_frame.setObjectName("coord_frame")
        
        self.op_1_check = QtWidgets.QCheckBox(self.coord_frame)
        self.op_1_check.setGeometry(QtCore.QRect(int(20 * sf_x), int(20 * sf_y), int(241 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.op_1_check.setFont(font)
        self.op_1_check.setObjectName("op_1_check")
        
        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(90 * sf_x), int(310 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self.select_method)
        
        self.building_value_op1 = QtWidgets.QLabel(self.coord_frame)
        self.building_value_op1.setGeometry(QtCore.QRect(int(240 * sf_x), int(50 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.building_value_op1.setFont(font)
        self.building_value_op1.setObjectName("building_value_op1")
        
        self.selected_value_op1 = QtWidgets.QLineEdit(self.coord_frame)
        self.selected_value_op1.setGeometry(QtCore.QRect(int(240 * sf_x), int(90 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.selected_value_op1.setFont(font)
        self.selected_value_op1.setObjectName("selected_value_op1")
        
        self.n_building_selected = QtWidgets.QLabel(self.coord_frame)
        self.n_building_selected.setGeometry(QtCore.QRect(int(20 * sf_x), int(90 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.n_building_selected.setFont(font)
        self.n_building_selected.setObjectName("n_building_selected")
        
        self.n_building_op1_label = QtWidgets.QLabel(self.coord_frame)
        self.n_building_op1_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(50 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.n_building_op1_label.setFont(font)
        self.n_building_op1_label.setObjectName("n_building_op1_label")
        
        
        self.op2_check = QtWidgets.QCheckBox(self.coord_frame)
        self.op2_check.setGeometry(QtCore.QRect(int(20 * sf_x), int(160 * sf_y), int(281 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.op2_check.setFont(font)
        self.op2_check.setObjectName("op2_check")
        
        self.distance_value = QtWidgets.QLineEdit(self.coord_frame)
        self.distance_value.setGeometry(QtCore.QRect(int(240 * sf_x), int(190 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.distance_value.setFont(font)
        self.distance_value.setObjectName("distance_value")
        
        self.n_building_op2_button = QtWidgets.QPushButton(self.coord_frame)
        self.n_building_op2_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(250 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.n_building_op2_button.setFont(font)
        self.n_building_op2_button.setObjectName("n_building_op2_button")
        
        self.distance_label = QtWidgets.QLabel(self.coord_frame)
        self.distance_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(190 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        font.setBold(True)
        font.setWeight(75)
        self.distance_label.setFont(font)
        self.distance_label.setObjectName("distance_label")
        
        self.backg_6 = QtWidgets.QLabel(self.coord_frame)
        self.backg_6.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(351 * sf_x), int(131 * sf_y)))
        self.backg_6.setStyleSheet("background-color: rgb(255, 224, 185);")
        self.backg_6.setText("")
        self.backg_6.setObjectName("backg_6")
        
        self.backg_7 = QtWidgets.QLabel(self.coord_frame)
        self.backg_7.setGeometry(QtCore.QRect(int(10 * sf_x), int(150 * sf_y), int(351 * sf_x), int(151 * sf_y)))
        self.backg_7.setStyleSheet("background-color: rgb(199, 205, 255);")
        self.backg_7.setText("")
        self.backg_7.setObjectName("backg_7")
        
        self.max_building = QtWidgets.QLabel(self.coord_frame)
        self.max_building.setGeometry(QtCore.QRect(int(250 * sf_x), int(230 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.max_building.setFont(font)
        self.max_building.setObjectName("max_building")
        
        self.max_value = QtWidgets.QLabel(self.coord_frame)
        self.max_value.setGeometry(QtCore.QRect(int(300 * sf_x), int(230 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.max_value.setFont(font)
        self.max_value.setObjectName("max_value")
        
        self.min_building = QtWidgets.QLabel(self.coord_frame)
        self.min_building.setGeometry(QtCore.QRect(int(250 * sf_x), int(260 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.min_building.setFont(font)
        self.min_building.setObjectName("min_building")
        
        self.min_value = QtWidgets.QLabel(self.coord_frame)
        self.min_value.setGeometry(QtCore.QRect(int(300 * sf_x), int(260 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_x))
        self.min_value.setFont(font)
        self.min_value.setObjectName("min_value")

        
        self.backg_7.raise_()
        self.backg_6.raise_()
        self.save_button.raise_()
        self.building_value_op1.raise_()
        self.selected_value_op1.raise_()
        self.n_building_selected.raise_()
        self.n_building_op1_label.raise_()
        self.distance_value.raise_()
        self.n_building_op2_button.raise_()
        self.distance_label.raise_()
        self.op_1_check.raise_()
        self.op2_check.raise_()
        self.max_building.raise_()
        self.max_value.raise_()
        self.min_building.raise_()
        self.min_value.raise_()

        PolygonSetting.setLayout(QtWidgets.QVBoxLayout())  # Set layout before adding widgets
        PolygonSetting.layout().addWidget(self.coord_frame)  # Add main frame to dialog

        self.retranslateUi(PolygonSetting)
        QtCore.QMetaObject.connectSlotsByName(PolygonSetting)

    def retranslateUi(self, PolygonSetting):
        _translate = QtCore.QCoreApplication.translate
        PolygonSetting.setWindowTitle(_translate("PolygonSetting", "Extrapolation Options"))
        self.save_button.setText(_translate("PolygonSetting", "Save and continue"))
        self.building_value_op1.setText(_translate("PolygonSetting", "0000"))
        self.selected_value_op1.setText(_translate("PolygonSetting", "10"))
        self.n_building_selected.setText(_translate("PolygonSetting", "N° Neighbors selected"))
        self.n_building_op1_label.setText(_translate("PolygonSetting", "N° Neighbors available:"))
        self.distance_value.setText(_translate("PolygonSetting", "1"))
        self.n_building_op2_button.setText(_translate("PolygonSetting", "N° Neighbors available:"))
        self.distance_label.setText(_translate("PolygonSetting", "Distance [km]:"))
        self.op_1_check.setText(_translate("PolygonSetting", "Option 1: N° neighbors"))
        self.op2_check.setText(_translate("PolygonSetting", "Option 2: Distance limitation"))
        self.max_building.setText(_translate("PolygonSetting", "Max:"))
        self.max_value.setText(_translate("PolygonSetting", "0000"))
        self.min_building.setText(_translate("PolygonSetting", "Min:"))
        self.min_value.setText(_translate("PolygonSetting", "0000"))
        
    
    def select_method(self):
        # Check how many checkboxes are checked
        checked_count = sum([self.op_1_check.isChecked(), 
                             self.op2_check.isChecked()])
        
        if checked_count > 1:
            # Show a warning if more than one checkbox is checked
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("You can only select one method at a time.")
            msg.setWindowTitle("Selection Warning")
            msg.exec_()
        else:
            # Print the selected method if only one checkbox is checked
            if self.op_1_check.isChecked():
                self.neighbor_method = 1
                self.accept()  
            if self.op2_check.isChecked():
                self.neighbor_method = 2
                self.accept()
                
    
    def distance_neighbor(self, building_extra_path, example_building_path):
        building_no_info = pd.read_csv(building_extra_path)
        building_reference = pd.read_csv(example_building_path)

        n_neighbor = []
        neighbor_method = 2
        # Iterate over each building with no image
        for idx, input_row in building_no_info.iterrows():
            # Find 3 nearest neighbors using geodesic distance
            nearest_neighbors = find_nearest_neighbors_geodesic(input_row, building_reference, 
                                                                n_neighbor, neighbor_method,
                                                                float(self.distance_value.text()))
            n_neighbor.append(nearest_neighbors.shape[0])
            
            
        max_neighbor = max(n_neighbor)
        min_neighbor = min(n_neighbor)

        self.max_value.setText(str(max_neighbor))
        self.min_value.setText(str(min_neighbor))





