from PyQt5 import QtCore, QtGui, QtWidgets
from neighbor_building_extrapolation_feature import find_nearest_neighbors_geodesic
import pandas as pd

class extrapolation_options_window(QtWidgets.QDialog):  # Inherit from QDialog
    def __init__(self, parent=None, n_neighbors=None):
        super().__init__(parent)
        self.n_neighbors = n_neighbors  # Data passed from main window
        self.neighbor_data = None
        self.setupUi(self)  # Call setupUi and pass self (QDialog instance)


    def setupUi(self, PolygonSetting):  # PolygonSetting is now self
        PolygonSetting.setObjectName("PolygonSetting")
        PolygonSetting.resize(400, 350)
        self.coord_frame = QtWidgets.QWidget(PolygonSetting)
        self.coord_frame.setObjectName("coord_frame")
        
        self.op_1_check = QtWidgets.QCheckBox(self.coord_frame)
        self.op_1_check.setGeometry(QtCore.QRect(20, 20, 241, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.op_1_check.setFont(font)
        self.op_1_check.setObjectName("op_1_check")
        
        self.save_buttom = QtWidgets.QPushButton(self.coord_frame)
        self.save_buttom.setGeometry(QtCore.QRect(90, 290, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.save_buttom.setFont(font)
        self.save_buttom.setObjectName("save_buttom")
        self.save_buttom.clicked.connect(self.select_method)
        
        self.building_value_op1 = QtWidgets.QLabel(self.coord_frame)
        self.building_value_op1.setGeometry(QtCore.QRect(240, 50, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.building_value_op1.setFont(font)
        self.building_value_op1.setObjectName("building_value_op1")
       
        self.selected_value_op1 = QtWidgets.QLineEdit(self.coord_frame)
        self.selected_value_op1.setGeometry(QtCore.QRect(240, 90, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.selected_value_op1.setFont(font)
        self.selected_value_op1.setObjectName("selected_value_op1")
       
        self.n_building_selected = QtWidgets.QLabel(self.coord_frame)
        self.n_building_selected.setGeometry(QtCore.QRect(20, 90, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.n_building_selected.setFont(font)
        self.n_building_selected.setObjectName("n_building_selected")
        
        self.n_building_op1_label = QtWidgets.QLabel(self.coord_frame)
        self.n_building_op1_label.setGeometry(QtCore.QRect(20, 50, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.n_building_op1_label.setFont(font)
        self.n_building_op1_label.setObjectName("n_building_op1_label")
        
        self.building_value_op2 = QtWidgets.QLabel(self.coord_frame)
        self.building_value_op2.setGeometry(QtCore.QRect(240, 230, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.building_value_op2.setFont(font)
        self.building_value_op2.setObjectName("building_value_op2")
        
        self.op2_check = QtWidgets.QCheckBox(self.coord_frame)
        self.op2_check.setGeometry(QtCore.QRect(20, 160, 281, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.op2_check.setFont(font)
        self.op2_check.setObjectName("op2_check")  
        
        self.distance_value = QtWidgets.QLineEdit(self.coord_frame)
        self.distance_value.setGeometry(QtCore.QRect(240, 190, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.distance_value.setFont(font)
        self.distance_value.setObjectName("distance_value")
        # self.distance_value.textChanged.connect(self.distance_neighbor)
        
        self.n_building_op2_button = QtWidgets.QPushButton(self.coord_frame)
        self.n_building_op2_button.setGeometry(QtCore.QRect(20, 230, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.n_building_op2_button.setFont(font)
        self.n_building_op2_button.setObjectName("n_building_op2_button ")
        
        self.distance_label = QtWidgets.QLabel(self.coord_frame)
        self.distance_label.setGeometry(QtCore.QRect(20, 190, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.distance_label.setFont(font)
        self.distance_label.setObjectName("distance_label")
        
        self.backg_6 = QtWidgets.QLabel(self.coord_frame)
        self.backg_6.setGeometry(QtCore.QRect(10, 10, 351, 131))
        self.backg_6.setStyleSheet("background-color: rgb(255, 224, 185);")
        self.backg_6.setText("")
        self.backg_6.setObjectName("backg_6")
        
        self.backg_7 = QtWidgets.QLabel(self.coord_frame)
        self.backg_7.setGeometry(QtCore.QRect(10, 150, 351, 121))
        self.backg_7.setStyleSheet("background-color: rgb(199, 205, 255);")
        self.backg_7.setText("")
        self.backg_7.setObjectName("backg_7")
        
        self.backg_7.raise_()
        self.backg_6.raise_()
        self.save_buttom.raise_()
        self.building_value_op1.raise_()
        self.selected_value_op1.raise_()
        self.n_building_selected.raise_()
        self.n_building_op1_label.raise_()
        self.building_value_op2.raise_()
        self.distance_value.raise_()
        self.n_building_op2_button.raise_()
        self.distance_label.raise_()
        self.op_1_check.raise_()
        self.op2_check.raise_()

        PolygonSetting.setLayout(QtWidgets.QVBoxLayout())  # Set layout before adding widgets
        PolygonSetting.layout().addWidget(self.coord_frame)  # Add main frame to dialog

        self.retranslateUi(PolygonSetting)
        QtCore.QMetaObject.connectSlotsByName(PolygonSetting)

    def retranslateUi(self, PolygonSetting):
        _translate = QtCore.QCoreApplication.translate
        PolygonSetting.setWindowTitle(_translate("PolygonSetting", "Extrapolation Options"))
        self.save_buttom.setText(_translate("PolygonSetting", "Save and continue"))
        self.building_value_op1.setText(_translate("PolygonSetting", "0000"))
        self.selected_value_op1.setText(_translate("PolygonSetting", "10"))
        self.n_building_selected.setText(_translate("PolygonSetting", "N° Neighbors selected"))
        self.n_building_op1_label.setText(_translate("PolygonSetting", "N° Neighbors available:"))
        self.building_value_op2.setText(_translate("PolygonSetting", "0000"))
        self.distance_value.setText(_translate("PolygonSetting", "1"))
        self.n_building_op2_button.setText(_translate("PolygonSetting", "N° Neighbors available:"))
        self.distance_label.setText(_translate("PolygonSetting", "Distance [km]:"))
        self.op_1_check.setText(_translate("PolygonSetting", "Option 1: N° neighbors"))
        self.op2_check.setText(_translate("PolygonSetting", "Option 2: Distance limitation"))
        
    
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

        n_neighbor = 10
        neighbor_method = 2
        # Iterate over each building with no image
        for idx, input_row in building_no_info.iterrows():
            # Find 3 nearest neighbors using geodesic distance
            nearest_neighbors = find_nearest_neighbors_geodesic(input_row, building_reference, 
                                                                n_neighbor, neighbor_method,
                                                                float(self.distance_value.text()))
            
        self.building_value_op2.setText(str(nearest_neighbors.shape[0]))


                

                



