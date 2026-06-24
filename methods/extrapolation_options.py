from PyQt5 import QtCore, QtGui, QtWidgets
from methods.knn_extrapolation_feature import knn_options_window
from methods.stratified_extrapolation_feature import stratified_extrapolation
import sys 
import numpy as np

class ExtrapolationOptions(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.method = parent
        
        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Design screen size
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

        self.setWindowTitle("Setting Extrapolation Method")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(1060 * sf_x), int(790 * sf_y))

        self.method_frame = QtWidgets.QWidget(self)

        title_font = QtGui.QFont()
        title_font.setPointSize(int(12 * sf_font))
        title_font.setBold(True)

        label_font = QtGui.QFont()
        label_font.setPointSize(int(10 * sf_font))
        label_font.setBold(True)
        self.sf_font = sf_font

        self.w_tittle = QtWidgets.QLabel(self.method_frame)
        self.w_tittle.setGeometry(QtCore.QRect(int(370 * sf_x), int(0 * sf_y), int(301 * sf_x), int(41 * sf_y)))
        self.w_tittle.setFont(title_font)
        self.w_tittle.setText("Setting Extrapolation Method")

        self.backg_1 = QtWidgets.QLabel(self.method_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(40 * sf_y), int(1011 * sf_x), int(351 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(255, 224, 185);")

        self.knn_check = QtWidgets.QCheckBox(self.method_frame)
        self.knn_check.setGeometry(QtCore.QRect(int(30 * sf_x), int(50 * sf_y), int(231 * sf_x), int(21 * sf_y)))
        self.knn_check.setFont(label_font)
        self.knn_check.setText("K-NN with Soft Voting")

        self.knn_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.knn_descrip.setGeometry(QtCore.QRect(int(30 * sf_x), int(80 * sf_y), int(981 * sf_x), int(301 * sf_y)))
        self.knn_descrip.setObjectName("knn_descrip")
        
        fs = int(10 * sf_font)
        
        self.knn_descrip.setHtml(f"""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html>
<head>
<meta name="qrichtext" content="1" />
<style>
p, li {{ white-space: pre-wrap; }}
</style>
</head>
<body style="font-family:'MS Shell Dlg 2'; font-size:{fs}pt; font-weight:400; font-style:normal;">

<p align="justify" style="margin-top:6px; margin-bottom:6px; text-indent:0px;">
<span style="font-size:{fs}pt;">This method considers the distance to the </span>
<span style="font-size:{fs}pt; font-weight:600;">K nearest neighbors</span>
<span style="font-size:{fs}pt;"> and uses </span>
<span style="font-size:{fs}pt; font-weight:600;">soft voting</span>
<span style="font-size:{fs}pt;">, meaning that closer neighbors carry more weight in determining the final class. It is based on the assumption that nearby buildings are more likely to share similar characteristics and applies </span>
<span style="font-size:{fs}pt; font-weight:600;">inverse kernel weighting</span>
<span style="font-size:{fs}pt;"> to reflect this relationship.</span>
</p>

<p align="justify" style="margin-top:6px; margin-bottom:6px; text-indent:0px;">
<span style="font-size:{fs}pt;">The process begins with an </span>
<span style="font-size:{fs}pt; font-weight:600;">initial sample representing 10% of the population</span>
<span style="font-size:{fs}pt;">. Building information can be provided either by uploading a CSV file or by using the built-in </span>
<span style="font-size:{fs}pt; font-weight:600;">deep learning model</span>
<span style="font-size:{fs}pt;"> integrated into the tool.</span>
</p>

<p align="justify" style="margin-top:6px; margin-bottom:6px; text-indent:0px;">
<span style="font-size:{fs}pt; font-weight:600;">Convergence</span>
<span style="font-size:{fs}pt;"> is evaluated based on the stability of the variable of interest—in this case, </span>
<span style="font-size:{fs}pt; font-weight:600;">building taxonomies</span>
<span style="font-size:{fs}pt;">. If the distribution of taxonomies changes by no more than </span>
<span style="font-size:{fs}pt; font-weight:600;">5%</span>
<span style="font-size:{fs}pt;"> in the subsequent iteration, convergence is assumed. Otherwise, the sample size is incrementally increased by </span>
<span style="font-size:{fs}pt; font-weight:600;">5%</span>
<span style="font-size:{fs}pt;"> in each iteration (e.g., 1st iteration = 10%, 2nd iteration = 15%, and so on) until convergence is reached.</span>
</p>

<p align="justify" style="margin-top:6px; margin-bottom:0px; text-indent:0px;">
<span style="font-size:{fs}pt;">Once convergence is achieved, the </span>
<span style="font-size:{fs}pt; font-weight:600;">sampling stage is complete</span>
<span style="font-size:{fs}pt;">, and the </span>
<span style="font-size:{fs}pt; font-weight:600;">extrapolation process</span>
<span style="font-size:{fs}pt;"> begins. At this stage, the tool analyzes the buildings of interest, calculates the </span>
<span style="font-size:{fs}pt; font-weight:600;">geodesic distance</span>
<span style="font-size:{fs}pt;"> to all available sampled buildings, and selects the 10 closest ones to estimate a </span>
<span style="font-size:{fs}pt; font-weight:600;">probabilistic distribution of likely taxonomies</span>
<span style="font-size:{fs}pt;">.</span>
</p>

</body>
</html>
""")

        self.backg_2 = QtWidgets.QLabel(self.method_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(420 * sf_y), int(1011 * sf_x), int(300 * sf_y)))
        self.backg_2.setStyleSheet("background-color: rgb(215, 213, 255);")

        self.stratified_check = QtWidgets.QCheckBox(self.method_frame)
        self.stratified_check.setGeometry(QtCore.QRect(int(30 * sf_x), int(430 * sf_y), int(201 * sf_x), int(21 * sf_y)))
        self.stratified_check.setFont(label_font)
        self.stratified_check.setText("Stratified Sampling")

        self.stratified_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.stratified_descrip.setGeometry(QtCore.QRect(int(30 * sf_x), int(455 * sf_y), int(981 * sf_x), int(250 * sf_y)))
        self.stratified_descrip.setObjectName("stratified_descrip")
        
        fs = int(10 * sf_font)
        
        self.stratified_descrip.setHtml(f"""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html>
<head>
<meta name="qrichtext" content="1" />
<style>
p, li {{ white-space: pre-wrap; }}
</style>
</head>

<body style="font-family:'MS Shell Dlg 2'; font-size:{fs}pt; font-weight:400; font-style:normal;">

<p align="justify" style="margin-top:6px; margin-bottom:6px; text-indent:0px;">
<span style="font-size:{fs}pt;">This method uses a </span>
<span style="font-size:{fs}pt; font-weight:600;">stratified sampling process based on Scheaffer et al. (1986)</span>
<span style="font-size:{fs}pt;">, aimed at estimating the distribution of building taxonomy classes. It begins by calculating a </span>
<span style="font-size:{fs}pt; font-weight:600;">pilot sample size</span>
<span style="font-size:{fs}pt;"> using a conservative formula that assumes maximum uncertainty in class proportions. Based on this pilot sample, the method estimates class proportions.</span>
</p>

<p align="justify" style="margin-top:6px; margin-bottom:6px; text-indent:0px;">
<span style="font-size:{fs}pt;">The process adheres to the principles of stratified sampling, ensuring that each class (or stratum) is proportionally represented according to its estimated frequency and variance. Users can iteratively upload </span>
<span style="font-size:{fs}pt; font-weight:600;">new CSV files (manually)</span>
<span style="font-size:{fs}pt;"> or use the built-in </span>
<span style="font-size:{fs}pt; font-weight:600;">deep learning model</span>
<span style="font-size:{fs}pt;"> to classify images and expand the sample—</span>
<span style="font-size:{fs}pt; font-weight:600;">increasing by a step value provided by the user (5% by default) of the population per iteration</span>
<span style="font-size:{fs}pt;">.</span>
</p>

<p align="justify" style="margin-top:6px; margin-bottom:6px; text-indent:0px;">
<span style="font-size:{fs}pt;">After each iteration, the method checks whether the estimated class proportions have </span>
<span style="font-size:{fs}pt; font-weight:600;">converged</span>
<span style="font-size:{fs}pt;">, meaning they remain stable across samples. If convergence is achieved, the process stops; otherwise, sampling continues, ensuring both </span>
<span style="font-size:{fs}pt; font-weight:600;">statistical robustness</span>
<span style="font-size:{fs}pt;"> and </span>
<span style="font-size:{fs}pt; font-weight:600;">data efficiency</span>
<span style="font-size:{fs}pt;">.</span>
</p>

</body>
</html>
""")

        self.load_button = QtWidgets.QPushButton(self.method_frame)
        self.load_button.setGeometry(QtCore.QRect(int(300 * sf_x), int(730 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.load_button.setFont(label_font)
        self.load_button.setText("Load files")
        self.load_button.clicked.connect(self.select_method)

        self.save_button = QtWidgets.QPushButton(self.method_frame)
        self.save_button.setGeometry(QtCore.QRect(int(590 * sf_x), int(730 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(label_font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.save_and_continue)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.method_frame)

    # ==============================================================
    # Extrapolation method functions
    # ==============================================================
    
    def select_method(self):
        """
        Validates that only one extrapolation method is selected, opens the corresponding setup 
        dialog, retrieves the required input parameters, and stores the selected method settings.
        """
        checked_count = sum([self.knn_check.isChecked(), self.stratified_check.isChecked()])

        if checked_count > 1:
            QtWidgets.QMessageBox.warning(self, "Selection Warning", "You can only select one method at a time.")
            
        #####################################################################################################    
        ########################## --------- KNN method ---------------###################################### 
        #####################################################################################################  
        elif self.knn_check.isChecked():
            dialog = knn_options_window(parent=self)
            # This should be passed at this stage because it has an additional layer compared with the other methods
            self.insp_method = self.method.insp_method
            self.load_check = True
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                try:
                    # KNN Manual
                    self.info_existing = dialog.info_existing
                    self.info_pending = dialog.info_pending
                    self.extrapolation_name = dialog.output_manual_value.text()
                    self.output_path = dialog.folder_path
                    self.k_value = dialog.k_value_manual.value()
                    self.extrapolation_mode = 2 # for manual and DL knn
                    # KNN DL method
                    try:
                        self.coord_reference = dialog.coord_reference
                        self.knn_dl_saved_path = dialog.knn_dl_saved_path
                        self.coord_reference_building_feature_path = dialog.coord_reference_building_feature_path
                        self.k_value = dialog.k_value_dl.value()
                        self.use_coord_reference = True
                    except (AttributeError, FileNotFoundError):
                        self.coord_reference = None
                        self.use_coord_reference = False
                        
                    QtWidgets.QMessageBox.information(
                        self,
                        "Success",
                        "✅ Setup complete!\n\n"
                        "Please click **Save and continue** button."
                    )
                except Exception:
                    QtWidgets.QMessageBox.warning(
                        self, "Inspection Method Error",
                        "An error occurred while reading the input files. Please click **Load files** button \n" 
                        "again and confirm that all files follow the required structure, or upload your data"
                    )
                           
        #####################################################################################################    
        ########################## ---------Stratified method ---------------################################   
        #####################################################################################################      
        elif self.stratified_check.isChecked():
            dialog = stratified_extrapolation(parent=self)
            # This should be passed at this stage because it has an additional layer compared with the other methods
            self.insp_method = self.method.insp_method
            self.load_check = True
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                try:
                    self.extrapolation_mode = dialog.stratified_mode
                    self.data_population = dialog.data_population
                    self.folder_path_new = dialog.folder_path_new

                    #Stratified manually
                    self.initial_fraction= dialog.ini_fract_new_value.value()
                    self.step_fraction= dialog.step_new_value.value()
                    self.max_fraction= dialog.max_frac_new_value.value()
                    self.max_iterations=dialog.n_iter_new_value.value()
                    self.stability_threshold=dialog.threshold_new_value.value()
                    # Stratified dl
                    self.feature_strata = dialog.feature_strata
                    
                    QtWidgets.QMessageBox.information(
                        self,
                        "Success",
                        "✅ Setup complete!\n\n"
                        "Please click **Save and continue** button.")
                        
                except Exception:
                    QtWidgets.QMessageBox.warning(
                        self, "Inspection Method Error",
                        "An error occurred while reading the input files. Please click **Load files** button \n" 
                        "again and confirm that all files follow the required structure, or upload your data"
                    )
            

            
    def save_and_continue(self):
        """
        Confirms the model input setup and closes the dialog if a method has been successfully loaded.
        """
        try:
            if self.load_check:
                self.accept()
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Inspection Method Error", "Please select inspection method.")
