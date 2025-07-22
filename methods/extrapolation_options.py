from PyQt5 import QtCore, QtGui, QtWidgets
from methods.neighbor_building_extrapolation_feature import data_options_window
import ctypes
import os

class ExtrapolationOptions(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # DPI Scaling
        LOGPIXELSX = 88
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        scale = 1.25 / (dpi / 96)  # 96 DPI = 100%
        sf_x = screen_width / 1920
        sf_y = screen_height / 1080
        sf_font = sf_x * scale

        self.setWindowTitle("Setting Extrapolation Method")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(1062 * sf_x), int(862 * sf_y))

        self.method_frame = QtWidgets.QWidget(self)

        title_font = QtGui.QFont()
        title_font.setPointSize(int(12 * sf_font))
        title_font.setBold(True)

        label_font = QtGui.QFont()
        label_font.setPointSize(int(10 * sf_font))
        label_font.setBold(True)

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
        self.knn_descrip.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">This method considers the distance to the </span><span style=\" font-size:10pt; font-weight:600;\">K nearest neighbors</span><span style=\" font-size:10pt;\"> and uses </span><span style=\" font-size:10pt; font-weight:600;\">soft voting</span><span style=\" font-size:10pt;\">, meaning that closer neighbors carry more weight in determining the final class. It is based on the assumption that nearby buildings are more likely to share similar characteristics and applies </span><span style=\" font-size:10pt; font-weight:600;\">inverse kernel weighting</span><span style=\" font-size:10pt;\"> to reflect this relationship.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">The process begins with an </span><span style=\" font-size:10pt; font-weight:600;\">initial sample representing 10% of the population</span><span style=\" font-size:10pt;\">. Building information can be provided either by uploading a CSV file or by using the built-in </span><span style=\" font-size:10pt; font-weight:600;\">deep learning model</span><span style=\" font-size:10pt;\"> integrated into the tool.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">Convergence</span><span style=\" font-size:10pt;\"> is evaluated based on the stability of the variable of interest—in this case, </span><span style=\" font-size:10pt; font-weight:600;\">building taxonomies</span><span style=\" font-size:10pt;\">. If the distribution of taxonomies changes by no more than </span><span style=\" font-size:10pt; font-weight:600;\">5%</span><span style=\" font-size:10pt;\"> in the subsequent iteration, convergence is assumed. Otherwise, the sample size is incrementally increased by </span><span style=\" font-size:10pt; font-weight:600;\">5%</span><span style=\" font-size:10pt;\"> in each iteration (e.g., 1st iteration = 10%, 2nd iteration = 15%, and so on) until convergence is reached.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Once convergence is achieved, the </span><span style=\" font-size:10pt; font-weight:600;\">sampling stage is complete</span><span style=\" font-size:10pt;\">, and the </span><span style=\" font-size:10pt; font-weight:600;\">extrapolation process</span><span style=\" font-size:10pt;\"> begins. At this stage, the tool analyzes the buildings of interest, calculates the </span><span style=\" font-size:10pt; font-weight:600;\">geodesic distance</span><span style=\" font-size:10pt;\"> to all available sampled buildings, and selects the 10 closest ones to estimate a </span><span style=\" font-size:10pt; font-weight:600;\">probabilistic distribution of likely taxonomies</span><span style=\" font-size:10pt;\">.</span></p></body></html>")
        self.backg_2 = QtWidgets.QLabel(self.method_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(420 * sf_y), int(1011 * sf_x), int(371 * sf_y)))
        self.backg_2.setStyleSheet("background-color: rgb(215, 213, 255);")

        self.stratified_check = QtWidgets.QCheckBox(self.method_frame)
        self.stratified_check.setGeometry(QtCore.QRect(int(30 * sf_x), int(430 * sf_y), int(201 * sf_x), int(21 * sf_y)))
        self.stratified_check.setFont(label_font)
        self.stratified_check.setText("Stratified Sampling")

        self.stratified_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.stratified_descrip.setGeometry(QtCore.QRect(int(30 * sf_x), int(460 * sf_y), int(981 * sf_x), int(321 * sf_y)))
        self.stratified_descrip.setObjectName("stratified_descrip")
        self.stratified_descrip.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">This method uses a </span><span style=\" font-size:10pt; font-weight:600;\">stratified sampling process based on Scheaffer et al. (1986)</span><span style=\" font-size:10pt;\">, aimed at estimating the distribution of building taxonomy classes. It begins by calculating a </span><span style=\" font-size:10pt; font-weight:600;\">pilot sample size</span><span style=\" font-size:10pt;\"> using a conservative formula that assumes maximum uncertainty in class proportions. Based on this pilot sample, the method estimates class proportions.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">The process adheres to the principles of stratified sampling, ensuring that each class (or stratum) is proportionally represented according to its estimated frequency and variance. Users can iteratively upload new CSV files or use the built-in </span><span style=\" font-size:10pt; font-weight:600;\">deep learning model</span><span style=\" font-size:10pt;\"> to classify images and expand the sample—</span><span style=\" font-size:10pt; font-weight:600;\">increasing by 5% of the population per iteration</span><span style=\" font-size:10pt;\">.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">After each iteration, the method checks whether the estimated class proportions have </span><span style=\" font-size:10pt; font-weight:600;\">converged</span><span style=\" font-size:10pt;\">, meaning they remain stable across samples. If convergence is achieved, the process stops; otherwise, sampling continues, ensuring both </span><span style=\" font-size:10pt; font-weight:600;\">statistical robustness</span><span style=\" font-size:10pt;\"> and </span><span style=\" font-size:10pt; font-weight:600;\">data efficiency</span><span style=\" font-size:10pt;\">.</span></p>\n"
"<p align=\"justify\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Once convergence is reached, the </span><span style=\" font-size:10pt; font-weight:600;\">assignment process</span><span style=\" font-size:10pt;\"> begins. This step uses a </span><span style=\" font-size:10pt; font-weight:600;\">hierarchical fallback strategy</span><span style=\" font-size:10pt;\"> that depends on the availability of information. The method prioritizes the use of </span><span style=\" font-size:10pt; font-weight:600;\">specific local data</span><span style=\" font-size:10pt;\"> where available, and progressively falls back to </span><span style=\" font-size:10pt; font-weight:600;\">more general or global information</span><span style=\" font-size:10pt;\"> when needed, ensuring the most accurate classification possible based on the data at hand.</span></p></body></html>")
        
        self.load_button = QtWidgets.QPushButton(self.method_frame)
        self.load_button.setGeometry(QtCore.QRect(int(300 * sf_x), int(800 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.load_button.setFont(label_font)
        self.load_button.setText("Load files")
        self.load_button.clicked.connect(self.select_method)

        self.save_button = QtWidgets.QPushButton(self.method_frame)
        self.save_button.setGeometry(QtCore.QRect(int(590 * sf_x), int(800 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(label_font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.save_and_continue)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.method_frame)


    def select_method(self):
        checked_count = sum([self.knn_check.isChecked(), self.stratified_check.isChecked()])

        if checked_count > 1:
            QtWidgets.QMessageBox.warning(self, "Selection Warning", "You can only select one method at a time.")
        elif self.knn_check.isChecked():
            dialog = data_options_window(parent=self)
            self.load_check = True
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                try:
                    self.info_existing = dialog.info_existing
                    self.info_pending = dialog.info_pending
                    self.extrapolation_name = dialog.output_manual_value.text()
                    QtWidgets.QMessageBox.information(
                        self,
                        "Success",
                        "✅ Setup complete!\n\n"
                        "Please click **Save and continue** button."
                    )
                    
                except Exception:
                    QtWidgets.QMessageBox.warning(
                        self, "Inspection Method Error",
                        "Please upload the input files. Press again the **Load files** button"
                    )
        elif self.stratified_check.isChecked():
            QtWidgets.QMessageBox.warning(
                self, "Inspection Method Error",
                "This option is currently unavailable"
            )
            
            
    def save_and_continue(self):
        try:
            if self.load_check:
                self.accept()
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Inspection Method Error", "Please select inspection method.")
