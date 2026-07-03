"""Provide the dialog used to configure building-data extrapolation methods."""

import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from methods.knn_extrapolation_feature import knn_options_window
from methods.stratified_extrapolation_feature import StratifiedExtrapolation

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 120
LOGPIXELSX = 88


class ExtrapolationOptions(QtWidgets.QDialog):
    """Display and configure the available extrapolation methods.

    The dialog allows the user to select either K-nearest neighbours with
    soft voting or stratified sampling. It then opens the corresponding
    configuration dialog and stores the selected parameters.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that also provides the current inspection method.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.method = parent
        self.load_check = False

        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        scale_x = screen_width / DESIGN_WIDTH
        scale_y = screen_height / DESIGN_HEIGHT
        scale_factor = np.sqrt(scale_x * scale_y)
        dpi = self._get_screen_dpi(screen)
        scale_dpi = DESIGN_DPI / dpi

        scale_x = scale_factor
        scale_y = scale_factor
        scale_font = scale_factor * scale_dpi
        self.sf_font = scale_font

        self.setWindowTitle("Setting Extrapolation Method")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(1060 * scale_x), int(700 * scale_y))

        self.method_frame = QtWidgets.QWidget(self)
        title_font = self._create_font(12, scale_font)
        label_font = self._create_font(10, scale_font)

        self._create_title(scale_x, scale_y, title_font)
        self._create_knn_section(scale_x, scale_y, scale_font, label_font)
        self._create_stratified_section(
            scale_x,
            scale_y,
            scale_font,
            label_font,
        )
        self._create_buttons(scale_x, scale_y, label_font)

        dialog_layout = QtWidgets.QVBoxLayout()
        dialog_layout.addWidget(self.method_frame)
        self.setLayout(dialog_layout)

    @staticmethod
    def _get_screen_dpi(screen):
        """Return a reliable DPI value for the active screen."""
        if sys.platform.startswith("win"):
            import ctypes

            device_context = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(
                device_context,
                LOGPIXELSX,
            )
            ctypes.windll.user32.ReleaseDC(0, device_context)
            return dpi

        dpi = screen.logicalDotsPerInch()
        if not 60 <= dpi <= 200:
            dpi = screen.physicalDotsPerInch()
        return dpi

    @staticmethod
    def _create_font(point_size, scale_font):
        """Create a bold font scaled to the current display."""
        font = QtGui.QFont()
        font.setPointSize(int(point_size * scale_font))
        font.setBold(True)
        return font

    def _create_title(self, scale_x, scale_y, title_font):
        """Create the dialog title."""
        self.w_tittle = QtWidgets.QLabel(self.method_frame)
        self.w_tittle.setGeometry(
            QtCore.QRect(
                int(370 * scale_x),
                int(0 * scale_y),
                int(301 * scale_x),
                int(41 * scale_y),
            )
        )
        self.w_tittle.setFont(title_font)
        self.w_tittle.setText("Setting Extrapolation Method")

    def _create_knn_section(self, scale_x, scale_y, scale_font, label_font):
        """Create the K-nearest-neighbours option and description."""
        self.backg_1 = QtWidgets.QLabel(self.method_frame)
        self.backg_1.setGeometry(
            QtCore.QRect(
                int(10 * scale_x),
                int(40 * scale_y),
                int(1010 * scale_x),
                int(320 * scale_y),
            )
        )
        self.backg_1.setStyleSheet("background-color: rgb(255, 224, 185);")

        self.knn_check = QtWidgets.QRadioButton(self.method_frame)
        self.knn_check.setGeometry(
            QtCore.QRect(
                int(30 * scale_x),
                int(50 * scale_y),
                int(231 * scale_x),
                int(21 * scale_y),
            )
        )
        self.knn_check.setFont(label_font)
        self.knn_check.setText("K-NN with Soft Voting")

        self.knn_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.knn_descrip.setGeometry(
            QtCore.QRect(
                int(30 * scale_x),
                int(80 * scale_y),
                int(980 * scale_x),
                int(260 * scale_y),
            )
        )
        self.knn_descrip.setObjectName("knn_descrip")
        font_size = int(10 * scale_font)
        self.knn_descrip.setHtml(self._knn_description_html(font_size))

    def _create_stratified_section(
        self,
        scale_x,
        scale_y,
        scale_font,
        label_font,
    ):
        """Create the stratified-sampling option and description."""
        self.backg_2 = QtWidgets.QLabel(self.method_frame)
        self.backg_2.setGeometry(
            QtCore.QRect(
                int(10 * scale_x),
                int(380 * scale_y),
                int(1010 * scale_x),
                int(240 * scale_y),
            )
        )
        self.backg_2.setStyleSheet("background-color: rgb(215, 213, 255);")

        self.stratified_check = QtWidgets.QRadioButton(self.method_frame)
        self.stratified_check.setGeometry(
            QtCore.QRect(
                int(30 * scale_x),
                int(390 * scale_y),
                int(201 * scale_x),
                int(21 * scale_y),
            )
        )
        self.stratified_check.setFont(label_font)
        self.stratified_check.setText("Stratified Sampling")

        self.stratified_descrip = QtWidgets.QTextBrowser(self.method_frame)
        self.stratified_descrip.setGeometry(
            QtCore.QRect(
                int(30 * scale_x),
                int(415 * scale_y),
                int(980 * scale_x),
                int(185 * scale_y),
            )
        )
        self.stratified_descrip.setObjectName("stratified_descrip")
        font_size = int(10 * scale_font)
        self.stratified_descrip.setHtml(self._stratified_description_html(font_size))

    def _create_buttons(self, scale_x, scale_y, label_font):
        """Create and connect the dialog action buttons."""
        self.load_button = QtWidgets.QPushButton(self.method_frame)
        self.load_button.setGeometry(
            QtCore.QRect(
                int(300 * scale_x),
                int(640 * scale_y),
                int(191 * scale_x),
                int(31 * scale_y),
            )
        )
        self.load_button.setFont(label_font)
        self.load_button.setText("Load Files")
        self.load_button.clicked.connect(self.select_method)

        self.save_button = QtWidgets.QPushButton(self.method_frame)
        self.save_button.setGeometry(
            QtCore.QRect(
                int(590 * scale_x),
                int(640 * scale_y),
                int(191 * scale_x),
                int(31 * scale_y),
            )
        )
        self.save_button.setFont(label_font)
        self.save_button.setText("Save and Continue")
        self.save_button.clicked.connect(self.save_and_continue)

    @staticmethod
    def _html_document(font_size, body):
        """Wrap description content in the HTML expected by QTextBrowser."""
        return (
            '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" '
            '"http://www.w3.org/TR/REC-html40/strict.dtd">'
            '<html><head><meta name="qrichtext" content="1" />'
            "<style>p, li { white-space: pre-wrap; }</style></head>"
            f"<body style=\"font-family:'MS Shell Dlg 2'; "
            f"font-size:{font_size}pt; font-weight:400; "
            f'font-style:normal;">{body}</body></html>'
        )

    @classmethod
    def _knn_description_html(cls, font_size):
        """Return the formatted K-nearest-neighbours description."""
        body = """
<p align="justify">
This method considers the distance to the
<b> K nearest neighbors</b> and uses <b>soft voting</b>, meaning that closer
neighbors carry more weight in determining the final class. It assumes that
nearby buildings are more likely to share similar characteristics and applies
<b> inverse kernel weighting</b> to reflect this relationship.
</p>
<p align="justify">
The process begins with an <b>initial sample representing 10% of the population
</b>. Building information can be provided by uploading a CSV file
or by using the built-in <b>deep learning model</b>.
</p>
<p align="justify">
<b>Convergence</b> is evaluated from the stability of the variable of interest,
which is the distribution of <b>building taxonomies</b>. Convergence is assumed
when the distribution changes by no more than <b>5%</b> in the next iteration.
 Otherwise, the sample increases by 5% of the population per iteration.
</p>
<p align="justify">
After convergence, the tool calculates the <b>geodesic distance</b> from each
building of interest to the sampled buildings. It uses the 10 closest samples
to estimate a <b>probabilistic distribution of likely taxonomies</b>.
</p>
"""
        return cls._html_document(font_size, body)

    @classmethod
    def _stratified_description_html(cls, font_size):
        """Return the formatted stratified-sampling description."""
        body = """
<p align="justify">
This method uses a <b>stratified sampling process based on Scheaffer et al.
(1986)</b> to estimate the distribution of building-taxonomy classes. It first
calculates a <b>pilot sample size</b> using a conservative formula that assumes
maximum uncertainty in the class proportions.
</p>
<p align="justify">
Each class is represented according to its estimated frequency and variance.
Users can upload <b>new CSV files manually</b> or use the built-in
<b> deep learning model</b> to expand the sample by a user-defined percentage of
the population in each iteration.
</p>
<p align="justify">
After each iteration, the method checks whether the estimated class
proportions have <b>converged</b>. Sampling stops when the proportions remain
stable; otherwise, it continues to improve <b>statistical robustness</b> and
<b> data efficiency</b>.
</p>
"""
        return cls._html_document(font_size, body)

    def select_method(self):
        """Open the configuration dialog for the selected method."""
        checked_count = sum(
            [
                self.knn_check.isChecked(),
                self.stratified_check.isChecked(),
            ]
        )

        if checked_count > 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Selection Warning",
                "You can only select one method at a time.",
            )
            return

        if self.knn_check.isChecked():
            self._configure_knn_method()
        elif self.stratified_check.isChecked():
            self._configure_stratified_method()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Selection Warning",
                "Please select an extrapolation method.",
            )

    def _configure_knn_method(self):
        """Open the K-nearest-neighbours configuration dialog."""
        dialog = knn_options_window(parent=self)
        self.insp_method = self.method.insp_method

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            self.info_existing = dialog.info_existing
            self.info_pending = dialog.info_pending
            self.extrapolation_name = dialog.output_manual_value.text()
            self.output_path = dialog.folder_path
            self.k_value = dialog.k_value_manual.value()
            self.extrapolation_mode = 2

            try:
                self.coord_reference = dialog.coord_reference
                self.knn_dl_saved_path = dialog.knn_dl_saved_path
                self.coord_reference_building_feature_path = (
                    dialog.coord_reference_building_feature_path
                )
                self.k_value = dialog.k_value_dl.value()
                self.use_coord_reference = True
            except (AttributeError, FileNotFoundError):
                self.coord_reference = None
                self.use_coord_reference = False

            self.load_check = True
            self._show_success_message()
        except (AttributeError, FileNotFoundError, TypeError, ValueError) as error:
            print(f"Error while reading K-NN inputs: {error}")
            self._show_input_error()

    def _configure_stratified_method(self):
        """Open the stratified-sampling configuration dialog."""
        dialog = StratifiedExtrapolation(parent=self)
        self.insp_method = self.method.insp_method

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            self.extrapolation_mode = dialog.stratified_mode
            self.data_population = dialog.population_data
            self.folder_path_new = dialog.folder_path_new
            self.initial_fraction = dialog.ini_fract_new_value.value()
            self.step_fraction = dialog.step_new_value.value()
            self.max_fraction = dialog.max_frac_new_value.value()
            self.max_iterations = dialog.n_iter_new_value.value()
            self.stability_threshold = dialog.threshold_new_value.value()
            self.feature_strata = dialog.feature_strata
            self.prefix_strata = dialog.output_new_value.text()
            self.load_check = True
            self._show_success_message()
        except (AttributeError, FileNotFoundError, TypeError, ValueError) as error:
            print(f"Error while reading stratified-sampling inputs: {error}")
            self._show_input_error()

    def _show_success_message(self):
        """Inform the user that the method setup was completed."""
        QtWidgets.QMessageBox.information(
            self,
            "Success",
            "Setup complete!\n\nPlease click the Save and Continue button.",
        )

    def _show_input_error(self):
        """Inform the user that the selected input files are invalid."""
        QtWidgets.QMessageBox.warning(
            self,
            "Inspection Method Error",
            (
                "An error occurred while reading the input files. Please click "
                "the Load files button again and confirm that all files follow "
                "the required structure, or upload your data again."
            ),
        )

    def save_and_continue(self):
        """Accept the dialog after a method has been configured successfully."""
        if self.load_check:
            self.accept()
            return

        QtWidgets.QMessageBox.warning(
            self,
            "Inspection Method Error",
            (
                "Please select and configure an extrapolation method. "
                "View the input requirements by clicking the "
                "'Load Files' button."
            ),
        )
