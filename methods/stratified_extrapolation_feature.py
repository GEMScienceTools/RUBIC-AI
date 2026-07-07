import ctypes
import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from methods.dl_prediction_models import (
    predict_block_position_img,
    predict_code_img,
    predict_llrs_img,
    predict_material_img,
    predict_n_stories_img,
    predict_occupancy_img,
    predict_roof_material_img,
    predict_roof_shape_img,
)
from methods.get_building_orientation import get_street_view_image
from methods.utilities import select_output_folder, upload_csv

DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080
DESIGN_DPI = 96 * 1.25
LOGPIXELS_X = 88
MIN_VALID_DPI = 60
MAX_VALID_DPI = 200


class StratifiedExtrapolation(QtWidgets.QDialog):
    """Configure stratified extrapolation settings and sampling options."""

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.method = parent

        ## Get screen resolution
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Scale the GUI based on resolution
        sf_x = screen_width / DESIGN_WIDTH
        sf_y = screen_height / DESIGN_HEIGHT
        sf_factor = np.sqrt(sf_x * sf_y)

        # DPI-based scale
        # Get a reliable DPI value
        if sys.platform.startswith("win"):
            # Windows: use ctypes to get real DPI
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELS_X)
            ctypes.windll.user32.ReleaseDC(0, hdc)
        else:
            # macOS / Linux: start with logical DPI
            dpi = screen.logicalDotsPerInch()
            # If logical DPI looks weird, fallback to physical
            if dpi < MIN_VALID_DPI or dpi > MAX_VALID_DPI:
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
        self.setObjectName("DataSetting")
        self.resize(int(640 * sf_x), int(680 * sf_y))
        self.setWindowTitle("Setting Input Files")

        # === UI Elements Start ===
        self.data_frame = QtWidgets.QWidget(self)
        self.data_frame.setObjectName("data_frame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.data_frame)

        # Title Label
        self.w_title = QtWidgets.QLabel(self.data_frame)
        self.w_title.setGeometry(
            QtCore.QRect(
                int(220 * sf_x), int(0 * sf_y), int(191 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(12 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.w_title.setFont(font)
        self.w_title.setObjectName("w_title")

        # Save Button
        self.save_button = QtWidgets.QPushButton(self.data_frame)
        self.save_button.setGeometry(
            QtCore.QRect(
                int(210 * sf_x), int(620 * sf_y), int(191 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self.select_method)

        # Table Widget
        self.tableWidget = QtWidgets.QTableWidget(self.data_frame)
        self.tableWidget.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(400 * sf_y), int(590 * sf_x), int(211 * sf_y)
            )
        )
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        # Output Label - New
        self.output_label_new = QtWidgets.QLabel(self.data_frame)
        self.output_label_new.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(55 * sf_y), int(121 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_new.setFont(font)
        self.output_label_new.setObjectName("output_label_new")

        # Output Value - New
        self.output_new_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_new_value.setGeometry(
            QtCore.QRect(
                int(180 * sf_x), int(55 * sf_y), int(331 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.output_new_value.setFont(font)
        self.output_new_value.setObjectName("output_new_value")

        # Background 2
        self.backg_2 = QtWidgets.QLabel(self.data_frame)
        self.backg_2.setGeometry(
            QtCore.QRect(
                int(10 * sf_x), int(40 * sf_y), int(591 * sf_x), int(351 * sf_y)
            )
        )
        self.backg_2.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        self.backg_2.lower()

        # Population Button - New
        self.population_new_button = QtWidgets.QPushButton(self.data_frame)
        self.population_new_button.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(185 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.population_new_button.setFont(font)
        self.population_new_button.setObjectName("population_new_button")
        self.population_new_button.clicked.connect(self.data_population)

        # Population Path - New
        self.population_new_path = QtWidgets.QLabel(self.data_frame)
        self.population_new_path.setGeometry(
            QtCore.QRect(
                int(290 * sf_x), int(190 * sf_y), int(291 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.population_new_path.setFont(font)
        self.population_new_path.setObjectName("population_new_path")

        # Saved Path - New
        self.saved_path_new = QtWidgets.QLabel(self.data_frame)
        self.saved_path_new.setGeometry(
            QtCore.QRect(
                int(290 * sf_x), int(100 * sf_y), int(291 * sf_x), int(21 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.saved_path_new.setFont(font)
        self.saved_path_new.setObjectName("saved_path_new")

        # Output Path Button - New
        self.output_path_new_button = QtWidgets.QPushButton(self.data_frame)
        self.output_path_new_button.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(95 * sf_y), int(241 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.output_path_new_button.setFont(font)
        self.output_path_new_button.setObjectName("output_path_new_button")
        # self.output_path_new_button.clicked.connect(self.select_output_folder_new)
        self.output_path_new_button.clicked.connect(self._on_select_output_folder)

        self.extra_label_new = QtWidgets.QLabel(self.data_frame)
        self.extra_label_new.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(140 * sf_y), int(181 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.extra_label_new.setFont(font)
        self.extra_label_new.setObjectName("extra_label_new")

        self.extrap_mode_new = QtWidgets.QComboBox(self.data_frame)
        self.extrap_mode_new.setGeometry(
            QtCore.QRect(
                int(230 * sf_x), int(140 * sf_y), int(201 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.extrap_mode_new.setFont(font)
        self.extrap_mode_new.setObjectName("extrap_mode_new")
        self.extrap_mode_new.addItem("Deep learning models", 0)
        self.extrap_mode_new.addItem("Manually", 1)

        self.ini_fract_new_label = QtWidgets.QLabel(self.data_frame)
        self.ini_fract_new_label.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(300 * sf_y), int(141 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.ini_fract_new_label.setFont(font)
        self.ini_fract_new_label.setObjectName("ini_fract_new_label")

        self.ini_fract_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.ini_fract_new_value.setGeometry(
            QtCore.QRect(
                int(180 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.ini_fract_new_value.setFont(font)
        self.ini_fract_new_value.setMaximum(0.9)
        self.ini_fract_new_value.setSingleStep(0.05)
        self.ini_fract_new_value.setProperty("value", 0.1)
        self.ini_fract_new_value.setObjectName("ini_fract_new_value")

        self.step_new_label = QtWidgets.QLabel(self.data_frame)
        self.step_new_label.setGeometry(
            QtCore.QRect(
                int(270 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.step_new_label.setFont(font)
        self.step_new_label.setObjectName("step_new_label")

        self.step_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.step_new_value.setGeometry(
            QtCore.QRect(
                int(330 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.step_new_value.setFont(font)
        self.step_new_value.setMaximum(0.33)
        self.step_new_value.setSingleStep(0.05)
        self.step_new_value.setProperty("value", 0.05)
        self.step_new_value.setObjectName("step_new_value")

        self.max_frac_new_label = QtWidgets.QLabel(self.data_frame)
        self.max_frac_new_label.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(340 * sf_y), int(121 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.max_frac_new_label.setFont(font)
        self.max_frac_new_label.setObjectName("max_frac_new_label")

        self.max_frac_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.max_frac_new_value.setGeometry(
            QtCore.QRect(
                int(160 * sf_x), int(340 * sf_y), int(71 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.max_frac_new_value.setFont(font)
        self.max_frac_new_value.setMaximum(1.0)
        self.max_frac_new_value.setSingleStep(0.05)
        self.max_frac_new_value.setProperty("value", 0.30)
        self.max_frac_new_value.setObjectName("max_frac_new_value")

        self.n_iter_new_label = QtWidgets.QLabel(self.data_frame)
        self.n_iter_new_label.setGeometry(
            QtCore.QRect(
                int(420 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.n_iter_new_label.setFont(font)
        self.n_iter_new_label.setObjectName("n_iter_new_label")

        self.n_iter_new_value = QtWidgets.QSpinBox(self.data_frame)
        self.n_iter_new_value.setGeometry(
            QtCore.QRect(
                int(690 * sf_x), int(300 * sf_y), int(51 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.n_iter_new_value.setFont(font)
        self.n_iter_new_value.setMaximum(100)
        self.n_iter_new_value.setProperty("value", 5)
        self.n_iter_new_value.setObjectName("n_iter_new_value")

        self.threshold_new = QtWidgets.QLabel(self.data_frame)
        self.threshold_new.setGeometry(
            QtCore.QRect(
                int(250 * sf_x), int(340 * sf_y), int(181 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.threshold_new.setFont(font)
        self.threshold_new.setObjectName("threshold_new")

        self.threshold_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.threshold_new_value.setGeometry(
            QtCore.QRect(
                int(430 * sf_x), int(340 * sf_y), int(71 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.threshold_new_value.setFont(font)
        self.threshold_new_value.setMaximum(0.20)
        self.threshold_new_value.setSingleStep(0.005)
        self.threshold_new_value.setProperty("value", 0.05)
        self.threshold_new_value.setObjectName("threshold_new_value")

        # --- Features of Interest filter (Excel-like) ---
        self.features_label = QtWidgets.QLabel(self.data_frame)
        self.features_label.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(220 * sf_y), int(181 * sf_x), int(31 * sf_y)
            )
        )
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        self.features_label.setFont(font)
        self.features_label.setText("Features of interest:")

        self.features_btn = CheckFilterButton(
            values=[""], parent=self.data_frame, text="Feature strata"
        )
        self.features_btn.setGeometry(
            QtCore.QRect(
                int(230 * sf_x), int(220 * sf_y), int(161 * sf_x), int(31 * sf_y)
            )
        )

        self.features_selected = QtWidgets.QLabel(self.data_frame)
        self.features_selected.setGeometry(
            QtCore.QRect(
                int(40 * sf_x), int(255 * sf_y), int(540 * sf_x), int(31 * sf_y)
            )
        )
        self.features_selected.setText("Selected: ----")

        self.features_btn.selection_changed.connect(self.on_features_changed)
        self.feature_strata = []  # initialize

        self.w_title.setText("Setting Input Files")
        self.save_button.setText("Save and Continue")
        self.output_label_new.setText("Output name:")
        self.output_new_value.setText("rubic_ai")
        self.population_new_button.setText("Upload population data")
        self.population_new_path.setText("filename.csv")
        self.saved_path_new.setText("path/where/save/the/results")
        self.output_path_new_button.setText("Select output folder")
        self.extra_label_new.setText("Extrapolation mode:")
        self.ini_fract_new_label.setText("Initial fraction:")
        self.step_new_label.setText("Step:")
        self.max_frac_new_label.setText("Max fraction:")
        self.n_iter_new_label.setText("N° iter:")
        self.threshold_new.setText("Maximum threshold:")

        # ==============================================================
        # Stratified method functions
        # ==============================================================

    def select_method(self):
        """Set the extrapolation mode and close the dialog."""
        if self.extrap_mode_new.currentData() == 0:
            # stratified deep learning
            self.stratified_mode = 0
            self.accept()
        else:
            # stratified manually
            self.stratified_mode = 1
            self.accept()

    def _on_upload_csv(self):
        """Open a CSV file and update the interface."""
        upload_csv(self)

    def _on_select_output_folder(self):
        """Select and display the output folder."""
        self.folder_path_new, self.display_folder = select_output_folder(self)
        self.saved_path_new.setText(self.display_folder)

    def data_population(self):
        """Load population data and update feature options."""
        self.population_data = upload_csv(self)
        unique_vals = [
            "material",
            "llrs",
            "code_level",
            "n_stories",
            "occupancy",
            "block_position",
            "roof_shape",
            "roof_material",
        ]
        self.features_btn.set_values(unique_vals)

    def on_features_changed(self, vals):
        """Update and display the selected stratification features."""
        self.feature_strata = list(vals)  # keep a copy
        self.features_selected.setText(
            "Selected: " + (", ".join(vals) if vals else "(none)")
        )


# ==============================================================
# Filter Selector by Feature
# ==============================================================


class CheckFilterPopup(QtWidgets.QWidget):
    """Display a searchable checklist in a popup widget."""

    # Signal emitted when the user confirms the selection.
    # It sends a list with the selected values.
    selection_changed = QtCore.pyqtSignal(list)

    def __init__(self, values=None, parent=None):
        """Popup widget that behaves like a filterable checklist.

        Main purpose:
        - Display a small popup window with a search bar
        - Allow the user to check/uncheck multiple values
        - Include a '(Select All)' option for visible items
        - Emit the final selected values when the user clicks OK
        """
        super().__init__(parent, flags=QtCore.Qt.Popup)

        # Configure the popup appearance:
        # - Popup behavior closes naturally when focus is lost
        # - Frameless style gives a cleaner dropdown-like look
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint, True)
        self.resize(220, 280)

        # Main vertical layout for all controls inside the popup
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(6, 6, 6, 6)

        # Search field used to filter the list of available values
        self.search = QtWidgets.QLineEdit(self)
        self.search.setPlaceholderText("Search")
        self.search.setClearButtonEnabled(True)
        vbox.addWidget(self.search)

        # Tri-state checkbox:
        # - Checked: all visible items selected
        # - Unchecked: no visible items selected
        # - PartiallyChecked: only some visible items selected
        self.selectAll = QtWidgets.QCheckBox("(Select All)", self)
        self.selectAll.setTristate(True)
        vbox.addWidget(self.selectAll)

        # List widget that shows all possible values as checkable items
        self.list = QtWidgets.QListWidget(self)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        vbox.addWidget(self.list, 1)

        # OK / Cancel buttons:
        # - OK confirms selection and emits the signal
        # - Cancel closes the popup without emitting anything
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self
        )
        btns.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f0f0f0;
                border: 1px solid #999;
                padding: 4px 12px;
            }
        """)
        vbox.addWidget(btns)

        # Internal storage of all unique values currently loaded
        self._all_values = []

        # If initial values are provided, populate the list
        if values:
            self.set_values(values)

        # Connections between UI events and internal logic
        self.search.textChanged.connect(self._apply_filter)
        self.selectAll.stateChanged.connect(self._toggle_all)
        self.list.itemChanged.connect(self._update_select_all_state)
        btns.accepted.connect(self._emit_and_close)
        btns.rejected.connect(self.close)

    def set_values(self, values):
        """Load values into the popup list.

        Steps:
        - Convert all values to string
        - Remove None values
        - Remove duplicates
        - Sort them alphabetically
        - Rebuild the checklist
        """
        uniq = sorted({str(v) for v in values if v is not None})
        self._all_values = uniq
        self._rebuild_list(uniq)

    def selected_values(self):
        """Return the currently checked values that are visible in the list.

        Note:
        Hidden items are excluded, which means this function respects
        the current search/filter state.
        """
        vals = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == QtCore.Qt.Checked and not it.isHidden():
                vals.append(it.text())
        return vals

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _rebuild_list(self, vals):
        """Recreate the checklist items from scratch.

        Each value is added as a checkable item and is checked by default.
        Signals are temporarily blocked to avoid triggering update logic
        during the rebuilding process.
        """
        self.list.blockSignals(True)
        self.list.clear()

        for txt in vals:
            it = QtWidgets.QListWidgetItem(txt)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            self.list.addItem(it)

        self.list.blockSignals(False)
        self._update_select_all_state()

    def _apply_filter(self, text):
        """Filter the list items according to the search text.

        Behavior:
        - Converts the search text to lowercase for case-insensitive matching
        - Hides items that do not contain the typed text
        - Keeps matching items visible
        - Updates the '(Select All)' state based only on visible items
        """
        text = text.strip().lower()

        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(text not in it.text().lower())
        self.list.blockSignals(False)

        self._update_select_all_state()

    def _visible_items(self):
        """Return a list of all items that are currently visible.

        This is useful because filtering hides some items, and many actions
        such as 'Select All' should only affect visible ones.
        """
        return [
            self.list.item(i)
            for i in range(self.list.count())
            if not self.list.item(i).isHidden()
        ]

    def _toggle_all(self, state):
        """Check or uncheck all visible items depending on the '(Select All)' state.

        Important:
        - If the checkbox is in PartiallyChecked state, do nothing
          because that state is informative, not an action target.
        - Only visible items are modified.
        """
        if state == QtCore.Qt.PartiallyChecked:
            return

        target = (
            QtCore.Qt.Checked if state == QtCore.Qt.Checked else QtCore.Qt.Unchecked
        )

        self.list.blockSignals(True)
        for it in self._visible_items():
            it.setCheckState(target)
        self.list.blockSignals(False)

        self._update_select_all_state()

    def _update_select_all_state(self):
        """Update the tri-state '(Select All)' checkbox based on visible items.

        Logic:
        - If no visible items exist, set it to Unchecked
        - If all visible items are checked, set it to Checked
        - If some are checked, set it to PartiallyChecked
        - If none are checked, set it to Unchecked
        """
        vis = self._visible_items()

        if not vis:
            self.selectAll.setCheckState(QtCore.Qt.Unchecked)
            return

        checks = [it.checkState() == QtCore.Qt.Checked for it in vis]

        if all(checks):
            self.selectAll.setCheckState(QtCore.Qt.Checked)
        elif any(checks):
            self.selectAll.setCheckState(QtCore.Qt.PartiallyChecked)
        else:
            self.selectAll.setCheckState(QtCore.Qt.Unchecked)

    def _emit_and_close(self):
        """Emit the final selected values and close the popup.

        This method is called when the user clicks OK.
        """
        self.selection_changed.emit(self.selected_values())
        self.close()


class CheckFilterButton(QtWidgets.QToolButton):
    """Open and manage a searchable checklist popup."""

    # Signal re-emitted from the popup so external widgets can react
    # when the selection is confirmed.
    selection_changed = QtCore.pyqtSignal(list)

    def __init__(self, values=None, parent=None, text="Feature strata"):
        """Tool button that opens the CheckFilterPopup.

        Main purpose:
        - Act as the visible control in the interface
        - Open the popup below the button when clicked
        - Provide a simple way to access the selected values
        - Re-emit the popup selection_changed signal
        """
        super().__init__(parent)

        # Configure button text and style
        self.setText(text)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setArrowType(QtCore.Qt.DownArrow)

        # Create the popup associated with this button
        self._popup = CheckFilterPopup(values, self)

        # When the button is clicked, manually show the popup
        self.clicked.connect(self._show_popup)

        # Re-emit the popup signal so users of this button do not need
        # to access the popup object directly
        self._popup.selection_changed.connect(self.selection_changed)

    def set_values(self, values):
        """Update the values available in the popup checklist."""
        self._popup.set_values(values)

    def selected_values(self):
        """Return the values currently selected in the popup."""
        return self._popup.selected_values()

    def _show_popup(self):
        """Show the popup directly below the button.

        Steps:
        - Convert the button's local position to global screen coordinates
        - Move the popup just under the button
        - Display the popup
        """
        pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        self._popup.move(pos)
        self._popup.show()


# ==============================================================================
# Stratified function for calling gui_methods.py
# ==============================================================================


# ========== Iterative sampling using existing labels ==========
def iterative_distribution_stability_manual(
    id_feature,
    data: pd.DataFrame,
    id_column,
    initial_fraction,
    step_fraction,
    max_fraction,
    stability_threshold,
    max_iterations,
    random_state: int = 42,
):
    """Sample data iteratively until feature proportions stabilize.

    Stop when the class proportions are stable or the iteration limit is reached.
    """
    population_size = len(data)
    all_sampled = pd.DataFrame(columns=data.columns)
    previous_dist = None
    iteration = 0

    # Shuffle dataset
    np.random.seed(random_state)
    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )
    while iteration < max_iterations:
        current_fraction = min(
            initial_fraction + step_fraction * iteration, max_fraction
        )
        target_size = int(population_size * current_fraction)

        # Select next sample
        remaining = shuffled_data[
            ~shuffled_data[id_column].isin(all_sampled[id_column])
        ]
        next_sample = remaining.head(target_size - len(all_sampled))

        if next_sample.empty:
            break

        all_sampled = pd.concat([all_sampled, next_sample], ignore_index=True)

        # Compute current distribution
        current_counts = (
            all_sampled[id_feature].value_counts(normalize=True).sort_index()
        )
        current_dist = current_counts.to_dict()

        # Check stabilization
        if previous_dist is not None:
            all_keys = set(previous_dist) | set(current_dist)
            max_change = max(
                abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys
            )
            print(
                f"Iteration {iteration + 1}: "
                f"Sample size = {len(all_sampled)}, "
                f"Max Δ = {max_change:.4f}"
            )

            if max_change < stability_threshold:
                print("✅ Class proportions stabilized.")
                return all_sampled, current_dist, len(all_sampled)

        previous_dist = current_dist
        iteration += 1

    print("⚠️ Reached max iterations or sample limit without convergence.")
    return all_sampled, current_dist, len(all_sampled)


# ========== Main Iterative Sampling Function ==========
def iterative_label_discovery_cached_fractional(
    id_feature,
    data: pd.DataFrame,
    labeling_function,
    id_column: str = "id",
    initial_fraction: float = 0.10,
    step_fraction: float = 0.05,
    max_fraction: float = 1.00,
    stability_threshold: float = 0.05,
    max_iterations: int = 20,
    random_state: int = 42,
):
    """Label increasing data fractions until label proportions stabilize.

    Stop when the proportions are stable or the iteration limit is reached.
    """
    population_size = len(data)
    all_labeled = pd.DataFrame(
        columns=[id_column, id_feature]
    )  # Initialize labeled dataset
    previous_dist = None  # Store label distribution from previous iteration
    iteration = 0  # Iteration counter
    # Shuffle the dataset for randomized sampling
    np.random.seed(random_state)
    shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )
    # === Iterative sampling loop ===
    while iteration < max_iterations:
        # Calculate current target sample size
        current_fraction = min(
            initial_fraction + step_fraction * iteration, max_fraction
        )
        target_size = min(int(population_size * current_fraction), population_size)
        # Filter out already labeled IDs and select next batch
        already_labeled_ids = set(all_labeled[id_column])
        next_sample = shuffled_data[
            ~shuffled_data[id_column].isin(already_labeled_ids)
        ].head(target_size - len(all_labeled))
        if next_sample.empty:
            break  # Stop if no more samples to process
        # Apply labeling function to new samples
        next_sample[id_feature] = next_sample[id_column].apply(labeling_function)
        all_labeled = pd.concat([all_labeled, next_sample], ignore_index=True)

        # Calculate class distribution
        current_counts = (
            all_labeled[id_feature].value_counts(normalize=True).sort_index()
        )
        current_dist = current_counts.to_dict()

        # Check for stabilization in label distribution
        if previous_dist is not None:
            all_keys = set(previous_dist) | set(current_dist)
            print("Previous dist: ")
            print(previous_dist)
            max_change = max(
                abs(previous_dist.get(k, 0) - current_dist.get(k, 0)) for k in all_keys
            )
            print(
                f"Iteration {iteration + 1}: "
                f"Sample size = {len(all_labeled)}, "
                f"Max Δ = {max_change:.4f}"
            )

            if max_change < stability_threshold:
                print("✅ Class proportions stabilized.")
                return all_labeled, current_dist, len(all_labeled)

        previous_dist = current_dist
        iteration += 1

    print("⚠️ Reached max iterations or sample limit without convergence.")
    return all_labeled, current_dist, len(all_labeled)


def _is_valid_img(arr):
    """Return True if arr is a non-empty HxWx3 uint8 NumPy image."""
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim == 3
        and arr.shape[2] in (3, 4)
        and arr.size > 0
    )


# ========== Labeling Function ==========
def labeling_function(image_id, id_feature, data_building):
    """Predict one building attribute from a Street View image."""
    row = data_building.loc[data_building["id"] == image_id, ["latitude", "longitude"]]
    if row.empty:
        print(f"[WARN] Missing lat/lon for id={image_id}")
        return None

    location = (
        float(row.iloc[0]["latitude"]),
        float(row.iloc[0]["longitude"]),
    )

    with open("methods/gsv_api_key.txt", encoding="utf-8") as file:
        api_key = file.read().strip()

    try:
        _url, result, _year = get_street_view_image(location, api_key, 0, 5, 120)
        image = result[1] if isinstance(result, tuple) else result
    except (
        AttributeError,
        ConnectionError,
        IndexError,
        OSError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as error:
        print(f"[WARN] GSV fetch failed at {location}: {error}")
        return None

    if not _is_valid_img(image):
        return None

    prediction_map = {
        "material": (
            predict_material_img,
            ["CR", "HYB(MCF;MUR)", "INF", "MCF", "MR", "MUR", "S", "W"],
        ),
        "llrs": (
            predict_llrs_img,
            ["LDUAL", "LFBR", "LFINF", "LFM", "LN", "LWAL", "LWAL"],
        ),
        "code_level": (
            predict_code_img,
            ["CDH", "CDL", "CDM", "CDN"],
        ),
        "n_stories": (
            predict_n_stories_img,
            ["10-12", "13+", "1", "2", "3", "4", "5", "6-7", "8-9"],
        ),
        "occupancy": (
            predict_occupancy_img,
            ["COM", "IND", "MIX(RES;COM)", "RES"],
        ),
        "block_position": (
            predict_block_position_img,
            ["BP1", "BP2", "BP3", "BPD"],
        ),
        "roof_shape": (
            predict_roof_shape_img,
            ["RSH1", "RSH2", "RSH3", "RSH5", "RSH7"],
        ),
        "roof_material": (
            predict_roof_material_img,
            ["RMN", "RMT1", "RMT6"],
        ),
    }

    try:
        predictor, classes = prediction_map[id_feature]
    except KeyError as error:
        raise ValueError(f"Unsupported feature: {id_feature}") from error

    prediction_index = predictor(image, 3, None, None)
    return classes[prediction_index]
