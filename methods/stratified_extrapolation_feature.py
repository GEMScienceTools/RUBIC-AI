import os
import sys 
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd


class stratified_extrapolation(QtWidgets.QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window

         ## Get screen resolution
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
        self.setObjectName("DataSetting")
        self.resize(int(640 * sf_x), int(680 * sf_y))
        self.setWindowTitle("Setting input files")
        
        # === UI Elements Start ===
        self.data_frame = QtWidgets.QWidget(self)
        self.data_frame.setObjectName("data_frame")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.data_frame)
        
        # Title Label
        self.w_title = QtWidgets.QLabel(self.data_frame)
        self.w_title.setGeometry(QtCore.QRect(int(220 * sf_x), int(0 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(12 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.w_title.setFont(font)
        self.w_title.setObjectName("w_title")
        
        # Save Button
        self.save_button = QtWidgets.QPushButton(self.data_frame)
        self.save_button.setGeometry(QtCore.QRect(int(210 * sf_x), int(620 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.save_button.setFont(font)
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self.select_method)
        
        # Table Widget
        self.tableWidget = QtWidgets.QTableWidget(self.data_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(10 * sf_x), int(400 * sf_y), int(590 * sf_x), int(211 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        # Output Label - New
        self.output_label_new = QtWidgets.QLabel(self.data_frame)
        self.output_label_new.setGeometry(QtCore.QRect(int(40 * sf_x), int(55 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.output_label_new.setFont(font)
        self.output_label_new.setObjectName("output_label_new")
        
        # Output Value - New
        self.output_new_value = QtWidgets.QLineEdit(self.data_frame)
        self.output_new_value.setGeometry(QtCore.QRect(int(180 * sf_x), int(55 * sf_y), int(331 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.output_new_value.setFont(font)
        self.output_new_value.setObjectName("output_new_value")
        
        # Background 2
        self.backg_2 = QtWidgets.QLabel(self.data_frame)
        self.backg_2.setGeometry(QtCore.QRect(int(10 * sf_x), int(40 * sf_y), int(591 * sf_x), int(351 * sf_y)))
        self.backg_2.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.backg_2.setText("")
        self.backg_2.setObjectName("backg_2")
        self.backg_2.lower()
        
        # Population Button - New
        self.population_new_button = QtWidgets.QPushButton(self.data_frame)
        self.population_new_button.setGeometry(QtCore.QRect(int(40 * sf_x), int(185 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.population_new_button.setFont(font)
        self.population_new_button.setObjectName("population_new_button")
        self.population_new_button.clicked.connect(self.data_population)
        
        # Population Path - New
        self.population_new_path = QtWidgets.QLabel(self.data_frame)
        self.population_new_path.setGeometry(QtCore.QRect(int(290 * sf_x), int(190 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.population_new_path.setFont(font)
        self.population_new_path.setObjectName("population_new_path")
        
        # Saved Path - New
        self.saved_path_new = QtWidgets.QLabel(self.data_frame)
        self.saved_path_new.setGeometry(QtCore.QRect(int(290 * sf_x), int(100 * sf_y), int(291 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.saved_path_new.setFont(font)
        self.saved_path_new.setObjectName("saved_path_new")
        
        # Output Path Button - New
        self.output_path_new_button = QtWidgets.QPushButton(self.data_frame)
        self.output_path_new_button.setGeometry(QtCore.QRect(int(40 * sf_x), int(95 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(False)
        font.setWeight(50)
        self.output_path_new_button.setFont(font)
        self.output_path_new_button.setObjectName("output_path_new_button")
        self.output_path_new_button.clicked.connect(self.select_output_folder_new)
        
        self.extra_label_new = QtWidgets.QLabel(self.data_frame)
        self.extra_label_new.setGeometry(QtCore.QRect(int(40 * sf_x), int(140 * sf_y), int(181 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.extra_label_new.setFont(font)
        self.extra_label_new.setObjectName("extra_label_new")
        
        self.extrap_mode_new = QtWidgets.QComboBox(self.data_frame)
        self.extrap_mode_new.setGeometry(QtCore.QRect(int(230 * sf_x), int(140 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.extrap_mode_new.setFont(font)
        self.extrap_mode_new.setObjectName("extrap_mode_new")
        self.extrap_mode_new.addItem("Deep learning models", 0)
        self.extrap_mode_new.addItem("Manually", 1)
        
        self.ini_fract_new_label = QtWidgets.QLabel(self.data_frame)
        self.ini_fract_new_label.setGeometry(QtCore.QRect(int(40 * sf_x), int(300 * sf_y), int(141 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.ini_fract_new_label.setFont(font)
        self.ini_fract_new_label.setObjectName("ini_fract_new_label")
        
        self.ini_fract_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.ini_fract_new_value.setGeometry(QtCore.QRect(int(180 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.ini_fract_new_value.setFont(font)
        self.ini_fract_new_value.setMaximum(0.9)
        self.ini_fract_new_value.setSingleStep(0.05)
        self.ini_fract_new_value.setProperty("value", 0.1)
        self.ini_fract_new_value.setObjectName("ini_fract_new_value")
        
        self.step_new_label = QtWidgets.QLabel(self.data_frame)
        self.step_new_label.setGeometry(QtCore.QRect(int(270 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.step_new_label.setFont(font)
        self.step_new_label.setObjectName("step_new_label")
        
        self.step_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.step_new_value.setGeometry(QtCore.QRect(int(330 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.step_new_value.setFont(font)
        self.step_new_value.setMaximum(0.33)
        self.step_new_value.setSingleStep(0.05)
        self.step_new_value.setProperty("value", 0.05)
        self.step_new_value.setObjectName("step_new_value")
        
        self.max_frac_new_label = QtWidgets.QLabel(self.data_frame)
        self.max_frac_new_label.setGeometry(QtCore.QRect(int(40 * sf_x), int(340 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.max_frac_new_label.setFont(font)
        self.max_frac_new_label.setObjectName("max_frac_new_label")
        
        self.max_frac_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.max_frac_new_value.setGeometry(QtCore.QRect(int(160 * sf_x), int(340 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.max_frac_new_value.setFont(font)
        self.max_frac_new_value.setMaximum(1.0)
        self.max_frac_new_value.setSingleStep(0.05)
        self.max_frac_new_value.setProperty("value", 0.30)
        self.max_frac_new_value.setObjectName("max_frac_new_value")
        
        self.n_iter_new_label = QtWidgets.QLabel(self.data_frame)
        self.n_iter_new_label.setGeometry(QtCore.QRect(int(420 * sf_x), int(300 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.n_iter_new_label.setFont(font)
        self.n_iter_new_label.setObjectName("n_iter_new_label")
        
        self.n_iter_new_value = QtWidgets.QSpinBox(self.data_frame)
        self.n_iter_new_value.setGeometry(QtCore.QRect(int(690 * sf_x), int(300 * sf_y), int(51 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.n_iter_new_value.setFont(font)
        self.n_iter_new_value.setMaximum(100)
        self.n_iter_new_value.setProperty("value", 5)
        self.n_iter_new_value.setObjectName("n_iter_new_value")
        
        self.threshold_new = QtWidgets.QLabel(self.data_frame)
        self.threshold_new.setGeometry(QtCore.QRect(int(250 * sf_x), int(340 * sf_y), int(181 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.threshold_new.setFont(font)
        self.threshold_new.setObjectName("threshold_new")
        
        self.threshold_new_value = QtWidgets.QDoubleSpinBox(self.data_frame)
        self.threshold_new_value.setGeometry(QtCore.QRect(int(430 * sf_x), int(340 * sf_y), int(71 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.threshold_new_value.setFont(font)
        self.threshold_new_value.setMaximum(0.20)
        self.threshold_new_value.setSingleStep(0.005)
        self.threshold_new_value.setProperty("value", 0.05)
        self.threshold_new_value.setObjectName("threshold_new_value")
        
        # --- Features of Interest filter (Excel-like) ---
        self.features_label = QtWidgets.QLabel(self.data_frame)
        self.features_label.setGeometry(QtCore.QRect(int(40 * sf_x), int(220 * sf_y), int(181 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        self.features_label.setFont(font)
        self.features_label.setText("Features of interest:")
        
        self.features_btn = CheckFilterButton(values=[""], parent=self.data_frame, text="Feature strata")
        self.features_btn.setGeometry(QtCore.QRect(int(230 * sf_x), int(220 * sf_y), int(161 * sf_x), int(31 * sf_y)))
        
        self.features_selected = QtWidgets.QLabel(self.data_frame)
        self.features_selected.setGeometry(QtCore.QRect(int(40 * sf_x), int(255 * sf_y), int(540 * sf_x), int(31 * sf_y)))
        self.features_selected.setText("Selected: ----")
        
        self.features_btn.selectionChanged.connect(self.on_features_changed)
        self.feature_strata = []   # initialize

        self.w_title.setText("Setting input files")
        self.save_button.setText("Save and continue")
        self.output_label_new.setText("Output name:")
        self.output_new_value.setText("new_strat")
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
        
    def select_method(self):
        if self.extrap_mode_new.currentData() == 0:
            # stratified deep learning
            self.stratified_mode = 0
            self.accept()
        else:
            self.stratified_mode = 1
            # stratified manually
            self.accept()
                
    def preview_data(self, database):
        if hasattr(self, 'df') and not self.df.empty:
            preview_df = database.head(10)  # Only show first 10 rows

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
            QtWidgets.QMessageBox.warning(self, "No Data", "No data available to preview. Please upload a valid CSV first.")

    def upload_csv(self, label):
        
        options = QtWidgets.QFileDialog.Options()
        # File path
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        # Send path to UI
        self.file_local_csv = file_path
            
        if file_path:
            try:
                # Upload csv with building coordinates
                self.df = pd.read_csv(file_path)
                file_path = file_path.rsplit("/", 1)[-1]
                label.setText(file_path)
            except FileNotFoundError:
                label.setText("File not found.")
            except pd.errors.ParserError:
                label.setText("Error parsing CSV file. Check the format.")
            except Exception as e: #catch other exceptions
                label.setText(f"An error occurred: {e}")
        else:
            label.setText("No file selected.")
            QtWidgets.QMessageBox.warning(self, "Input Error", "No file selected.")

        return self.df
    
    def select_output_folder_new(self):
        """Open a folder selection dialog and display the selected folder in a text output."""
        self.folder_path_new = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
        if self.folder_path_new:  # If a folder is selected
            folder_display = os.path.basename(self.folder_path_new)    
            self.saved_path_new.setText(folder_display)
    
    def data_population(self):
        self.data_population = self.upload_csv(self.population_new_path)
        self.preview_data(self.data_population)
        unique_vals = ["material", "llrs", "code_level","n_stories","occupancy","block_position",
                       "roof_shape", "roof_material"]
        self.features_btn.set_values(unique_vals)
 
    def on_features_changed(self, vals):
        self.feature_strata = list(vals)  # keep a copy
        self.features_selected.setText(
            "Selected: " + (", ".join(vals) if vals else "(none)")
        )

class CheckFilterPopup(QtWidgets.QWidget):
    selectionChanged = QtCore.pyqtSignal(list)

    def __init__(self, values=None, parent=None):
        super().__init__(parent, flags=QtCore.Qt.Popup)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint, True)
        self.resize(220, 280)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(6, 6, 6, 6)

        self.search = QtWidgets.QLineEdit(self)
        self.search.setPlaceholderText("Search")
        self.search.setClearButtonEnabled(True)
        vbox.addWidget(self.search)

        self.selectAll = QtWidgets.QCheckBox("(Select All)", self)
        self.selectAll.setTristate(True)
        vbox.addWidget(self.selectAll)

        self.list = QtWidgets.QListWidget(self)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        vbox.addWidget(self.list, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                          QtWidgets.QDialogButtonBox.Cancel, self)
        vbox.addWidget(btns)

        self._all_values = []
        if values:
            self.set_values(values)

        self.search.textChanged.connect(self._apply_filter)
        self.selectAll.stateChanged.connect(self._toggle_all)
        self.list.itemChanged.connect(self._update_select_all_state)
        btns.accepted.connect(self._emit_and_close)
        btns.rejected.connect(self.close)

    def set_values(self, values):
        uniq = sorted({str(v) for v in values if v is not None})
        self._all_values = uniq
        self._rebuild_list(uniq)

    def selected_values(self):
        vals = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == QtCore.Qt.Checked and not it.isHidden():
                vals.append(it.text())
        return vals

    # internals
    def _rebuild_list(self, vals):
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
        text = text.strip().lower()
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(text not in it.text().lower())
        self.list.blockSignals(False)
        self._update_select_all_state()

    def _visible_items(self):
        return [self.list.item(i) for i in range(self.list.count()) if not self.list.item(i).isHidden()]

    def _toggle_all(self, state):
        if state == QtCore.Qt.PartiallyChecked:
            return
        target = QtCore.Qt.Checked if state == QtCore.Qt.Checked else QtCore.Qt.Unchecked
        self.list.blockSignals(True)
        for it in self._visible_items():
            it.setCheckState(target)
        self.list.blockSignals(False)
        self._update_select_all_state()

    def _update_select_all_state(self):
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
        self.selectionChanged.emit(self.selected_values())
        self.close()


class CheckFilterButton(QtWidgets.QToolButton):
    selectionChanged = QtCore.pyqtSignal(list)

    def __init__(self, values=None, parent=None, text="Feature strata"):
        super().__init__(parent)
        self.setText(text)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setArrowType(QtCore.Qt.DownArrow)
        self._popup = CheckFilterPopup(values, self)
        self.clicked.connect(self._show_popup)
        self._popup.selectionChanged.connect(self.selectionChanged)

    def set_values(self, values):
        self._popup.set_values(values)

    def selected_values(self):
        return self._popup.selected_values()

    def _show_popup(self):
        pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        self._popup.move(pos)
        self._popup.show()
