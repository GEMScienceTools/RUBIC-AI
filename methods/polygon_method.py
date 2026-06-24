from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import os
import sys
import numpy as np
from typing import Optional, Union
    
from methods.gui_gis import GUI_geofiles
from methods.utilities import select_output_folder, preview_data, upload_csv, mode_use, save_coordinates

class PolygonSetting(QtWidgets.QDialog):
    def __init__(self, parent=None, method=None):
        super().__init__(parent)
        self.method = method

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
        self.sf_font = sf_font
        self.setWindowTitle("Polygon Method Input")
        self.setWindowIcon(QtGui.QIcon("help_img/RUBIC_logo.png"))
        self.resize(int(618 * sf_x), int(708 * sf_y))

        self.coord_frame = QtWidgets.QWidget(self)
        # Background color
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.backg_1 = QtWidgets.QLabel(self.coord_frame)
        self.backg_1.setGeometry(QtCore.QRect(int(10 * sf_x), int(10 * sf_y), int(601 * sf_x), int(651 * sf_y)))
        self.backg_1.setStyleSheet("background-color: rgb(255, 224, 185);")
        
        # Title
        self.polygon_label = QtWidgets.QLabel(self.coord_frame)
        self.polygon_label.setGeometry(QtCore.QRect(int(210 * sf_x), int(20 * sf_y), int(221 * sf_x), int(21 * sf_y)))
        title_font = QtGui.QFont()
        title_font.setPointSize(int(10 * sf_font))
        title_font.setBold(True)
        title_font.setItalic(True)
        title_font.setUnderline(True)
        self.polygon_label.setFont(title_font)
        self.polygon_label.setText("Polygon Method Input")

        # output label
        self.output_label_polygon = QtWidgets.QLabel(self.coord_frame)
        self.output_label_polygon.setGeometry(QtCore.QRect(int(20 * sf_x), int(50 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        label_font = QtGui.QFont()
        label_font.setPointSize(int(10 * sf_font))
        label_font.setBold(True)
        self.output_label_polygon.setFont(label_font)
        self.output_label_polygon.setText("Output name:")

        self.output_polygon = QtWidgets.QLineEdit(self.coord_frame)
        self.output_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(50 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.output_polygon.setFont(font)
        self.output_polygon.setText("polygon_building")

        self.path_out_folder_bt = QtWidgets.QPushButton(self.coord_frame)
        self.path_out_folder_bt.setGeometry(QtCore.QRect(int(20 * sf_x), int(95 * sf_y), int(231 * sf_x), int(31 * sf_y)))
        self.path_out_folder_bt.setFont(font)
        self.path_out_folder_bt.setText("Select output folder")
        self.path_out_folder_bt.clicked.connect(self._on_select_output_folder)

        self.output_folder_value = QtWidgets.QLabel(self.coord_frame)
        self.output_folder_value.setGeometry(QtCore.QRect(int(270 * sf_x), int(100 * sf_y), int(331 * sf_x), int(21 * sf_y)))
        self.output_folder_value.setFont(font)
        self.output_folder_value.setText("path/where/you/want/to/save/your/results")
        
        self.polygon_source_label = QtWidgets.QLabel(self.coord_frame)
        self.polygon_source_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(140 * sf_y), int(151 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.polygon_source_label.setFont(font)
        self.polygon_source_label.setObjectName("polygon_source_label")
        self.polygon_source_label.setText("Polygon source:")
        
        self.polygon_source_value = QtWidgets.QComboBox(self.coord_frame)
        self.polygon_source_value.setGeometry(QtCore.QRect(int(170 * sf_x), int(140 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.polygon_source_value.setFont(font)
        self.polygon_source_value.setObjectName("polygon_source_value")
        self.polygon_source_value.addItem("Vertex coordinates", 0)
        self.polygon_source_value.addItem("Existing polygon", 1)
        
        self.input_button_polygon = QtWidgets.QPushButton(self.coord_frame)
        self.input_button_polygon.setGeometry(QtCore.QRect(int(20 * sf_x), int(190 * sf_y), int(201 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.input_button_polygon.setFont(font)
        self.input_button_polygon.setObjectName("input_button_polygon")
        self.input_button_polygon.setText("Upload polygon file")
        self.input_button_polygon.clicked.connect(self.upload_input)
        
        self.polygon_path_value = QtWidgets.QLabel(self.coord_frame)
        self.polygon_path_value.setGeometry(QtCore.QRect(int(240 * sf_x), int(190 * sf_y), int(291 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.polygon_path_value.setFont(font)
        self.polygon_path_value.setObjectName("polygon_path_value")
        self.polygon_path_value.setText("filename.csv or polygon.gpkg")
        
        # Footprint Mode Label
        self.footprint_label = QtWidgets.QLabel(self.coord_frame)
        self.footprint_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(240 * sf_y), int(151 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.footprint_label.setFont(font)
        self.footprint_label.setObjectName("footprint_label")
        self.footprint_label.setText("Footprint source:")
           
        # Footprint Mode ComboBox
        self.footprint_mode = QtWidgets.QComboBox(self.coord_frame)
        self.footprint_mode.setGeometry(QtCore.QRect(int(180 * sf_x), int(240 * sf_y), int(241 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.footprint_mode.setFont(font)
        self.footprint_mode.setObjectName("footprint_mode")
        self.footprint_mode.addItem("OpenStreetMap",0)
        self.footprint_mode.addItem("Overture",1)

        self.load_data_button = QtWidgets.QPushButton(self.coord_frame)
        self.load_data_button.setGeometry(QtCore.QRect(int(20 * sf_x), int(290 * sf_y), int(271 * sf_x), int(31 * sf_y)))
        self.load_data_button.setFont(label_font)
        self.load_data_button.setText("Get footprints available")
        self.load_data_button.clicked.connect(self._on_save_coordinates)
        self.load_data_button.clicked.connect(self.building_polulation)
        
        self.footprint_progress_label = QtWidgets.QLabel(self.coord_frame)
        self.footprint_progress_label.setGeometry(QtCore.QRect(int(450 * sf_x), int(290 * sf_y), int(151 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.footprint_progress_label.setFont(font)
        self.footprint_progress_label.setObjectName("footprint_progress_label")
        self.footprint_progress_label.setText("status")

        self.footprint_progress = QtWidgets.QProgressBar(self.coord_frame)
        self.footprint_progress.setGeometry(QtCore.QRect(int(310 * sf_x), int(295 * sf_y), int(121 * sf_x), int(21 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.footprint_progress.setFont(font)
        self.footprint_progress.setProperty("value", 0)
        self.footprint_progress.setObjectName("footprint_progress")
        
        self.n_building = QtWidgets.QLabel(self.coord_frame)
        self.n_building.setGeometry(QtCore.QRect(int(20 * sf_x), int(330 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        self.n_building.setFont(label_font)
        self.n_building.setText("N° footprints:")

        self.building_value_polygon = QtWidgets.QLabel(self.coord_frame)
        self.building_value_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(330 * sf_y), int(211 * sf_x), int(31 * sf_y)))
        self.building_value_polygon.setFont(font)
        self.building_value_polygon.setText("0000")

        self.sample_size = QtWidgets.QLabel(self.coord_frame)
        self.sample_size.setGeometry(QtCore.QRect(int(20 * sf_x), int(370 * sf_y), int(121 * sf_x), int(31 * sf_y)))
        self.sample_size.setFont(label_font)
        self.sample_size.setText("Sample size:")

        self.sample_size_polygon = QtWidgets.QLineEdit(self.coord_frame)
        self.sample_size_polygon.setGeometry(QtCore.QRect(int(160 * sf_x), int(370 * sf_y), int(111 * sf_x), int(31 * sf_y)))
        self.sample_size_polygon.setFont(font)
        self.sample_size_polygon.setText("10")

        self.collection_mode = QtWidgets.QComboBox(self.coord_frame)
        self.collection_mode.setGeometry(QtCore.QRect(int(250 * sf_x), int(410 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        self.collection_mode.setFont(font)
        self.collection_mode.setObjectName("collection_mode")
        self.collection_mode.addItem("Manual")
        self.collection_mode.addItem("AI Powered")
        
        self.feature_collection_label = QtWidgets.QLabel(self.coord_frame)
        self.feature_collection_label.setGeometry(QtCore.QRect(int(20 * sf_x), int(410 * sf_y), int(221 * sf_x), int(31 * sf_y)))
        font = QtGui.QFont()
        font.setPointSize(int(10 * sf_font))
        font.setBold(True)
        font.setWeight(75)
        self.feature_collection_label.setFont(font)
        self.feature_collection_label.setObjectName("feature_collection_label")
        self.feature_collection_label.setText("Feature collection mode:")
        
        self.tableWidget = QtWidgets.QTableWidget(self.coord_frame)
        self.tableWidget.setGeometry(QtCore.QRect(int(20 * sf_x), int(450 * sf_y), int(581 * sf_x), int(192 * sf_y)))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        
        self.save_button = QtWidgets.QPushButton(self.coord_frame)
        self.save_button.setGeometry(QtCore.QRect(int(220 * sf_x), int(670 * sf_y), int(191 * sf_x), int(31 * sf_y)))
        self.save_button.setFont(label_font)
        self.save_button.setText("Save and continue")
        self.save_button.clicked.connect(self.building_sample)

        # ==============================================================
        # Polygon method function
        # ==============================================================
        

    def _on_select_output_folder(self):
        """
        Opens a folder selection dialog, stores the selected output folder path, and displays 
        the folder name in the interface.
        """
        self.method.output_folder_value, self.display_folder = select_output_folder(self)
        self.output_folder_value.setText(self.display_folder)
            
        
    def upload_input (self):
        """
        Calls the appropriate input loading method based on the selected polygon source.
        """
        if self.polygon_source_value.currentIndex() == 0:
            upload_csv(self)
        elif self.polygon_source_value.currentIndex() == 1:
            self.select_polygon()
 
            
    def select_polygon(self):
        """
        Opens a dialog to select a polygon file, loads and saves the boundary layer, and updates 
        the interface with the selected file information.
        """
        output_file_existing = os.path.join(self.method.output_folder_value, f"{self.output_polygon.text()}_boundary.gpkg")
        # Open file dialog restricted to .shp and .gpkg
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Polygon File",
            "",
            "Vector files (*.shp *.gpkg)"
        )
        if file_path:
            # Show only file name (not full path) in the GUI element
            file_display = os.path.basename(file_path)
            self.polygon_path_value.setText(file_display)
            gdf = self.load_polygon_layer(
                path = file_path,
                expected_crs = "EPSG:4326",
                fix_invalid=True
            )
            self.save_polygon_layer(
                            gdf,  
                            output_path=output_file_existing,
                            layer="boundary",
                            overwrite=True
                        )
            self.boundary_path = output_file_existing
        self.df = pd.DataFrame({"Polygon": ["OK"]})
        preview_data(self)
            

    def _on_save_coordinates(self):
        """
        Validates input coordinate data, converts latitude/longitude values into
        point or polygon geometries, and saves the result as a GeoPackage file.
        """
        save_coordinates(self)


    def building_polulation(self):
        """
        Downloads or loads building footprints for the selected boundary, processes and filters 
        the retrieved geometries, saves the results as a GeoPackage file, and updates the 
        progress indicators in the interface.
        """
        try:
            self.population = GUI_geofiles.download_building_footprints(self)
            self.building_value_polygon.setText(str(self.population))
        except (AttributeError, TypeError):
            QMessageBox.warning(
                self,
                "Input Error",
                (
                    "Some required inputs are missing or invalid. "
                    "Please review all fields and check the coordinates "
                    "file for inconsistencies."
                ),
            )

    def building_sample(self):
        """
        Retrieve the function that extracts a random sample of footprints (or the entire population), estimates 
        their centroids, which will be used later to retrieve the corresponding GSV images.
        """
        try:
            GUI_geofiles.extract_random_subset(self, self.sample_size_polygon.text())
            GUI_geofiles.create_centroid_layer(self)
            self.method.output_polygon = self.output_polygon.text()
            mode_use(self)
            
            if self.building_value_polygon.text()=="0000":
                QMessageBox.warning(self,
                                    "Error",
                                    "Building footprints have not been generated.\n"
                                    "Please click *Get footprints available* before saving and continuing.")
            else:
                self.accept()
        except (AttributeError, TypeError):
            QMessageBox.warning(
                self,
                "Input Error",
                (
                    "Some required inputs are missing or invalid. "
                    "Please review all fields and check the coordinates "
                    "file for inconsistencies."
                ),
            )

    def to_crs_safe(self, gdf, expected_crs):
        """
        Safer CRS comparison/reprojection across older GeoPandas/pyproj combos.
        """
        try:
            # In very old versions .crs can be dict-like (e.g., {'init': 'epsg:4326'})
            cur = gdf.crs
            if cur is None or expected_crs is None:
                return gdf
    
            cur_str = str(cur).lower()
            exp_str = str(expected_crs).lower()
    
            if cur_str != exp_str:
                # Let GeoPandas handle the conversion if possible
                return gdf.to_crs(expected_crs)
            return gdf
        except Exception:
            # If anything fails, just return original (we’ll already warn above)
            return gdf
    
    
    def load_polygon_layer(
        self,
        path: str,
        layer: Optional[str] = None,
        expected_crs: Optional[Union[str, int]] = None,
        fix_invalid: bool = True,
        ) -> Optional[gpd.GeoDataFrame]:
        """
        Load a polygon layer from a Shapefile (.shp) or GeoPackage (.gpkg) using GeoPandas 1.1.1-compatible calls.
        Shows QMessageBox warnings on any issue and returns None on failure.
    
        Parameters
        ----------
        self : QWidget
            Parent for QMessageBox.
        path : str
            Full path to .shp or .gpkg
        layer : str, optional
            GeoPackage layer name (if None, reads the first one).
        expected_crs : str|int, optional
            Target CRS (e.g. 'EPSG:4326' or 4326). If provided, attempts reprojection.
        fix_invalid : bool, default True
            If True, tries buffer(0) fix for invalid polygons.
    
        Returns
        -------
        GeoDataFrame or None
        """
    
        # --- Basic file checks
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "File not found", f"The path does not exist:\n{path}")
            return None
    
        ext = os.path.splitext(path)[1].lower()
        if ext not in {".shp", ".gpkg"}:
            QMessageBox.warning(
                self,
                "Unsupported file type",
                f"Unsupported extension: {ext}\nProvide a Shapefile (.shp) or GeoPackage (.gpkg).",
            )
            return None
    
        # --- Read with GeoPandas (1.1.1-friendly)
        try:
            if ext == ".gpkg":
                gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
            else:
                gdf = gpd.read_file(path)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error reading file",
                f"GeoPandas/Fiona could not read the file.\n\nFile: {path}\nError:\n{e}"
            )
            return None
    
        # --- Validate structure
        if gdf is None or len(gdf) == 0:
            QMessageBox.warning(self, "Empty layer", "The file was read but contains no features.")
            return None
    
        if "geometry" not in gdf.columns:
            QMessageBox.warning(self, "No geometry column", "The layer does not contain a geometry column.")
            return None
    
        # Drop null geometries (older-safe)
        gdf = gdf[gdf.geometry.notnull()]
    
        # Filter to polygons only (older-safe)
        geom_type = gdf.geometry.geom_type  # GeoSeries
        poly_mask = geom_type.isin(["Polygon", "MultiPolygon"])
        gdf_polys = gdf.loc[poly_mask].copy()
        if len(gdf_polys) == 0:
            QMessageBox.warning(
                self,
                "No polygon features",
                "The layer has no Polygon/MultiPolygon geometries."
            )
            return None
    
        # --- Optional fix for invalid geometries (older-safe)
        if fix_invalid:
            try:
                invalid_mask = ~gdf_polys.geometry.is_valid
                invalid_count = int(invalid_mask.sum())
                if invalid_count > 0:
                    # buffer(0) is the classic fix that works with old stacks
                    gdf_polys.loc[invalid_mask, "geometry"] = gdf_polys.loc[invalid_mask, "geometry"].buffer(0)
                    still_invalid = int((~gdf_polys.geometry.is_valid).sum())
                    if still_invalid > 0:
                        QMessageBox.warning(
                            self,
                            "Invalid geometries",
                            f"{still_invalid} geometries remain invalid after attempting to fix them."
                        )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Geometry repair failed",
                    f"Attempt to repair invalid geometries failed.\nError:\n{e}"
                )
    
        # --- CRS handling (older-safe)
        if gdf_polys.crs is None:
            QMessageBox.warning(
                self,
                "Missing CRS",
                "The layer has no defined CRS. Consider defining it before use."
            )
        elif expected_crs is not None:
            try:
                before = str(gdf_polys.crs)
                gdf_polys = self.to_crs_safe(gdf_polys, expected_crs)
                after = str(gdf_polys.crs)
                if before == after and str(before).lower() != str(expected_crs).lower():
                    # Reprojection did not happen when it probably should have
                    QMessageBox.warning(
                        self,
                        "CRS reprojection warning",
                        f"Requested CRS {expected_crs} but the layer remains in {before}."
                    )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "CRS reprojection error",
                    f"Failed to reproject to {expected_crs}.\nError:\n{e}"
                )
    
        # Final strictness: ensure all are Polygon/MultiPolygon instances
        # (avoid using Shapely 2.x vectorized APIs, stick to instance checks)
        try:
            for geom in gdf_polys.geometry:
                if not isinstance(geom, (Polygon, MultiPolygon)):
                    QMessageBox.warning(
                        self,
                        "Mixed geometries",
                        "Some features are not polygonal after filtering. Please verify the source data."
                    )
                    break
        except Exception:
            # If Shapely/GeoPandas mismatch causes type checks to fail, continue with best effort
            pass
    
        return gdf_polys
      
        
    def save_polygon_layer(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: str,
        layer: str = "polygons",
        overwrite: bool = True,
        ) -> bool:
        """
        Save a polygon GeoDataFrame to a GeoPackage (.gpkg) or Shapefile (.shp).
    
        Parameters
        ----------
        self : QWidget
            Parent for QMessageBox.
        gdf : GeoDataFrame
            Polygon/MultiPolygon features to save.
        output_path : str
            Target path ending with .gpkg or .shp.
        layer : str, default "polygons"
            Layer name for GeoPackage (ignored for Shapefile).
        overwrite : bool, default True
            If True and target exists, remove/overwrite. If False, try to adapt
            (append unique layer name for GPKG; refuse for SHP).
    
        Returns
        -------
        bool
            True on success, False if something went wrong.
        """
        # --- Basic checks
        if gdf is None or gdf.empty:
            QMessageBox.warning(self, "Nothing to save", "The GeoDataFrame is empty.")
            return False
    
        if not output_path:
            QMessageBox.warning(self, "Invalid path", "Please provide a valid output path.")
            return False
    
        ext = os.path.splitext(output_path)[1].lower()
        if ext not in {".gpkg", ".shp"}:
            QMessageBox.warning(
                self,
                "Unsupported format",
                "Output must be a GeoPackage (.gpkg) or a Shapefile (.shp)."
            )
            return False
    
        # --- Work on a copy & sanitize schema
        gdf = gdf.copy()
    
        # 1) Avoid writing the index (esp. if named 'fid')
        if gdf.index.name in ("fid", "FID", "ogc_fid", "OBJECTID"):
            gdf.reset_index(drop=True, inplace=True)
    
        # 2) If user column is 'fid', rename to avoid GPKG reserved FID conflict
        if "fid" in gdf.columns:
            gdf.rename(columns={"fid": "fid_attr"}, inplace=True)
    
        # 3) Sanitize dtypes (strings, ints, floats, datetimes->text, bool->ints)
        cleaned = {}
        for col in gdf.columns:
            if col == gdf.geometry.name:
                cleaned[col] = gdf[col]
                continue
    
            s = gdf[col]
            try:
                if pd.api.types.is_integer_dtype(s) and s.isna().any():
                    cleaned[col] = s.astype("float64")
                elif pd.api.types.is_integer_dtype(s):
                    cleaned[col] = s.astype("int64")
                elif pd.api.types.is_bool_dtype(s):
                    cleaned[col] = s.astype("int8")
                elif pd.api.types.is_datetime64_any_dtype(s):
                    cleaned[col] = pd.to_datetime(s).dt.strftime("%Y-%m-%d %H:%M:%S")
                elif pd.api.types.is_object_dtype(s):
                    cleaned[col] = s.astype(str)
                else:
                    cleaned[col] = s
            except Exception:
                # As a last resort, cast to string
                cleaned[col] = s.astype(str)
    
        gdf = gpd.GeoDataFrame(cleaned, geometry=gdf.geometry, crs=gdf.crs)
    
        # 4) Handle existing targets
        if os.path.exists(output_path):
            if overwrite:
                try:
                    os.remove(output_path) if ext == ".gpkg" else None
                    # For SHP: it’s multiple sidecar files; safest is to overwrite in place.
                    # Most drivers will overwrite; if not, we fall back to removing below.
                except Exception as e:
                    QMessageBox.warning(self, "Overwrite failed", f"Could not remove existing file.\n\n{e}")
                    return False
            else:
                if ext == ".shp":
                    QMessageBox.warning(
                        self,
                        "File exists",
                        "The Shapefile already exists and overwrite=False. Choose another path or enable overwrite."
                    )
                    return False
                else:
                    # For GPKG, try to make a unique layer name
                    base = layer or "layer"
                    candidate = base
                    n = 1
                    # We can't list layers easily without additional deps; just suffix a number.
                    while True:
                        try:
                            gdf.to_file(output_path, driver="GPKG", layer=candidate, index=False)
                            return True
                        except Exception as e:
                            # If it's a schema conflict on an existing layer, try a new name
                            n += 1
                            candidate = f"{base}_{n}"
                            if n > 50:  # avoid infinite loops
                                QMessageBox.warning(
                                    self,
                                    "Save failed",
                                    f"Could not find a free layer name in the GPKG.\nLast error:\n{e}"
                                )
                                return False
    
        # 5) Write
        try:
            if ext == ".gpkg":
                gdf.to_file(output_path, driver="GPKG", layer=layer or "polygons", index=False)
            else:  # .shp
                # Some drivers choke if sidecar files exist; try removing SHP set when overwriting.
                if overwrite:
                    for side in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"):
                        p = output_path.replace(".shp", side)
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                gdf.to_file(output_path, driver="ESRI Shapefile", index=False)
            return True
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Could not write the file.\n\nTarget: {output_path}\nError:\n{e}")
            return False

        
  
