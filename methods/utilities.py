"""
utilities.py
============
This module provides general-purpose utility functions for file selection, data preview,
CSV uploading, coordinate saving, and collection mode configuration within the GUI.
"""

from PyQt5 import QtWidgets
import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Polygon

def select_output_folder(parent=None):
    """
    Opens a folder selection dialog, stores the selected output folder path, and displays 
    the folder name in the interface.
    """
    folder = QtWidgets.QFileDialog.getExistingDirectory(None , "Select Output Folder")
    if folder:
        return folder, os.path.basename(folder)
    return None, None

    
def preview_data(self):
    """
    Displays a preview of the loaded data in the table widget, showing up to the first 
    10 rows and adjusting the table font settings for readability.
    """
    if hasattr(self, 'df') and not self.df.empty:
        preview_df = self.df.head(10)  # Only show first 10 rows

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


def upload_csv(self):
    """
    Opens a dialog to select a CSV file, loads and validates the input data, stores the file 
    path, and updates the interface with the selected file name.
    """
    options = QtWidgets.QFileDialog.Options()
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
    if self.method.insp_method == 0:
        display_path = self.polygon_path_value
    elif self.method.insp_method == 1:
        display_path = self.specific_path
    elif self.method.insp_method == 2:
        self.method.file_local_csv = file_path
        display_path = self.local_path
    elif self.method.insp_method == 3:
        try: 
            if self.extrap_mode_new.currentData() == 0:
                # Stratified DL
                display_path = self.population_new_path
            else:
                # Stratified Manually
                display_path = self.population_new_path
        except:
            # KNN
            display_path = self.label_path
        
    if file_path:
        try:
            self.df = pd.read_csv(file_path)
            display_path.setText(os.path.basename(file_path))
            preview_data(self)
        except Exception as e:
            display_path.setText(f"Error: {str(e)}")
            QtWidgets.QMessageBox.warning(self, "Input Error", "Invalid file selected or parsing error.")
    else:
        display_path.setText("No file selected.")
        QtWidgets.QMessageBox.warning(self, "Input Error", "No file selected.")
    
    if self.method.insp_method == 3:
        return self.df
      
    
def save_coordinates(self):
    """
    Validates input coordinate data, converts latitude/longitude values into
    point or polygon geometries, and saves the result as a GeoPackage file.
    """
    def build_output_path(folder, name, suffix=""):
        filename = f"{name}{suffix}.gpkg"
        return os.path.join(folder, filename)

    def remove_if_exists(path):
        if os.path.exists(path):
            os.remove(path)

    def validate_columns(df, required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            QtWidgets.QMessageBox.warning(self, "Missing Columns", f"Required columns missing: {', '.join(missing)}")
            return False
        return True

    def save_gpkg(gdf, path, driver="GPKG", layer=None):
        kwargs = {"driver": driver}
        if layer:
            kwargs["layer"] = layer
        gdf.to_file(path, **kwargs)

    try:
        df = self.df
        output_folder = self.method.output_folder_value
    
        if self.method.insp_method == 0:
            if self.polygon_source_value.currentIndex() == 1:
                return
            
            if not validate_columns(df, ['id', 'latitude', 'longitude']):
                return
    
            try:
                output_gpkg = build_output_path(output_folder, self.output_polygon, "_boundary")
                self.boundary_path = output_gpkg
                remove_if_exists(output_gpkg)
    
                polygon = Polygon(zip(df["longitude"], df["latitude"]))
                gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:4326")
                save_gpkg(gdf, output_gpkg, layer="polygon_layer")
    
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Could not generate GPKG:\n{e}")
    
        elif self.method.insp_method in (1, 2):
            if self.method.insp_method == 1:
                output_gpkg = build_output_path(output_folder, self.output_specific.text())
            else:
                output_gpkg = build_output_path(output_folder, self.output_local.text())
            remove_if_exists(output_gpkg)
    
            if not validate_columns(df, ['id', 'latitude', 'longitude']):
                return
    
            try:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                    crs="EPSG:4326"
                )
                save_gpkg(gdf, output_gpkg)
    
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Could not generate GPKG:\n{e}")
                return
    
            if self.method.insp_method == 1:
                self.method.specific_output_name = self.output_specific
            else:
                self.method.local_output_name = self.output_local
            mode_use(self)
            self.accept()

    except Exception as e:
        QtWidgets.QMessageBox.warning(
            self, "Input Error",
            f"Some required inputs are missing or invalid.\n\nDetails: {e}")
        
        
def mode_use(self):
    """
    Sets the collection mode based on the selected option in the interface.
    """
    if self.collection_mode.currentText() == "Manual":
        self.ai_value = False 
    elif self.collection_mode.currentText() == "AI Powered":
        self.ai_value = True
        
