# ============================================================
# Building Footprint Downloader with Progress Bar (PyQt5)
# Author: Daniel Gómez
# ============================================================

import os
import subprocess
import time
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QProgressBar, QLabel, QComboBox
)


# ============================================================
# Worker Thread
# ============================================================
class BuildingFootprintWorker(QThread):
    progress = pyqtSignal(int, str)  # percentage, message
    finished = pyqtSignal(int)       # number of buildings found (or 0 if none)

    def __init__(self, boundary_path, output_file, mode):
        super().__init__()
        self.boundary_path = boundary_path
        self.output_file = output_file
        self.mode = mode  # 0 = OSM, 1 = Overture

    def run(self):
        """Run long building footprint download process."""
        start_time = time.time()

        if not os.path.exists(self.boundary_path):
            self.progress.emit(0, "❌ Boundary file not found.")
            self.finished.emit(0)
            return

        # ----------------------------------------------------
        # MODE 0: OSMnx download
        # ----------------------------------------------------
        if self.mode == 0:
            self.progress.emit(5, "🧭 Reading boundary file...")
            gdf = gpd.read_file(self.boundary_path)
            if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            polygon = gdf.union_all()
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            ox.settings.overpass_endpoint = "https://overpass-api.de/api/interpreter"
            ox.settings.timeout = 180

            def get_osm_buildings(poly):
                """Safe wrapper for OSM building download."""
                try:
                    if hasattr(ox, "geometries_from_polygon"):
                        return ox.geometries_from_polygon(poly, tags={"building": True})
                    else:
                        return ox.features_from_polygon(poly, tags={"building": True})
                except Exception as e:
                    print(f"⚠️ Skipping polygon due to error: {e}")
                    return gpd.GeoDataFrame()

            building_list = []
            if isinstance(polygon, MultiPolygon):
                total = len(polygon.geoms)
                for i, poly in enumerate(polygon.geoms, 1):
                    pct = int(10 + (i / total) * 70)
                    self.progress.emit(pct, f"⬇️ Querying sub-polygon {i}/{total}...")
                    gdf_part = get_osm_buildings(poly)
                    if not gdf_part.empty:
                        building_list.append(gdf_part)
            else:
                self.progress.emit(50, "⬇️ Querying single polygon...")
                gdf_part = get_osm_buildings(polygon)
                if not gdf_part.empty:
                    building_list.append(gdf_part)

            if not building_list:
                self.progress.emit(100, "⚠️ No buildings found.")
                self.finished.emit(0)
                return

            buildings = gpd.GeoDataFrame(pd.concat(building_list, ignore_index=True))
            buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])]

            # Area filter
            self.progress.emit(85, "🧹 Filtering small buildings (<20 m²)...")
            buildings = buildings.to_crs("EPSG:3857")
            buildings["area_m2"] = buildings.geometry.area
            before = len(buildings)
            buildings = buildings[buildings["area_m2"] > 20]
            after = len(buildings)
            print(f"Filtered buildings by area: {before} → {after}")
            buildings = buildings.to_crs("EPSG:4326")

            buildings.to_file(self.output_file, driver="GPKG")
            self.progress.emit(100, f"✅ Saved {len(buildings)} OSM buildings.")
            self.finished.emit(len(buildings))

        # ----------------------------------------------------
        # MODE 1: Overture Maps
        # ----------------------------------------------------
        elif self.mode == 1:
            self.progress.emit(10, "🌍 Preparing Overture Maps download...")
            gdf = gpd.read_file(self.boundary_path)
            if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            polygon = gdf.union_all()
            minx, miny, maxx, maxy = polygon.bounds
            bbox_target = f"--bbox={minx},{miny},{maxx},{maxy}"
            temp_geojson = "buildings_bbox.geojson"

            cmd = [
                "overturemaps", "download", bbox_target,
                "-f", "geojson", "--type=building", "-o", temp_geojson
            ]

            self.progress.emit(30, "⬇️ Downloading from Overture Maps...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                self.progress.emit(100, f"❌ Overture Maps download failed: {e}")
                self.finished.emit(0)
                return

            if not os.path.exists(temp_geojson):
                self.progress.emit(100, "⚠️ No building data file found.")
                self.finished.emit(0)
                return

            self.progress.emit(60, "✂️ Clipping buildings to polygon...")
            buildings_ini = gpd.read_file(temp_geojson)
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

            try:
                buildings = gpd.overlay(buildings_ini, polygon_gdf, how="intersection")
            except Exception as e:
                self.progress.emit(100, f"⚠️ Error clipping buildings: {e}")
                self.finished.emit(0)
                return

            if buildings.empty:
                self.progress.emit(100, "⚠️ No buildings inside polygon.")
                self.finished.emit(0)
                return

            self.progress.emit(85, "🧹 Cleaning geometries...")
            buildings = buildings.to_crs("EPSG:3857")
            buildings["area_m2"] = buildings.geometry.area
            before = len(buildings)
            buildings = buildings[buildings["area_m2"] > 20]
            after = len(buildings)
            print(f"Filtered buildings by area: {before} → {after}")
            buildings = buildings.to_crs("EPSG:4326")

            buildings.to_file(self.output_file, driver="GPKG")
            self.progress.emit(100, f"✅ Saved {len(buildings)} Overture buildings.")
            self.finished.emit(len(buildings))


# ============================================================
# Main GUI
# ============================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Building Footprint Downloader")

        layout = QVBoxLayout(self)

        # Label
        self.label_status = QLabel("Ready.")
        layout.addWidget(self.label_status)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Mode Selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("OSM (OpenStreetMap)", 0)
        self.mode_selector.addItem("Overture Maps", 1)
        layout.addWidget(self.mode_selector)

        # Start Button
        self.button_start = QPushButton("Start Download")
        layout.addWidget(self.button_start)
        self.button_start.clicked.connect(self.start_download)

        self.worker = None

    def start_download(self):
        """Initialize and start the download worker."""
        boundary_path = r"C:\Users\User\Documents\GitHub\RUBIC-AI\demos\polygon_method\existing_polygon_example.gpkg"
        output_file = r"C:\Users\User\Downloads\proof_buildings_footprint.gpkg"
        mode = self.mode_selector.currentData()

        if not os.path.exists(boundary_path):
            self.label_status.setText("❌ Boundary file not found.")
            return

        self.label_status.setText("Starting download...")
        self.progress_bar.setValue(0)
        self.button_start.setEnabled(False)

        self.worker = BuildingFootprintWorker(boundary_path, output_file, mode)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.download_finished)
        self.worker.start()

    def update_progress(self, percent, message):
        """Update progress bar and message."""
        self.progress_bar.setValue(percent)
        self.label_status.setText(message)

    def download_finished(self, count):
        """Handle completion."""
        self.progress_bar.setValue(100)
        if count > 0:
            self.label_status.setText(f"✅ Download complete: {count} buildings saved.")
        else:
            self.label_status.setText("⚠️ No buildings found.")
        self.button_start.setEnabled(True)


# ============================================================
# Run Application
# ============================================================
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(420, 200)
    window.show()
    sys.exit(app.exec_())
