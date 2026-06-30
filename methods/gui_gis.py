"""Provide GIS-related methods for processing building footprints.

This module downloads building footprints, extracts random subsets, and
creates centroid layers from geographic data.
"""

import os
import subprocess

import geopandas as gpd
import osmnx as ox
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from shapely.geometry import MultiPolygon, Polygon


class GUI_geofiles:  # noqa: N801
    """Provide GIS-processing methods used by the graphical interface."""

    def __init__(self, ui):
        """Store a reference to the user-interface components."""
        self.ui = ui

    # City boundary shapefile
    def download_building_footprints(self):  # noqa: C901
        """Download, process, filter, and save building footprints."""
        output_file = (
            self.method.output_folder_value
            + "/"
            + self.output_polygon.text()
            + "_buildings_footprint.gpkg"
        )

        # Check for existing footprint file
        if os.path.exists(output_file):
            buildings = gpd.read_file(output_file)
            self.footprint_progress.setValue(100)
            self.footprint_progress_label.setText("Done!")
            QMessageBox.information(
                self,
                "Success",
                "Footprints successfully generated",
            )
            return len(buildings)

        # Ensure boundary exists
        if os.path.exists(self.boundary_path):
            # ==========================================================
            # MODE 0: OpenStreetMap
            # ==========================================================
            if self.footprint_mode.currentData() == 0:
                # Load and ensure EPSG:4326
                gdf = gpd.read_file(self.boundary_path)
                if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                    print("Reprojecting to EPSG:4326...")
                    gdf = gdf.to_crs("EPSG:4326")

                # Union all geometries
                polygon = gdf.union_all()
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)

                self.footprint_progress.setValue(5)
                self.footprint_progress_label.setText("in progress ...")

                # Settings for reliability
                ox.settings.overpass_endpoint = (
                    "https://overpass-api.de/api/interpreter"
                )
                ox.settings.timeout = 180

                # Function to safely query OSM
                def get_osm_buildings(poly):
                    try:
                        if hasattr(ox, "geometries_from_polygon"):
                            return ox.geometries_from_polygon(
                                poly,
                                tags={"building": True},
                            )
                        return ox.features_from_polygon(
                            poly,
                            tags={"building": True},
                        )
                    except Exception as error:
                        print(f"⚠️ Skipping polygon due to error: {error}")
                        return gpd.GeoDataFrame()

                self.footprint_progress.setValue(30)

                # Handle Polygon / MultiPolygon
                building_list = []
                if isinstance(polygon, MultiPolygon):
                    number_of_parts = len(polygon.geoms)
                    print(
                        "Detected MultiPolygon with "
                        f"{number_of_parts} parts..."
                    )
                    for index, poly in enumerate(polygon.geoms, 1):
                        print(
                            "  → Querying sub-polygon "
                            f"{index}/{number_of_parts}..."
                        )
                        gdf_part = get_osm_buildings(poly)
                        if not gdf_part.empty:
                            building_list.append(gdf_part)
                elif isinstance(polygon, Polygon):
                    print("Detected single Polygon...")
                    gdf_part = get_osm_buildings(polygon)
                    if not gdf_part.empty:
                        building_list.append(gdf_part)
                else:
                    print(
                        "Error: Input geometry is neither Polygon nor "
                        "MultiPolygon."
                    )
                    return None

                if not building_list:
                    print("No buildings found. Check your area or OSM coverage.")
                    return None

                self.footprint_progress.setValue(60)

                # Merge all parts
                buildings = gpd.GeoDataFrame(
                    pd.concat(building_list, ignore_index=True)
                )
                buildings = buildings[
                    buildings.geom_type.isin(["Polygon", "MultiPolygon"])
                ]

                # Clean invalid or reserved column names
                reserved_names = {
                    "Type",
                    "FID",
                    "Geometry",
                    "geom",
                    "geometry",
                    "FIXME",
                }
                clean_columns = []
                for column in buildings.columns:
                    if column in reserved_names or not column.isidentifier():
                        new_column = f"{column}_field"
                    else:
                        new_column = column
                    clean_columns.append(new_column)
                buildings.columns = clean_columns

                self.footprint_progress.setValue(70)

                # Ensure the active geometry column is properly set
                geometry_column = None
                for column in buildings.columns:
                    if "geom" in column.lower():
                        geometry_column = column
                        break

                if geometry_column is not None:
                    buildings = buildings.set_geometry(geometry_column)
                else:
                    raise ValueError(
                        "No geometry column found in the GeoDataFrame!"
                    )

                self.footprint_progress.setValue(80)

                # Area filter: greater than 50 square metres
                buildings = buildings.to_crs("EPSG:3857")
                buildings["area_m2"] = buildings.geometry.area
                before = len(buildings)
                buildings = buildings[buildings["area_m2"] > 50]
                after = len(buildings)
                print(
                    "Filtered buildings by area: "
                    f"{before} → {after} (>50 m²)"
                )
                buildings = buildings.to_crs("EPSG:4326")

                # Ensure an ID column exists
                fid_columns = [
                    column
                    for column in buildings.columns
                    if column.lower().startswith("fid")
                ]

                if fid_columns:
                    buildings["id"] = buildings[fid_columns[0]]
                elif "osmid" in buildings.columns:
                    buildings["id"] = buildings["osmid"]
                elif "id" not in buildings.columns:
                    buildings["id"] = range(1, len(buildings) + 1)

                buildings["id"] = buildings["id"].astype(str)

                # Save output
                buildings.to_file(output_file, driver="GPKG")
                self.footprint_progress.setValue(100)
                self.footprint_progress_label.setText("Done!")
                QMessageBox.information(
                    self,
                    "Success",
                    "Footprints successfully generated",
                )
                return len(buildings)

            # ==========================================================
            # MODE 1: Overture Maps
            # ==========================================================
            if self.footprint_mode.currentData() == 1:
                gdf = gpd.read_file(self.boundary_path)
                if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                    print("Reprojecting to EPSG:4326...")
                    gdf = gdf.to_crs("EPSG:4326")

                polygon = gdf.union_all()
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)

                self.footprint_progress.setValue(10)
                self.footprint_progress_label.setText("in progress ...")

                min_x, min_y, max_x, max_y = polygon.bounds
                bbox_target = (
                    f"--bbox={min_x},{min_y},{max_x},{max_y}"
                )

                temp_geojson = "buildings_bbox.geojson"
                print("Downloading buildings from Overture Maps...")
                command = [
                    "overturemaps",
                    "download",
                    bbox_target,
                    "-f",
                    "geojson",
                    "--type=building",
                    "-o",
                    temp_geojson,
                ]
                self.footprint_progress.setValue(20)

                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as error:
                    print(f"Overture Maps download failed: {error}")
                    return None

                if not os.path.exists(temp_geojson):
                    print("No building data file found after download.")
                    return None

                buildings_ini = gpd.read_file(temp_geojson)
                if buildings_ini.empty:
                    print("No buildings returned from Overture Maps.")
                    return None

                self.footprint_progress.setValue(70)

                print("✂️ Clipping buildings to custom polygon...")
                polygon_gdf = gpd.GeoDataFrame(
                    geometry=[polygon],
                    crs="EPSG:4326",
                )
                buildings_ini = buildings_ini.to_crs(polygon_gdf.crs)

                try:
                    buildings = gpd.overlay(
                        buildings_ini,
                        polygon_gdf,
                        how="intersection",
                    )
                except Exception as error:
                    print(f"⚠️ Error clipping buildings: {error}")
                    return None

                if buildings.empty:
                    print("No buildings found within the polygon area.")
                    return None

                # Clean and prepare geometries
                self.footprint_progress.setValue(80)
                buildings = buildings[
                    buildings.geom_type.isin(["Polygon", "MultiPolygon"])
                ]
                buildings["geometry"] = buildings["geometry"].buffer(0)

                # Clean invalid or reserved column names
                reserved_names = {
                    "Type",
                    "FID",
                    "Geometry",
                    "geom",
                    "geometry",
                    "FIXME",
                }
                clean_columns = []
                for column in buildings.columns:
                    if column in reserved_names or not column.isidentifier():
                        new_column = f"{column}_field"
                    else:
                        new_column = column
                    clean_columns.append(new_column)
                buildings.columns = clean_columns

                self.footprint_progress.setValue(85)

                # Ensure the active geometry column is correctly set
                geometry_column = None
                for column in buildings.columns:
                    if "geom" in column.lower():
                        geometry_column = column
                        break

                if geometry_column is not None:
                    buildings = buildings.set_geometry(geometry_column)
                else:
                    raise ValueError(
                        "No geometry column found in the GeoDataFrame!"
                    )

                # Drop unnecessary columns
                drop_columns = [
                    column
                    for column in buildings.columns
                    if column.upper() in ["AREA", "FIXME", "NOTE"]
                ]
                buildings = buildings.drop(
                    columns=drop_columns,
                    errors="ignore",
                )

                self.footprint_progress.setValue(90)

                # Area filter: greater than 50 square metres
                buildings = buildings.to_crs("EPSG:3857")
                buildings["area_m2"] = buildings.geometry.area
                before = len(buildings)
                buildings = buildings[buildings["area_m2"] > 50]
                after = len(buildings)
                print(
                    "Filtered buildings by area: "
                    f"{before} → {after} (>50 m²)"
                )
                buildings = buildings.to_crs("EPSG:4326")

                # Add an ID column
                fid_columns = [
                    column
                    for column in buildings.columns
                    if column.lower().startswith("fid")
                ]

                if fid_columns:
                    buildings["id"] = buildings[fid_columns[0]]
                elif "osmid" in buildings.columns:
                    buildings["id"] = buildings["osmid"]
                elif "id" not in buildings.columns:
                    buildings["id"] = range(1, len(buildings) + 1)

                buildings["id"] = buildings["id"].astype(str)

                # Save to file
                buildings.to_file(output_file, driver="GPKG")
                self.footprint_progress.setValue(100)
                self.footprint_progress_label.setText("Done!")
                os.remove(temp_geojson)

                QMessageBox.information(
                    self,
                    "Success",
                    "Footprints successfully generated",
                )
                return len(buildings)

        return None

    # Random subset of buildings
    def extract_random_subset(self, sample_size):
        """Extract and save a random subset of building footprints."""
        footprint = (
            self.method.output_folder_value
            + "/"
            + self.output_polygon.text()
            + "_buildings_footprint.gpkg"
        )
        output_file = (
            self.method.output_folder_value
            + "/"
            + self.output_polygon.text()
            + "_subset_footprints.gpkg"
        )
        seed = 10

        if os.path.exists(output_file):
            QMessageBox.warning(
                self,
                "Sample Size Warning",
                "A sample size file already exists.\n\n"
                "Any changes to the sample size have not been applied.\n"
                "If you want to use a different sample size, please "
                "delete the existing files first.",
            )
        else:
            sample_size = int(sample_size)
            gdf = gpd.read_file(footprint)

            if sample_size > len(gdf):
                QMessageBox.warning(
                    self,
                    "Sample Size Warning",
                    f"Sample size {sample_size} exceeds the number of "
                    f"features in the dataset ({len(gdf)}).",
                )

            subset = gdf.sample(n=sample_size, random_state=seed)
            subset.to_file(
                output_file,
                driver="GPKG",
                layer="random_subset",
            )

    # Create a point layer and extract subset coordinates
    def create_centroid_layer(self):
        """Create and save a centroid layer from selected footprints."""
        subset_file = (
            self.method.output_folder_value
            + "/"
            + self.output_polygon.text()
            + "_subset_footprints.gpkg"
        )
        output_file = (
            self.method.output_folder_value
            + "/"
            + self.output_polygon.text()
            + "_subset_centroids.gpkg"
        )

        if not os.path.exists(output_file):
            buildings = gpd.read_file(subset_file)
            buildings["centroid"] = buildings.geometry.centroid
            centroids = gpd.GeoDataFrame(
                buildings.drop(columns="geometry"),
                geometry=buildings["centroid"],
                crs=buildings.crs,
            )

            if "centroid" in centroids.columns:
                centroids = centroids.drop(columns=["centroid"])

            centroids["latitude"] = centroids.geometry.y
            centroids["longitude"] = centroids.geometry.x
            centroids.to_file(
                output_file,
                driver="GPKG",
                layer="centroids",
            )
