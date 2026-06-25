"""Provide GIS utilities for building-footprint processing.

This module contains PyQt5-compatible methods for downloading building
footprints from OpenStreetMap or Overture Maps, extracting random samples,
and creating centroid layers for subsequent building inspections.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from shapely.geometry import MultiPolygon, Polygon

DESIGN_CRS = "EPSG:4326"
AREA_CRS = "EPSG:3857"
MIN_BUILDING_AREA_M2 = 50
OSM_TIMEOUT_SECONDS = 180
OSM_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
RANDOM_SEED = 10
TEMP_OVERTURE_FILE = Path("buildings_bbox.geojson")


class GUIGeofiles:
    """Manage GIS files used by the RUBIC-AI interface.

    Parameters
    ----------
    ui : object
        User-interface object that provides the widgets and paths required by
        the GIS workflow.
    """

    def __init__(self, ui: Any) -> None:
        """Initialize the GIS helper with a reference to the interface."""
        self.ui = ui

    def download_building_footprints(self) -> int | None:
        """Download or load building footprints for the selected boundary.

        The selected data source can be OpenStreetMap or Overture Maps. The
        resulting geometries are cleaned, filtered by area, assigned an ID,
        and saved as a GeoPackage.

        Returns
        -------
        int or None
            Number of building footprints saved, or ``None`` when the process
            cannot be completed.
        """
        output_file = self._footprint_output_path()

        if output_file.exists():
            buildings = gpd.read_file(output_file)
            self._finish_footprint_process()
            return len(buildings)

        boundary_path = Path(self.boundary_path)
        if not boundary_path.exists():
            QMessageBox.warning(
                self,
                "Boundary Error",
                "The selected boundary file does not exist.",
            )
            return None

        boundary = self._load_boundary(boundary_path)
        mode = self.footprint_mode.currentData()

        if mode == 0:
            buildings = self._download_osm_buildings(boundary)
        elif mode == 1:
            buildings = self._download_overture_buildings(boundary)
        else:
            QMessageBox.warning(
                self,
                "Selection Error",
                "Please select a valid footprint source.",
            )
            return None

        if buildings is None or buildings.empty:
            return None

        buildings = self._prepare_buildings(buildings)
        if buildings.empty:
            QMessageBox.warning(
                self,
                "Footprint Error",
                "No valid building footprints remained after filtering.",
            )
            return None

        buildings.to_file(output_file, driver="GPKG")
        self._finish_footprint_process()
        return len(buildings)

    def _footprint_output_path(self) -> Path:
        """Return the output path for the building-footprint GeoPackage."""
        filename = f"{self.output_polygon.text()}_buildings_footprint.gpkg"
        return Path(self.method.output_folder_value) / filename

    def _load_boundary(self, boundary_path: Path):
        """Load, reproject, and merge the selected boundary geometries."""
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_gdf.crs is None or boundary_gdf.crs.to_string() != DESIGN_CRS:
            print(f"Reprojecting boundary to {DESIGN_CRS}...")
            boundary_gdf = boundary_gdf.to_crs(DESIGN_CRS)

        polygon = boundary_gdf.union_all()
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon

    def _download_osm_buildings(self, polygon):
        """Download building footprints from OpenStreetMap."""
        self.footprint_progress.setValue(5)
        self.footprint_progress_label.setText("In progress...")

        ox.settings.overpass_endpoint = OSM_OVERPASS_ENDPOINT
        ox.settings.timeout = OSM_TIMEOUT_SECONDS
        self.footprint_progress.setValue(30)

        polygons = self._split_polygon(polygon)
        if polygons is None:
            return None

        building_parts = []
        for index, part in enumerate(polygons, start=1):
            print(f"Querying polygon {index}/{len(polygons)}...")
            buildings = self._query_osm_buildings(part)
            if not buildings.empty:
                building_parts.append(buildings)

        if not building_parts:
            print("No buildings found. Check the area or OSM coverage.")
            return None

        self.footprint_progress.setValue(60)
        return gpd.GeoDataFrame(
            pd.concat(building_parts, ignore_index=True),
            crs=building_parts[0].crs,
        )

    @staticmethod
    def _query_osm_buildings(polygon):
        """Query OpenStreetMap for building geometries inside a polygon."""
        try:
            if hasattr(ox, "features_from_polygon"):
                return ox.features_from_polygon(
                    polygon,
                    tags={"building": True},
                )
            return ox.geometries_from_polygon(
                polygon,
                tags={"building": True},
            )
        except (ValueError, TypeError, RuntimeError) as error:
            print(f"Skipping polygon because of an OSM error: {error}")
            return gpd.GeoDataFrame()

    @staticmethod
    def _split_polygon(polygon) -> list[Polygon] | None:
        """Convert a polygon or multipolygon into a list of polygons."""
        if isinstance(polygon, MultiPolygon):
            print(f"Detected MultiPolygon with {len(polygon.geoms)} parts.")
            return list(polygon.geoms)
        if isinstance(polygon, Polygon):
            print("Detected a single Polygon.")
            return [polygon]

        print("Input geometry is neither Polygon nor MultiPolygon.")
        return None

    def _download_overture_buildings(self, polygon):
        """Download and clip building footprints from Overture Maps."""
        self.footprint_progress.setValue(10)
        self.footprint_progress_label.setText("In progress...")

        min_x, min_y, max_x, max_y = polygon.bounds
        bbox_argument = f"--bbox={min_x},{min_y},{max_x},{max_y}"
        command = [
            "overturemaps",
            "download",
            bbox_argument,
            "-f",
            "geojson",
            "--type=building",
            "-o",
            str(TEMP_OVERTURE_FILE),
        ]

        self.footprint_progress.setValue(20)
        print("Downloading buildings from Overture Maps...")

        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            print(f"Overture Maps download failed: {error}")
            return None

        if not TEMP_OVERTURE_FILE.exists():
            print("No building-data file was created after the download.")
            return None

        try:
            buildings = gpd.read_file(TEMP_OVERTURE_FILE)
            if buildings.empty:
                print("No buildings were returned from Overture Maps.")
                return None

            self.footprint_progress.setValue(70)
            polygon_gdf = gpd.GeoDataFrame(
                geometry=[polygon],
                crs=DESIGN_CRS,
            )
            buildings = buildings.to_crs(polygon_gdf.crs)
            buildings = gpd.overlay(
                buildings,
                polygon_gdf,
                how="intersection",
            )
        except (OSError, ValueError) as error:
            print(f"Error processing Overture Maps data: {error}")
            return None
        finally:
            TEMP_OVERTURE_FILE.unlink(missing_ok=True)

        if buildings.empty:
            print("No buildings were found within the selected boundary.")
            return None

        return buildings

    def _prepare_buildings(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clean, filter, and assign identifiers to building geometries."""
        self.footprint_progress.setValue(70)
        buildings = buildings[
            buildings.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()

        if buildings.empty:
            return buildings

        buildings.geometry = buildings.geometry.buffer(0)
        buildings = self._clean_column_names(buildings)
        buildings = self._set_geometry_column(buildings)

        self.footprint_progress.setValue(85)
        columns_to_drop = [
            column
            for column in buildings.columns
            if column.upper() in {"AREA", "FIXME", "NOTE"}
        ]
        buildings = buildings.drop(columns=columns_to_drop, errors="ignore")
        buildings = self._filter_by_area(buildings)
        buildings = self._add_identifier(buildings)
        return buildings

    @staticmethod
    def _clean_column_names(
        buildings: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Rename invalid or reserved GeoPackage column names."""
        reserved_names = {
            "Type",
            "FID",
            "Geometry",
            "geom",
            "geometry",
            "FIXME",
        }
        geometry_name = buildings.geometry.name
        renamed_columns = {}

        for column in buildings.columns:
            if column == geometry_name:
                continue
            if column in reserved_names or not column.isidentifier():
                renamed_columns[column] = f"{column}_field"

        return buildings.rename(columns=renamed_columns)

    @staticmethod
    def _set_geometry_column(
        buildings: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Ensure that the active geometry column is correctly configured."""
        current_geometry = buildings.geometry.name
        if current_geometry in buildings.columns:
            return buildings.set_geometry(current_geometry)

        geometry_columns = [
            column for column in buildings.columns if "geom" in column.lower()
        ]
        if not geometry_columns:
            raise ValueError("No geometry column found in the GeoDataFrame.")
        return buildings.set_geometry(geometry_columns[0])

    def _filter_by_area(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove building geometries below the configured area threshold."""
        self.footprint_progress.setValue(90)
        projected = buildings.to_crs(AREA_CRS)
        projected["area_m2"] = projected.geometry.area
        original_count = len(projected)
        projected = projected[projected["area_m2"] > MIN_BUILDING_AREA_M2]
        filtered_count = len(projected)
        print(
            "Filtered buildings by area: "
            f"{original_count} -> {filtered_count} "
            f"(>{MIN_BUILDING_AREA_M2} m²)"
        )
        return projected.to_crs(DESIGN_CRS)

    @staticmethod
    def _add_identifier(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create a string identifier column for each building."""
        fid_columns = [
            column for column in buildings.columns if column.lower().startswith("fid")
        ]

        if fid_columns:
            buildings["id"] = buildings[fid_columns[0]]
        elif "osmid" in buildings.columns:
            buildings["id"] = buildings["osmid"]
        elif "id" not in buildings.columns:
            buildings["id"] = range(1, len(buildings) + 1)

        buildings["id"] = buildings["id"].astype(str)
        return buildings

    def _finish_footprint_process(self) -> None:
        """Update the interface after footprint generation is complete."""
        self.footprint_progress.setValue(100)
        self.footprint_progress_label.setText("Done!")
        QMessageBox.information(
            self,
            "Success",
            "Footprints successfully generated.",
        )

    def extract_random_subset(self, sample_size: int | str) -> None:
        """Extract and save a random subset of building footprints.

        Parameters
        ----------
        sample_size : int or str
            Number of building footprints to include in the sample.
        """
        footprint_file = Path(self.method.output_folder_value) / (
            f"{self.output_polygon.text()}_buildings_footprint.gpkg"
        )
        output_file = Path(self.method.output_folder_value) / (
            f"{self.output_polygon.text()}_subset_footprints.gpkg"
        )

        if output_file.exists():
            QMessageBox.warning(
                self,
                "Sample Size Warning",
                "A sample-size file already exists.\n\n"
                "Changes to the sample size have not been applied.\n"
                "Delete the existing file before selecting a new sample size.",
            )
            return

        sample_size = int(sample_size)
        footprints = gpd.read_file(footprint_file)
        if sample_size > len(footprints):
            QMessageBox.warning(
                self,
                "Sample Size Warning",
                f"Sample size {sample_size} exceeds the number of available "
                f"features ({len(footprints)}).",
            )
            return

        subset = footprints.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        )
        subset.to_file(
            output_file,
            driver="GPKG",
            layer="random_subset",
        )

    def create_centroid_layer(self) -> None:
        """Create and save centroid points for the sampled footprints."""
        subset_file = Path(self.method.output_folder_value) / (
            f"{self.output_polygon.text()}_subset_footprints.gpkg"
        )
        output_file = Path(self.method.output_folder_value) / (
            f"{self.output_polygon.text()}_subset_centroids.gpkg"
        )

        if output_file.exists():
            return

        buildings = gpd.read_file(subset_file)
        if buildings.crs is None:
            raise ValueError("The subset layer does not define a coordinate system.")

        original_crs = buildings.crs
        projected = buildings.to_crs(AREA_CRS)
        projected["centroid"] = projected.geometry.centroid

        centroids = gpd.GeoDataFrame(
            projected.drop(columns="geometry"),
            geometry=projected["centroid"],
            crs=AREA_CRS,
        ).to_crs(original_crs)

        centroids = centroids.drop(columns=["centroid"], errors="ignore")
        geographic_centroids = centroids.to_crs(DESIGN_CRS)
        centroids["latitude"] = geographic_centroids.geometry.y
        centroids["longitude"] = geographic_centroids.geometry.x
        centroids.to_file(
            output_file,
            driver="GPKG",
            layer="centroids",
        )


# Backward-compatible alias for existing imports.
GUI_geofiles = GUIGeofiles
