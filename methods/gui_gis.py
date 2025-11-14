# GUI pyqt5 libraries
from PyQt5.QtWidgets import QMessageBox

# Libries for shape and geopackage creation
import osmnx as ox
import geopandas as gpd

# Utilities libraries
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

# For overture
import os
import subprocess

class GUI_geofiles:
    def __init__(self, ui):
        self.ui = ui  # Link to the UI components
          
    ############ City boundary shape file ################
    def download_building_footprints(self):
        # Create output file for building footprints
        self.city_method = self.city
        self.country_method = self.country
        output_file = (
            self.method.output_folder_value + "/" +
            self.output_polygon.text() + "_buildings_footprint.gpkg"
        )
    
        # Check for existing footprint file
        if os.path.exists(output_file):
            buildings = gpd.read_file(output_file)
            return len(buildings)
        else:
            # Ensure boundary exists
            if os.path.exists(self.boundary_path):
                # ==============================================================
                # MODE 0: OpenStreetMap
                # ==============================================================
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
                    ox.settings.overpass_endpoint = "https://overpass-api.de/api/interpreter"
                    ox.settings.timeout = 180

                    # Function to safely query OSM
                    def get_osm_buildings(poly):
                        try:
                            if hasattr(ox, "geometries_from_polygon"):
                                return ox.geometries_from_polygon(poly, tags={"building": True})
                            else:
                                return ox.features_from_polygon(poly, tags={"building": True})
                        except Exception as e:
                            print(f"⚠️ Skipping polygon due to error: {e}")
                            return gpd.GeoDataFrame()

                    self.footprint_progress.setValue(30)
                    # Handle Polygon / MultiPolygon
                    building_list = []
                    if isinstance(polygon, MultiPolygon):
                        print(f"Detected MultiPolygon with {len(polygon.geoms)} parts...")
                        for i, poly in enumerate(polygon.geoms, 1):
                            print(f"  → Querying sub-polygon {i}/{len(polygon.geoms)}...")
                            gdf_part = get_osm_buildings(poly)
                            if not gdf_part.empty:
                                building_list.append(gdf_part)
                    elif isinstance(polygon, Polygon):
                        print("Detected single Polygon...")
                        gdf_part = get_osm_buildings(polygon)
                        if not gdf_part.empty:
                            building_list.append(gdf_part)
                    else:
                        print("Error: Input geometry is neither Polygon nor MultiPolygon.")
                        return None

                    if not building_list:
                        print("No buildings found. Check your area or OSM coverage.")
                        return None

                    self.footprint_progress.setValue(60)
                    ## Merge all parts
                    buildings = gpd.GeoDataFrame(pd.concat(building_list, ignore_index=True))
                    buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])]

                    # 🧹 Clean invalid or reserved column names
                    reserved_names = {"Type", "FID", "Geometry", "geom", "geometry", "FIXME"}
                    clean_columns = []
                    for col in buildings.columns:
                        if col in reserved_names or not col.isidentifier():
                            new_col = f"{col}_field"
                        else:
                            new_col = col
                        clean_columns.append(new_col)
                    buildings.columns = clean_columns
                    self.footprint_progress.setValue(70)
                    
                    # ✅ Ensure the active geometry column is properly set
                    geom_col = None
                    for c in buildings.columns:
                        if "geom" in c.lower():
                            geom_col = c
                            break
                    
                    if geom_col is not None:
                        buildings = buildings.set_geometry(geom_col)
                    else:
                        raise ValueError("No geometry column found in the GeoDataFrame!")
                    self.footprint_progress.setValue(80)
                    
                    # ✅ AREA FILTER (greater than 16 m²)
                    buildings = buildings.to_crs("EPSG:3857")  # project to meters
                    buildings["area_m2"] = buildings.geometry.area
                    before = len(buildings)
                    buildings = buildings[buildings["area_m2"] > 20]
                    after = len(buildings)
                    print(f"Filtered buildings by area: {before} → {after} (>{20} m²)")
                    buildings = buildings.to_crs("EPSG:4326")  # revert to geographic

                    # Save output
                    buildings.to_file(output_file, driver="GPKG")
                    self.footprint_progress.setValue(100)
                    self.footprint_progress_label.setText("Done!")
                    return len(buildings)
                # ==============================================================
                # MODE 1: Overture Maps
                # ==============================================================
                elif self.footprint_mode.currentData() == 1:
                    gdf = gpd.read_file(self.boundary_path)
                    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                        print("Reprojecting to EPSG:4326...")
                        gdf = gdf.to_crs("EPSG:4326")

                    polygon = gdf.union_all()
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    self.footprint_progress.setValue(10)
                    self.footprint_progress_label.setText("in progress ...")
                    
                    minx, miny, maxx, maxy = polygon.bounds
                    bbox_target = f"--bbox={minx},{miny},{maxx},{maxy}"

                    temp_geojson = "buildings_bbox.geojson"
                    print("Downloading buildings from Overture Maps...")
                    command = [
                        "overturemaps", "download", bbox_target,
                        "-f", "geojson", "--type=building", "-o", temp_geojson
                    ]
                    self.footprint_progress.setValue(20)
                    
                    try:
                        subprocess.run(command, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Overture Maps download failed: {e}")
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
                    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
                    buildings_ini = buildings_ini.to_crs(polygon_gdf.crs)

                    try:
                        buildings = gpd.overlay(buildings_ini, polygon_gdf, how="intersection")
                    except Exception as e:
                        print(f"⚠️ Error clipping buildings: {e}")
                        return None

                    if buildings.empty:
                        print("No buildings found within the polygon area.")
                        return None

                    # ------------------------------------------------------------------
                    # Clean and prepare geometries
                    self.footprint_progress.setValue(80)
                    buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])]
                    buildings["geometry"] = buildings["geometry"].buffer(0)

                    # ------------------------------------------------------------------
                    # Clean invalid or reserved column names
                    reserved_names = {"Type", "FID", "Geometry", "geom", "geometry", "FIXME"}
                    clean_columns = []
                    for col in buildings.columns:
                        if col in reserved_names or not col.isidentifier():
                            new_col = f"{col}_field"
                        else:
                            new_col = col
                        clean_columns.append(new_col)
                    buildings.columns = clean_columns
                    self.footprint_progress.setValue(85)
                    
                    # ✅ Ensure the active geometry column is correctly set
                    geom_col = None
                    for c in buildings.columns:
                        if "geom" in c.lower():
                            geom_col = c
                            break
                    if geom_col is not None:
                        buildings = buildings.set_geometry(geom_col)
                    else:
                        raise ValueError("No geometry column found in the GeoDataFrame!")

                    # ------------------------------------------------------------------
                    # Drop unnecessary columns
                    drop_cols = [c for c in buildings.columns if c.upper() in ["AREA", "FIXME", "NOTE"]]
                    buildings = buildings.drop(columns=drop_cols, errors="ignore")
                    self.footprint_progress.setValue(90)
                    
                    # ------------------------------------------------------------------
                    # ✅ AREA FILTER (greater than 16 m²)
                    buildings = buildings.to_crs("EPSG:3857")  # project to meters
                    buildings["area_m2"] = buildings.geometry.area
                    before = len(buildings)
                    buildings = buildings[buildings["area_m2"] > 20]
                    after = len(buildings)
                    print(f"Filtered buildings by area: {before} → {after} (>{20} m²)")
                    buildings = buildings.to_crs("EPSG:4326")

                    # ------------------------------------------------------------------
                    # Save to file
                    buildings.to_file(output_file, driver="GPKG")
                    self.footprint_progress.setValue(100)
                    self.footprint_progress_label.setText("Done!")
                    os.remove(temp_geojson)
                    return len(buildings)
        
    ############ Random subset buildings ################  
    def extract_random_subset(self , sample_size):
        """
        Extract a random subset of building footprints and save it as a GeoPackage.
    
        This method selects a random sample of building footprints from a previously saved GeoPackage 
        and saves the subset to a new GeoPackage file. It ensures that the sample size does not 
        exceed the number of features in the dataset.
    
        Args:
            None. The method operates on instance attributes such as `city_method`, `country_method`, 
            and the output folder path provided in the UI.
    
        Returns:
            None. The random subset is saved to a GeoPackage file in the specified output folder.
    
        Effects:
            - Saves a randomly selected subset of building footprints to a new GeoPackage.
            - Logs messages about the saving process.
    
        Raises:
            ValueError: If the sample size exceeds the number of features in the dataset.
    
        Notes:
            - The random sample is controlled by a predefined sample size (`sample_size`) and seed 
              (`seed`) for reproducibility.
        """

        # Load buildng footprints
        footprint = self.method.output_folder_value+"/"+self.output_polygon.text()+"_buildings_footprint.gpkg"
        # Create output file for building footprints
        output_file= self.method.output_folder_value+"/"+self.output_polygon.text()+"_subset_footprints.gpkg"
        # Ensure sample size is not greater than the number of points in the dataset
        seed=10
        # Check if a subset file exists
        if os.path.exists(output_file):
            QMessageBox.warning(
                self,
                "Sample Size Warning",
                "A sample size file already exists.\n\n"
                "Any changes to the sample size have not been applied.\n"
                "If you want to use a different sample size, please delete the existing files first."
            )
        else:
            sample_size = int(sample_size)
            # Load the input point layer
            gdf = gpd.read_file(footprint)
            
            print("Sample size: ", sample_size)
            print("N° Buildings: ", len(gdf))
            
            if sample_size > len(gdf):
                QMessageBox.warning(
                    self,
                    "Sample Size Warning",
                    f"Sample size {sample_size} exceeds the number of features in the dataset ({len(gdf)}).")
                
            # Extract a random sample
            subset = gdf.sample(n=sample_size, random_state=seed)
            # Save the subset to a GeoPackage
            print(f"Saving random subset to: {output_file}")
            subset.to_file(output_file, driver="GPKG", layer="random_subset")

    ############ Create a point layer and extract the coordinates of a subset of buildings ################ 
    def create_centroid_layer(self):
        """
        Create a GeoPackage layer of centroids from building footprints, including latitude and longitude.
        
        This method calculates the centroids of building footprints from a subset GeoPackage file, 
        extracts their latitude and longitude, and saves the resulting data to a new GeoPackage layer. 
        If the centroid file already exists, it skips the execution.
        
        Args:
            None. The method operates on instance attributes such as `city_method`, `country_method`, 
            and the output folder path provided in the UI.
        
        Returns:
            None. The centroid data is saved to a GeoPackage file in the specified output folder.
        
        Effects:
            - Computes the centroids of building footprints.
            - Extracts latitude and longitude from centroid geometries.
            - Saves the centroids to a GeoPackage file.
        
        Notes:
            - The method checks for an existing centroid GeoPackage to avoid duplicate processing.
            - Ensures the output retains the same CRS as the input building footprints.
        """

        # Load selected subset building
        subset_file=self.method.output_folder_value+"/"+self.output_polygon.text()+"_subset_footprints.gpkg"
        # Create output file for building footprints
        output_file=self.method.output_folder_value+"/"+self.output_polygon.text()+"_subset_centroids.gpkg"
        # Check if a centroid file exists
        if os.path.exists(output_file):
            pass
        else:
            # Load the building footprints
            buildings = gpd.read_file(subset_file)
            # Calculate centroids
            print("Calculating centroids...")
            buildings["centroid"] = buildings.geometry.centroid
            # Create a new GeoDataFrame for the centroids
            centroids = gpd.GeoDataFrame(
                buildings.drop(columns="geometry"),  # Drop original geometry column
                geometry=buildings["centroid"],  # Set centroid as the active geometry
                crs=buildings.crs ) # Use the same CRS as the input  
            # Drop any remaining references to the original geometry column
            if "centroid" in centroids.columns:
                centroids = centroids.drop(columns=["centroid"])
            # Add latitude and longitude columns
            print("Extracting latitude and longitude...")
            centroids["latitude"] = centroids.geometry.y
            centroids["longitude"] = centroids.geometry.x
            # Save the centroids to a GeoPackage
            print(f"Saving centroids to: {output_file}")
            print("")
            print("DF: ")
            print(centroids)
            centroids.to_file(output_file, driver="GPKG", layer="centroids")
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    