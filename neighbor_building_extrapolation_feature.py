from geopy.distance import geodesic

# Function to calculate Geodesic distance (in km)
def geodesic_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).km

# Function to find 3 nearest neighbors using geodesic distance
def find_nearest_neighbors_geodesic(input_row, info_df, n_neighbors, neighbor_method, max_distance_km):
    if neighbor_method == 1:
        # Apply geodesic distance for each row in reference dataframe
        distances = info_df.apply(
            lambda row: geodesic_distance(
                input_row['latitude'],
                input_row['longitude'],
                row['Latitude'],
                row['Longitude']
            ), axis=1)
    
        nearest_indices = distances.nsmallest(n_neighbors).index
        
        # Extract neighbor data
        neighbor_data = info_df.loc[nearest_indices].copy()  # Use .copy() to avoid SettingWithCopyWarning
        
        # Add distance column to neighbor data
        neighbor_data['distance_km'] = distances.loc[nearest_indices].values
        
        # neighbor_data = info_df.loc[nearest_indices]
        
    elif neighbor_method == 2:
        # Calculate geodesic distance for each row in reference DataFrame
        distances = info_df.apply(
            lambda row: geodesic_distance(
                input_row['latitude'],
                input_row['longitude'],
                row['Latitude'],
                row['Longitude']
            ), axis=1
        )
    
        # Filter DataFrame based on max_distance_km
        filtered_df = info_df[distances <= max_distance_km].copy()
    
        # Optionally, you can add the distance as a column to the output
        filtered_df['distance_km'] = distances[distances <= max_distance_km].values
        neighbor_data = filtered_df
    
    return neighbor_data

# Function to compute taxonomy probabilities and preserve full structure of info_df
def compute_taxonomy_distribution_full_structure(nearest_neighbors, input_row):
    # Get counts of Taxonomy (normalized to sum to 1 as probability)
    taxonomy_counts = nearest_neighbors['Taxonomy'].value_counts(normalize=True)
    
    # Prepare list of rows to return
    distribution_rows = []
    
    # Loop through each unique Taxonomy found among nearest neighbors
    for taxonomy, prob in taxonomy_counts.items():
        # Take the first occurrence of this taxonomy in neighbors to extract corresponding categorical attributes
        taxonomy_row = nearest_neighbors[nearest_neighbors['Taxonomy'] == taxonomy].iloc[0]
        
        # Append row with all required columns
        distribution_rows.append({
            'ID': input_row['ID'],  # From input file
            'Latitude': input_row['latitude'],  # Use input coordinates
            'Longitude': input_row['longitude'],
            'Country': taxonomy_row['Country'],
            'City': taxonomy_row['City'],
            'LLRS Material': taxonomy_row['LLRS Material'],
            'LLRS': taxonomy_row['LLRS'],
            'Code Level': taxonomy_row['Code Level'],
            'Number of Stories': taxonomy_row['Number of Stories'],
            'Occupancy': taxonomy_row['Occupancy'],
            'Block Position': taxonomy_row['Block Position'],
            'Taxonomy': taxonomy,
            'Probability': prob  # Computed probability
        })
    
    return distribution_rows