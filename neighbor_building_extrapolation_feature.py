from geopy.distance import geodesic
from collections import defaultdict

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

# Function to compute taxonomy probabilities using inverse-distance weighted soft voting
def compute_taxonomy_distribution_full_structure(nearest_neighbors, input_row):
    """
    Computes taxonomy probabilities using weighted soft voting based on geodesic distance.

    Parameters:
    - nearest_neighbors: DataFrame containing the neighbors with 'Taxonomy' and 'distance_km' columns.
    - input_row: The row of the input point.
    - kernel: Kernel type ('inverse' or 'gaussian').
    - bandwidth: Bandwidth for the Gaussian kernel.

    Returns:
    - List of dictionaries, each representing a taxonomy and its probability, along with extra metadata.
    """
    class_weights = defaultdict(float)

    # Assign weights to each neighbor based on the chosen kernel
    for _, row in nearest_neighbors.iterrows():
        dist = row['distance_km']
        label = row['Taxonomy']
        
        # Compute weight based on kernel
        weight = 1 / (dist + 1e-6)  # Avoid division by zero
        class_weights[label] += weight

    # Normalize weights to create a probability distribution
    total_weight = sum(class_weights.values())
    probs = {label: weight / total_weight for label, weight in class_weights.items()}

    # Prepare output rows
    distribution_rows = []
    for taxonomy, prob in probs.items():
        # Take a representative row (first one with the taxonomy)
        taxonomy_row = nearest_neighbors[nearest_neighbors['Taxonomy'] == taxonomy].iloc[0]
        
        distribution_rows.append({
            'ID': input_row['ID'],
            'Latitude': input_row['latitude'],
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
            'Probability': prob
        })

    return distribution_rows