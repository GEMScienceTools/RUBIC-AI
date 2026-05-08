from pathlib import Path
import pandas as pd


# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------
CLASSIFICATION_CSV = Path("Local_images_AI_classification.csv")
COORDINATES_CSV = Path("data_ex2.csv")
OUTPUT_CSV = Path("results_prueba_Daniel.csv")

LAT_COL = "latitude"
LON_COL = "longitude"
FILENAME_COL = "image filename or link"
DATA_IMAGE_ID_COL = "id"      # Column in data_ex2.csv containing the filenames
COORD_DECIMALS = 8            # Increase/decrease depending on coordinate precision


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def replicate_classifications_by_coordinates(
    classification_csv: Path,
    coordinates_csv: Path,
    output_csv: Path,
    coord_decimals: int = 8,
) -> pd.DataFrame:

    # Read files
    classification_df = pd.read_csv(classification_csv)
    coordinates_df = pd.read_csv(coordinates_csv)

    # Check required columns
    required_classification_cols = {LAT_COL, LON_COL, FILENAME_COL}
    required_coordinates_cols = {DATA_IMAGE_ID_COL, LAT_COL, LON_COL}

    missing_classification = required_classification_cols - set(classification_df.columns)
    missing_coordinates = required_coordinates_cols - set(coordinates_df.columns)

    if missing_classification:
        raise ValueError(
            f"Missing columns in {classification_csv}: {sorted(missing_classification)}"
        )

    if missing_coordinates:
        raise ValueError(
            f"Missing columns in {coordinates_csv}: {sorted(missing_coordinates)}"
        )

    # Keep original order so the output follows the input classification order
    # and, within each classification row, the order in data_ex2.csv.
    classification_df = classification_df.copy()
    coordinates_df = coordinates_df.copy()
    classification_df["__classification_order"] = range(len(classification_df))
    coordinates_df["__coordinates_order"] = range(len(coordinates_df))

    # Create rounded coordinate keys for robust matching.
    classification_df["__lat_key"] = classification_df[LAT_COL].round(coord_decimals)
    classification_df["__lon_key"] = classification_df[LON_COL].round(coord_decimals)
    coordinates_df["__lat_key"] = coordinates_df[LAT_COL].round(coord_decimals)
    coordinates_df["__lon_key"] = coordinates_df[LON_COL].round(coord_decimals)

    # Rename data_ex2.csv id column to avoid conflict with the classification id.
    coordinates_df = coordinates_df.rename(columns={DATA_IMAGE_ID_COL: "__matched_filename"})

    # Merge: one classification row is repeated for every coordinate match.
    expanded_df = classification_df.merge(
        coordinates_df[["__matched_filename", "__lat_key", "__lon_key", "__coordinates_order"]],
        on=["__lat_key", "__lon_key"],
        how="inner",
    )

    # Replace the image filename/link with the matching filename from data_ex2.csv.
    expanded_df[FILENAME_COL] = expanded_df["__matched_filename"]

    # Restore stable order.
    expanded_df = expanded_df.sort_values(
        by=["__classification_order", "__coordinates_order"],
        kind="stable",
    )

    # Remove helper columns and preserve the original classification CSV columns.
    expanded_df = expanded_df[classification_df.drop(columns=[
        "__classification_order", "__lat_key", "__lon_key"
    ]).columns]

    # Save result.
    expanded_df.to_csv(output_csv, index=False)

    return expanded_df


if __name__ == "__main__":
    replicate_classifications_by_coordinates(
        classification_csv=CLASSIFICATION_CSV,
        coordinates_csv=COORDINATES_CSV,
        output_csv=OUTPUT_CSV,
        coord_decimals=COORD_DECIMALS,
    )
