"""Provide shared utility functions for the RUBIC-AI graphical interface."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from PyQt5 import QtWidgets
from shapely.geometry import Polygon

PREVIEW_ROW_LIMIT = 10
BASE_FONT_SIZE = 10
REQUIRED_COORDINATE_COLUMNS = ("id", "latitude", "longitude")


def select_output_folder(parent=None):
    """Open an output-folder dialog and return the selected path and name.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget for the dialog.

    Returns
    -------
    tuple[str | None, str | None]
        Selected folder path and folder name. Both values are ``None`` when
        the dialog is cancelled.
    """
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        parent,
        "Select Output Folder",
    )
    if folder:
        folder_path = Path(folder)
        return str(folder_path), folder_path.name

    return None, None


def preview_data(widget):
    """Display the first rows of the loaded data in the table widget.

    Parameters
    ----------
    widget : QWidget
        Object containing ``df``, ``tableWidget``, and ``sf_font``
        attributes.
    """
    if not hasattr(widget, "df") or widget.df.empty:
        QtWidgets.QMessageBox.warning(
            widget,
            "No Data",
            "No data available to preview. Please upload a valid CSV first.",
        )
        return

    preview_df = widget.df.head(PREVIEW_ROW_LIMIT)
    table = widget.tableWidget

    table.clear()
    table.setRowCount(len(preview_df))
    table.setColumnCount(len(preview_df.columns))
    table.setHorizontalHeaderLabels(preview_df.columns.astype(str).tolist())

    font_size = int(BASE_FONT_SIZE * widget.sf_font)
    _configure_table_headers(table, font_size)

    for row_index, row_values in enumerate(preview_df.itertuples(index=False)):
        for column_index, value in enumerate(row_values):
            item = QtWidgets.QTableWidgetItem(str(value))
            font = item.font()
            font.setPointSize(font_size)
            item.setFont(font)
            table.setItem(row_index, column_index, item)

    table.resizeColumnsToContents()


def _configure_table_headers(table, font_size):
    """Apply consistent font settings to table headers."""
    horizontal_header = table.horizontalHeader()
    horizontal_font = horizontal_header.font()
    horizontal_font.setPointSize(font_size)
    horizontal_font.setBold(True)
    horizontal_header.setFont(horizontal_font)

    vertical_header = table.verticalHeader()
    vertical_font = vertical_header.font()
    vertical_font.setPointSize(font_size)
    vertical_header.setFont(vertical_font)


def upload_csv(widget):
    """Open, validate, and preview a CSV file selected by the user.

    Parameters
    ----------
    widget : QWidget
        Object containing the current inspection settings and related labels.

    Returns
    -------
    pandas.DataFrame or None
        Loaded DataFrame for extrapolation workflows; otherwise ``None``.
    """
    options = QtWidgets.QFileDialog.Options()
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        widget,
        "Open CSV File",
        "",
        "CSV Files (*.csv);;All Files (*)",
        options=options,
    )

    display_label = _get_csv_display_label(widget, file_path)

    if not file_path:
        display_label.setText("No file selected.")
        QtWidgets.QMessageBox.warning(
            widget,
            "Input Error",
            "No file selected.",
        )
        return None

    try:
        widget.df = pd.read_csv(file_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        display_label.setText(f"Error: {error}")
        QtWidgets.QMessageBox.warning(
            widget,
            "Input Error",
            "Invalid file selected or parsing error.",
        )
        return None

    display_label.setText(Path(file_path).name)
    preview_data(widget)

    if widget.method.insp_method == 3:
        return widget.df

    return None


def _get_csv_display_label(widget, file_path):
    """Return the label used to display the selected CSV filename."""
    inspection_method = widget.method.insp_method

    if inspection_method == 0:
        return widget.polygon_path_value
    if inspection_method == 1:
        return widget.specific_path
    if inspection_method == 2:
        widget.method.file_local_csv = file_path
        return widget.local_path

    try:
        return widget.population_new_path
    except AttributeError:
        return widget.label_path


def save_coordinates(widget):
    """Validate coordinates and save them as a GeoPackage.

    Parameters
    ----------
    widget : QWidget
        Object containing the loaded DataFrame and current method settings.
    """
    try:
        data = widget.df
        output_folder = Path(widget.method.output_folder_value)
        inspection_method = widget.method.insp_method

        if inspection_method == 0:
            _save_polygon_coordinates(widget, data, output_folder)
        elif inspection_method in (1, 2):
            _save_point_coordinates(widget, data, output_folder)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        QtWidgets.QMessageBox.warning(
            widget,
            "Input Error",
            f"Some required inputs are missing or invalid.\n\nDetails: {error}",
        )


def _save_polygon_coordinates(widget, data, output_folder):
    """Save polygon-boundary coordinates to a GeoPackage."""
    if widget.polygon_source_value.currentIndex() == 1:
        return

    if not _validate_columns(widget, data, REQUIRED_COORDINATE_COLUMNS):
        return

    output_path = _build_output_path(
        output_folder,
        widget.output_polygon.text(),
        "_boundary",
    )
    widget.boundary_path = str(output_path)
    _remove_if_exists(output_path)

    try:
        polygon = Polygon(zip(data["longitude"], data["latitude"], strict=False))
        geodata = gpd.GeoDataFrame(
            {"geometry": [polygon]},
            crs="EPSG:4326",
        )
        _save_gpkg(geodata, output_path, layer="polygon_layer")
    except (OSError, TypeError, ValueError) as error:
        QtWidgets.QMessageBox.warning(
            widget,
            "Error",
            f"Could not generate GPKG:\n{error}",
        )


def _save_point_coordinates(widget, data, output_folder):
    """Save point coordinates to a GeoPackage and close the dialog."""
    if not _validate_columns(widget, data, REQUIRED_COORDINATE_COLUMNS):
        return

    if widget.method.insp_method == 1:
        output_name = widget.output_specific.text()
    else:
        output_name = widget.output_local.text()

    output_path = _build_output_path(output_folder, output_name)
    _remove_if_exists(output_path)

    try:
        geodata = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(
                data["longitude"],
                data["latitude"],
            ),
            crs="EPSG:4326",
        )
        _save_gpkg(geodata, output_path)
    except (OSError, TypeError, ValueError) as error:
        QtWidgets.QMessageBox.warning(
            widget,
            "Error",
            f"Could not generate GPKG:\n{error}",
        )
        return

    if widget.method.insp_method == 1:
        widget.method.specific_output_name = widget.output_specific
    else:
        widget.method.local_output_name = widget.output_local

    mode_use(widget)
    widget.accept()


def _build_output_path(folder, name, suffix=""):
    """Build a GeoPackage output path."""
    return Path(folder) / f"{name}{suffix}.gpkg"


def _remove_if_exists(path):
    """Remove an existing file."""
    path = Path(path)
    if path.exists():
        path.unlink()


def _validate_columns(widget, data, required_columns):
    """Validate that the DataFrame contains all required columns."""
    missing = [column for column in required_columns if column not in data.columns]
    if not missing:
        return True

    QtWidgets.QMessageBox.warning(
        widget,
        "Missing Columns",
        f"Required columns missing: {', '.join(missing)}",
    )
    return False


def _save_gpkg(geodata, path, driver="GPKG", layer=None):
    """Save a GeoDataFrame to a GeoPackage."""
    options = {"driver": driver}
    if layer:
        options["layer"] = layer

    geodata.to_file(path, **options)


def mode_use(widget):
    """Set the AI collection flag from the selected collection mode.

    Parameters
    ----------
    widget : QWidget
        Object containing ``collection_mode`` and ``ai_value`` attributes.
    """
    widget.ai_value = widget.collection_mode.currentText() == "AI Powered"
