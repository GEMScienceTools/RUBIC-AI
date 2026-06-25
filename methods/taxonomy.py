"""Validate, extract, and simplify GEM building taxonomy strings."""

from __future__ import annotations

from typing import Any

import pandas as pd
from openquake.gem_taxonomy import GemTaxonomy

MACRO_CLASSES = {
    "ADO|ST|E": "Adobe/Stone/Earth",
    "MUR": "Unreinforced Masonry",
    "MR|MCF": "Reinforced/Confined Masonry",
    "CR-": "Concrete lower quality",
    "CR+": "Concrete higher quality",
    "W": "Wood",
    "S": "Steel",
    "HYB": "Hybrid or mixed materials",
    "OT": "Others",
}

GEM_TAXONOMY = GemTaxonomy(vers="4.0")


def validate_gem_taxonomy(value: str) -> str | None:
    """Validate one taxonomy string against the GEM taxonomy standard.

    Args:
        value: Taxonomy string to validate. Any suffix after the first pipe
            character is ignored during validation.

    Returns
    -------
        An error message when the taxonomy is invalid; otherwise, ``None``.
    """
    taxonomy = value.split("|")[0]

    try:
        _, _, report = GEM_TAXONOMY.validate(taxonomy)
        if not report["is_canonical"]:
            raise ValueError(str(report))
    except ValueError as error:
        return f"{taxonomy}: {error}"

    return None


def get_canonical_taxonomy(value: str) -> str:
    """Return the canonical form of a GEM taxonomy string.

    Args:
        value: Taxonomy string that may include additional pipe-separated data.

    Returns
    -------
        Canonical taxonomy string with any additional suffix preserved.
    """
    parts = value.split("|")
    taxonomy = parts[0]
    remainder = parts[1:]

    _, _, report = GEM_TAXONOMY.validate(taxonomy)
    if report["is_canonical"]:
        return value

    canonical = report["canonical"]
    if remainder:
        return "|".join([canonical, *remainder])
    return canonical


def check_taxonomy(
    dataframe: pd.DataFrame,
    taxo_col: str = "TAXONOMY",
) -> list[str]:
    """Check that taxonomy strings follow the GEM taxonomy standard.

    Args:
        dataframe: DataFrame containing taxonomy strings.
        taxo_col: Name of the taxonomy column.

    Returns
    -------
        Empty list when every unique taxonomy is valid.

    Raises
    ------
        TypeError: If ``dataframe`` is not a pandas DataFrame.
        KeyError: If ``taxo_col`` is absent from the DataFrame.
        ValueError: If one or more taxonomy strings are invalid.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if taxo_col not in dataframe.columns:
        raise KeyError(f"The column '{taxo_col}' is not in the DataFrame")

    unique_taxonomies = dataframe[taxo_col].dropna().unique()
    check_list = [
        result
        for value in unique_taxonomies
        if (result := validate_gem_taxonomy(value)) is not None
    ]

    if check_list:
        for item in check_list:
            print(item)
        raise ValueError(
            f"Found {len(check_list)} invalid taxonomy strings: {check_list}"
        )

    return check_list


def extract_attributes(value: str) -> dict[str, Any]:
    """Extract taxonomy attributes from a GEM taxonomy string.

    Args:
        value: GEM taxonomy string.

    Returns
    -------
        Dictionary containing the extracted taxonomy attributes.
    """
    return GEM_TAXONOMY.split_by_attributes(value, "|", 0, "others")


def get_eq_design_attrs(value: Any) -> dict[str, str | None]:
    """Split an ``EQ_DESIGN`` value into earthquake-design attributes.

    Args:
        value: Pipe attribute containing plus-separated design atoms.

    Returns
    -------
        Dictionary containing earthquake-code, ductility, and other atoms.

    Raises
    ------
        ValueError: If the input is neither a string nor a missing value.
    """
    if pd.isna(value):
        return {"eq_code": None}
    if not isinstance(value, str):
        raise ValueError("Input value must be a string")

    attributes: dict[str, str | None] = {}
    for part in value.split("+"):
        atom = part.strip()
        if atom in {"CDN", "CDL", "CDM", "CDH"}:
            attributes["eq_code"] = atom
        elif atom in {"ERN", "ERL", "ERM", "ERH", "ERS"}:
            attributes["eq_erd"] = atom
        else:
            attributes["eq_others"] = atom

    return attributes


def add_tax_attrs(
    dataframe: pd.DataFrame,
    taxo_col: str = "TAXONOMY",
    attributes: str | list[str] | None = None,
) -> pd.DataFrame:
    """Add selected GEM taxonomy attributes to a DataFrame.

    Args:
        dataframe: Input DataFrame containing taxonomy strings.
        taxo_col: Name of the taxonomy column.
        attributes: Attribute name or list of attribute names to include. When
            omitted, all extracted attributes are included.

    Returns
    -------
        Copy of the input data merged with the selected taxonomy attributes.
    """
    unique_taxonomies = dataframe[[taxo_col]].drop_duplicates().reset_index(drop=True)
    extracted = pd.DataFrame(
        unique_taxonomies[taxo_col].apply(extract_attributes).tolist()
    )
    taxonomy_attributes = pd.concat([unique_taxonomies, extracted], axis=1)

    if "eq_design" in extracted.columns:
        earthquake_attributes = pd.DataFrame(
            extracted["eq_design"].apply(get_eq_design_attrs).tolist()
        )
        taxonomy_attributes = pd.concat(
            [taxonomy_attributes, earthquake_attributes],
            axis=1,
        )

    taxonomy_attributes.columns = taxonomy_attributes.columns.str.upper()
    taxonomy_attributes = taxonomy_attributes.rename(
        columns={"OCCUPANCY": "OCCUPANCY_TAXO"}
    )

    selected_attributes = _normalize_attribute_selection(
        attributes,
        taxonomy_attributes,
    )
    return dataframe.merge(
        taxonomy_attributes[selected_attributes],
        on=taxo_col,
        how="left",
    )


def _normalize_attribute_selection(
    attributes: str | list[str] | None,
    taxonomy_attributes: pd.DataFrame,
) -> list[str]:
    """Normalize the requested taxonomy attribute columns."""
    if isinstance(attributes, str):
        attributes = [attributes]

    if not attributes:
        return taxonomy_attributes.columns.tolist()

    selected = ["TAXONOMY", *attributes]
    if "OCCUPANCY" in selected:
        selected[selected.index("OCCUPANCY")] = "OCCUPANCY_TAXO"
    return selected


def macrotaxonomy(value: str) -> str:
    """Simplify a GEM taxonomy string into a macro taxonomy class.

    Args:
        value: GEM taxonomy string.

    Returns
    -------
        Macro taxonomy class code.

    Raises
    ------
        ValueError: If the taxonomy has no material or is unsupported.
    """
    attributes = extract_attributes(value)
    material = attributes.get("material")
    if material is None:
        raise ValueError(f"{value} does not have a material attribute")

    eq_attributes = get_eq_design_attrs(attributes.get("eq_design"))
    eq_code = eq_attributes.get("eq_code") or ""
    eq_erd = eq_attributes.get("eq_erd") or ""

    material_class = _classify_material(material)
    if material_class is not None:
        return material_class

    if material.startswith("CR"):
        return _classify_concrete(eq_code, eq_erd)

    raise ValueError(f"{value} not included in GEM MACRO_TAXONOMY")


def _classify_material(material: str) -> str | None:
    """Return a non-concrete macro class for a material string."""
    prefix_classes = (
        (("HYB",), "HYB"),
        (("S",), "S"),
        (("W",), "W"),
        (("MATO", "ME", "INF"), "OT"),
        (("EU", "ER", "E+"), "ADO|ST|E"),
        (("MR", "MCF"), "MR|MCF"),
        (("M+",), "MUR"),
    )

    if "+ADO" in material:
        return "ADO|ST|E"
    if material.startswith("MUR"):
        return "ADO|ST|E" if "+ST" in material else "MUR"

    for prefixes, macro_class in prefix_classes:
        if material.startswith(prefixes):
            return macro_class

    return None


def _classify_concrete(eq_code: str, eq_erd: str) -> str:
    """Classify concrete quality using code and ductility attributes."""
    if any(value in eq_code for value in ("CDN", "CDL")) or any(
        value in eq_erd for value in ("ERN", "ERL")
    ):
        return "CR-"

    if any(value in eq_code for value in ("CDM", "CDH")) or any(
        value in eq_erd for value in ("ERM", "ERH")
    ):
        return "CR+"

    return "CR"


def add_macro_taxonomy(
    data: pd.DataFrame,
    taxo_col: str = "TAXONOMY",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add a simplified ``MACRO_TAXONOMY`` column to a DataFrame.

    Args:
        data: Input DataFrame.
        taxo_col: Name of the taxonomy column.
        overwrite: Whether to replace an existing ``MACRO_TAXONOMY`` column.

    Returns
    -------
        DataFrame containing the ``MACRO_TAXONOMY`` column.

    Raises
    ------
        TypeError: If ``data`` is not a pandas DataFrame.
        KeyError: If ``taxo_col`` is absent from the DataFrame.
        ValueError: If the macro taxonomy column already exists and overwrite is
            disabled, or if an unsupported macro class is produced.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if taxo_col not in data.columns:
        raise KeyError(f"The column '{taxo_col}' is not in the DataFrame")

    dataframe = data.copy()
    original_columns = dataframe.columns.tolist()

    if "MACRO_TAXONOMY" in dataframe.columns:
        if not overwrite:
            raise ValueError(
                "Column 'MACRO_TAXONOMY' is already present. "
                "Set overwrite=True to replace it."
            )
        print("  Overwriting the MACRO_TAXONOMY column")
        dataframe = dataframe.drop(columns=["MACRO_TAXONOMY"])
    else:
        print("  Adding `MACRO_TAXONOMY` column")
        original_columns.append("MACRO_TAXONOMY")

    unique_taxonomies = dataframe[[taxo_col]].drop_duplicates().reset_index(drop=True)
    unique_taxonomies["MACRO_TAXONOMY"] = unique_taxonomies[taxo_col].apply(
        macrotaxonomy
    )

    _validate_macro_classes(unique_taxonomies["MACRO_TAXONOMY"])
    dataframe = dataframe.merge(unique_taxonomies, on=taxo_col, how="left")
    return dataframe[original_columns]


def _validate_macro_classes(macro_taxonomies: pd.Series) -> None:
    """Validate generated macro taxonomy classes."""
    unique_classes = set(macro_taxonomies)
    unique_classes.discard("CR")

    unsupported = unique_classes.difference(MACRO_CLASSES)
    if unsupported:
        raise ValueError(f"Unsupported macro taxonomy classes: {unsupported}")
