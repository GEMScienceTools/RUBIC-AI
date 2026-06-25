"""Convert OpenQuake NRML vulnerability and fragility XML data."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

NRML_NAMESPACE = {"nrml": "http://openquake.org/xmlns/nrml/0.5"}
REQUEST_TIMEOUT_SECONDS = 30


def _load_xml_content(xml_source: str) -> str:
    """Load XML content from a URL or local file."""
    if xml_source.startswith(("http://", "https://")):
        try:
            response = requests.get(
                xml_source,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not download XML from {xml_source!r}."
            ) from error
        return response.text

    try:
        return Path(xml_source).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Could not read the XML file {xml_source!r}.") from error


def _parse_xml(xml_content: str) -> ET.Element:
    """Parse XML text and return its root element."""
    try:
        return ET.fromstring(xml_content)
    except ET.ParseError as error:
        raise ValueError("The XML content could not be parsed.") from error


def _tag_matches(element: ET.Element, tag_name: str) -> bool:
    """Return whether an element matches a local XML tag name."""
    return element.tag == tag_name or element.tag.endswith(f"}}{tag_name}")


def _find_child(
    parent: ET.Element,
    tag_name: str,
) -> ET.Element | None:
    """Find a direct child by local XML tag name."""
    direct_match = parent.find(tag_name)
    if direct_match is not None:
        return direct_match

    return next(
        (child for child in parent if _tag_matches(child, tag_name)),
        None,
    )


def _find_all_children(
    parent: ET.Element,
    tag_name: str,
) -> list[ET.Element]:
    """Find all direct children matching a local XML tag name."""
    return [child for child in parent if _tag_matches(child, tag_name)]


def _parse_float_values(
    element: ET.Element,
    element_name: str,
) -> list[float]:
    """Parse whitespace-separated floating-point values from an XML element."""
    if element.text is None or not element.text.strip():
        raise ValueError(f"The {element_name!r} element contains no values.")

    try:
        return [float(value) for value in element.text.split()]
    except ValueError as error:
        raise ValueError(
            f"The {element_name!r} element contains a non-numeric value."
        ) from error


def _find_function_by_id(
    root: ET.Element,
    tag_name: str,
    function_id: str,
) -> ET.Element | None:
    """Find an NRML function element by tag name and identifier."""
    for element in root.iter():
        if _tag_matches(element, tag_name) and element.get("id") == function_id:
            return element
    return None


def _safe_output_filename(function_id: str) -> str:
    """Create a filesystem-safe CSV filename from a function identifier."""
    safe_id = function_id.replace("/", "_").replace("+", "_")
    return f"{safe_id}.csv"


def _save_dataframe(
    dataframe: pd.DataFrame,
    output_filename: str,
) -> None:
    """Save a DataFrame to CSV and report the resulting path."""
    dataframe.to_csv(output_filename, index=False)
    print(f"Data saved to: {output_filename}")


def extract_vulnerability_function(
    xml_url: str,
    function_id: str,
    save_csv: bool = False,
    output_filename: str | None = None,
) -> pd.DataFrame:
    """Extract one vulnerability function from an OpenQuake NRML XML file.

    Parameters
    ----------
    xml_url : str
        URL or local path to the XML file.
    function_id : str
        Identifier of the vulnerability function to extract.
    save_csv : bool, default=False
        Whether to save the extracted data as a CSV file.
    output_filename : str or None, default=None
        Output CSV path. When omitted, a safe name is derived from
        ``function_id``.

    Returns
    -------
    pandas.DataFrame
        Vulnerability data with ``IMT``, ``IML``, ``meanLR``, and ``covLR``
        columns.

    Raises
    ------
    RuntimeError
        If the XML source cannot be loaded.
    ValueError
        If the XML is invalid or the requested function is incomplete.
    """
    root = _parse_xml(_load_xml_content(xml_url))
    vulnerability_function = _find_function_by_id(
        root,
        "vulnerabilityFunction",
        function_id,
    )
    if vulnerability_function is None:
        raise ValueError(
            f"Vulnerability function with id {function_id!r} was not found."
        )

    imls_element = _find_child(vulnerability_function, "imls")
    mean_lrs_element = _find_child(vulnerability_function, "meanLRs")
    cov_lrs_element = _find_child(vulnerability_function, "covLRs")

    missing_elements = [
        name
        for name, element in (
            ("imls", imls_element),
            ("meanLRs", mean_lrs_element),
            ("covLRs", cov_lrs_element),
        )
        if element is None
    ]
    if missing_elements:
        missing = ", ".join(missing_elements)
        raise ValueError(f"The vulnerability function is missing: {missing}.")

    imls = _parse_float_values(imls_element, "imls")
    mean_lrs = _parse_float_values(mean_lrs_element, "meanLRs")
    cov_lrs = _parse_float_values(cov_lrs_element, "covLRs")

    if len({len(imls), len(mean_lrs), len(cov_lrs)}) != 1:
        raise ValueError(
            "Array length mismatch: "
            f"imls={len(imls)}, meanLRs={len(mean_lrs)}, "
            f"covLRs={len(cov_lrs)}."
        )

    imt = imls_element.get("imt", "")
    dataframe = pd.DataFrame(
        {
            "IMT": [imt] * len(imls),
            "IML": imls,
            "meanLR": mean_lrs,
            "covLR": cov_lrs,
        }
    )
    dataframe.attrs.update(
        {
            "function_id": function_id,
            "distribution": vulnerability_function.get("dist", ""),
            "imt": imt,
        }
    )

    if save_csv:
        filename = output_filename or _safe_output_filename(function_id)
        _save_dataframe(dataframe, filename)

    return dataframe


def _find_fragility_functions(root: ET.Element) -> list[ET.Element]:
    """Return all fragility function elements in an XML document."""
    functions = root.findall(
        ".//nrml:fragilityFunction",
        NRML_NAMESPACE,
    )
    if functions:
        return functions

    return [
        element for element in root.iter() if _tag_matches(element, "fragilityFunction")
    ]


def _fragility_records(
    fragility_function: ET.Element,
) -> list[dict[str, str | float]]:
    """Convert one fragility function element into long-format records."""
    imls_element = _find_child(fragility_function, "imls")
    if imls_element is None or imls_element.text is None:
        return []

    function_id = fragility_function.get("id", "")
    imt = imls_element.get("imt", "")
    imls = _parse_float_values(imls_element, "imls")
    records: list[dict[str, str | float]] = []

    for poes_element in _find_all_children(fragility_function, "poes"):
        if poes_element.text is None:
            continue

        limit_state = poes_element.get("ls", "")
        poes = _parse_float_values(poes_element, "poes")
        if len(poes) != len(imls):
            raise ValueError(
                f"Array length mismatch in function {function_id!r}, "
                f"limit state {limit_state!r}: imls={len(imls)}, "
                f"poes={len(poes)}."
            )

        records.extend(
            {
                "function_id": function_id,
                "imt": imt,
                "iml": iml,
                "limit_state": limit_state,
                "poe": poe,
            }
            for iml, poe in zip(imls, poes, strict=True)
        )

    return records


def fragility_xml_to_csv(
    xml_url: str,
    save_csv: bool = True,
    output_filename: str | None = None,
) -> pd.DataFrame:
    """Convert an OpenQuake discrete fragility XML file to long format.

    Parameters
    ----------
    xml_url : str
        URL or local path to the fragility XML file.
    save_csv : bool, default=True
        Whether to save the resulting DataFrame as a CSV file.
    output_filename : str or None, default=None
        Output CSV path. The default is ``fragility_from_xml.csv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``function_id``, ``imt``, ``iml``, ``limit_state``,
        and ``poe`` columns.

    Raises
    ------
    RuntimeError
        If the XML source cannot be loaded.
    ValueError
        If the XML is invalid or contains inconsistent fragility data.
    """
    root = _parse_xml(_load_xml_content(xml_url))
    functions = _find_fragility_functions(root)
    if not functions:
        raise ValueError("No fragilityFunction elements were found in the XML.")

    records = [
        record for function in functions for record in _fragility_records(function)
    ]
    columns = ["function_id", "imt", "iml", "limit_state", "poe"]
    dataframe = pd.DataFrame(records, columns=columns)

    if save_csv:
        filename = output_filename or "fragility_from_xml.csv"
        _save_dataframe(dataframe, filename)

    return dataframe
