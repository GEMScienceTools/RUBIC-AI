"""
xml_to_csv_converter.py
=======================
This module provides functions to extract and convert OpenQuake NRML vulnerability
and fragility XML data into pandas DataFrames or CSV files.
"""

import pandas as pd
import xml.etree.ElementTree as ET
import requests
from typing import Optional


def extract_vulnerability_function(xml_url: str, function_id: str, save_csv: bool = False, 
                                   output_filename: Optional[str] = None) -> pd.DataFrame:
    """
    Extract a specific vulnerability function from XML and convert to DataFrame.
    
    Parameters:
    -----------
    xml_url : str
        URL or local path to the XML file
    function_id : str
        ID of the vulnerability function to extract (e.g., "CR_LDUAL+CDH+DUH_H6/RES")
    save_csv : bool
        Whether to save the result as a CSV file
    output_filename : str, optional
        Name of the output CSV file (default: uses function_id as filename)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the vulnerability function data
    """
    
    # Fetch XML content
    try:
        if xml_url.startswith('http://') or xml_url.startswith('https://'):
            response = requests.get(xml_url)
            response.raise_for_status()
            xml_content = response.text
        else:
            with open(xml_url, 'r') as f:
                xml_content = f.read()
    except Exception as e:
        raise Exception(f"Error fetching XML: {str(e)}")
    
    # Parse XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise Exception(f"Error parsing XML: {str(e)}")
    
    # Find the specific vulnerability function
    # Handle both with and without namespaces
    vuln_func = None
    
    # Try to find with namespace
    namespaces = {'nrml': 'http://openquake.org/xmlns/nrml/0.5'}
    for func in root.findall('.//vulnerabilityFunction', namespaces):
        if func.get('id') == function_id:
            vuln_func = func
            break
    
    # If not found, try without namespace
    if vuln_func is None:
        for func in root.iter():
            if func.tag.endswith('vulnerabilityFunction') or func.tag == 'vulnerabilityFunction':
                if func.get('id') == function_id:
                    vuln_func = func
                    break
    
    if vuln_func is None:
        raise ValueError(f"Vulnerability function with id '{function_id}' not found in XML")
    
    # Extract data
    dist = vuln_func.get('dist', '')
    
    # Extract IMT (Intensity Measure Type)
    imls_elem = vuln_func.find('imls')
    if imls_elem is None:
        # Try finding by tag name ending (for namespaced XML)
        for child in vuln_func:
            if child.tag.endswith('imls') or child.tag == 'imls':
                imls_elem = child
                break
    
    if imls_elem is None:
        raise ValueError("No 'imls' element found in vulnerability function")
    
    imt = imls_elem.get('imt', '')
    imls_text = imls_elem.text.strip()
    imls = [float(x) for x in imls_text.split()]
    
    # Extract mean loss ratios
    mean_lrs_elem = vuln_func.find('meanLRs')
    if mean_lrs_elem is None:
        for child in vuln_func:
            if child.tag.endswith('meanLRs') or child.tag == 'meanLRs':
                mean_lrs_elem = child
                break
    
    if mean_lrs_elem is None:
        raise ValueError("No 'meanLRs' element found in vulnerability function")
    
    mean_lrs_text = mean_lrs_elem.text.strip()
    mean_lrs = [float(x) for x in mean_lrs_text.split()]
    
    # Extract coefficient of variation of loss ratios
    cov_lrs_elem = vuln_func.find('covLRs')
    if cov_lrs_elem is None:
        for child in vuln_func:
            if child.tag.endswith('covLRs') or child.tag == 'covLRs':
                cov_lrs_elem = child
                break
    
    if cov_lrs_elem is None:
        raise ValueError("No 'covLRs' element found in vulnerability function")
    
    cov_lrs_text = cov_lrs_elem.text.strip()
    cov_lrs = [float(x) for x in cov_lrs_text.split()]
    
    # Verify all arrays have the same length
    if not (len(imls) == len(mean_lrs) == len(cov_lrs)):
        raise ValueError(f"Array length mismatch: imls={len(imls)}, meanLRs={len(mean_lrs)}, covLRs={len(cov_lrs)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'IMT': [imt] * len(imls),
        'IML': imls,
        'meanLR': mean_lrs,
        'covLR': cov_lrs
    })
    
    # Add metadata as attributes
    df.attrs['function_id'] = function_id
    df.attrs['distribution'] = dist
    df.attrs['imt'] = imt
    
    # Save to CSV if requested
    if save_csv:
        if output_filename is None:
            # Create safe filename from function_id
            safe_id = function_id.replace('/', '_').replace('+', '_')
            output_filename = f"{safe_id}.csv"
        
        df.to_csv(output_filename, index=False)
        print(f"Data saved to: {output_filename}")
    
    return df


def fragility_xml_to_csv(xml_url: str,
                         save_csv: bool = True,
                         output_filename: Optional[str] = None) -> pd.DataFrame:
    """
    Convert an OpenQuake NRML *fragility* XML file (e.g., fragility_structural.xml)
    into a long-format CSV/DataFrame with **5 columns**:

        function_id | imt | iml | limit_state | poe

    Notes
    -----
    - This function targets "Discrete" fragility functions with:
        <imls imt="..."> ... </imls>
        <poes ls="..."> ... </poes>
      which is the common format in GEM/OpenQuake fragility models.
    - If a function has multiple <poes> (one per limit state), all are included.
    - Rows are stacked (long format): one row per (function_id, limit_state, iml).

    Parameters
    ----------
    xml_url : str
        URL or local path to the fragility XML file.
    save_csv : bool
        Whether to save the result as a CSV file.
    output_filename : str, optional
        Output CSV file name. Default: "fragility_from_xml.csv"

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [function_id, imt, iml, limit_state, poe]
    """

    # Fetch XML content
    try:
        if xml_url.startswith('http://') or xml_url.startswith('https://'):
            response = requests.get(xml_url)
            response.raise_for_status()
            xml_content = response.text
        else:
            with open(xml_url, 'r', encoding='utf-8') as f:
                xml_content = f.read()
    except Exception as e:
        raise Exception(f"Error fetching XML: {str(e)}")

    # Parse XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise Exception(f"Error parsing XML: {str(e)}")

    # Find fragility functions (with and without namespaces)
    namespaces = {'nrml': 'http://openquake.org/xmlns/nrml/0.5'}
    funcs = root.findall('.//nrml:fragilityFunction', namespaces)

    if not funcs:
        funcs = []
        for el in root.iter():
            if el.tag.endswith('fragilityFunction') or el.tag == 'fragilityFunction':
                funcs.append(el)

    if not funcs:
        raise ValueError("No 'fragilityFunction' elements found in XML")

    records = []

    for func in funcs:
        function_id = func.get('id', '')

        # Find <imls>
        imls_elem = None
        for child in func:
            if child.tag.endswith('imls') or child.tag == 'imls':
                imls_elem = child
                break

        if imls_elem is None or imls_elem.text is None:
            # Skip functions without IMLs
            continue

        imt = imls_elem.get('imt', '')
        imls = [float(x) for x in imls_elem.text.split()]

        # Find all <poes ls="...">
        poes_elems = []
        for child in func:
            if child.tag.endswith('poes') or child.tag == 'poes':
                poes_elems.append(child)

        for poes in poes_elems:
            limit_state = poes.get('ls', '')
            if poes.text is None:
                continue

            poe_vals = [float(x) for x in poes.text.split()]

            # Defensive check: sizes must match
            if len(poe_vals) != len(imls):
                raise ValueError(
                    f"Array length mismatch in function '{function_id}', "
                    f"limit_state='{limit_state}': imls={len(imls)} vs poes={len(poe_vals)}"
                )

            for iml, poe in zip(imls, poe_vals):
                records.append({
                    "function_id": function_id,
                    "imt": imt,
                    "iml": iml,
                    "limit_state": limit_state,
                    "poe": poe
                })

    df = pd.DataFrame(records, columns=["function_id", "imt", "iml", "limit_state", "poe"])

    # Save to CSV if requested
    if save_csv:
        if output_filename is None:
            output_filename = "fragility_from_xml.csv"
        df.to_csv(output_filename, index=False)
        print(f"Data saved to: {output_filename}")

    return df