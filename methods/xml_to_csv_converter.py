import pandas as pd
import xml.etree.ElementTree as ET
import requests
from typing import Optional, Union
import sys


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


def main():
    """
    Main function for command-line usage
    """
    # Example usage
    xml_url = "https://raw.githubusercontent.com/gem/global_vulnerability_model/refs/heads/v2025.0.0/Europe/Italy/vulnerability_structural.xml"
    function_id = "CR_LDUAL+CDH+DUH_H6/RES"
    
    print(f"Extracting vulnerability function: {function_id}")
    print(f"From: {xml_url}\n")
    
    # Extract and convert to DataFrame
    df = extract_vulnerability_function(xml_url, function_id, save_csv=True)
    
    # Display results
    print("\nDataFrame Preview:")
    print("=" * 80)
    print(f"Function ID: {df.attrs['function_id']}")
    print(f"Distribution: {df.attrs['distribution']}")
    print(f"IMT: {df.attrs['imt']}")
    print(f"Number of data points: {len(df)}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nLast 10 rows:")
    print(df.tail(10))
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame Statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
