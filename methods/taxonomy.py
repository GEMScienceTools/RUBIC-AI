"""
taxonomy.py
===========
This module provides utility functions for validating, extracting, and simplifying
building taxonomy strings following the GEM Taxonomy standard.
"""

import pandas as pd
from openquake.gem_taxonomy import GemTaxonomy


MACRO_CLASSES = {
    'ADO|ST|E' : 'Adobe/Stone/Earth',
    'MUR' : 'Unreinforced Masonry',
    'MR|MCF' : 'Reinforced/Confined Masonry',
    'CR-' : 'Concrete lower quality',
    'CR+' : 'Concrete higher quality',
    'W' : 'Wood',
    'S' : 'Steel',
    'HYB' : 'Hybrid or mixed materials',
    'OT' : 'Others',
    }

                 
gt = GemTaxonomy(vers="4.0")

def validate_gem_taxonomy(value):
    try:
        value = value.split('|')[0]  # Take only the first part of the taxonomy string
        _, _, report = gt.validate(value)
        if report['is_canonical'] is False:
            # Raise error if the taxonomy is not canonical
            raise ValueError(f'{report}')
    except ValueError as e:
        return f'{value}: {e}'
    

def get_canonical_taxonomy(value):
    taxo = value.split('|')[0]
    rest = value.split('|')[1:] if '|' in value else []

    _, _, report = gt.validate(taxo)
    if report['is_canonical'] is False:
        canonical = report['canonical']
        return canonical + '|' + '|'.join(rest) if rest else canonical
    else:
        return value


def check_taxonomy(df, taxo_col='TAXONOMY'):  
    '''
    Test that the taxonomy strings follow GEM Taxonomy standard.

    Parameters
    ----------
    df : a DataFrame object
        It should include a column for a given taxonomy string.
    taxo_col : str, default 'TAXONOMY'
        Column with taxonomy strings to be checked

    Returns
    -------
    check_list : list.
        List of building classes and associated error.

    '''
    assert isinstance(df, pd.DataFrame), "data must be a pandas DataFrame"

    unique_taxonomy = list(df[taxo_col].unique())
    check_list = [result for val in unique_taxonomy if (result := validate_gem_taxonomy(val)) is not None]

    if len(check_list) > 0:
        for item in check_list:
            print(item)
        raise ValueError(f'Found {len(check_list)} invalid taxonomy strings: {check_list}')
    else:
        return check_list


def extract_attributes(value):
    '''
    Extracts taxonomy attributes from a taxonomy string using the GemTaxonomy utility.
    Returns a dictionary of extracted attributes.
    '''
    # split_by_attributes(taxonomy_string, field_sep, taxonomy_field_idx, key_name)
    attr_dict = gt.split_by_attributes(value, '|', 0, 'others')
    return attr_dict


def get_eq_design_attrs(value):
    '''
    Function to split the EQ_DESIGN string (from the exposure summaries)
    '''

    # Check if the value is NaN or not a string
    if pd.isna(value):
        return {'eq_code': None}
    elif not isinstance(value, str) and not pd.isna(value):
        raise ValueError("Input value must be a string")
    
    # Get different atoms by splitting by '+' 
    parts = value.split('+')
    attrs = {}
    for part in parts:
        if part.strip() in ['CDN', 'CDL', 'CDM', 'CDH']:
            attrs['eq_code'] = part.strip()
        elif part.strip() in ['ERN', 'ERL', 'ERM', 'ERH', 'ERS']:
            attrs['eq_erd'] = part.strip()
        else:
            attrs['eq_others'] = part.strip()
    return attrs


def add_tax_attrs(df, taxo_col='TAXONOMY', attributes=None):
    '''
    Add taxonomy attributes to DataFrame

    df : DataFrame
        Input DataFrame
    taxo_col : str, default 'TAXONOMY'
        Column with taxonomy strings to be used as reference
    attrs : string or list, optional
        only return specific attribute columns.
    '''
    # Apply the function to the taxo_col
    df_unique = df[[taxo_col]].drop_duplicates().reset_index(drop=True)
    attrs = pd.DataFrame(df_unique[taxo_col].apply(extract_attributes).tolist())

    df_attrs = pd.concat([df_unique, attrs], axis=1)
    # Add EQ_DESIGN attributes if present
    if 'eq_design' in attrs.columns:
        eq_attrs = pd.DataFrame(attrs['eq_design'].apply(get_eq_design_attrs).tolist())
        df_attrs = pd.concat([df_attrs, eq_attrs], axis=1)

    df_attrs.columns = df_attrs.columns.str.upper()
    df_attrs = df_attrs.rename(columns={'OCCUPANCY': 'OCCUPANCY_TAXO'}) # It's creating duplicated columns with the model OCCUPANCY column

    # Select attributes to include in DataFrame
    if isinstance(attributes, str):
        attributes = [attributes]
    if attributes:
        attributes = ['TAXONOMY'] + attributes
        if 'OCCUPANCY' in attributes:
            attributes[attributes.index('OCCUPANCY')] = 'OCCUPANCY_TAXO'
    else:
        attributes = df_attrs.columns.tolist()
    
    df2 = df.merge(df_attrs[attributes], on=taxo_col, how='left')
    
    return df2


def macrotaxonomy(value):
    '''
    Function to simplify GEM taxonomy string
    '''
    atts = extract_attributes(value)

    # Get material
    if 'material' in atts:
        material = atts['material']
    else:
        raise ValueError(f'{value} does not have a material attribute')
    
    # Get EQ_DESIGN attributes
    if 'eq_design' in atts:
        eq_atts = get_eq_design_attrs(atts['eq_design'])
    else:
        eq_atts = {'eq_code': '', 'eq_erd': ''}
    if 'eq_code' in eq_atts.keys():
        eq_code = eq_atts['eq_code']
    else:
        eq_code = ''
    if 'eq_erd' in eq_atts.keys():
        eq_erd = eq_atts['eq_erd']
    else:
        eq_erd = ''
    
    # Assign macro taxonomy based on material and EQ_DESIGN
    if material.startswith('HYB'):
        return 'HYB'
    elif material.startswith('S'):
        return 'S'
    elif material.startswith('W'):
        return 'W'
    elif material.startswith(('MATO', 'ME', 'INF')):
        return 'OT'
    elif material.startswith(('EU', 'ER', 'E+')) or '+ADO' in material:
        return 'ADO|ST|E'
    elif material.startswith('MUR'):
        return 'ADO|ST|E' if '+ST' in material else 'MUR'
    elif material.startswith(('MR', 'MCF')):
        return 'MR|MCF'
    elif material.startswith('M+'):
        return 'MUR'
    elif material.startswith('CR'):
        if any(val in eq_code for val in ['CDN', 'CDL']):
            return 'CR-'
        elif any(val in eq_erd for val in ['ERN', 'ERL']):
            return 'CR-'
        elif any(val in eq_code for val in ['CDM', 'CDH']):
            return 'CR+'
        elif any(val in eq_erd for val in ['ERM', 'ERH']):
            return 'CR+'
        else:
            return 'CR'
    else:
        raise ValueError(f'{value} not included in GEM MACRO_TAXONOMY')


def add_macro_taxonomy(data, taxo_col='TAXONOMY', overwrite=False):
    '''
    Add a column ['MACRO_TAXONOMY'] with simplified GEM taxonomy

    Parameters
    ----------
    df : a DataFrame object
    taxo_col : str, default 'TAXONOMY'
    overwrite : bool, default False
        If True, overwrite existing "MACRO_TAXONOMY" column. Else, raise error.

    Returns
    -------
    df : DataFrame
        DataFrame with "MACRO_TAXONOMY" column.
    '''    
    
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert taxo_col in data.columns, f"The column '{taxo_col}' is not in the DataFrame"

    df = data.copy()
    cols = df.columns.tolist()

    if 'MACRO_TAXONOMY' in df.columns and not overwrite:
        raise ValueError('''
            Column `MACRO_TAXONOMY` already present in DataFrame.
            To re-assign macro taxonomy, include param `overwrite=True`
        ''')

    if 'MACRO_TAXONOMY' in df.columns and overwrite:
        print('  Overwriting the MACRO_TAXONOMY column')
        df = df.drop(columns=['MACRO_TAXONOMY'])
    else:
        print('  Adding `MACRO_TAXONOMY` column ')
        cols.append('MACRO_TAXONOMY')

    # Get unique TAXONOMY classes
    df_unique = df[[taxo_col]].drop_duplicates().reset_index(drop=True)
    df_unique['MACRO_TAXONOMY'] = df_unique[taxo_col].apply(macrotaxonomy)
    
    # Check unique macro classes
    unique_macro = set(df_unique['MACRO_TAXONOMY'])
    if 'CR' in unique_macro:
        unique_macro.remove('CR')
    msg = f'Error in MacroTaxonomy: {unique_macro.difference(MACRO_CLASSES)}'
    assert unique_macro.issubset(MACRO_CLASSES.keys()), msg

    df = df.merge(df_unique, on=taxo_col, how='left')

    return df[cols]
