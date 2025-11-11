from methods.taxonomy import check_taxonomy, add_tax_attrs
import pandas as pd
import re  # for pattern extraction

# 1) Create a small DataFrame with taxonomy strings
df = pd.DataFrame({
    "TAXONOMY": [
        "HYB(MCF;MUR)/LFM/CDH/H:8/BPD/RSH1+RMN/RES"
        #"HYB/LFM/CDH/H:8/BPD/RSH1+RMN/RES"
    ]
})

# 2) Check consistency (this will raise if something is wrong)
try:
    tax = check_taxonomy(df, taxo_col="TAXONOMY")
    print("All taxonomies are valid ✅")
except ValueError as e:
    print("There are invalid taxonomies ❌")
    # Convert the error to string
    err_str = str(e)
    
    # Extract the canonical value from the string using regex
    match = re.search(r"'canonical': '([^']+)'", err_str)
    if match:
        tax_canonical = match.group(1)
        print("Canonical taxonomy:", tax_canonical)
    else:
        tax_canonical = None
        print("No canonical value found.")
