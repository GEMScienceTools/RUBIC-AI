from methods.taxonomy import check_taxonomy, add_tax_attrs
import pandas as pd

# 1) Create a small DataFrame with taxonomy strings
df = pd.DataFrame({
    "TAXONOMY": [
        "CR/LDUAL/CDH/H:3/BP1/RES"      # invalid LLRS on purpose
    ]
})

# 2) Check consistency (this will raise if something is wrong)
try:
    check_taxonomy(df, taxo_col="TAXONOMY")
    print("All taxonomies are valid ✅")
except ValueError as e:
    print("There are invalid taxonomies ❌")
    print(e)

# 3) (optional) add parsed attributes to the dataframe
df2 = add_tax_attrs(df, taxo_col="TAXONOMY")
print(df2)
