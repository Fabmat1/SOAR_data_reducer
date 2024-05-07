import pandas as pd


table = pd.read_csv('output_2024_03.csv')

ordered_cols = ["name","source_id","ra","dec",
                "file","SPEC_CLASS","bp_rp","gmag","nspec",
                "pmra","pmra_error","pmdec","pmdec_error","parallax","parallax_error"]

table = table[ordered_cols]

table.to_csv('output_2024_03ordered.csv', index=False)