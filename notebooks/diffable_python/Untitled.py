# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import pandas as pd

htmlfiles = {f.split('_')[2].replace('.html',''):pd.read_html(f'../data/html/{f}') for f in os.listdir('../data/html/') if 'ccg' in f}

first = True
outer_df = None
for k,v in htmlfiles.items():
    for i,df in enumerate(v):
        inner_df = df.assign(practice=k).reset_index().rename(columns={'index':'rank'}).assign(high_low='h' if i==0 else 'l')
        outer_df = inner_df if first else pd.concat([outer_df,inner_df])
        first = False

outer_df.drop(columns='Plots').to_csv('../data/sproc_audit_ccg_original.csv',index=False)
