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

# **This requires > the standard 2GB of memory in docker. It worked with 3.5GB, but less may be possible.**

import pandas as pd
from tqdm.auto import tqdm
from ebmdatalab import bq
from lib.outliers import *

# +
with open("../data/static_outlier_sql/chem_per_para.sql") as sql:
    query = sql.read()
#chem_per_para = bq.cached_read(query, csv_path='data/chem_per_para.zip')

## reload specifying data type currently required
## due to https://github.com/ebmdatalab/datalab-pandas/issues/26
chem_per_para = pd.read_csv('../data/chem_per_para.zip',dtype={'subpara': str})
chem_per_para.head()

# +
## WHAT TO DO WHERE DENOMINATOR == 0?

# +
from lib.make_html import write_to_template
    
def loop_over_everything(df, entities):
    for ent_type in entities:
        entity_names = entity_names_query(ent_type)
        stats_class = StaticOutlierStats(
            df=df,
            entity_type=ent_type,
            num_code='chemical',
            denom_code='subpara'
        )
        stats = stats_class.get_table()
        
        table_high = create_out_table(
            df=stats,
            attr=stats_class,
            entity_type=ent_type,
            table_length=5,
            ascending=False
        )
        #print(table_high.info())
        table_low = create_out_table(
            df=stats,
            attr=stats_class,
            entity_type=ent_type,
            table_length=5,
            ascending=True
        )
        
        codes = stats.index.get_level_values(0).unique()[0:10]
        for code in tqdm(codes, desc=f'Writing HTML: {ent_type}'):
            output_file = f'static_{ent_type}_{code}'
            write_to_template(
                entity_names.loc[code,'name'],
                get_entity_table(table_high, stats_class, code),
                get_entity_table(table_low, stats_class, code),
                output_file,
            )


# -

loop_over_everything(chem_per_para, ['practice','pcn','ccg',])
