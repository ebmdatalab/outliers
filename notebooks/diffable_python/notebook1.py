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

# + active=""
# df = static_data_reshaping(chem_per_para, 'ccg', 'subpara', 'chemical')
# df.head(10)

# + active=""
# stats = get_stats(
#     df=df,
#     measure="ratio",
#     aggregators=["chemical"],
#     stat_parameters=None,
#     trim=True
# )
# stats.head()
# -

stats_class = StaticOutlierStats(
    df=chem_per_para,
    entity_type='ccg',
    num_code='chemical',
    denom_code='subpara'
)
stats = stats_class.get_table()
stats.head()

dist_table(
    stats,
    stats_class,
    '00C',
)

stats.loc[stats['z_score']>100000000000]

# +
from tqdm.notebook import tqdm
from lib.make_html import write_to_template

for x in ['ccg','pcn','practice']:
    entity_names = entity_names_query(x)
    stats_class = StaticOutlierStats(
        df=chem_per_para,
        entity_type=x,
        num_code='chemical',
        denom_code='subpara'
    )
    stats = stats_class.get_table()
    codes = stats.index.get_level_values(0).unique()[0:10]
    
    for code in tqdm(codes, desc=x):
        table_high = dist_table(
            stats,
            stats_class,
            code,
            ascending=False,
            table_length=5,
        )
        table_low = dist_table(
            stats,
            stats_class,
            code,
            ascending=True,
            table_length=5,
        )
        output_file = f'static_{x}_{code}'
        write_to_template(
            entity_names.loc[code,'name'],
            table_high,
            table_low,
            output_file,
        )
