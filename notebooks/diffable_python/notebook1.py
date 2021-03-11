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

# **This requires > the standard 2GB of memory in docker. It worked with 4GB.**

import pandas as pd
from ebmdatalab import bq
from lib.outliers import *
from lib.generate_measure_sql import get_measure_json, build_sql
from change_detection import functions as chg

# ## Load data

# +
with open("../data/static_outlier_sql/chem_per_para.sql") as sql:
    query = sql.read()
#chem_per_para = bq.cached_read(query, csv_path='data/chem_per_para.zip')

## reload specifying data type currently required
## due to https://github.com/ebmdatalab/datalab-pandas/issues/26
chem_per_para = pd.read_csv('../data/chem_per_para.zip', dtype={'subpara': str})
chem_per_para.head()

# +
## When not using cached data, this needs to be run first time
## after set up of docker environment (to authenticate BigQuery)
#from ebmdatalab import bq
#bq.cached_read("nothing",csv_path="nothing")
# -

measures = ["desogestrel", "trimethoprim",]
run_name = "practice_change_detection"
get_measure_json(measures, run_name)
build_sql(run_name)

change = chg.ChangeDetection(
    name=run_name,
    measure=True,
    custom_measure=True,
    direction="down",
    use_cache=True,
    overwrite=False,
    verbose=False,
    draw_figures="no")
change.run()

measure_changes = change.concatenate_outputs()
measure_changes.head()

import glob
pathname = f"data/{run_name}/**/bq_cache.csv"
glob.glob(pathname, recursive=True)

# +

sparkline_table(
    trimethoprim,
    'practice_change_detection/trimethoprim',
    'trimethoprim'
)
# -

# ## Generate HTML for practices, CCGs etc

# +
#loop_over_everything(chem_per_para, ['practice','pcn','ccg',])
