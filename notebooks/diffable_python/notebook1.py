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

# ## Load data

# +
with open("../data/static_outlier_sql/chem_per_para.sql") as sql:
    query = sql.read()
chem_per_para = bq.cached_read(query, csv_path='../data/chem_per_para.zip')

## reload specifying data type currently required
## due to https://github.com/ebmdatalab/datalab-pandas/issues/26
chem_per_para = pd.read_csv('../data/chem_per_para.zip',dtype={'subpara': str})
chem_per_para.head()
# -

# ## Generate HTML for practices, CCGs etc

loop_over_everything(chem_per_para, ['practice','pcn','ccg',])
