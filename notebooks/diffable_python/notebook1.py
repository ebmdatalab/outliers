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
from datetime import date
from dateutil.relativedelta import relativedelta

# ## Set parameters

# +
FMT = "%Y-%m-%d"
def minus_six_months(datestr):
    if not datestr:
        return None
    sma = date.fromisoformat(datestr) + relativedelta(months=-6)
    if sma < date.fromisoformat("2021-04-01"):
        sma = date.fromisoformat("2021-04-01")
    return date.strftime(sma, FMT)
end_date = date.strftime(date.today(), FMT)
start_date = six_months_ago = minus_six_months(end_date)

entities = ['practice','pcn','ccg',]
output_dir = '../data'
template_path = '../data/template.html'
# -

# ## Generate HTML for practices, CCGs etc

loop_over_everything(
        entities=entities,
        date_from=start_date,
        date_to=end_date,
        output_dir=output_dir,
        template_path=template_path,
    )
