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

from lib.outliers import Runner
from datetime import date

from_date = date(year=2021,month=6,day=1)
to_date = date(year=2021,month=12,day=1)
r = Runner(from_date=from_date,to_date=to_date,n_outliers=10,entities=["practice","ccg","pcn","stp"],force_rebuild=False,n_jobs=16)

r.run()
