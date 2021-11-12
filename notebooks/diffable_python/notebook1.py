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

from_date = date(year=2021,month=4,day=1)
to_date = date(year=2021,month=10,day=1)
r = Runner(from_date,to_date,5,["practice","ccg","pcn","stp"],False,100)

r.run()
