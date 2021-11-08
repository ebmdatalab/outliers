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

# +
# try to get just a single CCG's worth of data, not using BQ cache and compare KDE plots
# -

from lib.outliers import Runner, Plots
from matplotlib import pyplot as plt
import numpy as np

from_date = date(year=2021,month=4,day=1)
to_date = date(year=2021,month=10,day=1)
r = Runner(from_date,to_date,5,["practice","ccg","pcn","stp"],False)

r.build.run()

r.build.fetch_results()

r._run_item_report('ccg','04Y')

from ebmdatalab import bq
import pandas as pd

# +
for entity in ['practice','pcn','stp']:
    sql = f"""
                SELECT
                    chemical,
                    measure_array as `array`
                FROM
                    `ebmdatalab.outlier_detection.{entity}_measure_arrays`
                WHERE
                    build_id = 1;
            """
    csv_path = f"../data/bq_cache/{entity}_measure_arrays.zip"
    res = bq.cached_read(
        sql,
        csv_path,
        use_cache=False,
    )
# -


csv_path = f"../data/bq_cache/ccg_measure_arrays.zip"
res = bq.cached_read(
    sql,
    csv_path,
    use_cache=False,
)

res.array

res.array.values[0] #= res.array.apply(str)

res.array = res.array.apply(lambda x: np.fromstring(x[1:-1], sep=","))

self.results_measure_arrays[entity] = res.set_index("chemical")

distribution = sorted(res.query('chemical=="1001010AD"')['array'].values[0])

distribution

import seaborn as sns

~np.isnan(distribution)

figsize=(10, 2)
fig, ax = plt.subplots(1, 1, figsize=figsize)
#distribution = distribution[~np.isnan(distribution)]
sns.kdeplot(
    distribution,
    bw=Plots._bw_scott(distribution),
    ax=ax,
    linewidth=0.9,
    legend=False,
)

res.query('chemical=="1001010AD"')['array'].apply(str)

type(r.build.results_measure_arrays['ccg'].query('chemical=="0202010L0"')['array'].values[0][0])


