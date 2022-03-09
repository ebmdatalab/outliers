# -*- coding: utf-8 -*-
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
import pandas as pd

from_date = date(year=2021,month=4,day=1)
to_date = date(year=2021,month=8,day=1)
r = Runner(from_date,to_date,5,["practice","ccg","pcn","stp"],False)

r.run()

# +
### Extracting all the stored z scores etc across organisations
### so that summary statistics can be calculated

e_data = pd.concat(
    (d.assign(entity=e) for e, d in r.build.results.items())
)
# -

# ## Entity counts
#
# Counts of each kind of entity (i.e., organisation).

# +
### Summarising the number of each kind of entity (organisation)

e_counts = ( e_data.reset_index()[["practice","entity"]]
            .drop_duplicates()['entity']
            .value_counts()
            .to_frame()
            .rename( columns={'entity':'n'} ) )

e_counts
# -

# ## Chemical counts
#
# Counts of the number of chemicals for which we have data (Z scores etc)
# within each type of organisation.

# +
### Summarising the number of unique chemicals analysed within
### each type of organisation

c_counts = ( e_data.reset_index()[["chemical","entity"]]
            .drop_duplicates()['entity']
            .value_counts()
            .to_frame()
            .rename( columns={'entity':'chemicals'} ) )

c_counts

# +
### Combining the entity and chemical counts

all_counts = e_counts.join( c_counts )


# +
### Calculating summary statistics for the ratio and the Z score
### within each entity type

all_summary = e_data.groupby( "entity" )[["ratio","z_score"]].describe().reindex(['stp', 'ccg', 'pcn', 'practice']).stack(level=0)
all_summary = all_summary.rename( columns={"50%":"median"}, inplace=False )

### Defining which metrics will be displayed below
metrics_to_show = [ "n", "chemicals", "median","max","min","IQR" ]
# -


# ## Summary statistics for the z score in each organisation type

# +
### Extracting the summary statistics for the z scores
z_tmp = all_summary[all_summary.index.isin(["z_score"], level=1)]

### Calculating IQR, removing the row index and rounding to 2dp
z_summary = ( z_tmp
         .assign( IQR = z_tmp["75%"]-z_tmp["25%"] )
         .droplevel(level=1)
         .round(2) )

z_summary.join( all_counts )[metrics_to_show]
# -

# ## Summary statistics for the ratio in each organisation type

# +
### Extracting the summary statistics for the z scores
ratio_tmp = all_summary[all_summary.index.isin(["ratio"], level=1)]

### Calculating IQR, removing the row index and rounding to 2dp
ratio_summary = ( ratio_tmp
         .assign( IQR = ratio_tmp["75%"]-ratio_tmp["25%"] )
         .droplevel(level=1)
         .round(2) )

ratio_summary.join( all_counts )[metrics_to_show]
# -

