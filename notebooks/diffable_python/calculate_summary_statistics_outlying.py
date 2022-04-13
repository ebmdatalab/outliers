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

from_date = date(year=2021,month=6,day=1)
to_date = date(year=2021,month=12,day=1)
r = Runner(from_date,to_date,10,["practice","ccg","pcn","stp"],False)

r.build.run()
r.build.fetch_results()

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
# Counts of the number of unique outlying chemicals (i.e., those identified in the top/bottom
# 5 z scores) amongst all organisations of the given type.

# +
### Summarising the number of unique chemicals identified in the
### top/bottom five outliers amongst all organisations of the given type

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
### Defining which metrics will be displayed in the summary tables
metrics_to_show = [ "n", "chemicals", "median","max","min","IQR" ]


# +
### Calculating summary statistics for the Z scores for those chemicals
### identified in the TOP 5 in at least one organisation of the entity type.
### There are the chemicals displayed in the 'Higher than most' table.

overused_summary = e_data.query('z_score>0').groupby( "entity" )[["z_score"]].describe().reindex(['stp', 'ccg', 'pcn', 'practice']).stack(level=0)
overused_summary = overused_summary.rename( columns={"50%":"median"}, inplace=False )

# +
### Calculating summary statistics for the Z scores for those chemicals
### identified in the BOTTOM 5 in at least one organisation of the entity type.
### There are the chemicals displayed in the 'Lower than most' table.

underused_summary = e_data.query('z_score<0').groupby( "entity" )[["z_score"]].describe().reindex(['stp', 'ccg', 'pcn', 'practice']).stack(level=0)
underused_summary = underused_summary.rename( columns={"50%":"median"}, inplace=False )
# -

# ## Summary statistics for outlying Z scores in each organisation type
#
# ### Higher than most chemicals
#
# The table below summarises the Z scores for the high outlying (i.e., top 5) chemicals
# in each type of organisation. These are chemicals are seen to be used more often
# in a particular organisation than its peers.

# +
### Extracting the summary statistics for the z scores
overused_tmp = overused_summary[overused_summary.index.isin(["z_score"], level=1)]

### Calculating IQR, removing the row index and rounding to 2dp
overused_toprint = ( overused_tmp
         .assign( IQR = overused_tmp["75%"]-overused_tmp["25%"] )
         .droplevel(level=1)
         .round(2) )

overused_toprint.join( all_counts )[metrics_to_show]
# -

# ### Lower than most chemicals
#
# The table below summarises the Z scores for the low outlying (i.e., bottom 5) chemicals
# in each type of organisation. These are chemicals are seen to be used less often
# in a particular organisation than its peers.

# +
### Extracting the summary statistics for the z scores
underused_tmp = underused_summary[underused_summary.index.isin(["z_score"], level=1)]

### Calculating IQR, removing the row index and rounding to 2dp
underused_toprint = ( underused_tmp
         .assign( IQR = underused_tmp["75%"]-underused_tmp["25%"] )
         .droplevel(level=1)
         .round(2) )

underused_toprint.join( all_counts )[metrics_to_show]
# -
# ### Summary
#
# Below is a summary table that combines the 'Higher than most' and 'Lower than most'
# results displayed above.

pd.concat([overused_toprint.join( all_counts )[metrics_to_show],
           underused_toprint[metrics_to_show[2:]]],
          keys=["Higher than most", "Lower than most"],axis=1)


