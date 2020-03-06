from io import BytesIO
from base64 import b64encode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ebmdatalab import bq
# reset to matplotlib defaults rather than seaborn ones
plt.rcdefaults()
# Turn off the max column width so the images won't be truncated
pd.set_option('display.max_colwidth', None)
#Monkeypatch DataFrame so that instances can display charts in a notebook
pd.DataFrame._repr_html_ = lambda self: self.to_html(escape=False)

# Display pandas linebreaks properly
# Save the original `to_html` function to call it later
pd.DataFrame.base_to_html = pd.DataFrame.to_html
# Call it here in a controlled way
pd.DataFrame.to_html = (
    lambda df, *args, **kwargs: 
        (df.base_to_html(*args, **kwargs)
           .replace(r"\n", "<br/>"))
)



######## Plots ########

def bw_scott(x):
    """Adapted from https://www.statsmodels.org/stable/_modules/statsmodels/nonparametric/bandwidths.html
    previouly cause an issue where the IQR was 0 for many
    pandas.DataFrame.plot.kde does an okay job using scipy method,
    but haven't worked out how to do that 
    """
    def _select_sigma(X):
        #normalize = 1.349
        #IQR = (sap(X, 75) - sap(X, 25))/normalize
        #return np.minimum(np.std(X, axis=0, ddof=1), IQR)
        return np.std(X, axis=0, ddof=1)
    A = _select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** (-0.2)

def dist_plot(org_value,
              distribution,
              figsize=(3.5, 1),
              **kwargs):
    """ Draws a matplotlib plot with a kde curve and a line for
    an individual institution.
    
    Parameters
    ----------
    org_value : float
        Value of the individual institution to be highlighted.
    distribution : pandas series
        Values to be used to draw the distribution.
    figsize : tuple, optional
        Size of figure. The default is (3.5, 1).
    **kwags : to be passed to plt.subplots.

    Returns
    -------
    plt : matplotlib plot

    """
    fig, ax = plt.subplots(1,1,figsize=figsize,**kwargs)
    distribution = distribution[~np.isnan(distribution)]
    sns.kdeplot(distribution,bw=bw_scott(distribution),ax=ax,linewidth=0.9,legend=False)
    ax.axvline(org_value,color='r',linewidth=1)
    lower_limit = max(0,min(np.quantile(distribution, 0.001),org_value*0.9))
    upper_limit = max(np.quantile(distribution, 0.999),org_value*1.1)
    ax.set_xlim(lower_limit,upper_limit)
    ax = remove_clutter(ax)
    plt.close()
    return fig

def sparkline_plot(series,
                   figsize=(3.5, 1),
                   **kwags):
    """ Draws a sparkline plot to be used in a table.

    Parameters
    ----------
    series : pandas timeseries
        Timeseries to be plotted.
     figsize : tuple, optional
        Size of figure. The default is (3.5, 1).
    **kwags : to be passed to plt.subplots.

    Returns
    -------
    plt : matplotlib plot

    """
    fig, ax = plt.subplots(1,1,figsize=figsize,**kwags)
    series.reset_index().plot(ax=ax,linewidth=0.9)
    ax = remove_clutter(ax)
    return plt

def remove_clutter(ax):
    """ Removes axes and other clutter from the charts.
    
    Parameters
    ----------
    ax : matplotlib axis

    Returns
    -------
    ax : matplotlib axis

    """
    #ax.legend()
    #ax.legend_.remove()
    for k,v in ax.spines.items():
        v.set_visible(False)
    ax.tick_params(labelsize=5)
    ax.set_yticks([])
    #ax.set_xticks([])
    ax.xaxis.set_label_text('')
    plt.tight_layout()
    return ax

def html_plt(plt):
    """ Converts a matplotlib plot into an html image.
    
    Parameters
    ----------
    plt : matplotlib figure

    Returns
    -------
    html_plot : html image

    """
    img = BytesIO()
    plt.savefig(img, transparent=True)
    b64_plot = b64encode(img.getvalue()).decode()
    html_plot = f'<img src=\"data:image/png;base64,{b64_plot}"/>'

    return html_plot


######## Entity & bnf names ########
def get_entity_names(name, measure):
    entity_type = name.split('_')[0]
    entity_names = entity_names_query(entity_type)
    entity_names['code'] = entity_names.index
    measure_name = measure.split('_')[-1]
    entity_names['link'] = entity_names[['code','name']].apply(lambda x:
        f'<a href="https://openprescribing.net/measure/{measure_name}/{entity_type}/{x[0]}/">{x[1]}</a>',
        axis=1
    )
    return entity_names

def entity_names_query(entity_type):
    query = f"""
    SELECT
      DISTINCT code,
      name
    FROM
      ebmdatalab.hscic.{entity_type}s
    WHERE
      name IS NOT NULL
    """
    entity_names = bq.cached_read(
        query,
        csv_path=f'../data/{entity_type}_names.csv'
    )
    return entity_names.set_index('code')

def get_bnf_names(bnf_level):
    bnf_code = f'{bnf_level}_code'
    bnf_name = bnf_level
    names = bnf_query(bnf_code, bnf_name)
    return names

def bnf_query(bnf_code, bnf_name):
    query = f"""
    SELECT
      DISTINCT {bnf_code},
      {bnf_name} AS {bnf_name}_name
    FROM
      ebmdatalab.hscic.bnf
    WHERE
      {bnf_name} IS NOT NULL
    """
    bnf_names = bq.cached_read(
        query,
        csv_path=f'../data/{bnf_name}_names.csv'
    )
    bnf_names = pd.read_csv(f'../data/{bnf_name}_names.csv',dtype={bnf_code: str})
    return bnf_names.set_index(bnf_code)


######## Static outliers ########
def fill_zeroes(df, entity_type, denom_code, num_code):
    chems = df[[denom_code, num_code]].drop_duplicates()
    chems['tmp'] = 1
    entities = df[[entity_type]].drop_duplicates()
    entities['tmp'] = 1
    all_chem_map = entities.merge(chems,on='tmp').drop('tmp',axis=1)
    df = df.merge(all_chem_map,on=[entity_type, denom_code, num_code],how='outer')
    df = df.fillna(0)
    df['numerator'] = df['numerator'].astype(int)
    return df

def static_data_reshaping(df, entity_type, denom_code, num_code):
    ## Drop unnecessary columns
    df = df[[entity_type, num_code, 'numerator', denom_code]]
    
    ## Aggregate by pcn/ccg
    if entity_type!='practice':
        df = df.groupby([entity_type, denom_code, num_code],as_index=False).sum()
    
    ## Fill zeroes where there is some prescribing within that subpara
    df = fill_zeroes(df, entity_type, denom_code, num_code)
    
    ## Merge BNF names
    for x in [num_code, denom_code]:
        df = df.merge(
            get_bnf_names(x),
            how='left',
            left_on=x,
            right_index=True
        )

    ## Calculate denominator
    df = df.set_index([entity_type, denom_code, num_code])
    df['denominator'] = df.groupby([entity_type, denom_code]).sum()['numerator']
    df = df.reset_index()

    ## Calculate ratio
    df['ratio'] = df['numerator'] / df['denominator']

    return df.set_index([entity_type,num_code])

def trim_outliers(df, measure, aggregators):
    
    def trim_series(series):
        mask = (
            (series>=series.quantile(0.001)) &
            (series<=series.quantile(0.999))
        )
        return mask
    
    mask = df.groupby(aggregators)[measure].apply(lambda x: trim_series(x))
    return df.loc[mask]

def get_stats(df,
              measure,
              aggregators,
              stat_parameters=['skew',pd.DataFrame.kurt],
              trim=True):
    """ Generates pandas columns with various stats in.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the measure to be summarised and
        the aggregator column.
    measure : str, optional
        Column name to be summarised. The default is 'measure'.
    aggregators : [str], optional
        Column to use for aggregating data. The default is ['code'].

    Returns
    -------
    df : pandas DataFrame

    """
    if trim:
        stats = trim_outliers(df, measure, aggregators)
    else:
        stats = df
    #1 calculate stats
    if stat_parameters:
        stat_parameters = ['mean','std'] + stat_parameters
    else:
        stat_parameters = ['mean','std']
    stats = stats.groupby(aggregators).agg(stat_parameters)[measure]
    df = df.join(stats)
    #2 calculate the # of std deviations an entity is away from the mean
    df['z_score'] = (df[measure]-stats['mean'])/stats['std']
    #df['z_score'] = df['z_score'].abs() # change to absolute values

    ## Some chemicals had a std of ~0 making inf values
    ## I think this identifies where e.g. CCGs are the only ones to prescribe this.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

class StaticOutlierStats:
    def __init__(
            self,
            df,
            entity_type,
            num_code,
            denom_code,
            measure="ratio",
            stat_parameters=None,
            trim=True
            ):
        self.df = df
        self.entity_type = entity_type
        self.num_code = num_code
        self.denom_code = denom_code
        self.measure = measure
        self.stat_parameters = stat_parameters
        self.trim = trim

    def get_table(self):
        shaped_df = static_data_reshaping(
            self.df,
            self.entity_type,
            self.denom_code,
            self.num_code,
        )
        stats = get_stats(
            df=shaped_df,
            measure=self.measure,
            aggregators=self.num_code,
            stat_parameters=None,
            trim=True
        )
        return stats

def sort_pick_top(df, sort_col, ascending, entity_type, table_length):
    df = df.sort_values(sort_col, ascending=ascending)
    return df.groupby([entity_type]).head(table_length)
    
def join_numerator_array(big_df, filtered_df, measure):
    df = big_df[measure].unstack(level=0)
    series = df.apply(lambda r: tuple(r), axis=1).apply(np.array)
    series = pd.Series(series,index=df.index,name='array')
    return filtered_df.join(series)

def create_out_table(df, attr, entity_type, table_length, ascending):
    out_table = sort_pick_top(
        df,
        'z_score',
        ascending,
        entity_type,
        table_length
        )
    out_table = join_numerator_array(df, out_table, attr.measure)
    return out_table

def add_plots(df, measure):
    df['plots'] = df[[measure,'array']].apply(lambda x:
        html_plt(dist_plot(x[0],x[1])), axis=1)
    df = df.drop(columns='array')
    return df

col_names = {
    'chapter': ['BNF Chapter', 'Chapter Items'],
    'section': ['BNF Section', 'Section Items'],
    'para': ['BNF Paragraph', 'Paragraph Items'],
    'subpara': ['BNF Subparagraph', 'Subparagraph Items'],
    'chemical': ['BNF Chemical', 'Chemical Items'],
    'product': ['BNF Product', 'Product Items'],
    'presentation': ['BNF Presentation', 'Presentation Items']
}

def tidy_table(df, attr):
    df = df.round(decimals=2)
    df = df.drop(columns=attr.denom_code)
    df = df.rename(columns={
        f'{attr.num_code}_name': col_names[attr.num_code][0],
        'numerator': col_names[attr.num_code][1],
        f'{attr.denom_code}_name': col_names[attr.denom_code][0],
        'denominator': col_names[attr.denom_code][1],
        })
    df = df.set_index(col_names[attr.num_code][0])
    return df

def get_entity_table(df, attr, code):
    df_ent = df.loc[code].copy()
    df_ent = add_plots(df_ent, attr.measure)
    df_ent = tidy_table(df_ent, attr)
    return df_ent

######## Change outliers ########
def sparkline_series(df, column, subset=None):
    """ Creates a pandas series containing sparkline plots, based on a
    specific column in a dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the relevant column.
    column : str
        Column to use for drawing sparkline plots.
    subset : pandas index, optional
        Index matching the subset of rows that need to be in the
        column. The default is None.

    Returns
    -------
    pandas series containing columns and figures.

    """
    if subset is not None:
        index = subset
    else:
        index = df.index
    series = pd.Series(index=index,name='plots')
    for idx in index:
        plot = sparkline_plot(df.loc[idx,column])
        series.loc[idx] = html_plt(plot)
    df = df.join(series, how='right')
    df = df.round(decimals=2)
    series['one'] = 1
    return series

def sparkline_table(change_df, name, measure):
    data = pd.read_csv('../data/{}/bq_cache.csv'.format(name),index_col='code')
    data['month'] = pd.to_datetime(data['month'])
    data['rate'] = data['numerator'] / data['denominator']
    data = data.sort_values('month')
   
    filtered = change_df.loc[measure]
    
    #pick entities that start high
    mask = filtered['is.intlev.initlev'] > filtered['is.intlev.initlev'].quantile(0.8)
    filtered = filtered.loc[mask]
    
    #remove entities with a big spike
    mean_std_max = data['rate'].groupby(['code']).agg(['mean','std','max'])
    mask = mean_std_max['max'] < (mean_std_max['mean'] + (1.96*mean_std_max['std']))
    filtered = filtered.loc[mask]
    
    #drop duplicates
    filtered = filtered.loc[~filtered.index.duplicated(keep='first')]
    
    filtered = filtered.sort_values('is.intlev.levdprop', ascending=False).head(10)
    plot_series = sparkline_series(data, 'rate', subset=filtered.index)
    
    entity_names = get_entity_names(name, measure)

    #create output table
    out_table = filtered[['is.tfirst.big','is.intlev.levdprop']]
    out_table = out_table.join(entity_names['link'])
    out_table = out_table.join(plot_series)
    out_table = out_table.rename(columns={
        "is.tfirst.big": "Month when change detected",
         "is.intlev.levdprop": "Measured proportional change"
         })
    return out_table.set_index('link')

def month_integer_to_dates(input_df, change_df):
    change_df['min_month'] = input_df['month'].min()
    change_df['is.tfirst.big'] = change_df.apply(lambda x:
        x['min_month']
        + pd.DateOffset(months = x['is.tfirst.big']-1 ),
        axis=1)
    return change_df