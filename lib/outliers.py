from io import BytesIO
from base64 import b64encode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from ebmdatalab import bq
from lib.make_html import write_to_template
from lib.table_of_contents import MarkdownToC
from os import path

# reset to matplotlib defaults rather than seaborn ones
plt.rcdefaults()
# Turn off the max column width so the images won't be truncated
pd.set_option("display.max_colwidth", None)
# Monkeypatch DataFrame so that instances can display charts in a notebook
pd.DataFrame._repr_html_ = lambda self: self.to_html(escape=False)

# Display pandas linebreaks properly
# Save the original `to_html` function to call it later
pd.DataFrame.base_to_html = pd.DataFrame.to_html
# Call it here in a controlled way
pd.DataFrame.to_html = lambda df, *args, **kwargs: (
    df.base_to_html(*args, **kwargs).replace(r"\n", "<br/>")
)


######## Plots ########


def bw_scott(x):
    """Adapted from https://www.statsmodels.org/stable/_modules/statsmodels/nonparametric/bandwidths.html
    previouly cause an issue where the IQR was 0 for many
    pandas.DataFrame.plot.kde does an okay job using scipy method,
    but haven't worked out how to do that
    """

    def _select_sigma(X):
        # normalize = 1.349
        # IQR = (sap(X, 75) - sap(X, 25))/normalize
        # return np.minimum(np.std(X, axis=0, ddof=1), IQR)
        return np.std(X, axis=0, ddof=1)

    A = _select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** (-0.2)


def dist_plot(org_value, distribution, figsize=(3.5, 1), **kwargs):
    """Draws a matplotlib plot with a kde curve and a line for
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
    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    distribution = distribution[~np.isnan(distribution)]
    sns.kdeplot(
        distribution, bw=bw_scott(distribution), ax=ax, linewidth=0.9, legend=False
    )
    ax.axvline(org_value, color="r", linewidth=1)
    lower_limit = max(0, min(np.quantile(distribution, 0.001), org_value * 0.9))
    upper_limit = max(np.quantile(distribution, 0.999), org_value * 1.1)
    ax.set_xlim(lower_limit, upper_limit)
    ax = remove_clutter(ax)
    plt.close()
    return fig


def sparkline_plot(series, figsize=(3.5, 1), **kwags):
    """Draws a sparkline plot to be used in a table.

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
    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwags)
    series.reset_index().plot(ax=ax, linewidth=0.9)
    ax = remove_clutter(ax)
    return plt


def remove_clutter(ax):
    """Removes axes and other clutter from the charts.

    Parameters
    ----------
    ax : matplotlib axis

    Returns
    -------
    ax : matplotlib axis
    """
    # ax.legend()
    # ax.legend_.remove()
    for k, v in ax.spines.items():
        v.set_visible(False)
    ax.tick_params(labelsize=5)
    ax.set_yticks([])
    # ax.set_xticks([])
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    return ax


def html_plt(plt):
    """Converts a matplotlib plot into an html image.

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
    html_plot = f'<img class="zoom" src="data:image/png;base64,{b64_plot}"/>'

    return html_plot


######## Entity & bnf names ########
def get_entity_names(name, measure):
    """Takes entity name from entity_names_query and converts into a
    link to the corresponding measure on OpenPrescribing

    Parameters
    ----------
    name : str
        Name of practice/PCN/CCG etc
    measure : str
        Name of measure to create link to measure

    Returns
    -------
    link
        Link to the measure/entity on OpenPrescribing
    """
    entity_type = name.split("_")[0]
    entity_names = entity_names_query(entity_type)
    entity_names["code"] = entity_names.index
    measure_name = measure.split("_")[-1]
    entity_names["link"] = entity_names[["code", "name"]].apply(
        lambda x: f'<a href="https://openprescribing.net/measure/{measure_name}/{entity_type}/{x[0]}/">{x[1]}</a>',
        axis=1,
    )
    return entity_names


def entity_names_query(entity_type):
    """Queries the corresponding table for the entity and returns names with
    entity codes as the index

    Parameters
    ----------
    entity_type : str
        e.g. "ccg", "pcn", "practice"

    Returns
    -------
    pandas DataFrame
        code is the index and entity names are the column
    """
    query = f"""
    SELECT
      DISTINCT code,
      name
    FROM
      ebmdatalab.hscic.{entity_type}s
    WHERE
      name IS NOT NULL
    """
    entity_names = bq.cached_read(query, csv_path=f"../data/{entity_type}_names.csv")
    return entity_names.set_index("code")


def get_bnf_names(bnf_level):
    """Takes in input like "chemical" and passes the appropriate fields
    to bnf_query

    Parameters
    ----------
    bnf_level : str
        BNF level, allowable values from the bnf table in BQ are:
        "chapter", "section" ,"para", "subpara" ,"chemical" ,"product",
        "presentation"

    Returns
    -------
    pandas DataFrame
        Containing bnf_code as the index and bnf name as the only column
    """
    bnf_code = f"{bnf_level}_code"
    bnf_name = bnf_level
    names = bnf_query(bnf_code, bnf_name)
    return names


def bnf_query(bnf_code, bnf_name):
    """Queries bnf table in BQ and returns a list of BNF names
    mapped to BNF codes

    Parameters
    ----------
    bnf_code : str
        name of BNF code column
    bnf_name : str
        name of BNF name column

    Returns
    -------
    pandas DataFrame
        Containing bnf_code as the index and bnf name as the only column
    """
    query = f"""
    SELECT
      DISTINCT {bnf_code},
      {bnf_name} AS {bnf_name}_name
    FROM
      ebmdatalab.hscic.bnf
    WHERE
      {bnf_name} IS NOT NULL
    """
    bnf_names = bq.cached_read(query, csv_path=f"../data/{bnf_name}_names.csv")
    bnf_names = pd.read_csv(f"../data/{bnf_name}_names.csv", dtype={bnf_code: str})
    return bnf_names.set_index(bnf_code)


######## Static outliers ########
def fill_zeroes(df, entity_type, denom_code, num_code):
    """Adds missing rows with 0s to fill where there is prescribing in a
    specific denominator column, but no prescribing in the numerator

    Parameters
    ----------
    df : pandas df
        Dataframe containing numerator column, as well as entitiy codes,
        numerator codes and denominator codes.
    entity_type : str
        Column name for entities, e.g. 'ccg'
    denom_code : str
        Column name for denominator codes.
    num_code : str
        Column name for numerator codes.

    Returns
    -------
    pandas df
        dataframe filled with rows for every possible numerator code
    """
    chems = df[[denom_code, num_code]].drop_duplicates()
    chems["tmp"] = 1
    entities = df[[entity_type]].drop_duplicates()
    entities["tmp"] = 1
    all_chem_map = entities.merge(chems, on="tmp").drop("tmp", axis=1)
    df = df.merge(all_chem_map, on=[entity_type, denom_code, num_code], how="outer")
    df = df.fillna(0)
    df["numerator"] = df["numerator"].astype(int)
    return df


def static_data_reshaping(df, entity_type, denom_code, num_code):
    """Some data management to aggregate data, and calculate some columns

    Parameters
    ----------
    df : pandas df
        Dataframe obtained from the SQL query.
    entity_type : str
        Column name for entities, e.g. 'ccg'
    denom_code : str
        Column name for denominator codes.
    num_code : str
        Column name for numerator codes.

    Returns
    -------
    pandas df
        dataframe ready to be passed to get_stats
    """
    ## Drop unnecessary columns
    df = df[[entity_type, num_code, "numerator", denom_code]]

    ## Aggregate by pcn/ccg
    if entity_type != "practice":
        df = df.groupby([entity_type, denom_code, num_code], as_index=False).sum()

    ## Fill zeroes where there is some prescribing within that subpara
    df = fill_zeroes(df, entity_type, denom_code, num_code)

    ## Merge BNF names
    for x in [num_code, denom_code]:
        df = df.merge(get_bnf_names(x), how="left", left_on=x, right_index=True)

    ## Calculate denominator
    df = df.set_index([entity_type, denom_code, num_code])
    df["denominator"] = df.groupby([entity_type, denom_code]).sum()["numerator"]
    df = df.reset_index()

    ## Calculate ratio
    df["ratio"] = df["numerator"] / df["denominator"]

    return df.set_index([entity_type, num_code])


def trim_outliers(df, measure, aggregators):
    """Trims a small number of extreme values from a df, so that they
    don't affect the calculated z-score. This is only used in calculation of
    summary stats, extreme values are not excluded entirely.

    Parameters
    ----------
    df : pandas df
        Dataframe containing the column to be trimmed.
    measure : str
        Column to be trimmed.
    aggregators : list
        List of column(s) to group data, usually the numerator code.

    Returns
    -------
    pandas df
        trimmed df
    """

    def trim_series(series):
        """Trims extreme values from a series. This should not drop values where
        there are a large number of identical values (usually 0.0 or 1.0) at
        the extremes.

        Parameters
        ----------
        series : pandas Series
            Series to be trimmed.

        Returns
        -------
        pandas Series
            Trimmed Series.
        """
        mask = (series >= series.quantile(0.001)) & (series <= series.quantile(0.999))
        return mask

    mask = df.groupby(aggregators)[measure].apply(lambda x: trim_series(x))
    return df.loc[mask]


def get_stats(df, measure, aggregators, stat_parameters, trim=True):
    """Generates pandas columns with various stats in.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the measure to be summarised and
        the aggregator column.
    measure : str
        Column name to be summarised.
    aggregators : list
        Column(s) to use for aggregating data.
    stat_parameters : list
        Additional parameters to be calculated, e.g. ["skew", pd.DataFrame.kurt]
    trim : bool
        Say whether to trim the data before calculating stats

    Returns
    -------
    df : pandas DataFrame
    """
    if trim:
        stats = trim_outliers(df, measure, aggregators)
    else:
        stats = df
    # 1 calculate stats
    if stat_parameters:
        stat_parameters = ["mean", "std"] + stat_parameters
    else:
        stat_parameters = ["mean", "std"]
    stats = stats.groupby(aggregators).agg(stat_parameters)[measure]
    df = df.join(stats)
    # 2 calculate the # of std deviations an entity is away from the mean
    df["z_score"] = (df[measure] - stats["mean"]) / stats["std"]
    # df['z_score'] = df['z_score'].abs() # change to absolute values

    ## Some chemicals had a std of ~0 making inf values
    ## I think this identifies where e.g. CCGs are the only ones to prescribe this.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


class StaticOutlierStats:

    """Wrapper to take data from the SQL query and return stats dataframe

    Attributes
    ----------
    df : pandas df
        Dataframe from the SQL query
    entity_type : str
        Column name for entity type, e.g. 'ccg'
    measure : str
        Name of column to calculate stats on. Default is "ratio"
    num_code : str
        Column name for numerator codes
    denom_code : str
        Column name for denominator codes
    stat_parameters :
        Additional parameters to be calculated, e.g. ["skew", pd.DataFrame.kurt]
    trim : bool
        Say whether to trim the data before calculating stats
    """

    def __init__(
        self,
        df,
        entity_type,
        num_code,
        denom_code,
        measure="ratio",
        stat_parameters=None,
        trim=True,
    ):
        self.df = df
        self.entity_type = entity_type
        self.num_code = num_code
        self.denom_code = denom_code
        self.measure = measure
        self.stat_parameters = stat_parameters
        self.trim = trim

    def get_table(self):
        """Activate getting the stats table

        Returns
        -------
        pandas df
            Table of stats.
        """
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
            trim=True,
        )
        return stats


def sort_pick_top(df, sort_col, ascending, entity_type, table_length):
    """Sorts the df by a specified column, then picks the top X values for each
    entity

    Parameters
    ----------
    df : pandas df
        Input df.
    sort_col : str
        Name of column(s) on which to sort.
    ascending : bool
        Sort order to be passed to sort_values.
    entity_type : str
        Column name for entity type, e.g. 'ccg'
    table_length : int
        Number of rows to be returned for each entity

    Returns
    -------
    pandas df
        df containing selected rows for each entity
    """
    df = df.sort_values(sort_col, ascending=ascending)
    return df.groupby([entity_type]).head(table_length)


def join_measure_array(big_df, filtered_df, measure):
    """Adds a numpy array of measure values for each

    Parameters
    ----------
    big_df : pandas df
        Dataframe containing all values (from get_stats).
    filtered_df : pandas df
        Dataframe containing selcted rows from sort_pick_top.
    measure : str
        Column name for the arrays.

    Returns
    -------
    pandas df
        filtered_df with measure arrays joined on.
    """
    df = big_df[measure].unstack(level=0)
    series = df.apply(lambda r: tuple(r), axis=1).apply(np.array)
    series = pd.Series(series, index=df.index, name="array")
    return filtered_df.join(series)


def create_out_table(df, attr, entity_type, table_length, ascending):
    """Wrapper to create table for all entities, using sort_pick_top and
    join_measure_array

    Parameters
    ----------
    df : pandas df
        Dataframe containing all values (from get_stats).
    attr : StaticOutlierStats instance
        Contains attributes to be used in defining the tables.
    entity_type : str
        Column name for entity type, e.g. 'ccg'
    table_length : int
        Number of rows to be returned for each entity
    ascending : bool
        Sort order to be passed to sort_values.

    Returns
    -------
    pandas df
        Dataframe containing rows for all entities that will be made into HTML
        tables
    """
    out_table = sort_pick_top(df, "z_score", ascending, entity_type, table_length)
    out_table = join_measure_array(df, out_table, attr.measure)
    return out_table


def add_plots(df, measure):
    """Use the entity values and the measure array to draw a plot for each row
    in the dataframe.

    Parameters
    ----------
    df : pandas df
        Dataframe to have plots drawn in, from create_out_table
    measure : str
        Column name to be plotted for the entity

    Returns
    -------
    pandas df
        Dataframe with added plots
    """
    df["plots"] = df[[measure, "array"]].apply(
        lambda x: html_plt(dist_plot(x[0], x[1])), axis=1
    )
    df = df.drop(columns="array")
    return df


col_names = {
    "chapter": ["BNF Chapter", "Chapter Items"],
    "section": ["BNF Section", "Section Items"],
    "para": ["BNF Paragraph", "Paragraph Items"],
    "subpara": ["BNF Subparagraph", "Subparagraph Items"],
    "chemical": ["BNF Chemical", "Chemical Items"],
    "product": ["BNF Product", "Product Items"],
    "presentation": ["BNF Presentation", "Presentation Items"],
}


def tidy_table(df, attr):
    """Rounds figures, drops unnecessary columns and changes column names to be
    easier to read (according to col_names).

    Parameters
    ----------
    df : pandas df
        Input dataframe
    attr : StaticOutlierStats instance
        Contains attributes to be used in defining the tables.

    Returns
    -------
    pandas df
        Dataframe to be passed to the HTML template writer.
    """
    df = df.round(decimals=2)
    df = df.drop(columns=attr.denom_code)
    df = df.rename(
        columns={
            f"{attr.num_code}_name": col_names[attr.num_code][0],
            "numerator": col_names[attr.num_code][1],
            f"{attr.denom_code}_name": col_names[attr.denom_code][0],
            "denominator": col_names[attr.denom_code][1],
        }
    )
    df = df.set_index(col_names[attr.num_code][0])
    return df


def get_entity_table(df, attr, code):
    """Wrapper to take large input dataframe containing rows for all entities,
    and output table ready to be passed to HTML template.

    Parameters
    ----------
    df : pandas df
        Input dataframe
    attr : StaticOutlierStats instance
        Contains attributes to be used in defining the tables.
    code : str
        Code for the entity to be selected.

    Returns
    -------
    pandas df
        Dataframe to be passed to the HTML template writer.
    """
    df_ent = df.loc[code].copy()
    df_ent = add_plots(df_ent, attr.measure)
    df_ent = tidy_table(df_ent, attr)
    return df_ent


def loop_over_everything(
    df,
    entities,
    output_dir="../data",
    template_path="../data/template.html",
):
    """Loops over all entities to generate HTML for each.

    Parameters
    ----------
    df : pandas df
        Dataframe obtained from the SQL query.
    entities : list
        List of entities to write HTML for e.g. ['practice','pcn','ccg',]
    output_dir : str
        Directory for output
    template_path : str
        Path to jinja2 html template file
    """
    urlprefix = "https://htmlpreview.github.io/?" + \
        "https://raw.githubusercontent.com/ebmdatalab/outliers/master/"
    toc = MarkdownToC(urlprefix)

    for ent_type in entities:
        entity_names = entity_names_query(ent_type)
        stats_class = StaticOutlierStats(
            df=df,
            entity_type=ent_type,
            num_code="chemical",
            denom_code="subpara"
        )
        stats = stats_class.get_table()

        table_high = create_out_table(
            df=stats,
            attr=stats_class,
            entity_type=ent_type,
            table_length=5,
            ascending=False,
        )
        table_low = create_out_table(
            df=stats,
            attr=stats_class,
            entity_type=ent_type,
            table_length=5,
            ascending=True,
        )

        codes = stats.index.get_level_values(0).unique()[0:10]
        for code in tqdm(codes, desc=f"Writing HTML: {ent_type}"):
            output_file = path.join(
                    output_dir,
                    'html',
                    f"static_{ent_type}_{code}.html",)
            write_to_template(
                entity_names.loc[code, "name"],
                get_entity_table(table_high, stats_class, code),
                get_entity_table(table_low, stats_class, code),
                output_path=output_file,
                template_path=template_path,
            )
            toc.add_file(output_file, entity=ent_type)
    toc.write_toc(output_dir)


######## Change outliers ########
def sparkline_series(df, column, subset=None):
    """Creates a pandas series containing sparkline plots, based on a
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
    series = pd.Series(index=index, name="plots")
    for idx in index:
        plot = sparkline_plot(df.loc[idx, column])
        series.loc[idx] = html_plt(plot)
    df = df.join(series, how="right")
    df = df.round(decimals=2)
    series["one"] = 1
    return series


def sparkline_table(change_df, name, measure):
    data = pd.read_csv("../data/{}/bq_cache.csv".format(name), index_col="code")
    data["month"] = pd.to_datetime(data["month"])
    data["rate"] = data["numerator"] / data["denominator"]
    data = data.sort_values("month")

    filtered = change_df.loc[measure]

    # pick entities that start high
    mask = filtered["is.intlev.initlev"] > filtered["is.intlev.initlev"].quantile(0.8)
    filtered = filtered.loc[mask]

    # remove entities with a big spike
    mean_std_max = data["rate"].groupby(["code"]).agg(["mean", "std", "max"])
    mask = mean_std_max["max"] < (mean_std_max["mean"] + (1.96 * mean_std_max["std"]))
    filtered = filtered.loc[mask]

    # drop duplicates
    filtered = filtered.loc[~filtered.index.duplicated(keep="first")]

    filtered = filtered.sort_values("is.intlev.levdprop", ascending=False).head(10)
    plot_series = sparkline_series(data, "rate", subset=filtered.index)

    entity_names = get_entity_names(name, measure)

    # create output table
    out_table = filtered[["is.tfirst.big", "is.intlev.levdprop"]]
    out_table = out_table.join(entity_names["link"])
    out_table = out_table.join(plot_series)
    out_table = out_table.rename(
        columns={
            "is.tfirst.big": "Month when change detected",
            "is.intlev.levdprop": "Measured proportional change",
        }
    )
    return out_table.set_index("link")


def month_integer_to_dates(input_df, change_df):
    change_df["min_month"] = input_df["month"].min()
    change_df["is.tfirst.big"] = change_df.apply(
        lambda x: x["min_month"] + pd.DateOffset(months=x["is.tfirst.big"] - 1), axis=1
    )
    return change_df
