from datetime import date
from dateutil.relativedelta import relativedelta
from io import BytesIO
from base64 import b64encode
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from ebmdatalab import bq
from lib.make_html import write_to_template
from lib.outliers import StaticOutlierStats
from lib.table_of_contents import MarkdownToC
from os import path
from typing import List
import argparse


class Outliers:

    date_from: str
    date_to: str
    data_dir: str
    entities: List[str]
    output_dir: str
    template_path: str
    table_length: str
    table_length = 5
    df: pd.DataFrame

    COL_NAMES = {
        "chapter": ["BNF Chapter", "Chapter Items"],
        "section": ["BNF Section", "Section Items"],
        "para": ["BNF Paragraph", "Paragraph Items"],
        "subpara": ["BNF Subparagraph", "Subparagraph Items"],
        "chemical": ["BNF Chemical", "Chemical Items"],
        "product": ["BNF Product", "Product Items"],
        "presentation": ["BNF Presentation", "Presentation Items"],
    }

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
            Additional parameters to be calculated
            e.g. ["skew", pd.DataFrame.kurt]
        trim : bool
            Say whether to trim the data before calculating stats
        """

        def __init__(
            self,
            df,
            entity_type,
            num_code,
            denom_code,
            data_dir,
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
            self.data_dir = data_dir

        def get_table(self):
            """Activate getting the stats table

            Returns
            -------
            pandas df
                Table of stats.
            """
            shaped_df = self.__static_data_reshaping(
                self.df,
                self.entity_type,
                self.denom_code,
                self.num_code,
                self.data_dir,
            )
            stats = self.__get_stats(
                df=shaped_df,
                measure=self.measure,
                aggregators=self.num_code,
                stat_parameters=None,
                trim=True,
            )
            return stats

        def __static_data_reshaping(
            self, df, entity_type, denom_code, num_code, data_dir
        ):
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
            # Drop unnecessary columns
            df = df[[entity_type, num_code, "numerator", denom_code]]

            # Aggregate by pcn/ccg
            if entity_type != "practice":
                df = df.groupby(
                    [entity_type, denom_code, num_code], as_index=False
                ).sum()

            # Fill zeroes where there is some prescribing within that subpara
            df = StaticOutlierStats.__fill_zeroes(
                df, entity_type, denom_code, num_code
            )

            # Merge BNF names
            for x in [num_code, denom_code]:
                df = df.merge(
                    self.__get_bnf_names(x, data_dir),
                    how="left",
                    left_on=x,
                    right_index=True,
                )

            # Calculate denominator
            df = df.set_index([entity_type, denom_code, num_code])
            df["denominator"] = df.groupby([entity_type, denom_code]).sum()[
                "numerator"
            ]
            df = df.reset_index()

            # Calculate ratio
            df["ratio"] = df["numerator"] / df["denominator"]

            return df.set_index([entity_type, num_code])

        def __get_bnf_names(self, bnf_level):
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
            names = self.bnf_query(bnf_code, bnf_name)
            return names

        def bnf_query(self, bnf_code, bnf_name):
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
            csv_path = path.join(self.data_dir, "{bnf_name}_names.csv")
            bnf_names = bq.cached_read(query, csv_path=csv_path)
            bnf_names = pd.read_csv(csv_path, dtype={bnf_code: str})
            return bnf_names.set_index(bnf_code)

        @staticmethod
        # Static outliers
        def __fill_zeroes(df, entity_type, denom_code, num_code):
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
            df = df.merge(
                all_chem_map,
                on=[entity_type, denom_code, num_code],
                how="outer",
            )
            df = df.fillna(0)
            df["numerator"] = df["numerator"].astype(int)
            return df

        @staticmethod
        def __get_stats(df, measure, aggregators, stat_parameters, trim=True):
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
                Additional parameters to be calculated
                e.g. ["skew", pd.DataFrame.kurt]
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

            # Some chemicals had a std of ~0 making inf values
            # I think this identifies where e.g. CCGs
            # are the only ones to prescribe this.
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            return df

    # Plots
    @staticmethod
    def __remove_clutter(ax):
        """Removes axes and other clutter from the charts.

        Parameters
        ----------
        ax : matplotlib axis

        Returns
        -------
        ax : matplotlib axis
        """

        for _, v in ax.spines.items():
            v.set_visible(False)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
        # ax.set_xticks([])
        ax.xaxis.set_label_text("")
        plt.tight_layout()
        return ax

    @staticmethod
    def __html_plt(plt):
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
        html_plot = (
            f'<img class="zoom" src="data:image/png;base64,{b64_plot}"/>'
        )

        return html_plot

    @staticmethod
    def __bw_scott(x):
        """
        Scott's Rule of Thumb for bandwidth estimation

        Adapted from
        https://www.statsmodels.org/stable/_modules/statsmodels/nonparametric/bandwidths.html
        previouly cause an issue where the IQR was 0 for many
        pandas.DataFrame.plot.kde does an okay job using scipy method,
        but haven't worked out how to do that

        Parameters
        ----------
        x : array_like
            Array for which to get the bandwidth
        Returns
        -------
        bw : float
            The estimate of the bandwidth
        """

        def _select_sigma(X):
            # normalize = 1.349
            # IQR = (sap(X, 75) - sap(X, 25))/normalize
            # return np.minimum(np.std(X, axis=0, ddof=1), IQR)
            return np.std(X, axis=0, ddof=1)

        A = _select_sigma(x)
        n = len(x)
        return 1.059 * A * n ** (-0.2)

    @staticmethod
    def __dist_plot(org_value, distribution, figsize=(3.5, 1), **kwargs):
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
            distribution,
            bw=Outliers.__bw_scott(distribution),
            ax=ax,
            linewidth=0.9,
            legend=False,
        )
        ax.axvline(org_value, color="r", linewidth=1)
        lower_limit = max(
            0, min(np.quantile(distribution, 0.001), org_value * 0.9)
        )
        upper_limit = max(np.quantile(distribution, 0.999), org_value * 1.1)
        ax.set_xlim(lower_limit, upper_limit)
        ax = Outliers.__remove_clutter(ax)
        plt.close()
        return fig

    def add_plots(self, measure):
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
        self.df["plots"] = self.df[[measure, "array"]].apply(
            lambda x: Outliers.__html_plt(Outliers.__dist_plot(x[0], x[1])),
            axis=1,
        )
        self.df.drop(columns="array", inplace=True)
        # return df

    # data queries
    def __get_chems_per_para(self):
        """
        Gets prescription counts aggregated by chemical,ccg,pcn,practice

        Parameters
        ----------
        date_from : str
            ISO8601 format date indicating start of query range
        date_to : str
            ISO8601 format date indicating end of query range
        Returns
        -------
        chem_per_para : pandas DataFrame
        """

        query = f"""
        SELECT
            practice,
            pcn,
            ccg,
            chemical,
            SUBSTR(chemical, 1, 7) AS subpara,
            numerator
        FROM (
            SELECT
                practice,
                pcn_id AS pcn,
                ccg_id AS ccg,
                SUBSTR(bnf_code, 1, 9) AS chemical,
                SUM(items) AS numerator
            FROM
                ebmdatalab.hscic.normalised_prescribing_standard AS prescribing
            INNER JOIN
                ebmdatalab.hscic.practices AS practices
            ON
                practices.code = prescribing.practice
            WHERE
                practices.setting = 4
                AND practices.status_code ='A'
                AND month BETWEEN TIMESTAMP('{self.date_from}')
                AND TIMESTAMP('{self.date_to}')
                AND SUBSTR(bnf_code, 1, 2) <'18'
            GROUP BY
                chemical,
                ccg_id,
                pcn_id,
                practice
        )
        """
        csv_path = path.join(self.data_dir, "chem_per_para.zip")
        chem_per_para = bq.cached_read(query, csv_path=csv_path)

        # reload specifying data type currently required
        # due to https://github.com/ebmdatalab/datalab-pandas/issues/26
        chem_per_para = pd.read_csv(csv_path, dtype={"subpara": str})
        return chem_per_para

    def __entity_names_query(self, entity_type):
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
        entity_names = bq.cached_read(
            query,
            csv_path=path.join(self.data_dir, f"{entity_type}_names.csv"),
        )
        return entity_names.set_index("code")

    # dataframe processing
    def create_out_table(self, df, attr, entity_type, ascending):
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
        out_table = sort_pick_top(
            df, "z_score", ascending, entity_type, table_length
        )
        out_table = join_measure_array(df, out_table, attr.measure)
        return out_table

    # main method
    def loop_over_everything(self):
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
        self.df = self.__get_chems_per_para(
            self.date_from, self.date_to, self.output_dir
        )
        urlprefix = "https://raw.githack.com/ebmdatalab/outliers/master/"
        toc = MarkdownToC(urlprefix)

        for ent_type in self.entities:
            entity_names = self.__entity_names_query(ent_type, self.output_dir)
            stats_class = StaticOutlierStats(
                df=df,
                entity_type=ent_type,
                num_code="chemical",
                denom_code="subpara",
                data_dir=output_dir,
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
                    self.output_dir,
                    "html",
                    f"static_{ent_type}_{code}.html",
                )
                write_to_template(
                    entity_names.loc[code, "name"],
                    get_entity_table(table_high, stats_class, code),
                    get_entity_table(table_low, stats_class, code),
                    output_path=output_file,
                    template_path=self.template_path,
                )
                toc.add_file(output_file, entity=ent_type)
        toc.write_toc(self.output_dir)
