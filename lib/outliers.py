from base64 import b64encode
from io import BytesIO
from os import path
from matplotlib import pyplot as plt
import numpy as np
from ebmdatalab import bq
from typing import Dict, List
import pandas as pd
from datetime import date
import seaborn as sns
from lib.table_of_contents import TableOfContents
from lib.make_html import write_to_template
import traceback
from pqdm.processes import pqdm


class DatasetBuild:
    """
    Calls outlier dataset building stored procedure on bigquery
    fetches, and encapsulates results of this process.

    Attributes
    ----------
    from_date : datetime.date
        start date of outlier reporting period
    to_date : datetime.date
        end date of outlier reporting period
    n_outliers : int
        number of outliers to include in each "high" and "low" outtlier set
    entities : List[str]
        list of column names for entity types to report e.g. "ccg"
    force_rebuild : bool
        force rebuilding of outlier dataset within bigquery and rebuilding
        of local data caches.
    numerator_column : str
        column name for numerator values in ratio calculation
        N.B: not yet integrated into bigquery stored procedure
    denominator_column : str
        column name for denominator values in ratio calculation
        N.B: not yet integrated into bigquery stored procedure
    """

    # consts
    _DATEFMT = "%Y-%m-%d"
    _KNOWN_ENTITIES = ["practice", "ccg", "pcn"]

    def __init__(
        self,
        from_date: date,
        to_date: date,
        n_outliers: int,
        entities: List[str],
        force_rebuild: bool = False,
        numerator_column: str = "chemical",
        denominator_column: str = "subpara",
    ) -> None:
        assert (
            type(to_date) == date and type(from_date) == date
        ), "date args must be dates"
        assert to_date > from_date, "to date must be after from date"
        self.from_date = from_date
        self.to_date = to_date

        assert n_outliers > 0, "n must be greater than zero"
        self.n_outliers = n_outliers

        assert len(entities) > 0, "list of entities must be populated"
        for e in entities:
            assert e in self._KNOWN_ENTITIES, f"{e} not recognised entity"
        self.entities = entities

        self.force_rebuild = force_rebuild
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column

        self.results: Dict[str, pd.DataFrame] = {}
        self.results_items: Dict[str, pd.DataFrame] = {}
        self.results_measure_arrays: Dict[str, pd.DataFrame] = {}
        self.entity_hierarchy: Dict[str, Dict[str, List[str]]] = {}

    def run(self) -> None:
        """
        Execute outlier dataset build stored procedure

        Populates build_id attribute upon completion
        """
        sql = f"""
            CALL `ebmdatalab.outlier_detection.build_outliers`(
                '{self.from_date.strftime(self._DATEFMT)}',
                '{self.to_date.strftime(self._DATEFMT)}',
                {self.n_outliers},
                {str(self.force_rebuild).upper()});
            SELECT
                build_id
            FROM
                `ebmdatalab.outlier_detection.builds`
            WHERE
                from_date = '{self.from_date.strftime(self._DATEFMT)}'
                AND to_date = '{self.to_date.strftime(self._DATEFMT)}'
                AND n ={self.n_outliers}"""
        res = bq.cached_read(
            sql, csv_path="../data/bq_cache/build_result.zip", use_cache=False
        )
        self.build_id = res["build_id"].values[0]

    def fetch_results(self) -> None:
        """
        Runs results-fetching methods for each entity type + lookup tables
        """
        assert self.build_id, "build must be run before fetching results"
        for e in self.entities:
            self._get_entity_results(e)
            self._get_entity_items(e)
            self._get_entity_measure_arrays(e)
        self._get_lookups()
        self._get_hierarchy()

    def _get_hierarchy(self) -> None:
        """
        Gets ccg-pcn-practice hierachy as dictionary
        """
        sql = """
        SELECT
            code as `practice_code`,
            pcn_id as `pcn_code`,
            ccg_id as `ccg_code`
        FROM
            `ebmdatalab.hscic.practices`
        WHERE
            setting=4
            AND status_code = 'A'
            AND pcn_id is not null
        """
        csv_path = "../data/bq_cache/entity_hierarchy.zip"
        res: pd.DataFrame = bq.cached_read(
            sql,
            csv_path,
            use_cache=(not self.force_rebuild),
        )
        res = res.set_index(["ccg_code", "pcn_code"])

        # only include practices for which there are results
        res = res[
            res.practice_code.isin(
                self.results["practice"].index.get_level_values(0).unique()
            )
        ]

        # convert to hierarchial dict
        for ccg_code in res.index.get_level_values(0).unique():
            pcns = {}
            for pcn_code in res.loc[ccg_code, slice(None)].index.unique():
                pcns[pcn_code] = (
                    res.loc[ccg_code, slice(None)]
                    .query(f"pcn_code=='{pcn_code}'")
                    .practice_code.tolist()
                )
            self.entity_hierarchy[ccg_code] = pcns

    def _get_lookups(self) -> None:
        """
        Fetches entity code:name mapping tables for each entity, plus
        bnf code:name mapping tables for numerator and denominator
        """
        self.names = {e: self._entity_names_query(e) for e in self.entities}
        self.names[self.numerator_column] = self._get_bnf_names(
            self.numerator_column
        )
        self.names[self.denominator_column] = self._get_bnf_names(
            self.denominator_column
        )

    def _get_entity_results(self, entity: str) -> None:
        sql = f"""
        SELECT
            {entity},
            subpara,
            subpara_items,
            chemical,
            chemical_items,
            ratio,
            mean,
            std,
            z_score,
            rank_high,
            rank_low
        FROM
            `ebmdatalab.outlier_detection.{entity}_ranked`
        WHERE
            build_id = {self.build_id}
            AND (
                    rank_high <={self.n_outliers}
                    OR rank_low <={self.n_outliers}
                );
        """

        csv_path = f"../data/bq_cache/{entity}_results.zip"
        res = bq.cached_read(
            sql,
            csv_path,
            use_cache=(not self.force_rebuild),
        )
        # reload csv with correct datatypes
        # see https://github.com/ebmdatalab/datalab-pandas/issues/26
        res = pd.read_csv(
            csv_path,
            dtype={self.numerator_column: str, self.denominator_column: str},
        )
        res = res.set_index([entity, self.numerator_column])
        self.results[entity] = res

    def _get_entity_items(self, entity: str) -> None:
        sql = f"""
        SELECT
            {entity},
            bnf_code,
            bnf_name,
            chemical,
            high_low,
            numerator
        FROM
            `ebmdatalab.outlier_detection.{entity}_outlier_items`
        WHERE
            build_id = {self.build_id};
        """

        res = bq.cached_read(
            sql,
            f"../data/bq_cache/{entity}_items.zip",
            use_cache=(not self.force_rebuild),
        )
        self.results_items[entity] = res

    def _get_entity_measure_arrays(self, entity: str) -> None:
        sql = f"""
            SELECT
                chemical,
                measure_array as `array`
            FROM
                `ebmdatalab.outlier_detection.{entity}_measure_arrays`
            WHERE
                build_id = {self.build_id};
        """
        try:
            res = bq.cached_read(
                sql,
                f"../data/bq_cache/{entity}_measure_arrays.zip",
                use_cache=(not self.force_rebuild),
            )
        except Exception:
            print(f"Error getting BQ data for {entity}")
            traceback.print_stack()
        try:
            res.array = res.array.apply(
                lambda x: np.fromstring(x[1:-1], sep=",")
            )
        except Exception:
            print(f"Error doing array conversion for {entity}")
            traceback.print_stack()
        self.results_measure_arrays[entity] = res.set_index("chemical")

    def _entity_names_query(self, entity_type: str) -> pd.DataFrame:
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
            csv_path=f"../data/bq_cache/{entity_type}_names.zip",
            use_cache=(not self.force_rebuild),
        )
        return entity_names.set_index("code")

    def _get_bnf_names(self, bnf_level: str) -> pd.DataFrame:
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

    def bnf_query(self, bnf_code: str, bnf_name: str) -> pd.DataFrame:
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
        bnf_names = bq.cached_read(
            query,
            csv_path=f"../data/bq_cache/{bnf_name}_names.zip",
            use_cache=(not self.force_rebuild),
        )
        bnf_names = pd.read_csv(
            f"../data/bq_cache/{bnf_name}_names.zip", dtype={bnf_code: str}
        )
        return bnf_names.set_index(bnf_code)


class Report:
    """
    Formatted dataset for an individual instance of an entity
    Attributes
    ----------
    build: DatasetBuild
        Instance of a built and fetched outliers dataset
    entity_type : str
        Column name for entity type, e.g. 'ccg'
    entity_code : str
        Identifying code for entity
    entity_name : str
        Name of entity
    table_high: pandas.Dataframe
        Formatted table of "high" outliers for entity
    items_high : pandas.Dataframe
        Formatted table of prescription items pertaining to high outliers
    table_low: pandas.Dataframe
        Formatted table of "low" outliers for entity
    items_low : pandas.Dataframe
        Formatted table of prescription items pertaining to low outliers
    """

    # consts
    _COL_NAMES = {
        "chapter": ["BNF Chapter", "Chapter Items"],
        "section": ["BNF Section", "Section Items"],
        "para": ["BNF Paragraph", "Paragraph Items"],
        "subpara": ["BNF Subparagraph", "Subparagraph Items"],
        "chemical": ["BNF Chemical", "Chemical Items"],
        "product": ["BNF Product", "Product Items"],
        "presentation": ["BNF Presentation", "Presentation Items"],
    }

    def __init__(
        self,
        entity_type: str,
        entity_code: str,
        build: DatasetBuild,
    ) -> None:
        self.entity_type = entity_type
        self.entity_code = entity_code
        self.build = build

    def _ranked_dataset(self, h_l: str) -> pd.DataFrame:
        assert h_l in ["h", "l"], "high/low indicator must be 'h' or 'l'"
        rank_column = f"rank_{'high' if h_l == 'h' else 'low'}"
        return (
            self.build.results[self.entity_type]
            .query(f'{self.entity_type} == "{self.entity_code}"')
            .query(f"{rank_column} <= {self.build.n_outliers}")
            .copy()
            .sort_values(rank_column)
        )

    def _create_items_table(self, h_l: str) -> pd.DataFrame:
        assert h_l in ["h", "l"], "high/low indicator must be 'h' or 'l'"
        return (
            self.build.results_items[self.entity_type]
            .query(f'{self.entity_type} == "{self.entity_code}"')
            .query(f'high_low == "{h_l}"')
        )

    @staticmethod
    def _format_entity(entity: str) -> str:
        return entity.upper() if entity == "ccg" else entity

    @staticmethod
    def _format_denom(denominator_column: str, denominator_code: str) -> str:
        """
        formats BNF chapter/section/para/subpara strings for OP website
        e.g.: 030700 -> 3.7, 0601021 -> 6.1.2
        """
        if denominator_column in [
            "chapter",
            "section",
            "para",
            "subpara",
        ]:
            substrings = []
            for i in range(0, len(denominator_code), 2):
                sub = denominator_code[i : i + 2]
                if sub == "00" or len(sub) == 1:
                    continue
                substrings.append(sub.lstrip("0"))
            return ".".join(substrings)
        else:
            return denominator_code

    def _add_openprescribing_analyse_url(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate URL to OpenPrescribing analyse page for
        numerator and denominator highlighting entity
        Parameters
        ----------
        df : pandas df
            Input dataframe
        attr : StaticOutlierStats instance
            Contains attributes to be used in defining the tables.
        code : str
            ID of organisational entity to be highlighted

        Returns
        -------
        pandas df
            Dataframe with URL column added.
        """
        url_base = "https://openprescribing.net/analyse/#"
        url_selected = "&selectedTab=summary"

        url_org = (
            f"org={Report._format_entity(self.entity_type)}"
            f"&orgIds={self.entity_code}"
        )

        def build_url(x):
            """assembles url elements in order"""
            url_num = f"&numIds={x[self.build.numerator_column]}"
            formatted_denom = Report._format_denom(
                self.build.denominator_column, x[self.build.denominator_column]
            )
            url_denom = f"&denomIds={formatted_denom}"
            return url_base + url_org + url_num + url_denom + url_selected

        ix_col = df.index.name
        df = df.reset_index()
        df["URL"] = df.apply(build_url, axis=1)
        df = df.set_index(ix_col)
        return df

    def _tidy_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rounds figures, drops unnecessary columns and changes column names to be
        easier to read (according to col_names), reorders columns.

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
        for x in [self.build.numerator_column, self.build.denominator_column]:
            df = df.merge(
                self.build.names[x], how="left", left_on=x, right_index=True
            )
        df = df.drop(
            columns=[self.build.denominator_column, "rank_high", "rank_low"]
        )
        df = df.rename(
            columns={
                f"{self.build.numerator_column}_name": self._COL_NAMES[
                    self.build.numerator_column
                ][0],
                f"{self.build.numerator_column}_items": self._COL_NAMES[
                    self.build.numerator_column
                ][1],
                f"{self.build.denominator_column}_name": self._COL_NAMES[
                    self.build.denominator_column
                ][0],
                f"{self.build.denominator_column}_items": self._COL_NAMES[
                    self.build.denominator_column
                ][1],
            }
        )
        column_order = [
            self._COL_NAMES[self.build.numerator_column][0],
            self._COL_NAMES[self.build.numerator_column][1],
            self._COL_NAMES[self.build.denominator_column][0],
            self._COL_NAMES[self.build.denominator_column][1],
            "ratio",
            "mean",
            "std",
            "z_score",
            "plots",
            "URL",
        ]
        df = df[column_order]
        df = df.set_index(self._COL_NAMES[self.build.numerator_column][0])
        return df

    def _create_out_table(self, h_l: str) -> pd.DataFrame:
        df = self._ranked_dataset(h_l)
        df = df.reset_index().set_index("chemical")
        df = df.drop(columns=self.entity_type)
        df = df.join(self.build.results_measure_arrays[self.entity_type])
        df = Plots.add_plots(df, "ratio")
        df = self._add_openprescribing_analyse_url(df)
        df = self._tidy_table(df)
        return df

    def format(self) -> None:
        if self.entity_code in self.build.names[self.entity_type].index:
            self.entity_name = self.build.names[self.entity_type].loc[
                self.entity_code, "name"
            ]
        else:
            self.entity_name = "Unknown"
        self.table_high = self._create_out_table("h")
        self.table_low = self._create_out_table("l")
        self.items_high = self._create_items_table("h")
        self.items_low = self._create_items_table("l")


class Plots:
    """
    Collection of static methods for generation and formatting of
    distribution plots, and their appendment to dataframes
    """

    @staticmethod
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
            lambda x: Plots._html_plt(Plots._dist_plot(x[0], x[1])), axis=1
        )
        df = df.drop(columns="array")
        return df

    @staticmethod
    def _html_plt(plt):
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
    def _dist_plot(org_value, distribution, figsize=(3.5, 1), **kwargs):
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
            bw=Plots._bw_scott(distribution),
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
        ax = Plots._remove_clutter(ax)
        plt.close()
        return fig

    @staticmethod
    def _bw_scott(x):
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
    def _remove_clutter(ax):
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
        for _, v in ax.spines.items():
            v.set_visible(False)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
        # ax.set_xticks([])
        ax.xaxis.set_label_text("")
        plt.tight_layout()
        return ax


class Runner:
    """
    Constructs and runs dataset build, generates report datasets,
    populates template to form html reports, builds table of contents

    Attributes
    ----------
    from_date : datetime.date
        start date of outlier reporting period
    to_date : datetime.date
        end date of outlier reporting period
    n_outliers : int
        number of outliers to include in each "high" and "low" outtlier set
    entities : List[str]
        list of column names for entity types to report e.g. "ccg"
    force_rebuild : bool
        force rebuilding of outlier dataset within bigquery and rebuilding
        of local data caches.
    entity_limit : int
        limit generated entity reports to first n of each type
    output_dir : str
        path to output directory for html report files
    template_path : str
        path to jinja2 html template for reports
    url_prefix : str
        prefix for urls for links to report files within
        generated table of contents
    """

    def __init__(
        self,
        from_date: date,
        to_date: date,
        n_outliers: int,
        entities: List[str],
        force_rebuild: bool = False,
        entity_limit: int = None,
        output_dir="../data",
        template_path="../data/template.html",
        url_prefix="https://raw.githack.com/ebmdatalab/outliers/master/",
        n_jobs=8,
    ) -> None:
        self.build = DatasetBuild(
            from_date=from_date,
            to_date=to_date,
            n_outliers=n_outliers,
            entities=entities,
            force_rebuild=force_rebuild,
        )
        self.output_dir = output_dir
        self.template_path = template_path
        self.toc = TableOfContents(url_prefix=url_prefix)
        self.entity_limit = entity_limit
        self.n_jobs = n_jobs

    def run(self):
        # run main build process on bigquery and fetch results
        self.build.run()
        self.build.fetch_results()
        self._truncate_entites()
        self._trim_results()
        self.toc.hierarchy = self.build.entity_hierarchy

        # loop through entity types, generated a report for each entity item
        for e in self.build.entities:
            for f in self._run_entity_report(e):
                self.toc.add_item(**f)

        # write out toc
        self.toc.write_html(self.output_dir)
        self.toc.write_markdown(self.output_dir, True)

    def _truncate_entites(self):
        """
        Evenly discard entities throughout hierarchy so n<=limit for all levels
        """
        if not self.entity_limit:
            return

        # discard ccgs until limit
        while self.entity_limit < len(self.build.entity_hierarchy.keys()):
            self.build.entity_hierarchy.popitem()

        # if limit <= |ccgs| then only one practice per pcn per ccg
        if self.entity_limit <= len(self.build.entity_hierarchy.keys()):
            for ccg, pcns in self.build.entity_hierarchy.items():
                pcn_code = list(pcns.keys())[0]
                self.build.entity_hierarchy[ccg] = {
                    pcn_code: pcns[pcn_code][0:1]
                }
            return

        # if limit <= |pcns| then discard until limit
        # and then only one practice per pcn
        pcn_count = sum(
            [len(v.keys()) for v in self.build.entity_hierarchy.values()]
        )
        if self.entity_limit <= pcn_count:
            # discard pcns until limit
            while self.entity_limit < sum(
                [len(v.keys()) for v in self.build.entity_hierarchy.values()]
            ):
                for _, pcns in self.build.entity_hierarchy.items():
                    pcns.popitem()
            # one practice per pcn
            for _, pcns in self.build.entity_hierarchy.items():
                pcn_code = list(pcns.keys())[0]
                pcns = {pcn_code: pcns[pcn_code][0:1]}
            return

        # count bottom level of hierarchy
        def practice_count():
            return len(
                [
                    p
                    for q in [
                        y
                        for x in [
                            list(v.values())
                            for v in self.build.entity_hierarchy.values()
                        ]
                        for y in x
                    ]
                    for p in q
                ]
            )

        # if num practices over limit then discard evenly
        if practice_count() > self.entity_limit:
            while True:
                for ccg, pcns in self.build.entity_hierarchy.items():
                    for pcn, practices in pcns.items():
                        if len(practices) > 1:
                            self.build.entity_hierarchy[ccg][pcn] = practices[
                                :-1
                            ]
                        if practice_count() <= self.entity_limit:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

    def _trim_results(self):
        # trim build entity results to match limited entities
        ccgs = list(self.build.entity_hierarchy.keys())
        self.build.results["ccg"] = self.build.results["ccg"].loc[
            ccgs, slice(None)
        ]

        pcns = [
            y
            for x in [
                list(v.keys()) for v in self.build.entity_hierarchy.values()
            ]
            for y in x
        ]
        self.build.results["pcn"] = self.build.results["pcn"].loc[
            pcns, slice(None)
        ]

        practices = [p for q in [
            y
            for x in [
                list(v.values()) for v in self.build.entity_hierarchy.values()
            ]
            for y in x
        ] for p in q]

        self.build.results["practice"] = self.build.results["practice"].loc[
            practices, slice(None)
        ]

    def _run_item_report(self, entity, code):
        report = Report(
            entity_type=entity,
            entity_code=code,
            build=self.build,
        )
        report.format()
        output_file = path.join(
            self.output_dir,
            "html",
            f"static_{entity}_{code}.html",
        )
        write_to_template(
            entity_name=report.entity_name,
            tables_high=(report.table_high, report.items_high),
            tables_low=(report.table_low, report.items_low),
            output_path=output_file,
            template_path=self.template_path,
        )
        return {
            "code": code,
            "name": report.entity_name,
            "entity": entity,
            "file_path": output_file,
        }

    def _run_entity_report(self, entity):
        codes = self.build.results[entity].index.get_level_values(0).unique()
        kwargs = [{"entity": entity, "code": c} for c in codes]
        files = pqdm(
            kwargs,
            self._run_item_report,
            n_jobs=self.n_jobs,
            argument_type="kwargs",
        )
        return files
