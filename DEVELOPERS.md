# Architecture

The prescribing outlier report generation is designed around the concept of a "dataset build" with a configured set of parameters (number of outliers to report, date range, which entity types to report for) from which a set of static HTML files are generated according to this configuration. 

The building of the dataset from the NHS England primary care prescribing data is done within a stored procedure on Google BigQuery for performance and cost reasons. The large number of entities (~7000) results in a large number of repitious queries when run on a query-per-entity basis, resulting in unacceptable runtimes and Google Cloud Platform costs.

Building of the reports is controlled by a [runner](#runner) class, which handles the parameter set, building of the [dataset](#datasetbuild), [report](#report) generation, and construction of a [table of contents](#tableofcontents) for the generated HTML reports.

## BigQuery objects

The definition of the `build_outliers` stored procedure and supporting tables is given in the `.sql` files contained within `data/static_outiler_sql`.

### Tables

* `builds` *dataset build configuration*
  * `build_id` NUMERIC NOT NULL - *unique identifier for build*
  * `from_date` STRING NOT NULL - *start date for inclusion in dataset, YYYY-MM-DD* 
  * `to_date` STRING NOT NULL - *end date for inclusion in dataset, YYYY-MM-DD*
  * `n` NUMERIC NOT NULL - *number of outliers to include in build*
  
* `summed` *aggregate prescribing data at practice level within configured date range*
  * `build_id` int NOT NULL - *unique identifier for build, used as partitioning key*
  * `practice` STRING(6) - *practice identifier, FK to `ebmdatalab.hscic.practices`*
  * `chemical` STRING(9) - *BNF chemical identifier, zero padded*
  * `subpara` STRING(7) - *BNF subparagraph identifier, zero padded*
  * `numerator` NUMERIC - *quantity of items prescribed*
* `[entity]_ranked` *prescribing data aggregated to \[entity\]+chemical level, with chemical:subparagraph ratio, z-score and rank*
  * `build_id` INTEGER NOT NULL - *unique identifier for build, used as partitioning key*
  * `[entity]` STRING(6) NOT NULL - *entity identifier, FK to `ebmdatalab.hscic.practices/ccgs/pcns/stps` etc.*
  * `subpara` STRING(7) NOT NULL - *BNF subparagraph identifier, zero padded*
  * `subpara_items` NUMERIC NOT NULL - *Quantity of items prescribed by entity in this subparagraph*
  * `chemical` STRING(9) NOT NULL - *BNF chemical identifier, zero padded*
  * `chemical_items` NUMERIC NOT NULL - *Quantity of items prescribed by entity of this chemical*
  * `ratio` FLOAT64 NOT NULL - *Ratio of chemical items prescribed to subparagraph items prescribed*
  * `mean` FLOAT64 NOT NULL - *Mean chemical:subparagraph ratio for this chemical across all entities of this type*
  * `std` FLOAT64 NOT NULL - *Standard deviation of chemical:subparagraph ratios for this chemical across all entities of this type*
  * `z_score` FLOAT64 NOT NULL - *Z Score of chemical:subparagraph ratios for this chemical across all entities of this type**
  * `rank_high` NUMERIC NOT NULL - *Rank of the z score of this entity's chemical:subparagraph ratio for this chemical, descending order*
  * `rank_low` NUMERIC NOT NULL - *Rank of the z score of this entity's chemical:subparagraph ratio for this chemical, ascending order*
* `[entity]_outlier_items` *prescribing data aggregated to \[entity\]+BNF item level for identified high/low outliers*
  * `build_id` int NOT NULL - *unique identifier for build, used as partitioning key*
  * `[entity]` STRING(6) NOT NULL - *entity identifier, FK to `ebmdatalab.hscic.practices/ccgs/pcns/stps` etc.*
  * `bnf_code` STRING(15) NOT NULL - *BNF item identifier, zero padded*
  * `bnf_name` STRING(100) NOT NULL - *BNF item name*
  * `chemical` STRING(9) NOT NULL - *BNF chemical identifier, zero padded*
  * `high_low` STRING(1) NOT NULL - *H/L indicator of high or low outlier*
  * `numerator` NUMERIC NOT NULL - *Quantity of items prescribed*
* `[entity]_measure_arrays`
  * `build_id` int NOT NULL - *unique identifier for build, used as partitioning key*
  * `chemical` string(9) NOT NULL - *BNF chemical identifier, zero padded*
  * `measure_array` array\<float64\> NOT NULL - *Array of all chemical:subparagraph prescribing ratio z scores for this entity type within build date range for this chemical*

### build_outliers Stored Procedure

Aggregates prescribing data for the configured date period; calculates ratios, z scores, and outlier rankings

#### Parameters
* `p__from_date` STRING - *start date for outlier data generation YYYY-MM-DD format*
* `p__to_date` STRING - *end date for outlier data generation YYYY-MM-DD format*
* `p__n` : int - *number of outliers to include in high and low sets*
* `p__force` BOOLEAN - *force the regeneration of an outlier datasets if build matching above parameters exists*

#### Build Configuration
The first action of this procedure is to search the `builds` table for the supplied parameter values, if a match is found and the force parameter is false, then the procedure completes successfully with no further action.  If the force parameter is true, then all data in the `[entity]_ranked`, `[entity]_outlier_items`, and `[entity]_measure_arrays` tables for the matched `build_id` are deleted and the build process continues with the `v__build_id` variable set to the matched `build_id`. 

If no existing builds are found with the supplied parameter values, a new row with these values is added to the `builds` table and the `v__build_id` variable set to the inserted `build_id`.

#### Pre-aggregation
The dataset build process begins with aggregation of prescibing data within the configured date range to practice level. Prescription items are counted for BNF items in Chapters 17 and below, for practices with a setting of 4 and status code A. Practices without a CCG or STP are also excluded. The `summed` table is populated with these counts further aggregated to BNF Chemical and Subparagraph levels, and populated with the `v__build_id` variable to allow for this table to be partitioned on a build basis (for cost and performance reasons).

For chemicals where no prescribing has been made by a given practice, but any other practices have prescribed items in that chemical's subparagraph, rows with a `numerator` of 0 are added to the `summed` table to make calculations further down the process simpler. 

#### Ratios and Rankings
For each of the entity types (Practice, PCN, CCG, STP currently), data from the `summed` table is further aggregated (for higher-order entity types) and the following metrics calculated for each BNF chemical:
* the ratio of items prescribed to items prescribed for all other Chemicals in this BNF Subparagraph
* the mean of this ratio across all entites of this type
* the standard deviation of this ratio across all entites of this type

The results of these aggregations are then inserted into the `[entity]_ranked` tables alongside the value of the `v__build_id` variable.

#### Outlier Prescription Items
In order to support the modal list of specific items prescribed for a given outlying chemical, these are precalculated based on the rows in the `[entity]_ranked` tables with a high or low ranking less than or equal to this `build`'s configured `n` value. The item codes (to support linking out to openprescribing.net), names, and quantity prescribed are populated in the `[entity]_outlier_items` alongside the value of the `v__build_id` variable.

#### Measure Arrays
In order to support generation of the density plots in the outlier reports, an array of z scores for all entities of a given type is extracted from the `[entity]_ranked` tables on a per-chemical basis and stored as a native array of floating point numbers for maximum performance. These are then stored in the `[entity]_measure_arrays` tables alongside the value of the `v__build_id` variable.

## Report Generation Process
The output HTML reports are generated by a process split across a number of Python files, roughly separated into functional areas. These are the result of many interations of prototypes by many people over a long period of time and thus may not be the cleanest nor most efficient code.

### lib/outliers.py
#### Runner
The `Runner` class is the main entry point for this process. Instances are constructed with the parameters for the dataset build (date ranges, number of outliers, etc.), parameters governing the format of the reports (template paths, url prefices, output path where the files will be places), and a parameter to limit the number of reports of each entity type are generated (for small-scale testing of code changes).

Each instance exposes a single public `run()` method whose first action is to instantiate a `DatasetBuild` and execute the dataset builder stored procedure on BigQuery. Once this stored procedure is complete, its full results are fetched into memory within the `DatasetBuild` object.

If an entity limit is configured, these in-memory results are then truncated to that limit. Due to the hierarchical nature of the entity types (STP->CCG->PCN->Practice), this is a surprisingly complex process which ensures that the entities at each level are evenly distributed across entities at the parent level. There are [known issues](#known-issues) and inefficiencies within this process, but since this is only used during development it has not been prioritised for improvement.

Once the dataset is built and optionally truncated, the list of configured entity types is iterated, generating [`Report`s](#report) from the dataset, and then [rendering HTML](#libmakehtmlpy) from them. Within each entity type, these steps are performed in parallel using `pqdm` which also provides progress bars (where this is run in an interactive session). The number of concurrent jobs is configured by the `n_jobs` parameter of the `Runner` constructor. 

#### DatasetBuild
The `DatasetBuild` class represents an execution of the [dataset builder procedure](#buildoutliers-stored-procedure) and its resultant dataset. It is instantiated with the dataset-specific parameters, and its constructor performs some basic checks of these parameters and instantitates data structures for storage of the dataset itself. 

It exposes a public `run()` instance method, which connects to BigQuery and calls the [dataset builder procedure](#buildoutliers-stored-procedure) stored procedure.

The `fetch_results()` instance method iterates through the [entity result tables](#tables), performing a cached database read into the instances results data structures. The caching is performed by the [`cached_read()`](https://github.com/ebmdatalab/datalab-pandas/blob/master/ebmdatalab/bq.py#L27) method of the `BQ` module of the ebmdatalab [datalab-pandas](https://github.com/ebmdatalab/datalab-pandas) project. Without a force parameter, these reads should read from the local file cache (if present) if a given dataset's data has previously been fetched.

#### Report
The `Report` class is responsible for formatting a per-entity slice of the dataset ready for HTML generation. It is instantiated with a completed `DatasetBuild`, the type and identifier of an entity.

Each instance exposes a `format()` method which generates the formatted high- and low- outlier tables for an entity, along with their companion tables of individual prescription items.

#### Plots
The `Plots` class is a collection of static methods for formatting of the distribution plots used in the reports, and for appending them to dataframes used in HTML geneation.

### lib/make_html.py

Partially adapted from https://github.com/ebmdatalab/html-template-demo, this contains a number of helper methods for population of a Jinja2 template with data contained within a `Report` object. The most complex of which is `df_to_html()` which generates HTML tables from the Pandas Dataframes containing the low and high outliers. 

### lib/table_of_contents.py

#### TableOfContents
A `TableOfContents` instance is created by the `Runner.run()` and for each completed report, its path is passed to `add_item()`. The final HTML or markdown tables of contents are rendered by `write_html()` and `write_markdown()` respectively.

# Deployment to openprescribing.net

This project is designed for use in an interactive session, e.g. jupyter notebook (as per `notebooks/notebook1.ipynb`) during prototyping/development of these reports. The majority of the updates to data and precalculated metrics on openprescribing.net are done via Django [management commands](https://docs.djangoproject.com/en/4.0/howto/custom-management-commands/), and so the "production" version of these reports that appear on openprescribing are also generated using [such a command](https://github.com/ebmdatalab/openprescribing/blob/main/openprescribing/pipeline/management/commands/outlier_reports.py). 

This management command makes use of the [BigQuery objects](#bigquery-objects) described earlier in this document, but has some slight modifications to the Python code. The three report-generation Python files are combined into a single file and minimal Django management command logic is added. `pqdm` as used in standalone reports not suitable for this purpose, and so the underlying `ProcessPoolExecutor` is used instead. By default a lower degree of parallelism is used as not to adversely affect other running processes. Another change from this project is that the ebmdatalab cached bq read methods are replaced with raw BigQuery read methods from the BigQuery API. Repeated runs where caching beneficial unlikely in production so this is not conisdered to be a major issue.

# Known issues
* The entity list configuration within a `Runner` is not passed to builder stored procedure and so data tables in may be unneccesarily populated.
* Outlier data generation steps are statically defined for each entity type, additional entity types (e.g. regional teams) would require copying of code. It may be better to turn into dynamic SQL but performance penalty of doing so in BigQuery is currently unknown
* Use of the `entity_limit` parameter with a non-default `entities` list within `Runner` triggers a bug ( [#41](/../../issues/41)) within `truncate_entities()` due to assumption of a full entity hierarchy
