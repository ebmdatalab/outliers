# Architecture

The prescribing outlier report generation is designed around the concept of a "dataset build" with a configured set of parameters (number of outliers to report, date range, which entity types to report for) from which a set of static HTML files are generated according to this configuration. 

The building of the dataset from the NHS England primary care prescribing data is done within a stored procedure on Google BigQuery for performance and cost reasons. The large number of entities (~7000) results in a large number of repitious queries when run on a query-per-entity basis, resulting in unacceptable runtimes and Google Cloud Platform costs.

Building of the reports is controlled by a [runner](#runner) class, which handles the parameter set, building of the [dataset](#datasetbuild), [report](#report) generation, and construction of a [table of contents](#tableofcontents) for the generated HTML reports.

## BigQuery objects

The definition of the `build_outliers` stored procedure and supporting tables is given in the `.sql` files contained within `data/static_outiler_sql`.

### Tables

* `builds` *dataset build configuration*
  * `build_id` NUMERIC NOT NULL *unique identifier for build*
  * `from_date` STRING NOT NULL *start date for inclusion in dataset, YYYY-MM-DD* 
  * `to_date` STRING NOT NULL *end date for inclusion in dataset, YYYY-MM-DD*
  * `n` NUMERIC NOT NULL *number of outliers to include in build*
  
* `summed` *aggregate prescribing data at practice level within configured date range*
  * `build_id` int NOT NULL *unique identifier for build, used as partitioning key*
  * `practice` STRING(6) *practice identifier, FK to `ebmdatalab.hscic.practices`*
  * `chemical` STRING(9) *BNF chemical identifier, zero padded*
  * `subpara` STRING(7) *BNF subparagraph identifier, zero padded*
  * `numerator` NUMERIC *quantity of items prescribed*
* `[entity]_ranked` *prescribing data aggregated to \[entity\]+chemical level, with chemical:subparagraph ratio, z-score and rank*
  * `build_id` INTEGER NOT NULL *unique identifier for build, used as partitioning key*
  * `[entity]` STRING(6) NOT NULL *entity identifier, FK to `ebmdatalab.hscic.practices/ccgs/pcns/stps` etc.*
  * `subpara` STRING(7) NOT NULL *BNF subparagraph identifier, zero padded*
  * `subpara_items` NUMERIC NOT NULL *Quantity of items prescribed by entity in this subparagraph*
  * `chemical` STRING(9) NOT NULL *BNF chemical identifier, zero padded*
  * `chemical_items` NUMERIC NOT NULL *Quantity of items prescribed by entity of this chemical*
  * `ratio` FLOAT64 NOT NULL *Ratio of chemical items prescribed to subparagraph items prescribed*
  * `mean` FLOAT64 NOT NULL *Mean chemical:subparagraph ratio for this chemical across all entities of this type*
  * `std` FLOAT64 NOT NULL *Standard deviation of chemical:subparagraph ratios for this chemical across all entities of this type*
  * `z_score` FLOAT64 NOT NULL *Z Score of chemical:subparagraph ratios for this chemical across all entities of this type**
  * `rank_high` NUMERIC NOT NULL *Rank of the z score of this entity's chemical:subparagraph ratio for this chemical, descending order*
  * `rank_low` NUMERIC NOT NULL *Rank of the z score of this entity's chemical:subparagraph ratio for this chemical, ascending order*
* `[entity]_outlier_items` *prescribing data aggregated to \[entity\]+BNF item level for identified high/low outliers*
  * `build_id` int NOT NULL *unique identifier for build, used as partitioning key*
  * `[entity]` STRING(6) NOT NULL *entity identifier, FK to `ebmdatalab.hscic.practices/ccgs/pcns/stps` etc.*
  * `bnf_code` STRING(15) *BNF item identifier, zero padded*
  * `bnf_name` STRING(100) *BNF item name*
  * `chemical` STRING(9) *BNF chemical identifier, zero padded*
  * `high_low` STRING(1) *H/L indicator of high or low outlier*
  * `numerator` NUMERIC *Quantity of items prescribed*
* `[entity]_measure_arrays`
  * `build_id` int NOT NULL *unique identifier for build, used as partitioning key*
  * `chemical` string(9) not null *BNF chemical identifier, zero padded*
  * `measure_array` array\<float64\> *Array of all chemical:subparagraph prescribing ratio z scores for this entity type within build date range for this chemical*

## lib/outliers.py

### DatasetBuild
### Report
### Runner
### Plots

## lib/make_html.py

## lib/table_of_contents.py

### TableOfContents

# Deployment to openprescribing.net

# Known issues
* entity list configuration not passed to builder sproc
