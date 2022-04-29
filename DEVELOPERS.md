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
