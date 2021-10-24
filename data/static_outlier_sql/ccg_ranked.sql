CREATE TABLE ebmdatalab.outlier_detection.ccg_ranked (
    build_id INTEGER NOT NULL,
    ccg STRING(6) NOT NULL,
    subpara STRING(7) NOT NULL,
    subpara_items NUMERIC NOT NULL,
    chemical STRING(9) NOT NULL,
    chemical_items NUMERIC NOT NULL,
    ratio FLOAT64 NOT NULL,
    mean FLOAT64 NOT NULL,
    std FLOAT64 NOT NULL,
    z_score FLOAT64 NOT NULL,
    rank_high NUMERIC NOT NULL,
    rank_low NUMERIC NOT NULL
) 
PARTITION BY 
    RANGE_BUCKET(build_id, GENERATE_ARRAY(0, 10000, 1)) 
OPTIONS(require_partition_filter = TRUE);