CREATE TABLE ebmdatalab.outlier_detection.stp_outlier_items (
    build_id int NOT NULL,
    stp STRING(9),
    bnf_code STRING(15),
    bnf_name STRING(100),
    chemical STRING(9),
    high_low STRING(1),
    numerator NUMERIC
) 
PARTITION BY
    RANGE_BUCKET(build_id, GENERATE_ARRAY(0, 10000, 1))
OPTIONS(require_partition_filter = TRUE);