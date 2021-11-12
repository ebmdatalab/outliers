CREATE TABLE ebmdatalab.outlier_detection.summed (
    build_id int NOT NULL,
    practice STRING(6),
    chemical STRING(9),
    subpara STRING(7),
    numerator NUMERIC
) 
PARTITION BY 
    RANGE_BUCKET(build_id, GENERATE_ARRAY(0, 10000, 1)) 
OPTIONS(require_partition_filter = TRUE);