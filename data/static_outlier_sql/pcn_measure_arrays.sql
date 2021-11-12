CREATE TABLE `ebmdatalab.outlier_detection.pcn_measure_arrays`(
    build_id int64 not null,
    chemical string(9) not null,
    measure_array array<float64>
)
PARTITION BY 
    RANGE_BUCKET(build_id, GENERATE_ARRAY(0, 10000, 1)) 
OPTIONS(require_partition_filter = TRUE);