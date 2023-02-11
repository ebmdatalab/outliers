CREATE TABLE ebmdatalab.outlier_detection.builds (
    build_id NUMERIC NOT NULL,
    from_date STRING NOT NULL,
    to_date STRING NOT NULL,
    n NUMERIC NOT NULL,
    entities ARRAY<STRING>
)