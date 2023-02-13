CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.aggregate_outliers`(p__build_id INT64)
OPTIONS (strict_mode=true)
BEGIN
    DECLARE v__i INT64 DEFAULT 0;
    DECLARE v__entity STRING;
    DECLARE v__from_date STRING;
    DECLARE v__to_date STRING
    DECLARE v__n INT64;
    DECLARE v__entities ARRAY<STRING>;
    
    SET (v__from_date, v__to_date, v__n, v__entities) = (
        SELECT 
            from_date,
            to_date,
            n,
            v__entities
        FROM `ebmdatalab.outlier_detection.builds`
        WHERE build_id = p__build_id
    )

    WHILE v__i < ARRAY_LENGTH(v__entities) DO
        SET v__entity = v__entities[OFFSET(v__i)];

        -- entity ranking
        CALL insert_outlier_ranking(p__build_id, v__entity)

        --entity items   
        CALL insert_outlier_items(p__build_id, v__entity, v__from_date , v__to_date , v__n)

        -- entity measure arrays
        CALL insert_outlier_measure_arrays(p__build_id, v__entity, v__n)

        SET v__i = v__i + 1;
    END WHILE;
END