CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.delete_outliers`(p__build_id INT64)
OPTIONS (strict_mode=true)
BEGIN 
    DECLARE v__aggregate_table_suffixes ARRAY<string> DEFAULT ['ranked','outlier_items','measure_arrays'];
    DECLARE v__i INT64 DEFAULT 0;
    DECLARE v__j INT64 DEFAULT 0;
    DECLARE v__entities ARRAY<string>;

    SELECT v__entities = (
        SELECT entities 
        FROM `ebmdatalab.outlier_detection.builds` 
        WHERE build_id = p__build_id
    )

    DELETE `ebmdatalab.outlier_detection.summed`
    WHERE build_id = v__build_id;

    --loop through aggregated entity tables for this build
    WHILE v__i < ARRAY_LENGTH(v__entities) DO
        SET v__j =0;
        WHILE v__j < ARRAY_LENGTH(v__aggregate_table_suffixes) DO
            EXECUTE IMMEDIATE format("""
                DELETE ebmdatalab.outlier_detection.%s_%s
                WHERE build_id = @build_id
            """
            ,v__entities[OFFSET(v__i)]
            ,v__aggregate_table_suffixes[OFFSET(v__j)]
            ) using v__build_id as build_id;

            SET v__j = v__j + 1;
        END WHILE;

        SET v__i = v__i + 1;
    END WHILE;
END