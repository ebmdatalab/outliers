CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.insert_outlier_measure_arrays`(p__build_id INT64, p__entity STRING, p__n INT64)
OPTIONS (strict_mode=true)
BEGIN 
    EXECUTE IMMEDIATE FORMAT("""
        INSERT
            `ebmdatalab.outlier_detection.%s_measure_arrays` (
                build_id,
                chemical,
                measure_array
            )
        WITH ranked_chemicals AS (
            SELECT DISTINCT
                chemical
            FROM
                `ebmdatalab.outlier_detection.%s_ranked` AS r
            WHERE
                r.build_id = @build_id
                AND (
                    r.rank_high <= @n
                    OR r.rank_low <= @n
                )
        )
        SELECT 
            r.build_id,
            r.chemical,
            ARRAY_AGG(r.ratio) AS measure_array
        FROM
            `ebmdatalab.outlier_detection.%s_ranked` AS r
        INNER JOIN ranked_chemicals AS c
            ON r.chemical = c.chemical
        WHERE
            r.build_id = @build_id
        GROUP BY 
            r.build_id,
            r.chemical;""", 
        p__entity,
        p__entity,
        p__entity)
    USING p__build_id AS build_id, p__n AS n;
END