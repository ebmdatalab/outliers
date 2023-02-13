CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.insert_outlier_items`(p__build_id INT64, p__entity STRING, p__from_date STRING, p__to_date STRING, p__n INT64)
OPTIONS (strict_mode=true)
BEGIN 
    EXECUTE IMMEDIATE FORMAT("""
        INSERT `ebmdatalab.outlier_detection.%s_outlier_items` (
            build_id,
            %s,
            bnf_code,
            bnf_name,
            chemical,
            high_low,
            numerator
        ) 
        WITH outlier_entity_chemicals AS (
            SELECT
                r.%s as entity,
                r.chemical,
                CASE
                    WHEN r.rank_high <= @n THEN 'h'
                    ELSE 'l'
                END AS high_low
            FROM
                `ebmdatalab.outlier_detection.%s_ranked` AS r
            WHERE
                r.build_id = @build_id
                AND (
                    r.rank_high <= @n
                    OR r.rank_low <= @n
                )
        ),
        aggregated AS (
            SELECT
                m.entity,
                prescribing.bnf_code,
                prescribing.bnf_name,
                SUBSTR(bnf_code, 1, 9) AS chemical,
                o.high_low,
                SUM(items) AS numerator
            FROM
                `ebmdatalab.hscic.normalised_prescribing`AS prescribing
                INNER JOIN `ebmdatalab.hscic.practices` AS practices 
                    ON practices.code = prescribing.practice
                INNER JOIN `ebmdatalab.outlier_detection.vw_mapping_%s` m
                    ON m.practice = prescribing.practice
                INNER JOIN `outlier_entity_chemicals` AS o 
                    ON o.chemical = substr(bnf_code, 1, 9)
                    AND o.entity = m.entity
            WHERE
                MONTH BETWEEN TIMESTAMP(@from_date)
                AND TIMESTAMP(@to_date)
                AND practices.setting = 4
                    AND practices.status_code = 'A'
                    AND practices.pcn_id IS NOT NULL 
                    AND EXISTS ( 
                        SELECT 1 
                        FROM `ebmdatalab.hscic.ccgs` AS ccgs 
                        WHERE ccgs.stp_id IS NOT NULL 
                        AND practices.ccg_id = ccgs.code
                    )
            GROUP BY
                m.entity,
                prescribing.bnf_code,
                prescribing.bnf_name,
                SUBSTR(bnf_code, 1, 9),
                o.high_low)
        SELECT 
            @build_id, 
            *
        FROM aggregated""",
        p__entity,
        p__entity,
        p__entity,
        p__entity,
        p__entity)
    USING p__build_id AS build_id, p__n AS n, p__from_date AS from_date, p__to_date AS to_date;