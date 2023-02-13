CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.aggregate_outliers`(p__build_id INT64)
OPTIONS (strict_mode=true)
BEGIN
    DECLARE v__i INT64 DEFAULT 0;
    DECLARE v__entity STRING;
    
    WHILE v__i < ARRAY_LENGTH(p__entities) DO
        SET v__entity = p__entities[OFFSET(v__i)];

        -- entity ranking
        EXECUTE IMMEDIATE FORMAT("""
            INSERT
                `ebmdatalab.outlier_detection.%s_ranked` (
                    build_id,
                    %s,
                    subpara,
                    subpara_items,
                    chemical,
                    chemical_items,
                    ratio,
                    mean,
                    std,
                    z_score,
                    rank_high,
                    rank_low
                ) 
            WITH entity_chems_subparas AS (
                SELECT
                    m.entity,
                    s.subpara,
                    s.chemical,
                    SUM(s.numerator) as numerator
                FROM
                    `ebmdatalab.outlier_detection.summed` AS s
                JOIN
                    `ebmdatalab.outlier_detection.%s_mapping` as m
                    ON s.practice = p.practice
                WHERE
                    m.entity IS NOT NULL
                    AND s.subpara IS NOT NULL
                    AND s.build_id = @build_id
                GROUP BY
                    m.entity,
                    subpara,
                    chemical
            ),
            entity_ratios AS (
                SELECT
                    c.entity,
                    c.subpara,
                    c.chemical,
                    COALESCE(SAFE_DIVIDE(c.numerator,s.numerator),0) AS ratio,
                    c.numerator AS chemical_items,
                    s.numerator AS subpara_items
                FROM
                    entity_chems_subparas c
                    JOIN entity_chems_subparas s 
                        ON c.entity = s.entity
                        AND c.subpara = s.subpara
                WHERE
                    c.chemical IS NOT NULL
                    AND s.chemical IS NULL
            ),
            entity_chemical_trimvalues AS (
                SELECT
                    chemical,
                    APPROX_QUANTILES(ratio, 1000) [offset(999)] AS trim_high,
                    APPROX_QUANTILES(ratio, 1000) [offset(1)] AS trim_low
                FROM
                    entity_ratios
                GROUP BY
                    chemical
            ),
            entity_chemical_stats AS (
                SELECT
                    r.chemical,
                    avg(r.ratio) AS mean,
                    stddev(r.ratio) AS std
                FROM
                    entity_ratios r
                    INNER JOIN entity_chemical_trimvalues t 
                        ON r.chemical = t.chemical
                WHERE
                    r.ratio <= t.trim_high
                    AND r.ratio >= t.trim_low
                GROUP BY
                    r.chemical
            ),
            entity_zscores AS (
                SELECT
                    r.entity,
                    r.subpara,
                    r.subpara_items,
                    r.chemical,
                    r.chemical_items,
                    r.ratio,
                    s.mean,
                    COALESCE(s.std,0) AS std,
                    COALESCE(SAFE_DIVIDE((r.ratio - s.mean), s.std), 0) AS z_score
                FROM
                    entity_ratios r
                    INNER JOIN entity_chemical_stats s 
                        ON r.chemical = s.chemical
            )
            SELECT
                @build_id,
                entity,
                subpara,
                subpara_items,
                chemical,
                chemical_items,
                ratio,
                mean,
                std,
                z_score,
                DENSE_RANK() OVER (
                    PARTITION BY entity
                    ORDER BY
                        z_score DESC
                ) AS rank_high,
                DENSE_RANK() over (
                    PARTITION BY entity
                    ORDER BY
                        z_score ASC
                ) AS rank_low
            FROM
                entity_zscores;""",
            v__entity,
            v__entity,
            v__entity)
        using v__build_id as build_id;

        --entity items   
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
            v__entity,
            v__entity,
            v__entity,
            v__entity,
            v__entity)
        using v__build_id as build_id, p__n as n, p__from_date as from_date, p__to_date as to_date;

        -- entity measure arrays
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
            v__entity,
            v__entity,
            v__entity)
        using v__build_id as build_id, p__n as n;

        SET v__i = v__i + 1;
    END WHILE;
END