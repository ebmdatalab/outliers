CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.insert_outlier_ranking`(p__build_id INT64, p__entity STRING)
OPTIONS (strict_mode=true)
BEGIN 
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
        p__entity,
        p__entity,
        p__entity)
    USING p__build_id AS build_id;
END