CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.build_outliers`(p__from_date STRING, p__to_date STRING, p__n INT64, p__force BOOL, p__entities ARRAY<STRING>)
OPTIONS (strict_mode=true)
BEGIN 
    /*
    Build dataset of outlier prescribing patterns based on z-score of ratio of
    prescriptions made of BNF chemical to those of a BNF subparagraph.

    Outliers are reported at the entity levels configured in the builds table.
    Output from this procedure will be placed in 
    ebmdatalab.outlier_detection.{entity}_ranked tables for ranked chemical:subpara
    ratio outliers and ebmdatalab.outlier_detection.{entity}_outlier_items
    tables for BNF items pertaining to those outliers.

    A summation of prescribed items at per-practice, per-chemical level is
    calculated for the requested date range in order to optimise further
    calculations. This is stored per-build in ebmdatalab.outlier_detection.summed

    Dataset builds are tracked in the builds table based on start/end date for
    reporting period and the number of outliers reported for each entity.

    https://github.com/ebmdatalab/outliers/tree/master/data/static_outlier_sql/build_outliers.sql

    Parameters
    ----------
    p__from_date : string
        start date for outlier data generation YYYY-MM-DD format
    p__to_date : string
        end date for outlier data generation YYYY-MM-DD format
    p__n : int
        number of outliers to include in high and low sets
    p__force : boolean
        force the regeneration of an outlier datasets if build
        matching above parameters exists
    p__entities : array<string>
        which entity types should be reported on in this build
        e.g. STP, PCN, ICB, sub-ICB
    */


    DECLARE v__build_id INT64 DEFAULT NULL;
    DECLARE v__entity STRING;
    DECLARE v__i INT64;
    DECLARE v__j INT64;
    DECLARE v__aggregate_table_suffixes ARRAY<string>;
    SET v__aggregate_table_suffixes = ['ranked','outlier_items','measure_arrays'];

    --check to see if this build has been run already
    SET v__build_id =(
        SELECT
            b.build_id
        FROM
            `ebmdatalab.outlier_detection.builds` AS b
        WHERE
            b.from_date = p__from_date
            AND b.to_date = p__to_date
            AND b.n = p__n
            --clunky way of expressing: are all elements of p__entities in b.entities
            -- and vice-versa
            AND False not in (SELECT
                    LOGICAL_AND(p_e IN UNNEST(b.entities))
                FROM UNNEST(p__entities) p_e
                UNION DISTINCT
                SELECT
                    LOGICAL_AND(p_e IN UNNEST(p__entities))
                FROM UNNEST(b.entities) p_e)
    );

    -- already got data for this build?
    IF v__build_id IS NOT NULL THEN 
        --do nothing if not force-rebuild mode
        IF NOT p__force THEN RETURN; END IF;

        --if forced then delete the old data before we start
        DELETE `ebmdatalab.outlier_detection.summed`
        WHERE
            build_id = v__build_id;
        
        --delete from the aggregated entity tables for this build
        SET v__i =0;
        WHILE v__i < ARRAY_LENGTH(p__entities) DO
            SET v__j =0;
            WHILE v__j < ARRAY_LENGTH(v__aggregate_table_suffixes) DO
                EXECUTE IMMEDIATE format("""
                    DELETE ebmdatalab.outlier_detection.%s_%s
                    WHERE build_id = @build_id
                """
                ,p__entities[OFFSET(v__i)]
                ,v__aggregate_table_suffixes[OFFSET(v__j)]
                ) using v__build_id as build_id;

                SET v__j = v__j + 1;
            END WHILE;

            SET v__i = v__i + 1;
        END WHILE;
        
    --new build with requested configuration
    ELSE
        SET v__build_id = ( 
            SELECT
                1 + MAX(build_id)
            FROM
                `ebmdatalab.outlier_detection.builds`);

        INSERT
            `ebmdatalab.outlier_detection.builds`(build_id, to_date, from_date, n, entities)
        VALUES
            (v__build_id, p__to_date, p__from_date, p__n, p__entities);

    END IF;

    -- prescribing in date range summed to practice/chemical/subpara level
    INSERT
        `ebmdatalab.outlier_detection.summed` (
            build_id,
            practice,
            chemical,
            subpara,
            numerator
        ) 
    WITH cte AS (
        SELECT
            practice,
            SUBSTR(bnf_code, 1, 9) AS chemical,
            SUBSTR(bnf_code, 1, 7) AS subpara,
            items
        FROM
            `ebmdatalab.hscic.normalised_prescribing` AS prescribing
        WHERE
            MONTH BETWEEN TIMESTAMP(p__from_date)
            AND TIMESTAMP(p__to_date)
            AND SUBSTR(bnf_code, 1, 2) < '18'
            AND EXISTS (
                SELECT 1
                FROM `ebmdatalab.hscic.practices` AS practices
                WHERE practices.code = prescribing.practice
                AND practices.setting = 4
                AND practices.status_code = 'A'
                AND practices.pcn_id IS NOT NULL 
                AND EXISTS ( 
                    SELECT 1 
                    FROM `ebmdatalab.hscic.ccgs` AS ccgs 
                    WHERE ccgs.stp_id IS NOT NULL 
                    AND practices.ccg_id = ccgs.code
                )
            )
    ),
    agg AS (
        SELECT
            practice,
            chemical,
            subpara,
            SUM(items) AS numerator
        FROM
            cte
        GROUP BY
            ROLLUP(
                practice,
                subpara,
                chemical
            ))
    SELECT 
        v__build_id,
        *
    FROM agg;

    --add zero-numerator rows for every chemical+practice where no existing prescribing of that chemical
    -- but there exists prescribing of other chemicals in that subpara BY any practices
    INSERT
        `ebmdatalab.outlier_detection.summed`(
            build_id,
            practice,
            chemical,
            subpara,
            numerator
        )
    SELECT
        v__build_id,
        practice,
        chemical_code,
        subpara_code,
        0
    FROM
        (
            SELECT
                DISTINCT p.practice,
                b.chemical_code,
                b.subpara_code
            FROM
                (
                    SELECT
                        DISTINCT practice
                    FROM
                        `ebmdatalab.outlier_detection.summed`
                    WHERE
                        build_id = v__build_id
                ) p
                CROSS JOIN `ebmdatalab.hscic.bnf` AS b
            WHERE
                EXISTS(
                    SELECT
                        s.subpara
                    FROM
                        `ebmdatalab.outlier_detection.summed` AS s
                    WHERE
                        b.subpara_code = s.subpara
                        AND build_id = v__build_id
                )
            EXCEPT
                DISTINCT
            SELECT
                practice,
                chemical,
                subpara
            FROM
                `ebmdatalab.outlier_detection.summed`
            WHERE
                build_id = v__build_id
        ) x;

    set v__i = 0;
    WHILE v__i < ARRAY_LENGTH(p__entities) DO
        SET v__entity = p__entities[OFFSET(v__i)];

        -- entity ranking
        EXECUTE IMMEDIATE FORMAT(
            """
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
                entity_zscores;
            """,
            v__entity,
            v__entity,
            v__entity
        ) using v__build_id as build_id;

        --entity items   
        EXECUTE IMMEDIATE FORMAT("""INSERT
        `ebmdatalab.outlier_detection.%s_outlier_items` (
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
        using v__build_id as build_id, p__n as n, p__from_date as from_date, p__to_date as to_date ;

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
END;