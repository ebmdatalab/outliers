CREATE OR REPLACE PROCEDURE `ebmdatalab.outlier_detection.build_outliers`(p__from_date STRING, p__to_date STRING, p__n INT64, p__force BOOL)
BEGIN 
    /*
    Build dataset of outlier prescribing patterns based on z-score of ratio of
    prescriptions made of BNF chemical to those of a BNF subparagraph.

    Outliers are reported at CCG,PCN, and Practice entity levels.
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
    */


    DECLARE v__build_id INT64 DEFAULT NULL;

    --check to see if this build has been run already
    SET v__build_id =(
        SELECT
            b.build_id
        FROM
            `ebmdatalab.outlier_detection.builds` b
        WHERE
            b.from_date = p__from_date
            AND b.to_date = p__to_date
            AND b.n = p__n
    );

    -- already got data for this build?
    IF v__build_id IS NOT NULL THEN 
        --do nothing if not force-rebuild mode
        IF NOT p__force THEN RETURN; END IF;

        --if forced then delete the old data before we start
        DELETE `ebmdatalab.outlier_detection.summed`
        WHERE
            build_id = v__build_id;

        DELETE `ebmdatalab.outlier_detection.practice_ranked`
        WHERE
            build_id = v__build_id;

        DELETE `ebmdatalab.outlier_detection.pcn_ranked`
        WHERE
            build_id = v__build_id;

        DELETE `ebmdatalab.outlier_detection.ccg_ranked`
        WHERE
            build_id = v__build_id;

        DELETE `ebmdatalab.outlier_detection.practice_outlier_items`
        WHERE
            build_id = v__build_id;

        DELETE `ebmdatalab.outlier_detection.pcn_outlier_items`
        WHERE
            build_id = v__build_id;

        DELETE `ebmdatalab.outlier_detection.ccg_outlier_items`
        WHERE
            build_id = v__build_id;

    --new build with requested configuration
    ELSE
        SET v__build_id = ( 
            SELECT
                1 + max(b.build_id)
            FROM
                `ebmdatalab.outlier_detection.builds` b);

        INSERT
            `ebmdatalab.outlier_detection.builds`(build_id, to_date, from_date, n)
        VALUES
            (v__build_id, p__to_date, p__from_date, p__n);

    END IF;

    -- prescribing in date range summed to practice/chemical/subpara level
    INSERT
        ebmdatalab.outlier_detection.summed (
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
            ebmdatalab.hscic.normalised_prescribing AS prescribing
            INNER JOIN ebmdatalab.hscic.practices AS practices ON practices.code = prescribing.practice
        WHERE
            practices.setting = 4
            AND practices.status_code = 'A'
            AND MONTH BETWEEN TIMESTAMP(p__from_date)
            AND TIMESTAMP(p__to_date)
            AND SUBSTR(bnf_code, 1, 2) < '18'
    ),
    agg as (
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
    select 
    v__build_id,
    *
    from agg;

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
                CROSS JOIN `ebmdatalab.hscic.bnf` b
            WHERE
                EXISTS(
                    SELECT
                        s.subpara
                    FROM
                        `ebmdatalab.outlier_detection.summed` s
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

    --practice-level
    INSERT
        ebmdatalab.outlier_detection.practice_ranked (
            build_id,
            practice,
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
    WITH practice_chems_subparas AS (
        SELECT
            s.practice,
            s.subpara,
            s.chemical,
            s.numerator
        FROM
            ebmdatalab.outlier_detection.summed AS s
        WHERE
            s.practice IS NOT NULL
            AND s.subpara IS NOT NULL
            AND s.build_id = v__build_id
    ),
    practice_ratios AS (
        SELECT
            c.practice,
            c.subpara,
            c.chemical,
            COALESCE(SAFE_DIVIDE(c.numerator,s.numerator),0) AS ratio,
            c.numerator AS chemical_items,
            s.numerator AS subpara_items
        FROM
            practice_chems_subparas c
            JOIN practice_chems_subparas s 
                ON c.practice = s.practice
                AND c.subpara = s.subpara
        WHERE
            c.chemical IS NOT NULL
            AND s.chemical IS NULL
    ),
    practice_chemical_trimvalues AS (
        SELECT
            chemical,
            APPROX_QUANTILES(ratio, 1000) [offset(999)] AS trim_high,
            APPROX_QUANTILES(ratio, 1000) [offset(1)] AS trim_low
        FROM
            practice_ratios
        GROUP BY
            chemical
    ),
    practice_chemical_stats AS (
        SELECT
            r.chemical,
            avg(r.ratio) AS mean,
            stddev(r.ratio) AS std
        FROM
            practice_ratios r
            INNER JOIN practice_chemical_trimvalues t 
                ON r.chemical = t.chemical
        WHERE
            r.ratio <= t.trim_high
            AND r.ratio >= t.trim_low
        GROUP BY
            r.chemical
    ),
    practice_zscores AS (
        SELECT
            r.practice,
            r.subpara,
            r.subpara_items,
            r.chemical,
            r.chemical_items,
            r.ratio,
            s.mean,
            COALESCE(s.std,0) AS std,
            COALESCE(SAFE_DIVIDE((r.ratio - s.mean), s.std), 0) AS z_score
        FROM
            practice_ratios r
            INNER JOIN practice_chemical_stats s 
                ON r.chemical = s.chemical
    )
    SELECT
        v__build_id,
        practice,
        subpara,
        subpara_items,
        chemical,
        chemical_items,
        ratio,
        mean,
        std,
        z_score,
        DENSE_RANK() OVER (
            PARTITION BY practice
            ORDER BY
                z_score DESC
        ) AS rank_high,
        DENSE_RANK() over (
            PARTITION BY practice
            ORDER BY
                z_score ASC
        ) AS rank_low
    FROM
        practice_zscores;

    --practice-level items
    INSERT
        ebmdatalab.outlier_detection.practice_outlier_items (
            build_id,
            practice,
            bnf_code,
            bnf_name,
            chemical,
            high_low,
            numerator
        ) 
    WITH outlier_practice_chemicals AS (
        SELECT
            r.practice,
            r.chemical,
            CASE
                WHEN r.rank_high <= p__n THEN 'h'
                ELSE 'l'
            END AS high_low
        FROM
            ebmdatalab.outlier_detection.practice_ranked AS r
        WHERE
            r.build_id = v__build_id
            AND (
                r.rank_high <= p__n
                OR r.rank_low <= p__n
            )
    ),
    aggregated as (
        SELECT
            prescribing.practice,
            prescribing.bnf_code,
            prescribing.bnf_name,
            SUBSTR(bnf_code, 1, 9) AS chemical,
            o.high_low,
            SUM(items) AS numerator
        FROM
            ebmdatalab.hscic.normalised_prescribing AS prescribing
            INNER JOIN ebmdatalab.hscic.practices AS practices ON practices.code = prescribing.practice
            INNER JOIN outlier_practice_chemicals o ON o.chemical = substr(bnf_code, 1, 9)
            AND o.practice = practices.code
        WHERE
            MONTH BETWEEN TIMESTAMP(p__from_date)
            AND TIMESTAMP(p__to_date)
        GROUP BY
            prescribing.practice,
            prescribing.bnf_code,
            prescribing.bnf_name,
            SUBSTR(bnf_code, 1, 9),
            o.high_low)
    SELECT 
        v__build_id, 
        *
    FROM aggregated
        ;

    --practice-level measure arrays
    INSERT
        `ebmdatalab.outlier_detection.practice_measure_arrays` (
            build_id,
            chemical,
            measure_array
        )
    WITH ranked_chemicals as (
        SELECT DISTINCT
            chemical
        FROM
            `ebmdatalab.outlier_detection.practice_ranked` AS r
        WHERE
            r.build_id = v__build_id
            AND (
                r.rank_high <= p__n
                OR r.rank_low <= p__n
            )
    )
    SELECT 
        v__build_id,
        r.chemical,
        array_agg(r.z_score) as meausure_array
    FROM
        `ebmdatalab.outlier_detection.practice_ranked` r
    INNER JOIN ranked_chemicals as c
        ON r.chemical = c.chemical
    WHERE
        r.build_id = v__build_id
    GROUP BY 
        v__build_id,
        r.chemical;

    --pcn-level
    INSERT
        ebmdatalab.outlier_detection.pcn_ranked (
            build_id,
            pcn,
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
        ) WITH pcn_chems_subparas AS (
            SELECT
                p.pcn_id AS pcn,
                subpara,
                chemical,
                sum(numerator) AS numerator
            FROM
                ebmdatalab.outlier_detection.summed s
                JOIN `ebmdatalab.hscic.practices` p ON s.practice = p.code
            WHERE
                s.subpara IS NOT NULL
                AND s.build_id = v__build_id
            GROUP BY
                p.pcn_id,
                subpara,
                chemical
        ),
        pcn_ratios AS (
            SELECT
                c.pcn,
                c.subpara,
                c.chemical,
                c.numerator AS chemical_items,
                s.numerator AS subpara_items,
                COALESCE(SAFE_DIVIDE(c.numerator,s.numerator),0) AS ratio,
            FROM
                (
                    SELECT
                        pcn,
                        subpara,
                        chemical,
                        sum(numerator) AS numerator
                    FROM
                        pcn_chems_subparas
                    WHERE
                        chemical IS NOT NULL
                    GROUP BY
                        pcn,
                        subpara,
                        chemical
                ) c
                INNER JOIN (
                    SELECT
                        pcn,
                        subpara,
                        sum(numerator) AS numerator
                    FROM
                        pcn_chems_subparas
                    WHERE
                        chemical IS NULL
                    GROUP BY
                        pcn,
                        subpara
                ) s ON c.pcn = s.pcn
                AND c.subpara = s.subpara
        ),
        pcn_chemical_trimvalues AS (
            SELECT
                chemical,
                APPROX_QUANTILES(ratio, 1000) [offset(999)] AS trim_high,
                APPROX_QUANTILES(ratio, 1000) [offset(1)] AS trim_low
            FROM
                pcn_ratios
            GROUP BY
                chemical
        ),
        pcn_chemical_stats AS (
            SELECT
                r.chemical,
                avg(r.ratio) AS mean,
                stddev(r.ratio) AS std
            FROM
                pcn_ratios r
                INNER JOIN pcn_chemical_trimvalues t ON r.chemical = t.chemical
            WHERE
                r.ratio <= t.trim_high
                AND r.ratio >= t.trim_low
            GROUP BY
                r.chemical
        ),
        pcn_zscores AS (
            SELECT
                r.pcn,
                r.subpara,
                r.subpara_items,
                r.chemical,
                r.chemical_items,
                r.ratio,
                s.mean,
                COALESCE(s.std,0) AS std,
                COALESCE(SAFE_DIVIDE((r.ratio - s.mean), s.std), 0) AS z_score
            FROM
                pcn_ratios r
                INNER JOIN pcn_chemical_stats s 
                    ON r.chemical = s.chemical
        )
    SELECT
        v__build_id,
        pcn,
        subpara,
        subpara_items,
        chemical,
        chemical_items,
        ratio,
        mean,
        std,
        z_score,
        DENSE_RANK() over (
            PARTITION BY pcn
            ORDER BY
                z_score DESC
        ) AS rank_high,
        DENSE_RANK() over (
            PARTITION BY pcn
            ORDER BY
                z_score ASC
        ) AS rank_low
    FROM
        pcn_zscores;

    --pcn-level items
    INSERT
        ebmdatalab.outlier_detection.pcn_outlier_items (
            build_id,
            pcn,
            bnf_code,
            bnf_name,
            chemical,
            high_low,
            numerator
        ) 
    WITH outlier_pcn_chemicals AS (
        SELECT
            pcn,
            chemical,
            CASE
                WHEN rank_high <= p__n THEN 'h'
                ELSE 'l'
            END AS high_low
        FROM
            ebmdatalab.outlier_detection.pcn_ranked
        WHERE
            build_id = v__build_id
            AND (
                rank_high <= p__n
                OR rank_low <= p__n
            )
    ),
    aggregated as (
        SELECT
            practices.pcn_id AS pcn,
            prescribing.bnf_code,
            prescribing.bnf_name,
            SUBSTR(bnf_code, 1, 9) AS chemical,
            o.high_low,
            SUM(items) AS numerator
        FROM
            ebmdatalab.hscic.normalised_prescribing AS prescribing
            INNER JOIN ebmdatalab.hscic.practices AS practices 
                ON practices.code = prescribing.practice
            INNER JOIN outlier_pcn_chemicals o 
                ON o.chemical = substr(bnf_code, 1, 9)
                AND o.pcn = practices.pcn_id
        WHERE
            MONTH BETWEEN TIMESTAMP(p__from_date)
            AND TIMESTAMP(p__to_date)
        GROUP BY
            practices.pcn_id,
            prescribing.bnf_code,
            prescribing.bnf_name,
            SUBSTR(bnf_code, 1, 9),
            o.high_low
    )
    SELECT 
        v__build_id,
        *
    FROM
        aggregated;

     --pcn-level measure arrays
    INSERT
        `ebmdatalab.outlier_detection.pcn_measure_arrays` (
            build_id,
            chemical,
            measure_array
        )
    WITH ranked_chemicals as (
        SELECT DISTINCT
            chemical
        FROM
            `ebmdatalab.outlier_detection.pcn_ranked` AS r
        WHERE
            r.build_id = v__build_id
            AND (
                r.rank_high <= p__n
                OR r.rank_low <= p__n
            )
    )
    SELECT 
        v__build_id,
        r.chemical,
        array_agg(r.z_score) as meausure_array
    FROM
        `ebmdatalab.outlier_detection.pcn_ranked` r
    INNER JOIN ranked_chemicals as c
        ON r.chemical = c.chemical
    WHERE
        r.build_id = v__build_id
    GROUP BY 
        v__build_id,
        r.chemical;

    --ccg-level
    INSERT
        ebmdatalab.outlier_detection.ccg_ranked (
            build_id,
            ccg,
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
    WITH ccg_chems_subparas AS (
        SELECT
            p.ccg_id AS ccg,
            subpara,
            chemical,
            sum(numerator) AS numerator
        FROM
            ebmdatalab.outlier_detection.summed s
            JOIN `ebmdatalab.hscic.practices` p 
                ON s.practice = p.code
        WHERE
            s.subpara IS NOT NULL
            AND s.build_id = v__build_id
        GROUP BY
            p.ccg_id,
            s.subpara,
            s.chemical
    ),
    ccg_ratios AS (
        SELECT
            c.ccg,
            c.subpara,
            c.chemical,
            c.numerator AS chemical_items,
            s.numerator AS subpara_items,
            COALESCE(SAFE_DIVIDE(c.numerator,s.numerator),0) AS ratio,
        FROM
            (
                SELECT
                    ccg,
                    subpara,
                    chemical,
                    sum(numerator) AS numerator
                FROM
                    ccg_chems_subparas
                WHERE
                    chemical IS NOT NULL
                GROUP BY
                    ccg,
                    subpara,
                    chemical
            ) c
            JOIN (
                SELECT
                    ccg,
                    subpara,
                    sum(numerator) AS numerator
                FROM
                    ccg_chems_subparas
                WHERE
                    chemical IS NULL
                GROUP BY
                    ccg,
                    subpara
            ) s 
                ON c.ccg = s.ccg
            AND c.subpara = s.subpara
    ),
    ccg_chemical_trimvalues AS (
        SELECT
            chemical,
            APPROX_QUANTILES(ratio, 1000) [offset(999)] AS trim_high,
            APPROX_QUANTILES(ratio, 1000) [offset(1)] AS trim_low
        FROM
            ccg_ratios
        GROUP BY
            chemical
    ),
    ccg_chemical_stats AS (
        SELECT
            r.chemical,
            avg(r.ratio) AS mean,
            stddev(r.ratio) AS std
        FROM
            ccg_ratios r
            INNER JOIN ccg_chemical_trimvalues t 
                ON r.chemical = t.chemical
        WHERE
            r.ratio <= t.trim_high
            AND r.ratio >= t.trim_low
        GROUP BY
            r.chemical
    ),
    ccg_zscores AS (
        SELECT
            r.ccg,
            r.subpara,
            r.subpara_items,
            r.chemical,
            r.chemical_items,
            r.ratio,
            s.mean,
            COALESCE(s.std,0) AS std,
            COALESCE(SAFE_DIVIDE((r.ratio - s.mean), s.std), 0) AS z_score
        FROM
            ccg_ratios r
            INNER JOIN ccg_chemical_stats s 
                ON r.chemical = s.chemical
    )
    SELECT
        v__build_id, 
        ccg,
        subpara,
        subpara_items,
        chemical,
        chemical_items,
        ratio,
        mean,
        std,
        z_score,
        DENSE_RANK() over (
            PARTITION BY ccg
            ORDER BY
                z_score DESC
        ) AS rank_high,
        DENSE_RANK() over (
            PARTITION BY ccg
            ORDER BY
                z_score ASC
        ) AS rank_low
    FROM
        ccg_zscores;

    --ccg-level items
    INSERT
        `ebmdatalab.outlier_detection.ccg_outlier_items`(
            build_id,
            ccg,
            bnf_code,
            bnf_name,
            chemical,
            high_low,
            numerator
        ) 
    WITH outlier_ccg_chemicals AS (
        SELECT
            ccg,
            chemical,
            CASE
                WHEN rank_high <= p__n THEN 'h'
                ELSE 'l'
            END AS high_low
        FROM
            ebmdatalab.outlier_detection.ccg_ranked
        WHERE
            build_id = v__build_id
            AND (
                rank_high <= p__n
                OR rank_low <= p__n
            )
    ),
    aggregated as (
        SELECT
            practices.ccg_id AS ccg,
            prescribing.bnf_code,
            prescribing.bnf_name,
            SUBSTR(bnf_code, 1, 9) AS chemical,
            o.high_low,
            SUM(items) AS numerator
        FROM
            ebmdatalab.hscic.normalised_prescribing AS prescribing
            INNER JOIN ebmdatalab.hscic.practices AS practices 
                ON practices.code = prescribing.practice
            INNER JOIN outlier_ccg_chemicals o 
                ON o.chemical = substr(bnf_code, 1, 9)
                AND o.ccg = practices.ccg_id
        WHERE
            MONTH BETWEEN TIMESTAMP(p__from_date)
            AND TIMESTAMP(p__to_date)
        GROUP BY
            practices.ccg_id,
            prescribing.bnf_code,
            prescribing.bnf_name,
            SUBSTR(bnf_code, 1, 9),
            o.high_low
    )
    SELECT 
        v__build_id,
        *
    FROM aggregated;

     --ccg-level measure arrays
    INSERT
        `ebmdatalab.outlier_detection.ccg_measure_arrays` (
            build_id,
            chemical,
            measure_array
        )
    WITH ranked_chemicals as (
        SELECT DISTINCT
            chemical
        FROM
            `ebmdatalab.outlier_detection.ccg_ranked` AS r
        WHERE
            r.build_id = v__build_id
            AND (
                r.rank_high <= p__n
                OR r.rank_low <= p__n
            )
    )
    SELECT 
        v__build_id,
        r.chemical,
        array_agg(r.z_score) as meausure_array
    FROM
        `ebmdatalab.outlier_detection.ccg_ranked` r
    INNER JOIN ranked_chemicals as c
        ON r.chemical = c.chemical
    WHERE
        r.build_id = v__build_id
    GROUP BY 
        v__build_id,
        r.chemical;
END;