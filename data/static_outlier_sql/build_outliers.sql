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
        --else delete the old data before we start
        CALL delete_outliers(v__build_id)
        
        
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

    CALL aggregate_outliers(v__build_id)
END;