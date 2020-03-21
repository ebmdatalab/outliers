WITH practice_numerator AS (
    SELECT
        CAST(month AS DATE) AS month,
        practice AS code,
        SUM(quantity) AS numerator
    FROM hscic.normalised_prescribing_standard
    WHERE bnf_code LIKE '0703021Q0B%' --Desogestrel (all brands)
    GROUP BY month, code
),

practice_denominator AS (
    SELECT
        CAST(month AS DATE) AS month,
        practice AS code,
        SUM(quantity) AS denominator
    FROM hscic.normalised_prescribing_standard
    WHERE bnf_code LIKE '0703021Q0%' --Desogestrel (brand and generic)
    GROUP BY month, code
),

month_practice AS (
    SELECT
        m.month,
        p.id AS code
    FROM public_draft.practice AS p
    CROSS JOIN (
        SELECT DISTINCT month FROM practice_numerator
        UNION DISTINCT
        SELECT DISTINCT month FROM practice_denominator
        ORDER BY month
    ) AS m
    WHERE p.setting = 4
      AND p.status_code = "A"
)

SELECT
    mp.month,
    mp.code,
    COALESCE(n.numerator, 0) AS numerator,
    COALESCE(d.denominator, 0) AS denominator
FROM month_practice AS mp
LEFT JOIN practice_numerator AS n
    ON mp.month = n.month AND mp.code = n.code
LEFT JOIN practice_denominator AS d
    ON mp.month = d.month AND mp.code = d.code
