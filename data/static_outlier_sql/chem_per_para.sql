SELECT
  practice,
  pcn,
  ccg,
  chemical,
  SUBSTR(chemical, 1, 7) AS subpara,
  numerator
FROM (
  SELECT
    practice,
    pcn_id AS pcn,
    ccg_id AS ccg,
    SUBSTR(bnf_code, 1, 9) AS chemical,
    SUM(items) AS numerator
  FROM
    ebmdatalab.hscic.normalised_prescribing_standard AS prescribing
  INNER JOIN
    ebmdatalab.hscic.practices AS practices
  ON
    practices.code = prescribing.practice
  WHERE
    practices.setting = 4
    AND practices.status_code ='A'
    AND month BETWEEN TIMESTAMP('2019-07-01')
    AND TIMESTAMP('2019-12-01')
    AND SUBSTR(bnf_code, 1, 2) <'18'
  GROUP BY
    chemical,
    ccg_id,
    pcn_id,
    practice
)