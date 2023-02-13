CREATE VIEW
  `ebmdatalab.outlier_detection.vw_mapping_stp` AS (
  SELECT
    c.stp_id AS entity,
    p.code AS practice
  FROM
    `ebmdatalab.hscic.practices` AS p
  JOIN
    `ebmdatalab.hscic.ccgs` AS c
  ON
    p.ccg_id = c.code)