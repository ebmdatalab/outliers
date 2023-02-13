CREATE VIEW
  `ebmdatalab.outlier_detection.vw_mapping_ccg` AS (
  SELECT
    ccg_id AS entity,
    code AS practice
  FROM
    `ebmdatalab.hscic.practices` )