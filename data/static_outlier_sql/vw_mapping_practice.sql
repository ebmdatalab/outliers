CREATE VIEW
  `ebmdatalab.outlier_detection.vw_mapping_practice` AS (
  SELECT
    code AS entity,
    code AS practice
  FROM
    `ebmdatalab.hscic.practices` )