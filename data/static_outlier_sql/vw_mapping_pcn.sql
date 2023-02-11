CREATE VIEW
  `ebmdatalab.outlier_detection.vw_mapping_pcn` AS (
  SELECT
    pcn_id AS entity,
    code AS practice
  FROM
    `ebmdatalab.hscic.practices` )