-- Target horizon, reintroduced for the per-analyte abnormality track under
-- the redesigned cohort (symmetric, ICU-anchored feature window). The
-- feature window is [t0-24h, t0+24h); this horizon starts exactly where that
-- window's forward edge ends, so no feature-window lab can leak into a
-- target -- mirrors the original design's W/H split, just re-anchored after
-- the new wider feature window instead of the old forward-only one.
CREATE OR REPLACE TABLE `labmae.labpid.labs_horizon` AS
SELECT
  c.subject_id,
  c.hadm_id,
  l.specimen_id,
  l.itemid,
  l.charttime,
  l.valuenum,
  l.flag,
  TIMESTAMP_DIFF(l.charttime, c.t0, SECOND) / 3600.0 AS hours_from_t0
FROM `labmae.labpid.cohort` c
JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
  ON l.subject_id = c.subject_id
 AND l.charttime >= TIMESTAMP_ADD(c.t0, INTERVAL 24 HOUR)
 AND l.charttime <  TIMESTAMP_ADD(c.t0, INTERVAL 48 HOUR)
WHERE (l.hadm_id IS NULL OR l.hadm_id = c.hadm_id);

-- FIRST draw of analyte j in the horizon; A_j against the canonical
-- (itemid, gender) reference range from sql/03_ref_canon.sql.
CREATE OR REPLACE TABLE `labmae.labpid.targets_long` AS
WITH h AS (
  SELECT l.subject_id, l.itemid, l.valuenum, l.charttime, l.hours_from_t0, l.flag,
         r.canon_lower, r.canon_upper,
         ROW_NUMBER() OVER (PARTITION BY l.subject_id, l.itemid
                            ORDER BY l.charttime ASC) AS rn
  FROM `labmae.labpid.labs_horizon` l
  JOIN `labmae.labpid.cohort` c USING (subject_id)
  LEFT JOIN `labmae.labpid.ref_canon` r ON r.itemid = l.itemid AND r.gender = c.gender
  WHERE l.valuenum IS NOT NULL
)
SELECT subject_id, itemid, valuenum, charttime, hours_from_t0, flag,
       canon_lower, canon_upper,
       IF(canon_lower IS NULL, NULL,
          IF(valuenum < canon_lower OR valuenum > canon_upper, 1, 0)) AS abnormal
FROM h
WHERE rn = 1;
