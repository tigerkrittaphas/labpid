-- Single scan of hosp.labevents: every result for a cohort patient falling in
-- [t0, t0+48h). `phase` splits the feature window from the target horizon.
--   phase = 'W'   -> [t0, t0+24h)   feature window
--   phase = 'H'   -> [t0+24, t0+48) target horizon
-- Joined on subject_id + time (labevents.hadm_id is nullable); rows carrying a
-- conflicting hadm_id are excluded.
CREATE OR REPLACE TABLE `labmae.labpid.labs_48h`
PARTITION BY RANGE_BUCKET(itemid, GENERATE_ARRAY(50800, 51600, 25))
CLUSTER BY subject_id AS
SELECT
  c.subject_id,
  c.hadm_id,
  c.stay_id,
  c.t0,
  l.specimen_id,
  l.itemid,
  l.charttime,
  l.valuenum,
  l.ref_range_lower,
  l.ref_range_upper,
  l.flag,
  l.priority,
  l.order_provider_id,
  TIMESTAMP_DIFF(l.charttime, c.t0, SECOND) / 3600.0 AS hours_from_t0,
  IF(l.charttime < c.t_win_end, 'W', 'H') AS phase
FROM `labmae.labpid.cohort` c
JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
  ON l.subject_id = c.subject_id
 AND l.charttime >= c.t0
 AND l.charttime <  c.t_tgt_end
WHERE (l.hadm_id IS NULL OR l.hadm_id = c.hadm_id);
