-- Single scan of hosp.labevents: every result for a cohort patient falling in
-- the ICU-admission-centered window [t0-24h, t0+24h). No phase split -- the
-- target (in-hospital mortality) is read directly from admissions.hospital_
-- expire_flag, not derived from labs in a separate horizon, so there is no
-- feature/target time split to encode here anymore.
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
  TIMESTAMP_DIFF(l.charttime, c.t0, SECOND) / 3600.0 AS hours_from_t0
FROM `labmae.labpid.cohort` c
JOIN `physionet-data.mimiciv_3_1_hosp.labevents` l
  ON l.subject_id = c.subject_id
 AND l.charttime >= c.t_win_start
 AND l.charttime <  c.t_win_end
WHERE (l.hadm_id IS NULL OR l.hadm_id = c.hadm_id);
