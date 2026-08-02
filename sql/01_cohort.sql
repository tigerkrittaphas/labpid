-- T1 — Cohort: adult (anchor_age >= 18) ICU stays, FIRST stay per patient.
-- Anchor t0 = icu_intime. Feature window W = 24h, target horizon H = 24h.
-- One row per subject_id (first stay only) => hadm_id maps 1:1 to subject_id here.
CREATE OR REPLACE TABLE `labmae.labpid.cohort` AS
WITH ranked AS (
  SELECT
    s.subject_id,
    s.hadm_id,
    s.stay_id,
    s.intime  AS t0,
    s.outtime,
    s.los      AS icu_los_days,
    ROW_NUMBER() OVER (PARTITION BY s.subject_id ORDER BY s.intime ASC, s.stay_id ASC) AS rn
  FROM `physionet-data.mimiciv_3_1_icu.icustays` s
)
SELECT
  r.subject_id,
  r.hadm_id,
  r.stay_id,
  r.t0,
  TIMESTAMP_ADD(r.t0, INTERVAL 24 HOUR)  AS t_win_end,   -- t0 + W
  TIMESTAMP_ADD(r.t0, INTERVAL 48 HOUR)  AS t_tgt_end,   -- t0 + W + H
  r.outtime,
  r.icu_los_days,
  p.anchor_age,
  p.gender,
  p.dod,
  a.hospital_expire_flag,
  a.admission_type,
  a.race
FROM ranked r
JOIN `physionet-data.mimiciv_3_1_hosp.patients`   p USING (subject_id)
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (hadm_id)
WHERE r.rn = 1
  AND p.anchor_age >= 18;
