-- E1 -- eICU-CRD cohort: adult (age >= 18), FIRST ICU stay per patient
-- (uniquepid, spans hospitals). Lives in labmae.labpid_eicu -- a dataset
-- fully separate from labmae.labpid (MIMIC) so this script can never
-- CREATE OR REPLACE over the MIMIC-critical `cohort` table.
--
-- eICU already anchors every *offset column at unit (ICU) admission = 0
-- minutes, so there is no icu_intime-style t0 to compute the way MIMIC
-- needed one. `age` is a STRING in eICU/BigQuery -- ages over 89 are
-- top-coded as the literal text "> 89" (HIPAA de-identification), recoded
-- here to 90 rather than being SAFE_CAST into NULL and silently dropped.
-- hospitaldischargestatus has a small blank fraction (~1,751 rows) -- those
-- are missing outcome, not a third category, and are excluded.
CREATE OR REPLACE TABLE `labmae.labpid_eicu.cohort` AS
WITH parsed AS (
  SELECT *,
    CASE WHEN age = '> 89' THEN 90.0
         WHEN age IS NULL OR age = '' THEN NULL
         ELSE SAFE_CAST(age AS FLOAT64) END AS age_num
  FROM `physionet-data.eicu_crd.patient`
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY uniquepid
    ORDER BY hospitaldischargeyear ASC, patienthealthsystemstayid ASC, unitvisitnumber ASC
  ) AS rn
  FROM parsed
  WHERE age_num >= 18
    AND hospitaldischargestatus IN ('Alive', 'Expired')
)
SELECT
  patientunitstayid,
  patienthealthsystemstayid,
  uniquepid,
  hospitalid,
  age_num AS age,
  gender,
  ethnicity,
  IF(hospitaldischargestatus = 'Expired', 1, 0) AS hospital_expire_flag
FROM ranked
WHERE rn = 1;
