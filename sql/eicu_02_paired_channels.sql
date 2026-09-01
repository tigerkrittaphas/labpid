-- E2 -- eICU-CRD paired value/time channels: first & last draw per
-- (patientunitstayid, labname) in the window labresultoffset IN [-1440, 1440]
-- minutes (+-24h around unit admission = offset 0, eICU's built-in t0).
-- Combines what the MIMIC pipeline splits into mimic_02_labs_48h.sql (windowing)
-- + mimic_03_paired_channels.sql (pairing) into one script, since eicu_crd.lab is
-- already flat and offset-native -- no d_labitems-style join needed for
-- windowing.
--
-- Same first/last-BY-TIME-not-value logic as MIMIC's mimic_03_paired_channels.sql:
-- the ORDER BY never reads labresult, only labresultoffset (+ a stable
-- tiebreak on labid), so which draws survive never depends on how extreme
-- the result was. time_first/time_last are hours from unit admission
-- (labresultoffset / 60.0), matching MIMIC's hours_from_t0 unit convention.
CREATE OR REPLACE TABLE `labmae.labpid_eicu.paired_first_last` AS
WITH win AS (
  SELECT c.patientunitstayid, l.labid, l.labname, l.labresult,
         l.labresultoffset / 60.0 AS hours_from_t0
  FROM `labmae.labpid_eicu.cohort` c
  JOIN `physionet-data.eicu_crd.lab` l USING (patientunitstayid)
  WHERE l.labresultoffset BETWEEN -1440 AND 1440
    AND l.labresult IS NOT NULL
),
w AS (
  SELECT patientunitstayid, labname, labresult, hours_from_t0,
         ROW_NUMBER() OVER (PARTITION BY patientunitstayid, labname
                            ORDER BY hours_from_t0 ASC,  labid ASC)  AS rn_first,
         ROW_NUMBER() OVER (PARTITION BY patientunitstayid, labname
                            ORDER BY hours_from_t0 DESC, labid DESC) AS rn_last
  FROM win
)
SELECT
  patientunitstayid, labname,
  MAX(IF(rn_first = 1, labresult,      NULL)) AS value_first,
  MAX(IF(rn_first = 1, hours_from_t0,  NULL)) AS time_first,
  MAX(IF(rn_last  = 1, labresult,      NULL)) AS value_last,
  MAX(IF(rn_last  = 1, hours_from_t0,  NULL)) AS time_last
FROM w
GROUP BY 1, 2;
