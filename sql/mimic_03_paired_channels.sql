-- Paired value/time channels, first & last draw per (subject_id, itemid).
--
-- Each analyte contributes exactly 2 (value, time) pairs: the FIRST and LAST
-- result observed in the window [t0-24h, t0+24h), selected by TIME alone. This
-- is what keeps V and S honestly separable: the rule that decides which two draws
-- survive reads only charttime, never valuenum, so which draws are kept never
-- depends on how extreme/abnormal they were. (A min/max-by-value selection
-- would leak value information into which time gets reported -- "the time of
-- the worst result" is a fact about the value, not just about time.)
--
-- One row per (subject_id, itemid) that was drawn at least once in the
-- window; an analyte never ordered for a patient produces no row here, so the
-- natural representation of "not ordered" is absence, not a sentinel.
-- Downstream consumers choose complete-case (drop) or imputation (fill with
-- an explicit sentinel) -- that choice does not belong in this extraction.
--
-- time_first/time_last are hours_from_t0 (same convention as labs_48h) and now
-- range over [-24, 24) rather than [0, 24) -- draws in the 24h before ICU
-- admission carry negative time. When only one result was drawn in the window,
-- time_first = time_last and value_first = value_last by construction -- that
-- alone recovers "drawn once" without needing a separate count feature.
CREATE OR REPLACE TABLE `labmae.labpid.paired_first_last` AS
WITH ordered AS (
  -- a stable row id breaks charttime ties deterministically -- two results
  -- landing on the identical timestamp would otherwise leave rn_first/rn_last
  -- ambiguous, and ROW_NUMBER needs a total order to be reproducible.
  SELECT *, ROW_NUMBER() OVER () AS row_id
  FROM `labmae.labpid.labs_48h`
  WHERE valuenum IS NOT NULL
),
w AS (
  SELECT subject_id, itemid, valuenum, hours_from_t0,
         ROW_NUMBER() OVER (PARTITION BY subject_id, itemid
                            ORDER BY charttime ASC,  row_id ASC)  AS rn_first,
         ROW_NUMBER() OVER (PARTITION BY subject_id, itemid
                            ORDER BY charttime DESC, row_id DESC) AS rn_last
  FROM ordered
)
SELECT
  subject_id, itemid,
  MAX(IF(rn_first = 1, valuenum,      NULL)) AS value_first,
  MAX(IF(rn_first = 1, hours_from_t0, NULL)) AS time_first,
  MAX(IF(rn_last  = 1, valuenum,      NULL)) AS value_last,
  MAX(IF(rn_last  = 1, hours_from_t0, NULL)) AS time_last
FROM w
GROUP BY 1, 2;
