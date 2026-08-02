-- Rule 4 — harmonize reference ranges before any comparison.
-- Distinct ranges per analyte are (a) sex-specific and (b) era/site-specific
-- (e.g. Hgb male 13.7-17.5 vs 14.0-18.0; WBC upper 10 vs 11). Using the range
-- recorded on each row would let a site/era effect masquerade as abnormality.
-- Canonical range = the modal (lower, upper) pair per (itemid, gender) in-cohort.
CREATE OR REPLACE TABLE `labmae.labpid.ref_canon` AS
WITH counted AS (
  SELECT l.itemid, c.gender, l.ref_range_lower, l.ref_range_upper, COUNT(*) AS n
  FROM `labmae.labpid.labs_48h` l
  JOIN `labmae.labpid.cohort` c USING (subject_id)
  WHERE l.valuenum IS NOT NULL
    AND l.ref_range_lower IS NOT NULL AND l.ref_range_upper IS NOT NULL
  GROUP BY 1, 2, 3, 4
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY itemid, gender ORDER BY n DESC) AS rn,
         SUM(n) OVER (PARTITION BY itemid, gender) AS n_total
  FROM counted
)
SELECT itemid, gender,
       ref_range_lower AS canon_lower,
       ref_range_upper AS canon_upper,
       n AS n_modal, n_total,
       n / n_total AS modal_share
FROM ranked
WHERE rn = 1;
