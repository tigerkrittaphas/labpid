-- T3/T4/T5 — channel and target extraction, long format. Pivoted locally.
-- Everything below is restricted to phase 'W' except the target block.

-- ---------------------------------------------------------------- value channel
-- T3: per admission x analyte aggregates. No timing, no counts, no mask leaves
-- this block: `n_obs` stays out of the value channel by construction downstream.
CREATE OR REPLACE TABLE `labmae.labpid.value_long` AS
WITH w AS (
  SELECT subject_id, itemid, valuenum, hours_from_t0,
         ROW_NUMBER() OVER (PARTITION BY subject_id, itemid ORDER BY charttime ASC)  AS rn_first,
         ROW_NUMBER() OVER (PARTITION BY subject_id, itemid ORDER BY charttime DESC) AS rn_last
  FROM `labmae.labpid.labs_48h`
  WHERE phase = 'W' AND valuenum IS NOT NULL
)
SELECT
  subject_id, itemid,
  COUNT(*)                                          AS n_obs,
  MIN(valuenum)                                     AS v_min,
  MAX(valuenum)                                     AS v_max,
  AVG(valuenum)                                     AS v_mean,
  MAX(IF(rn_first = 1, valuenum, NULL))             AS v_first,
  MAX(IF(rn_last  = 1, valuenum, NULL))             AS v_last,
  -- OLS slope per hour; NULL when a single observation (variance 0) -> filled 0
  SAFE_DIVIDE(COVAR_POP(valuenum, hours_from_t0), VAR_POP(hours_from_t0)) AS v_slope
FROM w
GROUP BY 1, 2;

-- ------------------------------------------------------------ structure: global
-- T4: ordering-process features. No analyte values anywhere.
-- Timing is computed at the *specimen* level: one specimen = one draw.
CREATE OR REPLACE TABLE `labmae.labpid.struct_global` AS
WITH spec AS (
  SELECT subject_id, specimen_id,
         MIN(charttime)      AS spec_time,
         MIN(hours_from_t0)  AS spec_hours,
         LOGICAL_OR(priority = 'STAT') AS is_stat,
         COUNT(*)            AS n_results
  FROM `labmae.labpid.labs_48h`
  WHERE phase = 'W'
  GROUP BY 1, 2
),
gaps AS (
  SELECT subject_id, spec_hours,
         spec_hours - LAG(spec_hours) OVER (PARTITION BY subject_id ORDER BY spec_hours) AS gap
  FROM spec
),
gap_agg AS (
  SELECT subject_id,
         AVG(gap) AS gap_mean, MIN(gap) AS gap_min, STDDEV_POP(gap) AS gap_std
  FROM gaps WHERE gap IS NOT NULL GROUP BY 1
),
analyte_agg AS (
  SELECT subject_id, COUNT(DISTINCT itemid) AS n_distinct_analytes, COUNT(*) AS n_results_total
  FROM `labmae.labpid.labs_48h` WHERE phase = 'W' GROUP BY 1
)
SELECT
  s.subject_id,
  COUNT(*)                                                  AS n_specimens,
  a.n_results_total,
  a.n_distinct_analytes,
  MIN(s.spec_hours)                                         AS hours_to_first_draw,
  MAX(s.spec_hours)                                         AS hours_to_last_draw,
  g.gap_mean, g.gap_min, g.gap_std,
  SAFE_DIVIDE(g.gap_std, g.gap_mean)                        AS gap_cv,
  -- time-of-day: the 04:00-07:00 AM-lab spike is the routine signature;
  -- draws before 07:00 or after 19:00 are the off-hours signature.
  AVG(IF(EXTRACT(HOUR FROM s.spec_time) BETWEEN 4 AND 6, 1.0, 0.0))          AS frac_amlab,
  AVG(IF(EXTRACT(HOUR FROM s.spec_time) < 7
         OR EXTRACT(HOUR FROM s.spec_time) >= 19, 1.0, 0.0))                 AS frac_offhours,
  AVG(IF(s.is_stat, 1.0, 0.0))                                              AS frac_stat
FROM spec s
JOIN analyte_agg a USING (subject_id)
LEFT JOIN gap_agg g USING (subject_id)
GROUP BY s.subject_id, a.n_results_total, a.n_distinct_analytes,
         g.gap_mean, g.gap_min, g.gap_std;

-- --------------------------------------------- structure: discretionary ordering
-- Per-analyte counts for the discretionary set. The binary mask is derived from
-- the count locally. Values never appear here.
CREATE OR REPLACE TABLE `labmae.labpid.struct_disc_long` AS
SELECT subject_id, itemid, COUNT(*) AS n_obs
FROM `labmae.labpid.labs_48h`
WHERE phase = 'W'
GROUP BY 1, 2;

-- ----------------------------------------------------------------------- targets
-- T5: FIRST draw of analyte j in [t0+W, t0+W+H); A_j against the canonical range.
CREATE OR REPLACE TABLE `labmae.labpid.targets_long` AS
WITH h AS (
  SELECT l.subject_id, l.itemid, l.valuenum, l.charttime, l.hours_from_t0, l.flag,
         r.canon_lower, r.canon_upper,
         ROW_NUMBER() OVER (PARTITION BY l.subject_id, l.itemid
                            ORDER BY l.charttime ASC, l.labevent_order ASC) AS rn
  FROM (SELECT *, ROW_NUMBER() OVER () AS labevent_order
        FROM `labmae.labpid.labs_48h` WHERE phase = 'H' AND valuenum IS NOT NULL) l
  JOIN `labmae.labpid.cohort` c USING (subject_id)
  LEFT JOIN `labmae.labpid.ref_canon` r ON r.itemid = l.itemid AND r.gender = c.gender
)
SELECT subject_id, itemid, valuenum, charttime, hours_from_t0, flag,
       canon_lower, canon_upper,
       IF(canon_lower IS NULL, NULL,
          IF(valuenum < canon_lower OR valuenum > canon_upper, 1, 0)) AS abnormal
FROM h
WHERE rn = 1;
