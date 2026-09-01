# LABPID

Partial information decomposition of ICU laboratory records into a **value** channel
(what the numbers said) and a **structure** channel (when the draws happened), asking
what each one — and only their combination — knows about a patient outcome.

Every file in this repo backs a claim on the write-up:
**[The lab record is not one signal. It is at least four.](https://claude.ai/code/artifact/69ef672d-85b4-4f2a-b223-40190e7ab44e)**
Code that does not is in `archive/` (gitignored).

## The design

One patient contributes 48 numbers. For each of 12 matched analytes: the **first** and
**last** result in a ±24 h window around ICU admission (→ value channel, 24 dims) and the
**times** those two draws happened (→ structure channel, 24 dims). Nothing else — no
diagnoses, no notes, no vitals, no treatment. Four outcomes: in-hospital mortality, age
(≥65), sex, self-reported race.

The decomposition splits `I(Y; V, S)` into redundant / unique-to-value /
**unique-to-structure** / synergistic. Unique-to-structure is the component that cannot
be physiology, and it is the one the write-up is about.

| Database | n (matched panel) | Sites |
|---|---|---|
| MIMIC-IV v3.1 | 64,623 | 1 |
| eICU-CRD v2.0 | 120,266 | 198 pooled |

Both accessed via their PhysioNet BigQuery mirrors. `results/`, `out/`, `figures/`,
`archive/` and `Plan.md` are gitignored — no patient-level data or derived output is
committed.

## Layout

```
sql/     mimic_01..03            cohort -> 48 h lab extract -> paired first/last channels
         eicu_01..02             cohort -> paired channels (flat/offset-native, one step)
config/  config.yaml             run configuration (documentation; nothing reads it)
         *.resolved.json         analyte identity, resolved at runtime, never hardcoded
         *_bucket_map.json       race / ethnicity bucketing
src/     bqutil                  BigQuery + local parquet cache
         mimic_resolve_itemids   analyte resolution: MIMIC itemids
         eicu_resolve_labnames   analyte resolution: eICU labnames
         mimic_channels          channel construction, missingness arms, retargeting
         eicu_channels           the same, on eICU
         quantize                k-means discretisation, occupancy collapse
         pid_broja               BROJA-2PID + CCS atoms, solver cascade, permutation nulls
         gcmi_pid                Gaussian-copula PID -- the knob-free cross-check
         bootstrap_ci            patient bootstrap with the codebook refit per replicate
         manuscript_numbers      regenerates manuscript/NUMBERS.md from results/
         test_mimic_channels     channel-construction invariants (14 checks)
         test_eicu_channels      the same, on eICU
notebooks/  01..08               see the provenance table below
manuscript/ NUMBERS.md           canonical values, generated -- never hand-edit
            refs.bib
```

## Aggregation convention (fixed 2026-09-01)

The same eight cells have three defensible aggregations that differ by 1-3
points. Mixing them is how the write-up came to state MIMIC race synergy as both
36.9 and 24.3. One convention, everywhere:

| quantity | is | from |
|---|---|---|
| point estimate | mean over the 5 k-means seeds | `headline_matched_panel12.json` |
| interval | 95% patient bootstrap CI | `bootstrap_ci_matched_panel12.json` |
| floor | conditional-independence null, same 5 seeds | `pid_null_calibration.json` |

Bootstrap **medians are not point estimates** -- the resamples refit the codebook,
so the median sits 1-3 points off the seed mean. The floor applies to `u_str` and
`syn` only: the null forces those two to zero by construction, but leaves `red`
and `u_val` carrying the null's own composition, so subtracting theirs is
meaningless.

## Provenance: write-up section → code → result

| § | Claim | Code | Result |
|---|---|---|---|
| 00 | cohort + input table | `sql/*`, `mimic_channels` / `eicu_channels` | `out/*.parquet` |
| 01 | all four outcomes clear their permutation null (18–140×) | `notebooks/01` | `ijoint_gate_summary.json` |
| 02 | composition of `I(Y;V,S)`, k=13 matched panel | `notebooks/02`, `notebooks/03` | `headline_matched_panel12.json`, `composition_across_db_and_outcome.json` |
| 03 | unique-to-structure non-zero in MIMIC only | `src/bootstrap_ci.py` | `bootstrap_ci_matched_panel12.json`, `pid_null_calibration.json` |
| 05 | pooling removes it dose-dependently | `notebooks/05`, `notebooks/04` | `pooling_dose_response.json`, `eicu_within_hospital.json` |
| 07 | quantisation cost, synergy floor, k choice | `src/gcmi_pid.py`, `notebooks/06`, `notebooks/07` | `pid_gcmi_paired_channels.json`, `pid_null_calibration.json`, `k_resolution_headline_impact.json` |
| 02 | database explains more than outcome | `notebooks/08` | `composition_across_db_and_outcome.json` |

Every result the write-up cites now has a producer in this repo. The three that
did not -- `pid_null_calibration`, `k_resolution_headline_impact` and
`composition_across_db_and_outcome` -- were rebuilt on 2026-09-01 as notebooks 06,
07 and 08. The composition rebuild reproduces the lost original exactly; the
calibration deliberately does not, because the original ran a single seed and the
floor has to be measured on the same five seeds as the estimate it is subtracted
from.

## Running

```bash
# channel-construction invariants -- run these before trusting a changed result
PYTHONPATH=src .venv/bin/python src/test_mimic_channels.py
PYTHONPATH=src .venv/bin/python src/test_eicu_channels.py

# analyte resolution (writes config/*.resolved.json)
PYTHONPATH=src .venv/bin/python src/mimic_resolve_itemids.py
PYTHONPATH=src .venv/bin/python src/eicu_resolve_labnames.py

# bootstrap intervals for the matched panel (~12 min)
PYTHONPATH=src .venv/bin/python src/bootstrap_ci.py
```

Notebooks run in numeric order: `02_headline_broja_k13` (the k=13 protocol, which
supersedes the retired k=7 headline) → `03_matched_panel12` (which supersedes `02` for
every cross-database comparison) → `04_eicu_within_hospital`. `01_significance_gate` is
independent of the other three.

> `notebooks/01` reads `headline_k` out of `results/headline_result*.json`, which the
> archived k=7 notebooks wrote. Those JSONs are still on disk, so it runs — but its
> numbers are **k=7 on each database's own panel**, not the k=13 matched panel the
> write-up's §"Resolution" note claims for §01–03. Worth re-running at k=13.

## Reading the estimators

`pid_broja` carries two redundancy definitions. **BROJA** is primary: non-negative by
construction, exponential-cone solver with a cascade and a convergence check — never a
silent SLSQP fallback. **CCS** is the sensitivity check: pointwise, and negative atoms
are a property of the definition rather than a solver failure. Null calibration showed
CCS's unique-to-structure carries a −3.8% to −8.9% offset that varies by cell, so it is
not usable for that component in either direction.

`gcmi_pid` has no tunable knobs at all — no k, no seed, no occupancy gate — which is why
it is the check on whether a discrete result survives its quantisation family. It is a
lower bound (Gaussian part of the copula only) and it forces unique-to-structure to
exactly 0.0 by construction, so it can never corroborate that component.
