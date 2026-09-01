# LABPID

Partial information decomposition of ICU laboratory records into a **value** channel
(what the numbers said) and a **structure** channel (when the draws happened), asking
what each one — and only their combination — knows about a patient outcome.


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