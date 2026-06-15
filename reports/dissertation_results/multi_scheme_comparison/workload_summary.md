# Workload Summary

- number of schemes implemented: 8
- number of PatchCore layer variants completed: 4
- number of evaluation subsets: 3
- number of OOD types: 3
- number of OOD subtypes: 11
- number of figures generated: 12
- number of tables generated: 15
- total dataset images evaluated across completed scheme/evaluation runs: 36000
- training remained ID-only: yes, only `label=0` / `ood_type=id` train manifests were used.
- OOD was evaluation-only: yes, OOD categories were used for metrics grouping and stress testing only.
- model decision: binary ACCEPT valid FAF / REJECT invalid or OOD input.
- synthetic fallback warning: `test_id_synthetic_fallback.csv` is not real clinical FAF validation.
