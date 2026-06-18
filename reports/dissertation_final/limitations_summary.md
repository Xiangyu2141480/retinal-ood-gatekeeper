# Limitations Summary

| limitation | why_it_matters | safe_wording |
| --- | --- | --- |
| Synthetic ID fallback | `test_id_synthetic_fallback` is not real clinical FAF validation. | Proof-of-concept stress-test evidence, not clinical validation. |
| Curated OOD stress tests | OOD frequencies do not estimate clinical prevalence. | OOD sets are evaluation/stress-test categories. |
| Stage 2 is supervised explanation | Reason labels are used after rejection, not for Stage 1 fitting. | Optional post-rejection reason attribution only. |
| Reason labels are not diagnoses | Labels describe likely rejection causes, not disease status. | Likely explanations, not clinical diagnoses. |
| Parent-image-hash overlap | Generated artifact variants share parents across Stage 2 splits. | Stage 2 scores may be optimistic for parent-independent generalization. |
