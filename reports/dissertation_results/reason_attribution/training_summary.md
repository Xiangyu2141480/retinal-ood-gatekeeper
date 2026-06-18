# Reason Attribution Training Summary

This trains only the optional Stage 2 explanation module for already rejected OOD inputs.
Stage 1 remains the ID-only unsupervised binary OOD gatekeeper and is not retrained here.

- train rows: 1260
- validation rows: 420
- feature mode: image_statistics
- feature dimension: 24
- subtype classifier trained: True
- validation unknown predictions at gamma=0.5: 10
- validation metrics: `reports/generated/reason_attribution/validation/reason_family_metrics.csv`
