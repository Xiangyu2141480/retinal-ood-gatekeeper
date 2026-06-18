# Reason Attribution Evaluation Summary

Stage 2 was evaluated on OOD-only rows as a post-hoc explanation module. Stage 1 remains the ID-only unsupervised binary OOD gatekeeper.

- unknown threshold gamma: 0.50
- reason family accuracy: 0.8810
- reason family macro-F1: 0.8947
- hardest reason family by F1: modality_shift (F1=0.8136)
- unknown_ood rate at gamma=0.50: 0.0357
- subtype accuracy: 0.7429
- subtype macro-F1: 0.6724
- hardest subtype by F1: jpeg_compression (F1=0.2439)

Reason labels are likely explanations for rejection, not clinical diagnoses.
