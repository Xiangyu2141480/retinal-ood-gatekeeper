# Final Thesis Table Shortlist

Use these tables to keep the dissertation compact. The main text should focus on method selection,
threshold interpretation, and the Stage 1/Stage 2 boundary; the appendix can carry detailed subtype
and leakage checks.

## Main Text Tables

| filename | suggested chapter | placement | one-line message |
| --- | --- | --- | --- |
| `reports/dissertation_final/recommended_model_summary.md` | Results Overview | main text | Final recommended methods: Mahalanobis, PatchCore L3, `linear_svm`, and `hierarchical_classifier`. |
| `reports/dissertation_final/stage1_method_summary.md` | Stage 1 Results | main text | Compact Stage 1 comparison with AUROC, AUPRC, FPR@95%TPR, and recommended use. |
| `reports/dissertation_results/multi_scheme_comparison/model_selection_summary.md` | Stage 1 Results | main text | Explains why Mahalanobis is primary and PatchCore is complementary. |
| `reports/dissertation_results/robustness_analysis/threshold_policy_sweep.md` | Threshold Safety | main text | Separates research thresholds from ID-calibrated deployment-style thresholds. |
| `reports/dissertation_final/stage2_method_summary.md` | Stage 2 Results | main text | Compact Stage 2 method comparison for family and subtype attribution. |
| `reports/dissertation_results/reason_attribution_method_comparison/best_method_summary.md` | Stage 2 Results | main text | Records the final Stage 2 selected methods and headline metrics. |
| `reports/dissertation_final/limitations_summary.md` | Discussion | main text | Safe wording for synthetic fallback, proof-of-concept status, and Stage 2 limitations. |
| `reports/dissertation_final/claim_evidence_matrix.md` | Discussion / Appendix | main text or appendix | Maps each dissertation claim to committed evidence and the appropriate limitation. |

## Appendix Tables

| filename | suggested chapter | placement | one-line message |
| --- | --- | --- | --- |
| `reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.md` | Stage 1 Results | appendix | Full Stage 1 metrics for every evaluated scheme. |
| `reports/dissertation_results/multi_scheme_comparison/per_ood_type_by_scheme.md` | Stage 1 Results | appendix | Broad OOD category metrics by model. |
| `reports/dissertation_results/multi_scheme_comparison/per_ood_subtype_by_scheme.md` | Failure Analysis | appendix | Detailed subtype metrics for all Stage 1 methods. |
| `reports/dissertation_results/primary_balanced_by_subtype_10k/layer_ablation_table.md` | Methods / Results | appendix | PatchCore layer ablation table. |
| `reports/dissertation_results/robustness_analysis/bootstrap_ci.md` | Robustness | appendix | Bootstrap uncertainty for main OOD metrics. |
| `reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.md` | Discussion | appendix | Robustness findings linked to dissertation implications. |
| `reports/dissertation_results/robustness_analysis/result_provenance.md` | Reproducibility | appendix | Provenance for final result files and ID-only/OOD-only guardrails. |
| `reports/dissertation_results/reason_attribution_method_comparison/family_metrics_by_method.md` | Stage 2 Results | appendix | Full family-level Stage 2 method comparison. |
| `reports/dissertation_results/reason_attribution_method_comparison/subtype_metrics_by_method.md` | Stage 2 Results | appendix | Full subtype-level Stage 2 method comparison. |
| `reports/dissertation_final/subtype_label_mapping.md` | Stage 2 Results | appendix | Mapping for S1-S11 labels used in compact subtype confusion matrices. |
| `reports/dissertation_results/reason_attribution_method_comparison/leakage_sanity_check.md` | Limitations | appendix | Feature leakage, split disjointness, parent-hash overlap, and non-oracle hierarchy checks. |
