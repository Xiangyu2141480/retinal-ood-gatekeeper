# Final Evidence Index

| Claim | Evidence | Status |
|---|---|---|
| Stage 1 is ID-only and unsupervised. | `main.tex`; manifest-defined ID training rows; Stage 1 method descriptions | Unchanged by provenance audit |
| Synthetic FAF-like ID data come from UCL SynthEye. | `docs/dissertation/dataset_source_provenance.md`; `reports/audit/dataset_source_mapping.csv`; `sample.bib` entry `uclSyntheyeDataset2025` | Dataset-level confirmed |
| Sensory artefact OOD images are generated from held-out synthetic FAF-like parents. | `test_ood_full.csv` parent hashes and synthetic transforms; `src/retinal_ood/data/dissertation_dataset.py:659-708` | Generation-level confirmed |
| Imported OOD images are retained as public prepared source buckets. | `reports/audit/imported_ood_image_provenance.csv`; dataset builder copy logic | Source-bucket confirmed |
| Per-image upstream source mapping is not recoverable from the current committed package. | Missing `original_filename`, `original_relative_path`, and `source_url` fields; builder renaming logic | Documented limitation |
| Retinograd-AI is not benchmark data. | `docs/dissertation/dataset_source_provenance.md`; manuscript wording | Confirmed as related work/reference only |
