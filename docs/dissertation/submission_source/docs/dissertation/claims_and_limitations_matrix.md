# Claims and Limitations Matrix

| Statement | Supported? | Evidence | Limitation Wording |
|---|---|---|---|
| The ID collection is synthetic FAF-like data from UCL SynthEye. | Yes | `dataset_source_mapping.csv`; UCL RDR DOI record | This is not real clinical FAF validation. |
| Sensory artefacts add no new external image source. | Yes | Parent-image hashes and artifact-generation code | They remain synthetic transformations of held-out synthetic parents. |
| Imported OOD rows are from public/prepared source buckets. | Yes, at source-bucket level | `test_modality.csv`; `test_semantic.csv`; builder copy logic | Per-image upstream dataset mapping is unavailable. |
| Every imported OOD image can be mapped to APTOS/RFMiD/OLIVES/CIFAR/Open Images. | No | No original filename, original path, or source URL retained | Do not claim exact upstream datasets without recovering original import directories. |
| Retinograd-AI was used as benchmark data. | No | Provenance audit | Retinograd-AI is related work / FAF gradability reference only. |
| Stage 1 training used OOD labels or OOD images. | No | Manifest-defined Stage 1 fitting protocol | OOD labels are evaluation-only for Stage 1. |
