# Dataset Card: Dissertation FAF OOD v1

## Task

Unsupervised OOD gatekeeper for retinal FAF image quality control. Training uses
valid FAF only (`label=0`, `ood_type=id`). The system must reject invalid/OOD
inputs upstream of any diagnostic model.

## Important Warning

`test_id_synthetic_fallback.csv` is synthetic fallback data only. It is not real
clinical FAF validation and should be described as a proof-of-concept limitation.

## Counts

```json
{
  "train_id": {
    "rows": 700,
    "label": {
      "0": 700
    },
    "split": {
      "train": 700
    },
    "ood_type": {
      "id": 700
    },
    "ood_subtype": {
      "id": 700
    }
  },
  "val_id": {
    "rows": 150,
    "label": {
      "0": 150
    },
    "split": {
      "val": 150
    },
    "ood_type": {
      "id": 150
    },
    "ood_subtype": {
      "id": 150
    }
  },
  "test_id_synthetic_fallback": {
    "rows": 150,
    "label": {
      "0": 150
    },
    "split": {
      "test": 150
    },
    "ood_type": {
      "id": 150
    },
    "ood_subtype": {
      "id": 150
    }
  },
  "test_artifact": {
    "rows": 1200,
    "label": {
      "1": 1200
    },
    "split": {
      "test": 1200
    },
    "ood_type": {
      "sensory_artifact": 1200
    },
    "ood_subtype": {
      "text_watermark": 150,
      "rectangle_annotation": 150,
      "arrow_annotation": 150,
      "composite_layout": 150,
      "blur_artifact": 150,
      "border_crop": 150,
      "gaussian_noise": 150,
      "jpeg_compression": 150
    }
  },
  "test_ood_artifact": {
    "rows": 1200,
    "label": {
      "1": 1200
    },
    "split": {
      "test": 1200
    },
    "ood_type": {
      "sensory_artifact": 1200
    },
    "ood_subtype": {
      "text_watermark": 150,
      "rectangle_annotation": 150,
      "arrow_annotation": 150,
      "composite_layout": 150,
      "blur_artifact": 150,
      "border_crop": 150,
      "gaussian_noise": 150,
      "jpeg_compression": 150
    }
  },
  "test_modality": {
    "rows": 400,
    "label": {
      "1": 400
    },
    "split": {
      "test": 400
    },
    "ood_type": {
      "modality_shift": 400
    },
    "ood_subtype": {
      "colour_fundus": 200,
      "oct_screenshot": 200
    }
  },
  "test_ood_modality": {
    "rows": 400,
    "label": {
      "1": 400
    },
    "split": {
      "test": 400
    },
    "ood_type": {
      "modality_shift": 400
    },
    "ood_subtype": {
      "colour_fundus": 200,
      "oct_screenshot": 200
    }
  },
  "test_semantic": {
    "rows": 500,
    "label": {
      "1": 500
    },
    "split": {
      "test": 500
    },
    "ood_type": {
      "semantic_outlier": 500
    },
    "ood_subtype": {
      "cifar10_natural": 500
    }
  },
  "test_ood_semantic": {
    "rows": 500,
    "label": {
      "1": 500
    },
    "split": {
      "test": 500
    },
    "ood_type": {
      "semantic_outlier": 500
    },
    "ood_subtype": {
      "cifar10_natural": 500
    }
  },
  "test_ood": {
    "rows": 2100,
    "label": {
      "1": 2100
    },
    "split": {
      "test": 2100
    },
    "ood_type": {
      "sensory_artifact": 1200,
      "modality_shift": 400,
      "semantic_outlier": 500
    },
    "ood_subtype": {
      "text_watermark": 150,
      "rectangle_annotation": 150,
      "arrow_annotation": 150,
      "composite_layout": 150,
      "blur_artifact": 150,
      "border_crop": 150,
      "gaussian_noise": 150,
      "jpeg_compression": 150,
      "colour_fundus": 200,
      "oct_screenshot": 200,
      "cifar10_natural": 500
    }
  },
  "test_ood_full": {
    "rows": 2100,
    "label": {
      "1": 2100
    },
    "split": {
      "test": 2100
    },
    "ood_type": {
      "sensory_artifact": 1200,
      "modality_shift": 400,
      "semantic_outlier": 500
    },
    "ood_subtype": {
      "text_watermark": 150,
      "rectangle_annotation": 150,
      "arrow_annotation": 150,
      "composite_layout": 150,
      "blur_artifact": 150,
      "border_crop": 150,
      "gaussian_noise": 150,
      "jpeg_compression": 150,
      "colour_fundus": 200,
      "oct_screenshot": 200,
      "cifar10_natural": 500
    }
  },
  "test_ood_balanced_by_type": {
    "rows": 1200,
    "label": {
      "1": 1200
    },
    "split": {
      "test": 1200
    },
    "ood_type": {
      "modality_shift": 400,
      "semantic_outlier": 400,
      "sensory_artifact": 400
    },
    "ood_subtype": {
      "colour_fundus": 200,
      "oct_screenshot": 200,
      "cifar10_natural": 400,
      "arrow_annotation": 51,
      "blur_artifact": 49,
      "border_crop": 58,
      "composite_layout": 46,
      "gaussian_noise": 51,
      "jpeg_compression": 50,
      "rectangle_annotation": 52,
      "text_watermark": 43
    }
  },
  "test_ood_balanced_by_subtype": {
    "rows": 1650,
    "label": {
      "1": 1650
    },
    "split": {
      "test": 1650
    },
    "ood_type": {
      "sensory_artifact": 1200,
      "semantic_outlier": 150,
      "modality_shift": 300
    },
    "ood_subtype": {
      "arrow_annotation": 150,
      "blur_artifact": 150,
      "border_crop": 150,
      "cifar10_natural": 150,
      "colour_fundus": 150,
      "composite_layout": 150,
      "gaussian_noise": 150,
      "jpeg_compression": 150,
      "oct_screenshot": 150,
      "rectangle_annotation": 150,
      "text_watermark": 150
    }
  },
  "test_ood_smoke": {
    "rows": 22,
    "label": {
      "1": 22
    },
    "split": {
      "test": 22
    },
    "ood_type": {
      "sensory_artifact": 16,
      "semantic_outlier": 2,
      "modality_shift": 4
    },
    "ood_subtype": {
      "arrow_annotation": 2,
      "blur_artifact": 2,
      "border_crop": 2,
      "cifar10_natural": 2,
      "colour_fundus": 2,
      "composite_layout": 2,
      "gaussian_noise": 2,
      "jpeg_compression": 2,
      "oct_screenshot": 2,
      "rectangle_annotation": 2,
      "text_watermark": 2
    }
  }
}
```

## Privacy

This repository includes the actual final dissertation dataset images under `data/images/dissertation_v1/` through Git LFS.
Individual PNG/JPG/TIFF files are tracked by Git LFS and must not be stored as normal Git blobs.
Patient identifiers and clinical disease labels are not included.
