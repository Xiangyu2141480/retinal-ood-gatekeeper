# Dataset Plan：数据结构、划分和隐私规则

## 1. 数据原则

训练阶段只使用正常数据：valid synthetic Heidelberg FAF。

测试阶段必须包含：

- valid real FAF，用于衡量 sim-to-real false positive；
- OOD modality shift；
- OOD sensory artifacts；
- OOD semantic outliers。

## 2. 本地目录结构

```text
data/
  manifests/
    train_synthetic_faf.csv
    val_synthetic_faf.csv
    test_real_id.csv
    test_ood.csv
  images/
    synthetic_faf/
    real_faf/
    ood_modality/
    ood_artifact/
    ood_semantic/
```

`data/images/` 必须被 `.gitignore` 忽略。

## 3. Manifest schema

每个 CSV 至少包含：

| 字段 | 含义 | 示例 |
|---|---|---|
| image_path | 图片路径，可以是相对路径或绝对路径 | data/images/a.png |
| label | 0=ID, 1=OOD | 0 |
| split | train/val/test | train |
| source | synthetic_faf/real_faf/real_ood/public_ood | synthetic_faf |
| ood_type | id/modality_shift/sensory_artifact/semantic_outlier | id |
| patient_id | 可为空；真实数据建议匿名 hash | anon_001 |
| scanner | Heidelberg/Topcon/unknown | Heidelberg |
| notes | 备注 | watermark added |

## 4. 推荐划分

### Train

- 只包含 synthetic FAF ID。
- 不包含真实 OOD。
- 不包含 valid real FAF test。

### Validation

两种策略：

1. Strict unsupervised：只用 synthetic FAF validation 设置 ID quantile threshold。
2. Clinical validation：允许一个小的 calibration set，但要在论文里明确说明。

### Test

包含：

- real valid FAF；
- wrong modality retinal images；
- watermarked/text annotated FAF；
- non-retinal natural/medical images。

## 5. Synthetic artifacts 生成策略

为了测试 sensory artifact，可以从 test/validation 的 valid FAF 上生成以下 corruption：

- text watermark；
- red/white clinical annotation；
- black border/crop；
- image stitching / composite layout；
- blur/noise/compression artifacts。

不要从 training images 生成测试 OOD，以避免数据泄漏。

## 6. 数据泄漏检查

必须检查：

- 同一 patient_id 不跨 train/test；
- 同一原图的 corrupted version 不在 train 和 test 同时出现；
- synthetic generator seed 记录；
- 公共 natural image 只用于 semantic outlier test，不用于训练。

## 7. GitHub 展示方式

可以上传：

- manifest template；
- small toy images if they are non-medical and clearly marked as toy examples；
- generated plots without identifiable information；
- docs and code。

不要上传：

- 真实临床图片；
- 任何患者信息；
- 私有数据下载链接；
- checkpoint weights trained on private clinical data。
