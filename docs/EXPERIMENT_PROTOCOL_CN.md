# Experiment Protocol：实验设计和评价指标

## 1. 研究问题

### RQ1

Feature-space OOD detector 是否比 autoencoder reconstruction baseline 更适合 retinal FAF quality control？

### RQ2

在 synthetic FAF 训练、real FAF 测试时，哪一层 CNN features 最能平衡 sim-to-real gap 和 artifact sensitivity？

### RQ3

AUROC 是否掩盖了 clinical false positive 问题？AUPRC 和 FPR@95%TPR 是否提供更真实的评估？

## 2. 模型矩阵

| Model | Backbone | Layers | 目的 |
|---|---|---|---|
| Conv Autoencoder | custom CNN | pixel space | baseline，展示 reconstruction limitation |
| PatchCore-L1 | ResNet50 | layer1 | 测试浅层是否过拟合 synthetic noise |
| PatchCore-L2 | ResNet50 | layer2 | 中层候选 |
| PatchCore-L3 | ResNet50 | layer3 | 中层候选 |
| PatchCore-L2+L3 | ResNet50 | layer2+layer3 | 主模型 |
| PatchCore-L4 | ResNet50 | layer4 | 测试深层是否忽略局部 artifact |
| PaDiM | ResNet50 | layer2+layer3 | optional comparison |
| FastFlow | ResNet/ViT | multi-scale | optional advanced model |

## 3. 数据集矩阵

| Test subset | label | 目的 |
|---|---:|---|
| valid real FAF | 0 | 衡量正常临床图被误拒的比例 |
| colour fundus / IR | 1 | modality shift |
| watermarked / annotated FAF | 1 | sensory artifact |
| non-retinal natural/medical | 1 | semantic outlier |

## 4. 指标

### AUROC

总体区分能力，但在极度不平衡临床场景中可能过于乐观。

### AUPRC

更关注 anomaly prediction 的 precision/recall，不被大量 true negatives 稀释。

### FPR@95%TPR

临床解释：当系统调到能抓住 95% 异常时，会误拒多少正常图？

### Per-category metrics

每类 OOD 单独报告：

- modality_shift AUROC/AUPRC；
- sensory_artifact AUROC/AUPRC；
- semantic_outlier AUROC/AUPRC；
- valid real FAF score distribution。

## 5. 阈值策略

报告两类阈值：

1. Evaluation threshold：为计算 FPR@95%TPR，在 OOD score 上找 95% recall 对应的 threshold。
2. Deployment threshold：用 validation ID score 的 95% 或 99% quantile 作为 conservative operating threshold。

论文里要清楚区分：前者是评价指标，后者是部署策略。

## 6. 结果表模板

| Model | Layers | AUROC ↑ | AUPRC ↑ | FPR@95%TPR ↓ | Notes |
|---|---|---:|---:|---:|---|
| AE | pixel | | | | expected weak on watermarks |
| PatchCore | L1 | | | | shallow synthetic noise risk |
| PatchCore | L2 | | | | |
| PatchCore | L3 | | | | |
| PatchCore | L2+L3 | | | | expected main model |
| PatchCore | L4 | | | | deep layer artifact blind spot |

## 7. 必备图

1. Pipeline diagram。
2. Score distribution: ID vs OOD。
3. ROC and PR curves。
4. Layer ablation bar chart。
5. Heatmap grid: TP / FP / FN / TN examples。
6. Failure-case analysis figure。

## 8. 论文中如何解释预期结果

如果 PatchCore-L2+L3 最好：说明中层特征既不过度关注 synthetic pixel fingerprints，也不过度抽象到忽略局部 artifact。

如果 L1 在 synthetic validation 很好但 real FAF false positive 很高：这是 sim-to-real overfitting 的证据。

如果 L4 semantic outlier 好但 watermark 差：说明深层 semantic feature 对局部质量控制不够敏感。
