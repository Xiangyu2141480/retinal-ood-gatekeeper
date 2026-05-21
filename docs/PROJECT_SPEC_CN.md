# Project Specification：Retinal FAF OOD Gatekeeper

## 1. 项目目标

构建一个无监督 OOD 检测系统，用于在临床 FAF 图像进入诊断模型之前进行质量控制。

系统目标：

```text
输入图像 -> OOD gatekeeper -> valid FAF 放行 / invalid input 拒绝并提示人工复查
```

## 2. 非目标

本项目不做：

- 眼病分类；
- 病灶分割；
- 医疗诊断建议；
- 用所有异常类型训练一个普通监督分类器。

## 3. 用户故事

### 用户故事 1：临床上传正确 FAF

系统应输出：

```text
prediction = ACCEPT
anomaly_score = low
```

### 用户故事 2：上传 colour fundus

系统应输出：

```text
prediction = REJECT
reason = modality/domain shift likely
heatmap = areas with abnormal feature distances
```

### 用户故事 3：上传带水印 FAF

系统应输出：

```text
prediction = REJECT
heatmap = highlights watermark/text region
```

### 用户故事 4：上传自然图像

系统应输出：

```text
prediction = REJECT
anomaly_score = very high
```

## 4. 功能需求

### F1. 数据读取

- 从 CSV manifest 读取图片路径和标签。
- 支持 train/val/test split。
- 支持 OOD 类型字段。
- 支持灰度 FAF 自动扩展为 RGB，以适配 ImageNet backbone。

### F2. 预处理

- resize 到统一尺寸，例如 224 或 256。
- 归一化策略可配置：ImageNet normalize / minmax。
- 不允许在测试集上做训练增强。

### F3. 模型

必须实现：

- Autoencoder baseline。
- PatchCore main model。

可选实现：

- PaDiM。
- FastFlow。
- Student-Teacher KD。
- ViT feature extractor。

### F4. 训练

- PatchCore 只用 ID training data 建 memory bank。
- Autoencoder 只用 ID training data 学 reconstruction。
- 训练记录 config、seed、模型参数、时间戳。

### F5. 推理

每张图输出：

```json
{
  "image_path": "...",
  "anomaly_score": 0.0,
  "prediction": "ACCEPT|REJECT",
  "threshold": 0.0,
  "ood_type": "..."
}
```

### F6. 评估

输出：

- AUROC。
- AUPRC。
- FPR@95%TPR。
- confusion matrix。
- per-OOD-type metrics。
- score distribution plots。

### F7. 可解释性

至少输出：

- PatchCore patch-distance heatmap；
- 输入图 + heatmap overlay；
- top false positive / false negative examples。

## 5. 非功能需求

- 可复现：所有实验由 YAML config 控制。
- 可维护：模块化代码结构。
- 可测试：metric 和 dataset schema 至少有 unit tests。
- 隐私安全：不提交任何真实医学数据或私有路径。
- GitHub 可展示：README、docs、results tables、figures。

## 6. 成功标准

最低标准：

- 可跑通完整 pipeline。
- 生成一张主结果表。
- PatchCore 明显优于 AE baseline，尤其在 sensory artifacts 上。
- 能展示至少 8 张 heatmap 示例图。

高分标准：

- 有 layer ablation，证明 `layer2+layer3` 对 sim-to-real 更稳健。
- 有 per-category OOD 分析。
- 有失败案例讨论。
- 有清晰的 clinical threshold 解释。
