# Literature Review 详细解析：这个毕设到底要做什么

## 1. 一句话定义

你的毕设不是做一个眼病诊断模型，而是做一个**诊断模型前面的质量控制 gatekeeper**：任何临床上传的图像，都必须先经过这个 OOD detector；只有被判定为“有效 FAF 图像”的输入才能进入后续诊断模型，其他错误模态、带水印、文字标注、拼接图、自然图像等都要被拒绝。

## 2. 为什么这个项目重要

临床 AI 的危险点不是模型完全不会预测，而是它会对无效输入也强行预测。传统分类模型通常通过 softmax 输出类别概率；当输入是错误图像时，softmax 仍然会给出某个疾病类别的高置信度，形成 silent failure。

本项目要解决的是：

```text
无效临床输入 -> 不能直接进诊断模型 -> 先被 OOD gatekeeper 拦截
```

这本质上是一个 medical AI safety / quality assurance 项目。

## 3. 输入、输出、训练数据

### 输入

任意用户上传的图像，例如：

- 合格的 FAF 图像；
- Colour fundus / IR / OCT 截图等错误模态；
- 带文字、水印、临床标注、拼接布局的 FAF 图；
- 完全非眼底图像，例如动物、物体、MRI 等。

### 输出

每张图像输出三类结果：

```text
anomaly_score: float
prediction: ACCEPT or REJECT
heatmap: optional visual explanation
```

其中 `anomaly_score` 越高，越不像正常 FAF。

### 训练数据

训练阶段只能使用正常样本，也就是 valid synthetic Heidelberg FAF。不能把所有异常类型都作为监督分类数据训练，因为异常空间无限大，真实部署时永远会出现训练集中没有见过的错误。

## 4. 核心数学思想

正常 FAF 图像虽然像素维度很高，但它们落在一个低维的正常图像流形上。检测器要学习一个 scoring function：

```text
S(x) = image x 距离正常 FAF 流形有多远
```

决策规则：

```text
S(x) <= tau -> ACCEPT
S(x) >  tau -> REJECT
```

毕业论文中需要解释 `tau` 如何影响安全性：阈值太低会误拒很多好图；阈值太高会漏掉异常图。

## 5. 异常类型拆解

### 5.1 Domain / Modality Shift

例子：把 colour fundus 当成 FAF 上传，或者不同设备的 FAF。

难点：它们仍然有视网膜结构，血管和 optic disc 都像“眼底”，所以只看大结构的模型可能误判为正常。

### 5.2 Sensory Artifacts

例子：水印、文字、医生手写标注、拼接图。

难点：大部分图像仍然是正常 FAF，异常只占局部区域。Autoencoder 可能把水印平滑掉，导致 reconstruction error 不够高。

### 5.3 Semantic Outliers

例子：狗、猫、桌子、MRI、CT。

难点：人类一眼能看出来，但 CNN 可能只抓局部纹理，产生错误高置信度。

## 6. 为什么不推荐只做 Autoencoder

Autoencoder 的思路是：只学会重建正常 FAF，异常图重建不好，所以 error 高。

问题是 deep network 有 spectral bias，容易先学低频结构，忽略高频细节。对文字、水印这类局部高频异常，AE 可能直接平滑掉，整体 L2 error 仍然低，于是异常被放行。论文里这个点是你实现 baseline 时要重点展示的：AE 可以作为 baseline，但不应作为最终主方法。

## 7. 推荐主方法：PatchCore / Feature-space OOD

PatchCore 不比较像素，而比较 CNN feature patch。流程：

1. 用 ImageNet 预训练 CNN 抽特征。
2. 从正常 FAF 训练集提取 patch-level features。
3. 建立 normal memory bank。
4. 用 coreset subsampling 压缩 memory bank。
5. 测试时，每个 patch 找最近的 normal feature。
6. 最远/最异常的 patch distance 作为图像异常分数。

它更适合你的题目，因为水印、文字、错误模态会在局部或中层特征空间中形成明显距离峰值。

## 8. 你的真正研究贡献

建议把贡献写成三部分：

1. **实现贡献**：构建一个完整、可复现的 retinal FAF OOD gatekeeper pipeline。
2. **实验贡献**：比较 reconstruction baseline 和 feature-space model，证明 feature-space 更适合质量控制。
3. **分析贡献**：围绕 Sim-to-Real gap 做 layer ablation，证明中层 CNN 特征可能比浅层/深层更稳健。

## 9. 最小可行成果 MVP

MVP 只需要做到：

- 读 manifest 数据；
- 训练 autoencoder baseline；
- 训练 PatchCore；
- 在 real FAF + OOD 测试集上输出 score；
- 计算 AUROC、AUPRC、FPR@95%TPR；
- 生成 heatmap 或至少 top anomaly examples；
- 形成可复现实验表格。

## 10. 更高分拓展

完成 MVP 后再做：

- PaDiM 或 FastFlow；
- Knowledge Distillation；
- ViT backbone 对比 CNN backbone；
- Grad-CAM 或 multi-scale heatmap；
- scanner shift / device shift 分组评估；
- threshold calibration 章节。

## 11. 论文写作主线

你最终 dissertation 可以按这个逻辑写：

1. Clinical motivation: silent failure in retinal AI.
2. Problem definition: unsupervised one-class OOD detection.
3. Data and anomaly taxonomy.
4. Methods: AE baseline vs PatchCore feature-space detector.
5. Experiment protocol: synthetic training, real-world stress testing.
6. Metrics: AUROC is insufficient; AUPRC and FPR@95%TPR are clinically meaningful.
7. Results: quantitative tables + heatmaps + failure cases.
8. Discussion: Sim-to-Real gap, layer ablation, limitations.
9. Conclusion: quality-control gatekeeper improves safety of downstream retinal AI.
