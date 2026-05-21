# 结果解读模板：Retinal FAF OOD Gatekeeper

本文档用于把实验输出转换成 dissertation 中的 Results、Discussion、Limitations 和 Future Work 段落。它只讨论上游质量控制 gatekeeper，不应把系统描述成疾病分类器。

## 1. 写作原则

### 1.1 研究定位

本项目的目标不是判断眼底疾病，而是在诊断模型之前判断输入是否属于有效 FAF 图像分布。结果解读应始终围绕以下问题：

- 系统是否能接受 valid FAF。
- 系统是否能拒绝 colour fundus、IR、文本水印、拼接图、非视网膜自然图像等 OOD 输入。
- 训练是否只使用 label=0 的 ID/valid FAF 图像。
- 阈值选择是否会造成 valid FAF 被误拒。

可使用的表述：

> This system is evaluated as a pre-diagnostic quality-control gatekeeper rather than a disease classifier. A high anomaly score indicates that the input is unlikely to match the valid FAF training distribution, not that a disease is present.

### 1.2 不要过度声明

避免以下结论：

- “模型已经可临床部署。”
- “模型能诊断疾病。”
- “热力图证明了病灶位置。”
- “真实临床失败率已经被准确估计。”

更稳妥的表述：

> The results suggest that feature-space OOD detection is a promising gatekeeping strategy, but prospective validation on larger multi-scanner clinical data is required before deployment.

## 2. 指标解释模板

### 2.1 AUROC

含义：

AUROC 衡量 ID 和 OOD 样本在所有可能阈值下的整体排序能力。分数越高，说明 OOD 图像通常比 valid FAF 图像得到更高的 anomaly score。

论文写法模板：

> AUROC measures whether OOD inputs receive higher anomaly scores than valid FAF inputs across all thresholds. The observed AUROC of `[AUROC]` indicates that the detector has `[strong/moderate/weak]` global ranking ability. However, AUROC alone can be optimistic when the clinical operating point requires very low false rejection of valid FAF scans.

中文解释模板：

> AUROC 反映的是整体排序能力，而不是固定阈值下的临床可用性。若 AUROC 较高但 FPR@95%TPR 仍然偏高，说明模型虽然能大体区分 ID/OOD，但在高召回 OOD 的安全设置下仍会误拒较多 valid FAF。

### 2.2 AUPRC

含义：

AUPRC 更关注 OOD 作为 positive class 时的 precision-recall 表现。它比 AUROC 更能反映 OOD 类别稀少或类别不平衡时的实际报警质量。

论文写法模板：

> AUPRC treats OOD samples as the positive class and is therefore sensitive to whether high anomaly scores correspond to true invalid inputs. The AUPRC of `[AUPRC]` suggests that the detector's high-confidence rejections are `[mostly reliable/mixed/unreliable]` under the tested OOD composition.

中文解释模板：

> 如果 AUROC 高但 AUPRC 明显低，说明模型排序有一定能力，但高分样本中混入了较多 valid FAF，可能导致实际质控流程中的无效报警增加。

### 2.3 FPR@95%TPR

含义：

FPR@95%TPR 表示当系统调到能检出 95% OOD 输入时，有多少 valid FAF 会被误拒。这是质量控制 gatekeeper 最关键的安全指标之一。

论文写法模板：

> FPR@95%TPR estimates the fraction of valid FAF scans that would be rejected when the system is tuned to catch 95% of OOD inputs. A lower value is preferable because false rejection can interrupt downstream diagnostic workflows. The observed value of `[FPR@95TPR]` indicates `[acceptable / borderline / high]` false rejection pressure at a safety-oriented operating point.

中文解释模板：

> 对本项目而言，FPR@95%TPR 比单独 AUROC 更接近临床风险：如果该值较高，即使 OOD 检出率高，也意味着大量 valid FAF 会被挡在诊断模型之前。

### 2.4 Confusion Matrix

含义：

Confusion matrix 是在某个固定部署阈值下的 accept/reject 结果。需要说明阈值来源，例如 validation ID quantile 或人工配置阈值。

论文写法模板：

> At the selected threshold `[THRESHOLD]`, calibrated from `[THRESHOLD_SOURCE]`, the detector produced `[TN]` true accepts, `[FP]` false rejects, `[FN]` false accepts, and `[TP]` true rejects. In a gatekeeping context, false accepts are safety failures because invalid inputs pass downstream, while false rejects reduce workflow usability by rejecting valid FAF scans.

中文解释模板：

> 误接收 FN 和误拒 FP 的风险不同：FN 表示 OOD 图像进入后续诊断模型，可能造成 silent failure；FP 表示 valid FAF 被拒绝，主要影响工作流效率和用户信任。

### 2.5 Per-OOD Category Metrics

含义：

总指标可能掩盖不同 OOD 类型的差异。必须分别分析 modality shift、sensory artifact、semantic outlier 等类别。

论文写法模板：

> Per-category results show whether the detector is sensitive to all invalid input modes or only to the easiest distribution shifts. Strong performance on semantic outliers but weaker performance on sensory artifacts would suggest that the model captures global semantics but may miss local quality-control failures.

中文解释模板：

> 若 colour fundus/IR 表现很好但 text watermark 或 stitched layout 表现较差，说明模型更容易识别模态差异，却不一定可靠识别局部伪影或版式异常。

## 3. 主要结果写作模板

### 3.1 PatchCore 优于 Autoencoder

适用条件：

- PatchCore AUROC/AUPRC 高于 AE。
- PatchCore FPR@95%TPR 更低。
- AE heatmap 或 reconstruction error 对水印、文字、拼接图不敏感。

英文模板：

> PatchCore outperformed the convolutional autoencoder baseline across the main OOD metrics. This supports the hypothesis that feature-space patch embeddings are more suitable for FAF quality control than pixel-level reconstruction error. The autoencoder may reconstruct low-level structures without assigning sufficiently high anomaly scores to clinically invalid inputs, whereas PatchCore compares local CNN features against a normal FAF memory bank.

中文模板：

> PatchCore 相比 autoencoder baseline 表现更好，说明基于 CNN patch feature 的 feature-space OOD 检测更适合 FAF 质控任务。AE 的 reconstruction error 容易受到低层像素重建能力限制：模型可能重建出整体眼底结构，却没有把文字、水印或拼接布局赋予足够高的异常分数。PatchCore 则通过 normal FAF patch memory bank 进行最近邻距离比较，更直接地衡量输入局部特征是否偏离训练分布。

### 3.2 Autoencoder 失败或表现较弱

适用条件：

- AE 对 OOD 的 score separation 不明显。
- AE 对 watermark、annotation、composite layout 等 artifact 不敏感。
- AE 的 AUROC/AUPRC 低或 FPR@95%TPR 高。

英文模板：

> The autoencoder baseline showed limited suitability for this gatekeeping task. Reconstruction-based anomaly detection assumes that invalid inputs produce higher reconstruction error, but this assumption can fail when the model learns generic retinal structure or reconstructs artifacts without preserving their semantic invalidity. This explains the weaker separation between valid FAF and OOD categories.

中文模板：

> AE baseline 的结果说明，reconstruction error 并不总能代表输入是否适合进入诊断模型。对于眼底图像，autoencoder 可能学习到较通用的 retinal appearance，并对文字、水印或局部伪影产生可接受的重建，从而低估这些输入的异常程度。这是本项目选择 PatchCore 作为主方法的重要动机。

### 3.3 PatchCore-L2+L3 最好

适用条件：

- layer2+layer3 的 AUROC/AUPRC 最高或接近最高。
- FPR@95%TPR 相对较低。
- heatmap 对局部 artifact 有响应。

英文模板：

> The layer2+layer3 configuration achieved the best balance between modality-level discrimination and local artifact sensitivity. Mid-level CNN features appear to preserve enough spatial detail to detect quality-control artifacts while being more robust than shallow features to synthetic-to-real pixel-level differences.

中文模板：

> layer2+layer3 取得较好结果，说明中层 CNN 特征在本任务中提供了合适的折中：它们保留了足够空间信息用于检测局部 artifact，同时又不像浅层特征那样过度依赖 synthetic FAF 的像素纹理。因此，中层特征更有可能缓解 synthetic-to-real gap，并保持对无效输入的敏感性。

### 3.4 Layer1 表现好但 real FAF false positive 高

适用条件：

- layer1 在 synthetic/validation 上好。
- real FAF 上 false positive 或 FPR@95%TPR 高。

英文模板：

> Layer1 features may be overly sensitive to low-level texture, contrast, or acquisition differences. If layer1 performs well on synthetic validation data but rejects many valid real FAF scans, this is evidence of synthetic-to-real overfitting rather than clinically useful quality control.

中文模板：

> 如果 layer1 在 synthetic validation 上表现较好，但在 valid real FAF 上误拒率较高，这更可能说明浅层特征过拟合了 synthetic 图像的低层纹理、噪声或对比度特征，而不是学到了稳定的 FAF 质控标准。这一结果可作为 sim-to-real gap 的证据。

### 3.5 Layer4 对 semantic outlier 好但 artifact 差

适用条件：

- layer4 对 non-retinal natural images 表现好。
- layer4 对 watermark/text/composite artifact 表现弱。

英文模板：

> Layer4 features captured high-level semantic mismatch but were less sensitive to local quality-control artifacts. This suggests that deep CNN representations may be too invariant for detecting small text overlays, annotations, or stitched layouts that are clinically invalid despite preserving retinal semantics.

中文模板：

> layer4 若能很好识别非视网膜自然图像，却对水印、文字或拼接图不敏感，说明深层语义特征可能过于抽象。它们能发现“这不是眼底图”的大语义错误，但可能忽略“这是眼底图但不适合诊断模型”的局部质控问题。

## 4. Heatmap 解读模板

### 4.1 True Positive heatmap

> True positive examples show that the detector assigns high anomaly scores to invalid inputs. Heatmaps should be interpreted as localization of feature-space deviation rather than disease evidence.

中文模板：

> TP heatmap 说明模型在无效输入上产生了较高异常响应。需要强调的是，heatmap 展示的是相对 normal FAF feature memory 的偏离区域，不是病灶定位。

### 4.2 False Positive heatmap

> False positive examples reveal valid FAF scans that are rejected by the gatekeeper. These cases are important because they represent workflow cost and may indicate scanner shift, illumination differences, or synthetic-to-real mismatch.

中文模板：

> FP 样本是 valid FAF 被误拒的情况，应重点分析其是否存在 scanner difference、contrast shift、边缘裁剪、低照度或 synthetic-to-real gap。FP 不是“模型发现疾病”，而是模型认为图像质量/分布偏离 normal training memory。

### 4.3 False Negative heatmap

> False negatives are the most safety-critical errors because invalid inputs pass the gatekeeper. If heatmaps are diffuse or low-intensity on these examples, the model may lack sensitivity to that OOD category.

中文模板：

> FN 是安全性最关键的失败：无效图像被接受并进入后续诊断模型。若 FN 集中在某类 artifact，说明该类别需要更多 targeted augmentation、特征层调整或额外质控规则。

## 5. Limitations 模板

可直接改写使用：

> This study has several limitations. First, the detector is trained on synthetic valid FAF images, so the learned normal distribution may not fully represent the variability of real clinical FAF scans. Second, the OOD categories are curated stress tests and may not cover all invalid inputs seen in prospective deployment. Third, threshold calibration is based on available validation data, and the false rejection rate may change across scanners, acquisition protocols, or institutions. Fourth, heatmaps indicate feature-space deviation rather than causal clinical abnormalities. Finally, the system is evaluated as an upstream gatekeeper and does not replace clinical image quality assessment or disease diagnosis.

中文版本：

> 本研究存在若干限制。第一，模型主要从 synthetic valid FAF 学习 normal distribution，可能无法完全覆盖真实临床 FAF 的采集差异。第二，OOD 测试集是人工构建的压力测试集合，不能穷尽真实部署中的所有无效输入。第三，阈值基于当前 validation 数据校准，在不同 scanner、采集协议或机构中可能出现误拒率变化。第四，heatmap 只表示 feature-space deviation，不应解释为病灶定位。最后，本系统是诊断模型前的质量控制 gatekeeper，不能替代临床图像质量评估或疾病诊断。

## 6. Future Work 模板

可选方向：

- 使用多中心、多 scanner valid FAF 数据进行外部验证。
- 增加真实 clinical artifact 类别，如低质量曝光、运动模糊、遮挡、裁剪错误。
- 对 threshold calibration 做 scanner-specific 或 site-specific 分析。
- 比较 PatchCore、PaDiM、FastFlow、student-teacher 等 feature-space 方法。
- 增加 human-in-the-loop 审核流程，降低 false rejection 的工作流成本。
- 将 gatekeeper 与 downstream diagnostic model 串联，评估 OOD 拦截对诊断稳定性的影响。

英文模板：

> Future work should validate the gatekeeper on larger multi-centre FAF datasets, expand the OOD taxonomy to include more real acquisition failures, and study scanner-specific threshold calibration. It would also be valuable to evaluate whether rejecting high-scoring OOD inputs improves the robustness of downstream diagnostic models.

## 7. Results Section Checklist

写 Results 时建议依次检查：

- 是否说明 ID=valid FAF，OOD=invalid input，而不是 disease positive。
- 是否报告 AUROC、AUPRC、FPR@95%TPR，而不是只报告 AUROC。
- 是否包含 confusion matrix 和阈值来源。
- 是否包含 per-OOD-category metrics。
- 是否比较 AE baseline 与 PatchCore。
- 是否报告 layer1、layer2、layer3、layer4、layer2+layer3 消融。
- 是否解释 FP 和 FN 的不同风险。
- 是否明确 heatmap 不是病灶定位。
- 是否避免提交或展示患者标识、私有路径和原始医疗图像。

## 8. 一句话总结模板

PatchCore 胜出时：

> Overall, the results support feature-space PatchCore-style anomaly detection as a stronger upstream FAF quality-control gatekeeper than reconstruction-based autoencoding, especially when mid-level ResNet features are used.

AE 失败时：

> The autoencoder baseline demonstrates that low reconstruction error does not necessarily imply that an image is valid for diagnostic use.

层消融显示 sim-to-real gap 时：

> The layer ablation suggests that shallow features are vulnerable to synthetic-to-real texture mismatch, while very deep features may miss local quality-control artifacts; mid-level features provide the most useful compromise.
