# 十周毕设推进计划：Retinal FAF OOD Gatekeeper

## 1. 项目主线

本项目的主线是构建一个用于 retinal Fundus Autofluorescence，FAF，图像的 unsupervised out-of-distribution gatekeeper。它不是疾病分类器，而是放在下游诊断模型之前的输入质控模块。

核心目标：

- 只使用 valid FAF 图像训练。
- 拒绝 colour fundus、infrared、text/watermark artefact、composite layout、natural images 等 invalid/OOD 输入。
- 用 autoencoder 作为 reconstruction baseline。
- 用 PatchCore-style feature-space anomaly detection 作为主方法。
- 通过 layer ablation、per-OOD-category analysis、threshold analysis 和 heatmaps 支撑 dissertation 实验工作量。

## 2. 十周总览

| 周次 | 核心目标 | 具体任务 | 交付物 |
|---|---|---|---|
| Week 1 | 确认项目范围和数据方案 | 明确 research questions；确认 valid FAF / OOD taxonomy；整理公开数据源；跑通本地环境、`pytest`、`ruff check .`；确认 GitHub 不包含医学图像和私有路径 | 项目背景说明、数据源清单、环境可运行记录 |
| Week 2 | 建立数据集 v1 | 下载/整理 public datasets；将图像统一成 PNG/JPG/TIFF；建立 `data/images/` 和 `data/manifests/`；编写 train/val/test manifests；检查 label、split、category、patient_id 规则 | `train_synthetic_faf.csv`、`val_synthetic_faf.csv`、`test_id.csv`、`test_ood.csv` |
| Week 3 | 开始 coding 和 smoke evaluation | 用小数据集跑 PatchCore L2+L3；跑 autoencoder baseline 小实验；生成少量 heatmaps；修复 manifest/path/config 问题；锁定 dataset v1 | smoke experiment 结果、第一版 `runs/` 输出、可运行实验命令 |
| Week 4 | 主实验 | 跑 PatchCore L2+L3 主实验；跑 autoencoder baseline；保存 metrics、scores、ROC/PR curves、heatmaps；记录硬件、随机种子、配置 | 主结果表：AE vs PatchCore |
| Week 5 | 对比实验和 ablation | 跑 PatchCore layer1、layer2、layer3、layer4、layer2+layer3；按 OOD 类别分别评估；如果时间允许，实现/运行 global feature kNN baseline | layer ablation 表格、per-category metrics |
| Week 6 | Stress test 和 threshold analysis | 生成文字、水印、拼接、截图类 artifact；比较 evaluation threshold 和 deployment threshold；分析 FPR@95%TPR；整理 ACCEPT/REJECT examples | threshold analysis、artifact stress test 图 |
| Week 7 | 结果解释和 failure analysis | 选择 TP/FP/FN/TN heatmap examples；分析 PatchCore 失败案例；总结 synthetic-to-real gap；生成 dissertation figures 和 report tables | heatmap grid、failure-case figure、实验章节素材 |
| Week 8 | 完成论文初稿 | 写 Introduction、Literature Review、Methodology、Experiments、Results、Discussion、Conclusion；把图表嵌入论文；统一术语：gatekeeper / OOD / FAF / not disease classifier | dissertation full draft v1 |
| Week 9 | 论文优化 | 根据导师反馈修改结构；加强 novelty、method justification、baseline rationale；检查引用、图表编号、caption、实验复现描述；补充 limitation | dissertation draft v2 |
| Week 10 | 最终润色和答辩准备 | 做最终复现实验检查；整理 GitHub README；准备 demo UI 截图；准备答辩 slides；检查语法、格式、Turnitin 风险、附录命令 | final dissertation、slides、demo materials、reproducibility notes |

## 3. 每周执行细化

### Week 1：范围确认和数据路线

本周目标是把项目方向讲清楚，并确认后续实验不会偏离 gatekeeper 主题。

要完成：

- 写清楚 project background：本项目不是 disease classifier，而是 pre-diagnostic OOD gatekeeper。
- 锁定 research questions：
  - PatchCore 是否优于 autoencoder reconstruction baseline？
  - ResNet 哪一层特征最适合 FAF OOD gatekeeping？
  - AUROC、AUPRC、FPR@95%TPR 对 clinical safety 各自说明什么？
- 确认 OOD taxonomy：
  - `modality_shift`
  - `sensory_artifact`
  - `semantic_outlier`
  - 设备差异先作为 `scanner` metadata 和 failure analysis 记录，不单独扩展当前 OOD taxonomy
- 跑通基础命令：
  - `pytest`
  - `ruff check .`
- 记录公开数据候选，不下载或提交任何真实医学图像到 GitHub。

验收标准：

- 能用 2-3 分钟向导师解释项目背景、novelty 和 evaluation plan。
- 仓库测试通过。
- 数据源清单和隐私规则明确。

### Week 2：数据集 v1 和 manifest

本周目标是让实验数据可以被现有代码稳定读取。

要完成：

- 建立本地数据目录：

```text
data/
  images/
    synthetic_faf/
    real_faf/
    ood_colour_fundus/
    ood_ir/
    ood_artifact/
    ood_natural/
  manifests/
```

- 准备 manifests：
  - `data/manifests/train_synthetic_faf.csv`
  - `data/manifests/val_synthetic_faf.csv`
  - `data/manifests/test_id.csv`
  - `data/manifests/test_ood.csv`
- 确认 schema：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
```

- 使用 `label=0` 表示 valid FAF / ID。
- 使用 `label=1` 表示 OOD / invalid input。
- public/non-medical OOD 图像允许 `patient_id` 为空。

验收标准：

- 所有 manifest 可以被 dataset loader 读取。
- 没有 missing file、invalid split、invalid label、empty dataset 错误。
- 数据不进入 GitHub，只保存在本地或 Jackpot scratch/project storage。

### Week 3：smoke evaluation

本周目标是小规模跑通完整 pipeline，不追求最终分数。

要完成：

```bash
python scripts/train_patchcore.py --config configs/patchcore_l23.yaml
python scripts/evaluate.py --config configs/patchcore_l23.yaml --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz --save-heatmaps --heatmap-top-k 2
python scripts/train_autoencoder.py --config configs/autoencoder_baseline.yaml
python scripts/evaluate_autoencoder.py --config configs/autoencoder_baseline.yaml --checkpoint runs/autoencoder_baseline/model.pt
```

验收标准：

- PatchCore 和 autoencoder 都能生成 evaluation 输出。
- 至少有 `metrics.json`、`scores.csv`、ROC/PR 曲线或 heatmaps。
- 发现的路径、config、manifest 问题已经修正。
- dataset v1 锁定，后续不再频繁改变 split。

### Week 4：主实验

本周目标是完成 dissertation 的第一张核心结果表。

要完成：

- 跑 PatchCore L2+L3 主实验。
- 跑 autoencoder baseline。
- 保存每次实验的 resolved config、random seed、hardware note、metrics。
- 生成主结果表：

| Method | Feature space | AUROC | AUPRC | FPR@95%TPR | Notes |
|---|---|---:|---:|---:|---|
| Autoencoder | pixel reconstruction |  |  |  | baseline |
| PatchCore L2+L3 | patch-level CNN features |  |  |  | main method |

验收标准：

- 能初步回答：PatchCore 是否优于 autoencoder baseline？
- 主实验结果可以放进论文 Results 章节。

### Week 5：对比实验和 layer ablation

本周目标是体现实验工作量和方法选择依据。

P0 实验：

- PatchCore layer1
- PatchCore layer2
- PatchCore layer3
- PatchCore layer4
- PatchCore layer2+layer3

P1 实验：

- Global feature kNN baseline，用 whole-image feature 证明 patch-level detection 的必要性。

验收标准：

- 生成 layer ablation table。
- 生成 per-OOD-category metrics。
- 能解释为什么 layer2+layer3 是主模型，或者根据结果调整主模型选择。

### Week 6：stress test 和 threshold analysis

本周目标是让实验更贴近真实 gatekeeper 部署场景。

要完成：

- 生成或整理 artifact OOD：
  - text overlay
  - watermark
  - annotation
  - stitched/composite layout
  - screenshot-like border/crop
- 比较两种 threshold：
  - evaluation threshold：用于 FPR@95%TPR。
  - deployment threshold：使用 validation ID score 的 95% 或 99% percentile。
- 分析 anomaly score 是否随 artifact severity 增强而上升。

验收标准：

- 有 threshold comparison 表。
- 有 artifact severity 或 qualitative stress-test 图。
- 能解释 ACCEPT/REJECT trade-off。

### Week 7：结果解释和 failure analysis

本周目标是把实验结果转化成 dissertation discussion。

要完成：

- 选择 heatmap examples：
  - true positive OOD reject
  - true negative valid FAF accept
  - false positive valid FAF reject
  - false negative OOD accept
- 分析 failure cases：
  - real FAF 是否因为 synthetic-to-real gap 被误拒？
  - colour fundus / IR 是否因为 retinal structure 相似而难检？
  - local watermark 是否被 AE 忽略但被 PatchCore 检出？
- 生成最终 figures：

```bash
python scripts/generate_report_tables.py --runs-dir runs --out reports/generated/experiment_summary.md
python scripts/generate_dissertation_figures.py --runs-dir runs --out-dir reports/generated/figures
```

验收标准：

- Results 章节需要的表格和图基本齐全。
- Discussion 章节已有明确论点，而不是只描述数字。

### Week 8：论文初稿

本周目标是完成完整 dissertation draft v1。

建议章节：

1. Introduction
2. Literature Review
3. Problem Formulation
4. Data
5. Methods
6. Evaluation Protocol
7. Results
8. Discussion
9. Limitations
10. Conclusion

验收标准：

- 所有章节都有内容。
- 所有核心图表已经插入。
- 论文能完整讲清楚项目：background -> method -> experiments -> results -> discussion。

### Week 9：论文优化

本周目标是根据导师反馈提高论文质量。

重点修改：

- novelty 表达：强调应用场景、实验设计和系统实现，而不是声称发明新网络。
- baseline rationale：解释为什么 autoencoder 是合理 baseline。
- method justification：解释为什么 PatchCore 和 patch-level feature 适合 FAF gatekeeping。
- metric interpretation：解释 AUROC、AUPRC、FPR@95%TPR 的不同含义。
- limitation：如 synthetic-to-real gap、数据规模、无前瞻临床验证。

验收标准：

- 论文逻辑更顺。
- 图表 caption 能独立解释图的意义。
- 结果讨论避免过度 clinical claim。

### Week 10：最终润色和答辩准备

本周目标是最终提交和展示准备。

要完成：

- 最终运行：

```bash
pytest
ruff check .
```

- 检查 GitHub：
  - 不包含医学图像。
  - 不包含模型权重。
  - 不包含 patient data。
  - 不包含私有路径或 credentials。
- 准备 demo UI：

```bash
python scripts/serve_gatekeeper_app.py --config configs/patchcore_l23.yaml --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz
```

- 准备答辩 slides：
  - problem
  - novelty
  - method
  - experiments
  - results
  - limitations
  - future work

验收标准：

- final dissertation 可提交。
- slides 可用于 8-10 分钟展示。
- demo UI 有可展示截图或本地可运行版本。
- 代码和实验说明可复现。

## 4. 实验优先级

| 实验 | 目的 | 优先级 |
|---|---|---|
| Autoencoder vs PatchCore | 证明 feature-space method 是否优于 reconstruction baseline | P0 |
| PatchCore layer ablation | 比较 layer1、layer2、layer3、layer2+layer3、layer4 | P0 |
| Per-OOD-category evaluation | 分析 wrong modality、artefact、composite、natural images 的表现 | P0 |
| Threshold strategy comparison | 比较 FPR@95%TPR threshold 和 validation quantile deployment threshold | P0 |
| Heatmap qualitative analysis | 展示模型关注文字、水印、拼接边界或错误模态区域 | P0 |
| Global feature kNN baseline | 证明 patch-level features 比 whole-image features 更适合局部异常 | P1 |
| Artifact severity stress test | 展示 anomaly score 是否随异常强度合理上升 | P1 |
| Training data size ablation | 分析少量 valid FAF 是否足够建立 gatekeeper | P2 |
| Coreset ratio ablation | 分析 PatchCore memory bank 压缩对性能和效率的影响 | P2 |

## 5. 每周固定检查清单

每周结束时检查：

- 代码可以运行，关键命令有记录。
- `pytest` 和 `ruff check .` 在本周改代码后通过。
- 数据不进入 GitHub，只保留 manifest 示例和配置。
- 实验结果有 `metrics.json`、`scores.csv`、图表或日志。
- 新增实验能回答一个明确 research question。
- 论文至少同步更新一个章节、一个表格或一个结果解释段落。
- 所有结论都避免 clinical deployment 过度声明。

## 6. 风险和应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 真实 FAF 数据拿不到 | 无法充分验证 synthetic-to-real gap | 使用 synthetic FAF 完成 proof-of-concept，并把真实 FAF 缺失写入 limitation |
| OOD 数据格式混乱 | manifest 难以稳定读取 | 统一导出为 PNG/JPG/TIFF，再写 manifest |
| PatchCore 运行慢 | ablation 时间不够 | 先跑 L2+L3、AE、L2、L3，其他实验降为 optional |
| 指标表现不如预期 | 论文结论不够强 | 加强 failure analysis，讨论哪些 OOD 类型困难以及为什么 |
| 第 8 周论文没写完 | 第 9-10 周压力过大 | Week 4 开始同步写 Results 表格和 Methodology，不等实验全部结束 |

## 7. 最终成果

到第 10 周结束时，目标交付物包括：

- 一个完整的 retinal FAF OOD gatekeeper 代码仓库。
- 可复现的 manifest/config/runbook。
- Autoencoder baseline 和 PatchCore main method 的结果。
- Layer ablation、per-category metrics、threshold analysis 和 heatmaps。
- Dissertation final draft。
- 答辩 slides 和 demo UI 展示材料。
