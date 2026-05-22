# 本地运行与数据集选择指南

本文档回答三个问题：

- 这份代码需要什么样的数据集。
- 如何把数据整理成 manifest。
- 如何在本地跑出训练、评估、热力图、表格和论文图。

本项目是 FAF 图像质量控制 / OOD gatekeeper，不是疾病分类器。训练阶段只应使用 `label=0` 的 valid FAF 图像。

如果你想按时间线从公开数据一路跑到结果表、热力图和拖拽 UI，请优先看 `docs/EXPERIMENT_RUNBOOK_CN.md`。本文档保留为数据选择和命令速查。

## 1. 你应该找怎样的数据集

### 1.1 最理想的数据组成

最适配这份代码的数据集应包含以下四类：

| 用途 | label | 建议内容 | 是否用于训练 |
|---|---:|---|---|
| ID train | 0 | valid/gradable FAF，最好同一主要设备或协议 | yes |
| ID validation | 0 | valid/gradable FAF，用于阈值 calibration | no model fitting |
| ID test | 0 | valid real FAF，最好来自真实临床或不同 acquisition batch | no |
| OOD test | 1 | colour fundus、IR、text/watermark/composite artifact、non-retinal images | no |

最重要的一点：

> PatchCore 和 autoencoder 的训练 manifest 里可以混入其他行，但训练代码会只使用 `label=0`。为了论文更清楚，建议 train manifest 本身只放 ID/valid FAF。

### 1.2 ID / valid FAF 数据建议

优先级从高到低：

1. 你自己能合法使用的真实 FAF 数据，尤其是 Heidelberg Spectralis / HRA FAF。
2. 公开或可申请的 FAF 数据。
3. synthetic FAF 数据，用于 dissertation proof-of-concept。

公开 FAF 数据比 colour fundus 少很多。可以考虑：

- UCL Research Data Repository 的 SynthEye synthetic FAF 数据：适合作为 synthetic FAF proof-of-concept 或小规模 smoke experiment。
- Retinograd-AI 论文和仓库：它是 FAF gradability 方向的强参考，公开了代码/权重；但不要假设其内部临床 FAF 数据可直接下载。
- 如果你能通过学校/导师获得匿名 FAF，优先使用真实 valid FAF 作为 test ID，因为这正是 sim-to-real gap 要评估的地方。

### 1.3 OOD 数据建议

OOD 不需要和 FAF 同病种；它们代表“不能送进 FAF 诊断模型”的输入。

建议组合：

| OOD 类型 | 推荐来源 | manifest `ood_type` |
|---|---|---|
| colour fundus | RFMiD、APTOS 2019、IDRiD、DRIVE、STARE 等 colour fundus datasets | `modality_shift` |
| infrared / near-IR fundus | OLIVES 等 near-IR fundus 数据 | `modality_shift` |
| text/watermark/annotation | 在 held-out FAF test images 上人工生成文字、框、箭头、水印 | `sensory_artifact` |
| stitched/composite layout | 把 2-4 张 held-out images 拼接成 panel | `sensory_artifact` |
| non-retinal natural images | CIFAR-10、ImageNet subset、Open Images subset | `semantic_outlier` |

不要用 train FAF 生成 artifact test OOD。artifact 应从 validation/test ID 原图或独立图像生成，避免数据泄漏。

## 2. Manifest 要怎么写

代码当前要求 CSV 至少有这些列：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
```

`patient_id`、`scanner`、`notes` 可以为空。公共非医疗 OOD 图片不需要 patient_id。

加载器会检查：manifest 不能是空表，`label` 只能是 `0` 或 `1`，`split` 只能是 `train` / `val` / `test`，并且默认会检查图片文件真实存在。这样可以把数据整理错误和模型训练错误分开。

推荐本地结构：

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
    ood_colour_fundus/
    ood_ir/
    ood_artifact/
    ood_natural/
```

示例 `train_synthetic_faf.csv`：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
images/synthetic_faf/faf_0001.png,0,train,synthetic_faf,id,,,synthetic valid FAF
images/synthetic_faf/faf_0002.png,0,train,synthetic_faf,id,,,synthetic valid FAF
```

示例 `val_synthetic_faf.csv`：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
images/synthetic_faf/faf_val_0001.png,0,val,synthetic_faf,id,,,threshold calibration
```

示例 `test_real_id.csv`：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
images/real_faf/real_faf_0001.png,0,test,real_faf,id,anon_001,Heidelberg,valid real FAF
```

示例 `test_ood.csv`：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
images/ood_colour_fundus/aptos_0001.png,1,test,public_ood,modality_shift,,,colour fundus
images/ood_ir/olives_0001.png,1,test,public_ood,modality_shift,,,near-IR fundus
images/ood_artifact/watermark_0001.png,1,test,synthetic_artifact,sensory_artifact,,,text watermark
images/ood_natural/cifar_0001.png,1,test,public_ood,semantic_outlier,,,non-retinal natural image
```

## 3. 本地安装

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
ruff check .
```

macOS/Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
ruff check .
```

如果显存不够或没有 GPU，把 config 里的 `model.device` 改成 `cpu`，并降低 `data.image_size`、`data.batch_size`、`model.max_train_patches`。

## 4. PatchCore 主实验怎么跑

### 4.1 训练主模型

```bash
python scripts/train_patchcore.py --config configs/patchcore_l23.yaml
```

默认输出：

```text
runs/patchcore_resnet50_layer2_layer3/
  patchcore_memory.npz
  resolved_config.json
  training_metrics.json
```

### 4.2 评估主模型

```bash
python scripts/evaluate.py ^
  --config configs/patchcore_l23.yaml ^
  --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz ^
  --save-heatmaps ^
  --heatmap-top-k 5
```

macOS/Linux 写法：

```bash
python scripts/evaluate.py \
  --config configs/patchcore_l23.yaml \
  --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz \
  --save-heatmaps \
  --heatmap-top-k 5
```

默认输出：

```text
runs/patchcore_resnet50_layer2_layer3/evaluation/
  scores.csv
  metrics.json
  resolved_evaluation_config.json
  roc_curve.png
  pr_curve.png
  heatmaps/
```

### 4.3 本地拖拽式 gatekeeper 界面

在完成一次 evaluation 后，`metrics.json` 里会保存 threshold。然后可以启动本地网页：

```bash
python scripts/serve_gatekeeper_app.py \
  --config configs/patchcore_l23.yaml \
  --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz
```

Windows PowerShell 也可以写成一行：

```powershell
python scripts/serve_gatekeeper_app.py --config configs/patchcore_l23.yaml --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz
```

打开 `http://127.0.0.1:7860`，把图片拖进去即可。当前支持的输入文件是普通图片：

- `.png`
- `.jpg` / `.jpeg`
- `.tif` / `.tiff`
- `.bmp`

输出是二分类 gatekeeper decision：

- `ACCEPT: likely valid FAF`：分数低于 threshold，认为可以进入下游 FAF 诊断模型。
- `REJECT: OOD / invalid input`：分数高于或等于 threshold，认为应被拒绝。

它不会输出疾病类别，也不会把 OOD 自动细分成 colour fundus / IR / watermark 等类别；这些类别用于离线评估和 per-category metrics。

新版 UI 支持一次拖入多张图片、点击结果行查看 original/overlay、并导出本次 UI prediction CSV。上传图片只在本地 server 内存中处理，不会被 UI 写入仓库或保存到 `runs/`。

如果你没有先跑 evaluation，也可以手动传 threshold：

```bash
python scripts/serve_gatekeeper_app.py \
  --config configs/patchcore_l23.yaml \
  --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz \
  --threshold 12.34
```

在 Jackpot 上建议仍然默认只绑定本机端口，然后通过 SSH tunnel 访问；如果必须绑定外部地址，再使用 `--host 0.0.0.0`，并确保不要把私有医疗图片暴露到公网。

## 5. Layer ablation 怎么跑

逐个训练和评估：

```bash
python scripts/train_patchcore.py --config configs/patchcore_l1.yaml
python scripts/evaluate.py --config configs/patchcore_l1.yaml --checkpoint runs/patchcore_resnet50_layer1/patchcore_memory.npz

python scripts/train_patchcore.py --config configs/patchcore_l2.yaml
python scripts/evaluate.py --config configs/patchcore_l2.yaml --checkpoint runs/patchcore_resnet50_layer2/patchcore_memory.npz

python scripts/train_patchcore.py --config configs/patchcore_l3.yaml
python scripts/evaluate.py --config configs/patchcore_l3.yaml --checkpoint runs/patchcore_resnet50_layer3/patchcore_memory.npz

python scripts/train_patchcore.py --config configs/patchcore_l4.yaml
python scripts/evaluate.py --config configs/patchcore_l4.yaml --checkpoint runs/patchcore_resnet50_layer4/patchcore_memory.npz

python scripts/train_patchcore.py --config configs/patchcore_l23.yaml
python scripts/evaluate.py --config configs/patchcore_l23.yaml --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz
```

## 6. Autoencoder baseline 怎么跑

```bash
python scripts/train_autoencoder.py --config configs/autoencoder_baseline.yaml
```

当前 autoencoder 训练和 checkpoint 已实现，但统一 evaluation CLI 目前面向 PatchCore checkpoint。你可以把 autoencoder 结果作为 baseline smoke result 记录在 `training_metrics.json`，或者后续再补一个 autoencoder evaluation CLI。

## 7. 生成报告表和论文图

```bash
python scripts/generate_report_tables.py --runs-dir runs --out reports/generated/experiment_summary.md
python scripts/generate_dissertation_figures.py --runs-dir runs --out-dir reports/generated/figures
```

输出在 `reports/generated/`，该目录被 `.gitignore` 忽略。你可以把没有隐私信息的 summary 图表手动复制到 dissertation，不要提交原始医疗图像。

## 8. 最小可行实验建议

如果时间有限，做这个组合：

1. `train_synthetic_faf.csv`：至少 50-200 张 valid/synthetic FAF。
2. `val_synthetic_faf.csv`：至少 20-50 张 valid/synthetic FAF。
3. `test_real_id.csv`：尽可能使用真实 valid FAF，哪怕数量较少，也要明确 limitation。
4. `test_ood.csv`：
   - 50 张 colour fundus。
   - 50 张 near-IR 或其他 retinal modality。
   - 50 张 text/watermark/composite artifacts。
   - 50 张 non-retinal natural images。

论文里重点报告：

- PatchCore-L2+L3 vs autoencoder baseline。
- layer1/layer2/layer3/layer4/layer2+layer3 ablation。
- per-OOD-category metrics。
- FPR@95%TPR 和 confusion matrix。
- TP/FP/FN/TN heatmap examples。

## 9. 常见问题

### 9.1 为什么不要拿 colour fundus 当 ID 训练？

因为本项目要做 FAF gatekeeper。Colour fundus 是 wrong modality，应作为 OOD 测试，不应进入 ID training。

### 9.2 如果没有真实 FAF 怎么办？

可以用 synthetic FAF 做 proof-of-concept，但 dissertation 必须把它写成 limitation。最好至少找到少量真实 valid FAF 作为 test ID，用来展示 sim-to-real false positive。

### 9.3 OOD label 是疾病标签吗？

不是。`label=1` 表示“输入不适合作为 valid FAF 进入诊断模型”，不是“有病”。

### 9.4 可以提交数据到 GitHub 吗？

不要提交真实医疗图像、patient_id、私有下载链接、模型权重或本地绝对路径。只提交代码、docs、toy tests、无隐私的 aggregate plots。

## 10. 推荐候选数据源

使用前请检查每个数据源的 license、伦理要求和下载条款。

- SynthEye synthetic FAF dataset: https://rdr.ucl.ac.uk/articles/dataset/Synthetic_dataset_of_100_fundus_auto_fluorescence_of_inherited_retinal_disease/28604234
- Retinograd-AI FAF gradability reference: https://github.com/Eye2Gene/retinograd-ai
- Retinograd-AI paper: https://www.medrxiv.org/content/10.1101/2024.08.07.24311607v1
- RFMiD colour fundus dataset reference: https://www.nature.com/articles/s41598-023-38610-y
- APTOS 2019 colour fundus dataset: https://www.kaggle.com/c/aptos2019-blindness-detection/data
- OLIVES near-IR fundus/OCT dataset: https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset
- IRFundusSet harmonized public colour fundus collection: https://bilha-analytics.github.io/IRFundusSet/
