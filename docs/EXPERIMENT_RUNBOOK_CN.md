# 实验运行手册：从公开数据到本地/Jackpot 结果

这份手册按“公开数据优先”的路线写，目标是让你先把 dissertation 实验完整跑通。项目任务是 FAF 图像质量控制 gatekeeper，不是疾病分类器：模型只判断输入是否像 valid FAF，输出 `ACCEPT` 或 `REJECT`。

## 1. 你需要准备什么数据

最小可行实验需要四组图片：

| 数据 | label | split | 用途 | 推荐数量 |
|---|---:|---|---|---:|
| synthetic/valid FAF train | 0 | train | 训练 PatchCore normal memory | 100-500 |
| synthetic/valid FAF val | 0 | val | 估计 threshold | 30-100 |
| valid FAF test | 0 | test | 测 false rejection | 30-100 |
| OOD/invalid test | 1 | test | 测拒绝能力 | 每类 50+ |

如果暂时没有真实 FAF，先用 synthetic FAF 同时做 train/val/test smoke experiment，但论文里必须写清楚 limitation。真实 FAF 最好通过导师、学校或合规数据协议获取，且必须匿名化。

## 2. 数据从哪里找

### 2.1 ID / valid FAF

首选路线：

1. [UCL SynthEye synthetic FAF dataset](https://rdr.ucl.ac.uk/articles/dataset/Synthetic_dataset_of_100_fundus_auto_fluorescence_of_inherited_retinal_disease/28604234)：适合作为 proof-of-concept 的 synthetic FAF ID 数据。
2. 你能合规使用的真实匿名 FAF：最好用于 `test_real_id.csv`，用来评估 synthetic-to-real false positive。
3. [Retinograd-AI](https://github.com/Eye2Gene/retinograd-ai)：这是 FAF gradability 的重要参考，不要默认它提供可直接下载的临床 FAF 训练集。

### 2.2 OOD / invalid 输入

这些图片不用于训练，只用于 test OOD：

| OOD 类别 | 推荐来源 | manifest `ood_type` |
|---|---|---|
| colour fundus | [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)、RFMiD、IRFundusSet | `modality_shift` |
| infrared / near-IR | [OLIVES on Hugging Face](https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset) | `modality_shift` |
| natural images | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)、[Open Images](https://storage.googleapis.com/openimages/web/index.html) subset | `semantic_outlier` |
| watermark/text/composite | 从 held-out FAF/公开非隐私图片人工生成 | `sensory_artifact` |

不要把 colour fundus 当作 ID 训练数据。它是 wrong modality，应该作为 OOD。

## 3. 推荐本地目录

所有数据放在本地，不能提交 GitHub。

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

如果数据放在仓库外，例如 Jackpot scratch，把 YAML 里加上：

```yaml
data:
  root_dir: /path/to/private/retinal_ood_data
```

然后 manifest 的 `image_path` 写成相对 `root_dir` 的路径，例如 `images/synthetic_faf/faf_0001.png`。

## 4. Manifest 怎么写

CSV 必须包含：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
```

字段规则：

- `label=0`：valid FAF / ID。
- `label=1`：OOD / invalid input。
- `split`：只能是 `train`、`val`、`test`。
- `patient_id`：公共 OOD 或非医疗图片可以为空。
- `scanner`：未知可以为空或写 `unknown`。
- `notes`：记录来源、转换方式、artifact 类型即可，不要写私有下载链接。

示例 `train_synthetic_faf.csv`：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
images/synthetic_faf/faf_0001.png,0,train,synthetic_faf,id,,,SynthEye synthetic FAF
images/synthetic_faf/faf_0002.png,0,train,synthetic_faf,id,,,SynthEye synthetic FAF
```

示例 `test_ood.csv`：

```csv
image_path,label,split,source,ood_type,patient_id,scanner,notes
images/ood_colour_fundus/aptos_0001.png,1,test,public_ood,modality_shift,,,APTOS colour fundus
images/ood_ir/olives_0001.png,1,test,public_ood,modality_shift,,,OLIVES near-IR image
images/ood_natural/cifar_0001.png,1,test,public_ood,semantic_outlier,,,CIFAR-10 natural image
images/ood_artifact/watermark_0001.png,1,test,synthetic_artifact,sensory_artifact,,,text watermark artifact
```

## 5. Day-by-day 时间线

### Day 1：安装环境并确认代码可用

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
ruff check .
```

Linux / macOS / Jackpot：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
ruff check .
```

### Day 1-2：下载并整理数据

1. 下载 SynthEye synthetic FAF，放在本地或 Jackpot 私有目录，不要提交图片到 GitHub。
2. 用 `prepare_syntheye_dataset.py` 生成匿名化 ID 图片副本和 train/val/test manifests。
3. 下载 APTOS/RFMiD/IRFundusSet colour fundus，选 50-200 张放到 `data/images/ood_modality/colour_fundus/`。
4. 下载或导出 OLIVES near-IR 图片，放到 `data/images/ood_modality/infrared/`。
5. 选 CIFAR-10/Open Images 小子集，放到 `data/images/ood_semantic/natural/`。
6. 所有图片最好统一成 `.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff` 或 `.bmp`。

如果下载到的是 DICOM、NIfTI、PDF、parquet 或 Hugging Face dataset 格式，先导出成普通图片再写 manifest。

SynthEye 的 10 个 class folder 只用于 stratified split/audit，不作为疾病分类标签：

```bash
python scripts/prepare_syntheye_dataset.py \
  --input-dir "<path-to-syntheye_onefold_10class_100perclass>" \
  --out-dir data/images/synthetic_faf \
  --manifest-dir data/manifests \
  --seed 42
```

如果你需要生成 watermark/text/annotation/composite/blur/compression 这类 sensory artifact，可以从 held-out valid FAF manifest 生成：

```bash
python scripts/generate_artifacts.py \
  --input-manifest data/manifests/test_real_id.csv \
  --root-dir data \
  --out-dir data/images/ood_artifact \
  --out-manifest data/manifests/test_artifact.csv \
  --split test \
  --seed 42 \
  --artifacts text_watermark rectangle_annotation arrow_annotation composite_layout blur_artifact border_crop gaussian_noise jpeg_compression
```

从文件夹生成 wrong-modality / semantic-outlier manifests：

```bash
python scripts/build_ood_manifest.py \
  --root-dir data \
  --out-manifest data/manifests/test_modality.csv \
  --mapping images/ood_modality/colour_fundus=modality_shift \
  --mapping images/ood_modality/infrared=modality_shift \
  --mapping images/ood_modality/oct_screenshot=modality_shift

python scripts/build_ood_manifest.py \
  --root-dir data \
  --out-manifest data/manifests/test_semantic.csv \
  --mapping images/ood_semantic/natural=semantic_outlier \
  --mapping images/ood_semantic/non_retinal_medical=semantic_outlier
```

合并最终 OOD manifest：

```bash
python scripts/merge_manifests.py \
  --out data/manifests/test_ood.csv \
  data/manifests/test_modality.csv \
  data/manifests/test_semantic.csv \
  data/manifests/test_artifact.csv
```

不要从 train FAF 生成 test artifact，避免数据泄漏。

### Day 2：创建 manifest 并跑 smoke experiment

先用少量图片确认路径和格式没有问题：

```bash
python scripts/train_patchcore.py --config configs/patchcore_l23.yaml
python scripts/evaluate.py --config configs/patchcore_l23.yaml --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz --save-heatmaps --heatmap-top-k 2
```

如果报 missing files，优先检查 manifest 的 `image_path` 是否相对 `data.root_dir` 写对了。

### Day 3：跑 PatchCore L2+L3 主实验

```bash
python scripts/train_patchcore.py --config configs/patchcore_l23.yaml
python scripts/evaluate.py \
  --config configs/patchcore_l23.yaml \
  --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz \
  --save-heatmaps \
  --heatmap-top-k 5
```

主要输出：

```text
runs/patchcore_resnet50_layer2_layer3/
  patchcore_memory.npz
  resolved_config.json
  training_metrics.json
  evaluation/
    scores.csv
    metrics.json
    roc_curve.png
    pr_curve.png
    heatmaps/
```

同时跑 autoencoder baseline：

```bash
python scripts/train_autoencoder.py --config configs/autoencoder_baseline.yaml
python scripts/evaluate_autoencoder.py \
  --config configs/autoencoder_baseline.yaml \
  --checkpoint runs/autoencoder_baseline/model.pt
```

Autoencoder baseline 会生成 `scores.csv`、`metrics.json`、ROC/PR 曲线，但不会生成 PatchCore patch heatmap。

### Day 3-4：跑 layer ablation

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

### Day 4：生成表格和论文图

```bash
python scripts/generate_report_tables.py --runs-dir runs --out reports/generated/experiment_summary.md
python scripts/generate_dissertation_figures.py --runs-dir runs --out-dir reports/generated/figures
```

重点查看：

- `reports/generated/experiment_summary.md`
- `reports/generated/per_category_metrics.md`
- `reports/generated/figures/`

### Day 5：启动拖拽式 UI 做定性展示

先确保主实验已经跑过 evaluation，因为 UI 会读取 `metrics.json` 里的 threshold。

```bash
python scripts/serve_gatekeeper_app.py \
  --config configs/patchcore_l23.yaml \
  --checkpoint runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz
```

打开：

```text
http://127.0.0.1:7860
```

支持拖入多个 `.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff`、`.bmp` 文件。UI 会显示：

- `ACCEPT` / `REJECT`
- anomaly score
- threshold
- PatchCore overlay
- 可导出的 UI prediction CSV

## 6. Jackpot 上怎么跑

建议目录：

```text
/scratch/<your_user>/retinal_ood_data/
  images/
  manifests/

/scratch/<your_user>/retinal_ood_runs/
```

把 config 改成：

```yaml
data:
  root_dir: /scratch/<your_user>/retinal_ood_data
  train_manifest: manifests/train_synthetic_faf.csv
  val_manifest: manifests/val_synthetic_faf.csv
  test_id_manifest: manifests/test_real_id.csv
  test_ood_manifest: manifests/test_ood.csv

output:
  runs_dir: /scratch/<your_user>/retinal_ood_runs
```

如果通过 SSH tunnel 访问 UI：

```bash
python scripts/serve_gatekeeper_app.py \
  --config configs/patchcore_l23.yaml \
  --checkpoint /scratch/<your_user>/retinal_ood_runs/patchcore_resnet50_layer2_layer3/patchcore_memory.npz \
  --host 127.0.0.1 \
  --port 7860
```

在你的本地电脑开 tunnel：

```bash
ssh -L 7860:127.0.0.1:7860 <your_user>@<jackpot-host>
```

然后本地浏览器打开 `http://127.0.0.1:7860`。除非学校环境明确保护端口，否则不要使用 `--host 0.0.0.0`。

## 7. 结果怎么解读

核心文件是每个 run 的 `evaluation/metrics.json`：

- `global.auroc`：整体区分 ID/OOD 的能力。
- `global.auprc`：OOD 为正类时的 precision-recall 表现。
- `global.fpr_at_95_tpr`：在 95% OOD recall 下误拒 ID 的比例。
- `confusion_matrix.fp`：valid FAF 被误拒，论文里要分析 synthetic-to-real gap。
- `confusion_matrix.fn`：OOD 被放行，安全风险最高。
- `per_ood_type`：看 colour fundus、IR、artifact、natural image 哪类最难。

热力图只能解释“feature-space deviation”，不能解释为病灶定位。

## 8. 隐私和提交规则

可以提交：

- 代码。
- 文档。
- toy tests。
- 没有隐私信息的 aggregate plots / tables。

不要提交：

- 真实医疗图片。
- patient_id 或可识别信息。
- 私有下载链接。
- 本地绝对路径。
- 训练好的私有模型权重或 memory bank。
- 认证凭据。
