# GitHub Workflow：如何持续更新这个毕设项目

## 1. 推荐分支

```text
main        # 稳定版本，只合并通过测试的代码
dev         # 日常开发整合分支
feature/*   # 单个功能或实验
```

## 2. 推荐 Issue 分类

- `data`: 数据读取、manifest、预处理。
- `model`: PatchCore、AE、PaDiM、FastFlow。
- `eval`: metrics、threshold、plots。
- `viz`: heatmaps、figures。
- `docs`: README、报告、论文草稿。
- `bug`: 代码或实验错误。
- `privacy`: 数据泄漏检查。

## 3. Commit message 模板

```text
feat(model): implement patchcore memory bank
fix(metrics): correct fpr at 95 tpr thresholding
docs(report): add layer ablation interpretation
chore(ci): add pytest workflow
```

## 4. 每周更新节奏

### Week update template

```markdown
## Week X Update

### Completed
- ...

### Experiments run
- Model/config:
- Dataset split:
- Main metrics:

### Problems
- ...

### Next week
- ...
```

## 5. Pull Request checklist

```markdown
- [ ] No data or private paths committed.
- [ ] Configs are reproducible.
- [ ] Tests pass.
- [ ] Metrics are saved to JSON/CSV.
- [ ] README/docs updated if needed.
- [ ] Results are interpretable.
```

## 6. Codex 使用方式

推荐把每个 `docs/CODEX_TASKS.md` 中的任务复制成 GitHub Issue，然后让 Codex 按单个 issue 工作。不要一次让 Codex 完成整个毕设；要拆成小 PR，例如：

- PR 1: dataset loader。
- PR 2: metrics。
- PR 3: autoencoder。
- PR 4: PatchCore feature extractor。
- PR 5: PatchCore memory bank。
- PR 6: evaluation CLI。
- PR 7: heatmaps。
- PR 8: experiment runner。

这样每一步都能 review、测试、回滚。
