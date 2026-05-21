# Risk Register：项目风险和应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 没有足够真实 FAF 测试数据 | 无法评估 sim-to-real | 明确写 limitation；使用可获得的真实/公开数据；做 synthetic-real 特征可视化 |
| PatchCore memory bank 太大 | 运行慢/内存高 | coreset_ratio；max_train_patches；CPU/GPU 分批 |
| AE baseline 效果也不错 | 结论不明显 | 分 OOD 类型分析，重点看 watermark/local artifact |
| AUROC 很高但实际误拒严重 | 临床意义不足 | 必须报告 AUPRC 和 FPR@95%TPR |
| 训练数据和测试数据泄漏 | 结果无效 | patient_id/original_id 去重，artifact 不从 train 生成 |
| Codex 改坏代码 | 时间损失 | 单任务 PR、小步提交、测试、review |
| GitHub 泄露数据 | 严重合规问题 | .gitignore、manifest template、永不上传图片/权重/私有路径 |
| Heatmap 模糊 | 临床解释力弱 | 诚实讨论 limitation；使用 multi-scale 或 patch overlay |
