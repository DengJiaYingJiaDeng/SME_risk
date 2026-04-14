# 中小型企业信贷评估与动态违约预警（Dynamic Cox）

本项目实现：
- 分行业放贷决策（Approve / Reject）
- 放贷后动态违约预警（RED / AMBER / GREEN）
- 动态 Cox 风险建模
- 论文级评估指标与图表（C-index、ROC-AUC、KS、时变AUC、Brier、生存分层图、局部解释、策略折中曲线）

## 1. 数据与默认设置
- 数据目录：`E:\my model\SME_model\SME - Synthetic UK Businesses financial statistics`
- 默认贷款文件：`Loan_Augmented_RealSignals.csv`
- 如需回到原始贷款数据：`--loan-file Loan.csv`

## 2. 运行实验
```bash
python run_experiment.py
```

示例：
```bash
python run_experiment.py --loan-file Loan_Augmented_RealSignals.csv --evaluation-horizons 30,90,180,360
```

## 3. 扩充前后对比
```bash
python run_before_after_comparison.py
```

输出目录：`E:\my model\SME_model\sme_dynamic_cox_project\outputs\comparison`

关键文件：
- `before_vs_after_metrics.csv`
- `before_vs_after_metrics.md`
- `comparison_summary.json`

## 4. 新企业预测
使用新贷款申请 CSV 进行评分：
```bash
python predict_new_sme.py --new-loan-file "E:\path\to\new_loans.csv"
```

可选增量数据：
- `--new-business-file`
- `--new-credit-rating-file`
- `--new-credit-account-file`
- `--new-factoring-file`
- `--new-credit-card-file`

输出：`outputs/new_sme_predictions.csv`

## 5. 主要输出文件
- `metrics.json`
- `time_dependent_metrics.csv`
- `strategy_tradeoff_curve.csv`
- `stratified_survival_summary.csv`
- `local_explanations.csv`
- `partial_dependence_data.csv`
- `top_coefficients.png`
- `time_dependent_metrics_curve.png`
- `stratified_survival_curve.png`
- `strategy_tradeoff_curve.png`
- `local_explain_waterfall_*.png`
- `partial_dependence_plots.png`

