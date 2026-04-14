# 中小型企业信贷评估与动态违约预警（动态 Cox）

本项目用于课题《中小型企业信贷评估和违约风险预测》，实现：
- 分行业放贷决策（Approve / Reject）
- 放贷后动态违约预警（RED / AMBER / GREEN）
- 基于时间变化协变量的动态 Cox 建模

## 1. 数据
默认数据目录：
`E:\my model\SME_model\SME - Synthetic UK Businesses financial statistics`

当前训练管线已默认切换为扩充后贷款文件：
- `Loan_Augmented_RealSignals.csv`

如需使用原始贷款文件，可显式传参：
- `--loan-file Loan.csv`

## 2. 一键运行
在项目根目录执行：

```bash
python run_experiment.py
```

常用参数：

```bash
python run_experiment.py --loan-file Loan_Augmented_RealSignals.csv --output-dir "E:\my model\SME_model\sme_dynamic_cox_project\outputs\augmented"
```

## 3. 扩充前后对比
运行下面命令自动完成“扩充前 vs 扩充后”两组实验并输出对比表：

```bash
python run_before_after_comparison.py
```

对比结果目录：
`E:\my model\SME_model\sme_dynamic_cox_project\outputs\comparison`

关键文件：
- `before_vs_after_metrics.csv`
- `before_vs_after_metrics.md`
- `comparison_summary.json`

## 4. 核心输出
每次实验会产出：
- `metrics.json`
- `experiment_summary.txt`
- `industry_approval_policy.csv`
- `loan_decisions.csv`
- `industry_warning_policy.csv`
- `dynamic_warning_full.csv`
- `dynamic_warning_latest.csv`

