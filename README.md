# 中小型企业信贷评估与动态违约预警（动态 Cox 模型）

本项目针对课题《中小型企业信贷评估和违约风险预测》，实现了以下实验目标：

1. 分行业对 SME 贷款对象进行风险评估，形成“是否放贷”决策建议。
2. 对已放贷企业进行月度动态监测，输出未来 90 天违约预警（RED/AMBER/GREEN）。
3. 使用时间变化协变量的动态 Cox（`lifelines.CoxTimeVaryingFitter`）作为核心模型。

## 1. 数据输入

默认读取目录：

`E:\my model\SME_model\SME - Synthetic UK Businesses financial statistics`

核心表包括：

- `Loan.csv`（贷款、违约时间、还款状态）
- `Businesses.csv`（企业静态财务）
- `Credit_Rating.csv`（征信评分）
- `Credit_Account_History.csv`（账户历史）
- `Credit_Card_History.csv`（信用卡历史）
- `Factoring.csv`（保理信息）
- `Combined_macro_data.csv`（宏观月度变量）

## 2. 方法框架

### 2.1 动态 Cox 建模

- 以每笔贷款为个体（`loan_key`），按月展开为 start-stop 生存区间。
- 事件定义：`loan_status = Defaulted` 且存在 `loan_default_date`。
- 右删失：到 `loan_satisfaction_date` / `loan_date_due_to_close` / `snapshot_date` 的最早时间。
- 按行业 `industry_group` 进行 `strata`，允许不同行业有不同基准风险。

### 2.2 特征构建

- 静态财务：营收、成本、资产负债、流动比率、杠杆率等。
- 征信时序：截至当月最新征信（as-of merge）。
- 账户与信用卡时序：累计开户数、累计交易规模、逾期相关特征。
- 宏观时序：利率、CPI、GDP 及其变化率。

### 2.3 业务输出

- 行业放贷策略：训练集内按“收益-损失”目标自动寻优阈值。
- 动态预警策略：训练集内按 F1 自动寻优行业阈值，输出风险等级。

## 3. 运行方式

在项目目录下执行：

```bash
python run_experiment.py
```

可选参数示例：

```bash
python run_experiment.py --snapshot-date 2024-02-29 --warning-horizon-days 90 --test-size 0.3 --penalizer 0.25
```

## 4. 输出文件

输出目录默认：`E:\my model\SME_model\sme_dynamic_cox_project\outputs`

- `metrics.json`：整体实验指标（C-index、预警 AUC/F1 等）
- `experiment_summary.txt`：可直接用于实验小结
- `model_coefficients.csv`：模型系数
- `top_coefficients.png`：关键特征可视化
- `industry_approval_policy.csv`：行业放贷阈值策略
- `loan_decisions.csv`：每笔贷款的放贷建议（Approve/Reject）
- `industry_decision_summary.csv`：行业层面决策统计
- `industry_warning_policy.csv`：行业预警阈值策略
- `dynamic_warning_full.csv`：全量动态预警记录
- `dynamic_warning_latest.csv`：每笔贷款最新预警状态

## 5. 项目结构

```text
sme_dynamic_cox_project/
├─ run_experiment.py
├─ requirements.txt
├─ README.md
├─ outputs/
└─ src/sme_dynamic_cox/
   ├─ config.py
   ├─ data_io.py
   ├─ feature_builder.py
   ├─ model.py
   ├─ policy.py
   ├─ evaluation.py
   └─ pipeline.py
```

