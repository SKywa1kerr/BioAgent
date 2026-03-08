---
name: sanger-qc
description: |
  Sanger 测序质控与突变分析。将 AB1 测序文件与 GenBank 参考序列比对，
  检测碱基突变、氨基酸变异和移码，并由 AI 综合判断每个样本的基因是否正确。
  当用户需要分析 Sanger 测序结果、检查质粒是否正确时触发。
metadata:
  openclaw:
    emoji: "\U0001F9EC"
    requires:
      bins:
        - python
      env:
        - LLM_API_KEY
---

# Sanger 测序质控分析

对 Sanger 测序结果进行自动化质控与突变分析。

## 使用方法

在项目根目录执行 `python run.py`，通过参数控制分析行为：

```bash
# 基础数据集，含 LLM 判读
python run.py --skill sanger_qc --dataset base

# 仅生物信息学分析，跳过 LLM
python run.py --skill sanger_qc --dataset base --no-llm

# 指定模型和输出目录
python run.py --skill sanger_qc --dataset pro --model gpt-5 --output-dir ./my_output
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--skill` | 否 | 要运行的 skill，默认 `sanger_qc` |
| `--dataset` | 是 | 数据集名称：`base`、`pro`、`promax` |
| `--model` | 否 | LLM 模型名称，省略则自动检测 |
| `--output-dir` | 否 | 输出目录，默认 `outputs/<dataset>` |
| `--no-llm` | 否 | 跳过 LLM 判读，仅输出生物信息学结果 |

## 输入

- **GenBank 参考序列** (`.gb` / `.gbk`) — 质粒的理论正确序列
- **AB1 测序文件** (`.ab1`) — 测序公司返回的实际测序数据

## 输出

- `evidence.txt` — 生物信息学分析汇总表
- `result.txt` — 每个样本的最终判读（ok / wrong + 原因）
- `llm_raw.txt` — AI 原始回复（使用 LLM 时）
- `html/` — 碱基级比对可视化

## 判读规则

1. identity ≈ 1.0 且 cds_coverage > 0.55 且无移码且无 AA 变化 → **ok**
2. 有 AA 变异且 identity > 0.95 且变异数量少 (1-5) → **wrong** + 列出变异
3. 有移码 (frameshift) 且 cds_coverage > 0.55 → **wrong 移码错误**
4. cds_coverage < 0.55 且无变异无移码 → **ok 未测通**
5. identity < 0.85 且大量 AA 变异 (>40) → **wrong 重叠峰，比对失败**
6. identity < 0.85 且 AA 变异 25-40 → **wrong 重叠峰**
7. AA 变异集中在连续区域 → **wrong 片段缺失**
8. identity < 0.30 或 seq_length < 50 → **wrong 测序失败**

## 依赖

- biopython >= 1.80
- pandas >= 1.5
- openai >= 1.0
