---
name: sanger-qc
description: "Sanger sequencing QC and mutation analysis. Aligns AB1 sequencing files against GenBank references, detects mutations, and judges whether sample genes are correct. Use when: user mentions analyzing sequencing results, Sanger QC, checking plasmids, base/pro/promax datasets, AB1 files, or plasmid verification. NOT for: NGS, RNA-seq, general genome annotation, or protein structure prediction."
metadata:
  {
    "openclaw":
      {
        "emoji": "🧬",
        "requires": { "bins": ["python"] },
      },
  }
---

# Sanger 测序质控分析

自动化 Sanger 测序质控管道。数据已准备好，直接运行即可。

## When to Use

✅ **USE this skill when:**

- "分析 base 数据集" / "分析 Sanger 测序结果"
- "帮我看看测序结果对不对" / "检查质粒"
- "运行 sanger_qc" / "跑一下测序质控"
- "base/pro/promax 数据集分析"
- 任何涉及 Sanger 测序、AB1 文件、质粒验证的请求

## When NOT to Use

❌ **DON'T use this skill when:**

- NGS / RNA-seq 分析
- 通用基因组注释
- 蛋白结构预测

## Commands

**重要：所有命令必须在工作区根目录执行。数据文件已就绪，无需用户上传。**

### 完整分析（生物信息学 + AI 判读）

```bash
# base 数据集
cd {baseDir}/../.. && python run.py --dataset base

# pro 数据集
cd {baseDir}/../.. && python run.py --dataset pro

# promax 数据集
cd {baseDir}/../.. && python run.py --dataset promax
```

### 仅生物信息学分析（跳过 AI，不消耗 API 额度）

```bash
cd {baseDir}/../.. && python run.py --dataset base --no-llm
```

### 指定 LLM 模型

```bash
cd {baseDir}/../.. && python run.py --dataset base --model gpt-4.1-mini
```

## How to Handle User Requests

| 用户说 | 你应该执行 |
|--------|-----------|
| "分析 base 数据集" | `python run.py --dataset base` |
| "分析 pro 数据集" | `python run.py --dataset pro` |
| "分析测序结果" / "跑一下质控" | 询问数据集（base/pro/promax），然后执行 |
| "只做比对，不用 AI" | 加 `--no-llm` 参数 |
| "用 gpt-4.1-mini 分析" | 加 `--model gpt-4.1-mini` 参数 |

## Output

运行后会在 `outputs/<dataset>/` 目录生成：

| 文件 | 内容 |
|------|------|
| `evidence.txt` | 生物信息学分析汇总表（比对结果、突变列表） |
| `result.txt` | 每个样本的最终判读（ok / wrong + 原因） |
| `llm_raw.txt` | AI 原始回复（使用 LLM 时才有） |
| `html/` | 每个样本的碱基级比对可视化 |

## Presenting Results

运行完成后：

1. **读取 `outputs/<dataset>/result.txt`** 并展示给用户
2. 如有失败样本，简要说明原因（突变、移码、测序质量差等）
3. 如用户需要详细信息，读取 `outputs/<dataset>/evidence.txt`
4. 结果格式示例：
   ```
   C123-1 gene is ok
   C456-2 gene is wrong S334L K171M
   C789-1 gene is wrong 移码错误
   ```

## Judgment Rules（供参考）

1. identity ≈ 1.0 且 cds_coverage > 0.55 且无移码且无 AA 变化 → **ok**
2. 有 AA 变异且 identity > 0.95 且变异数量少 (1-5) → **wrong** + 列出变异
3. 有移码 (frameshift) 且 cds_coverage > 0.55 → **wrong 移码错误**
4. cds_coverage < 0.55 且无变异无移码 → **ok 未测通**
5. identity < 0.85 且大量 AA 变异 (>40) → **wrong 重叠峰，比对失败**
6. identity < 0.85 且 AA 变异 25-40 → **wrong 重叠峰**
7. AA 变异集中在连续区域 → **wrong 片段缺失**
8. identity < 0.30 或 seq_length < 50 → **wrong 测序失败**

## Notes

- 数据文件已在 `data/` 目录中，无需用户另外上传
- 需要 `.env` 中配置 `LLM_API_KEY` 才能使用 AI 判读（`--no-llm` 可跳过）
- 依赖：biopython >= 1.80, pandas >= 1.5, openai >= 1.0
