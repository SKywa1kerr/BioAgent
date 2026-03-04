# BioAgent — Sanger 测序质控与突变分析工具

BioAgent 是一个两阶段的 Sanger 测序分析流水线：

1. **生物信息学分析** — 将 AB1 测序文件与 GenBank 参考序列进行比对，检测碱基突变、氨基酸变异和移码
2. **AI 判读** — 将比对结果交给大模型，综合判断每个样本的基因是否正确（ok/wrong）

## 项目结构

```
BioAgent/
├── .env                  # API 配置文件（需要自行填写）
├── requirements.txt      # Python 依赖
├── run.py                # 主程序入口
├── core/
│   ├── alignment.py      # 生物信息学核心：序列比对、突变检测、氨基酸翻译
│   ├── evidence.py       # 将比对结果格式化为文本，供 AI 分析
│   └── llm_client.py     # LLM API 调用与结果解析（兼容 OpenAI 格式）
└── data/                 # 数据目录（需要自行放入）
    ├── gb/               # base 数据集的 GenBank 参考序列（.gb/.gbk）
    ├── ab1_files/        # base 数据集的 AB1 测序文件
    ├── gb_pro/           # pro 数据集的参考序列
    ├── ab1_files_pro/    # pro 数据集的测序文件
    ├── gb_promax/        # promax 数据集的参考序列
    └── ab1_files_promax/ # promax 数据集的测序文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要 Python 3.10+。依赖包：`biopython`、`pandas`、`openai`。

### 2. 配置 API

编辑项目根目录下的 `.env` 文件。支持任何 OpenAI 兼容的 API 服务。

**推荐：使用 OpenRouter（有免费模型）**

1. 注册 [openrouter.ai](https://openrouter.ai) 获取 API Key
2. 编辑 `.env`：

```env
LLM_API_KEY=sk-or-v1-xxxxxxx
LLM_BASE_URL=https://openrouter.ai/api/v1
```

> 注册即可使用免费模型（如 `google/gemma-3-27b-it:free`），无需充值。
> 充值任意金额（$1 起）可提升免费模型的每日请求上限（20 次 → 200 次），充值金额仅在使用付费模型时扣除。

**其他 API 服务：**

```env
# DeepSeek
LLM_API_KEY=sk-xxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1

# Anthropic 中转站
LLM_API_KEY=sk-xxxxxxx
LLM_BASE_URL=https://your-proxy.com/v1
```

### 3. 准备数据

将数据文件放入 `data/` 的对应子文件夹中：

| `--dataset` 参数 | 参考序列目录 | 测序文件目录 |
|------------------|-------------|------------|
| `base`           | `data/gb/`  | `data/ab1_files/` |
| `pro`       | `data/gb_pro/` | `data/ab1_files_pro/` |
| `promax` | `data/gb_promax/` | `data/ab1_files_promax/` |

**文件夹名是硬编码的，不能随意更改。** 如果你只有一组数据，放到 `data/gb/` 和 `data/ab1_files/` 里，用 `--dataset base` 运行即可。

每个数据集需要两类文件：
- **GenBank 参考序列**（`.gb` 或 `.gbk`）— 放在对应的 `gb*` 目录
- **AB1 测序文件**（`.ab1`）— 放在对应的 `ab1_files*` 目录

### 4. 运行

```bash
# 使用默认免费模型运行
python run.py --dataset base

# 指定其他模型
python run.py --dataset base --model meta-llama/llama-3.3-70b-instruct:free

# 仅运行生物信息学分析，跳过 AI 判读（不消耗 API 额度）
python run.py --dataset base --no-llm

# 指定输出目录
python run.py --dataset base --output-dir ./my_output
```

**命令行参数：**

| 参数 | 说明 |
|------|------|
| `--dataset` | 必填，`base` / `pro` / `promax` |
| `--model` | 模型名，默认 `google/gemma-3-27b-it:free` |
| `--output-dir` | 输出目录，默认 `outputs/<dataset>` |
| `--no-llm` | 跳过 LLM 阶段，只输出生物信息学分析结果 |

### 5. 输出

运行结果默认保存在 `outputs/<数据集名>/` 目录下：

| 文件 | 内容 |
|---|---|
| `evidence.txt` | 生物信息学分析的汇总表格 |
| `llm_raw.txt` | AI 的原始回复 |
| `result.txt` | 最终判读结果（每个样本一行） |
| `html/` | 每个样本的碱基级比对可视化（HTML 格式，可用浏览器打开） |

结果格式示例：

```
C123-1 gene is ok
C456-2 gene is wrong S334L K171M
C789-1 gene is wrong 移码错误
```

## 可用的免费模型（OpenRouter）

以下模型均可免费使用，推荐按能力从高到低选择：

| 模型 ID | 参数量 | 备注 |
|---------|--------|------|
| `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | 最强，限流较严 |
| `qwen/qwen3-coder:free` | 480B(A35B) | 中文理解好 |
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | 综合能力强 |
| `google/gemma-3-27b-it:free` | 27B | 默认模型，稳定可用 |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 24B | 速度快 |

> 免费模型有每日请求次数限制，被限流时程序会自动等待重试。

## 各模块说明

| 文件 | 功能 |
|---|---|
| `run.py` | 主入口。解析命令行参数，依次调用生物信息学分析和 AI 判读，输出结果 |
| `core/alignment.py` | 读取 GenBank 参考序列和 AB1 测序文件，执行双向（正向/反向互补）局部比对，计算 identity、CDS 覆盖率，检测碱基替换/插入/缺失、氨基酸变异、移码突变，并生成 HTML 可视化比对图 |
| `core/evidence.py` | 将 `alignment.py` 产出的结构化数据格式化为文本摘要，作为 AI 判读的输入 |
| `core/llm_client.py` | 封装 LLM API 的调用逻辑（兼容任何 OpenAI 格式的 API），包含质控判读的 System Prompt、429 限流自动重试、system prompt 不兼容时自动降级 |
