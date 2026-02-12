# BioAgent — Sanger 测序质控与突变分析工具

BioAgent 是一个两阶段的 Sanger 测序分析流水线：

1. **生物信息学分析** — 将 AB1 测序文件与 GenBank 参考序列进行比对，检测碱基突变、氨基酸变异和移码
2. **AI 判读** — 将比对结果交给 Claude 大模型，综合判断每个样本的基因是否正确（ok/wrong）

## 项目结构

```
BioAgent-release/
├── .env                  # API 配置文件（需要自行填写）
├── requirements.txt      # Python 依赖
├── run.py                # 主程序入口
├── core/
│   ├── alignment.py      # 生物信息学核心：序列比对、突变检测、氨基酸翻译
│   ├── evidence.py       # 将比对结果格式化为文本，供 AI 分析
│   └── llm_client.py     # Claude API 调用与结果解析
└── data/                 # 数据目录（需要自行放入）
    ├── gb/               # base 数据集的 GenBank 参考序列（.gb/.gbk）
    ├── ab1_files/        # base 数据集的 AB1 测序文件
    ├── gb_pro/           # pro 数据集的参考序列
    ├── ab1_files_pro/    # pro 数据集的测序文件
    ├── gb_promax/        # promax 数据集的参考序列
    └── ab1_files_promax/ # promax 数据集的测序文件
```

## 各模块说明

| 文件 | 功能 |
|---|---|
| `run.py` | 主入口。解析命令行参数，依次调用生物信息学分析和 AI 判读，输出结果 |
| `core/alignment.py` | 读取 GenBank 参考序列和 AB1 测序文件，执行双向（正向/反向互补）局部比对，计算 identity、CDS 覆盖率，检测碱基替换/插入/缺失、氨基酸变异、移码突变，并生成 HTML 可视化比对图 |
| `core/evidence.py` | 将 `alignment.py` 产出的结构化数据格式化为文本摘要，作为 AI 判读的输入 |
| `core/llm_client.py` | 封装 Claude API 的调用逻辑，包含质控判读的 System Prompt 和响应解析 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要 Python 3.10+。

### 2. 配置 API

编辑项目根目录下的 `.env` 文件：

```env
# 填写你的 API Key
ANTHROPIC_API_KEY=sk-xxxxxxx

# 如果你使用中转站/代理服务（非 Anthropic 官方 API），取消下面的注释并填写地址
# ANTHROPIC_BASE_URL=https://your-proxy-url.com
```

**说明：**
- 如果你使用的是 Anthropic **官方 API**，只需填写 `ANTHROPIC_API_KEY` 即可
- 如果你使用的是**中转站**（第三方代理），需要同时填写 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_BASE_URL`

### 3. 准备数据

程序通过 `data/` 下的**固定子文件夹名**来区分不同数据集。你需要手动创建这些子文件夹并放入对应文件：

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
# 分析 base 数据集（默认使用 claude-sonnet-4-5-20250929 模型）
python run.py --dataset base

# 分析 pro 数据集
python run.py --dataset pro

# 分析 promax 数据集
python run.py --dataset promax

# 指定其他模型
python run.py --dataset base --model claude-opus-4-6

# 仅运行生物信息学分析，跳过 AI 判读（不消耗 API 额度）
python run.py --dataset base --no-llm

# 指定输出目录
python run.py --dataset base --output-dir ./my_output
```

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
