# BioAgent — Sanger 测序质控分析平台

基于 [OpenClaw](https://openclaw.org) Skill 架构的 **Sanger 测序自动化质控管道**：AB1 测序文件 + GenBank 参考序列 → 生物信息学比对 → AI 智能判读 → 样本基因是否正确。

## 架构总览

```
                     run.py (入口调度)
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
     [Stage 1: 生信分析]      [Stage 2: AI 判读]
      alignment.py              llm_judge.py
      evidence.py                    │
             │               ┌───────┴───────┐
             ▼               ▼               ▼
       evidence.txt     OpenAI 兼容      Anthropic
       html/ 可视化    (DeepSeek 等)    (anyrouter)
                             │
                             ▼
                        result.txt
```

**两阶段设计**：Stage 1 纯生物信息学（确定性算法），Stage 2 由 LLM 综合判读（可选，`--no-llm` 可跳过）。

## 项目结构

```
BioAgent/
├── run.py                      # 入口调度器
├── requirements.txt            # Python 依赖
├── .env                        # API 配置（填入你的 key）
├── .env.example                # 配置模板
│
├── skills/                     # Skill 模块目录
│   └── sanger_qc/              # Sanger 测序质控技能
│       ├── SKILL.md            # 技能元数据 & OpenClaw 触发规则
│       ├── alignment.py        # 核心算法：比对、突变检测、AA 翻译、HTML 可视化
│       ├── evidence.py         # 结果格式化（LLM 输入 & 人类可读表格）
│       └── llm_judge.py        # LLM API 客户端（OpenAI/Anthropic 双格式）
│
├── data/                       # 输入数据
│   ├── gb/  +  ab1_files/             # base 数据集
│   ├── gb_pro/  +  ab1_files_pro/     # pro 数据集
│   └── gb_promax/  +  ab1_files_promax/  # promax 数据集
│
├── truth/                      # 真值数据（用于准确率验证）
├── outputs/                    # 运行结果输出
│
├── AGENTS.md                   # OpenClaw agent 行为准则
├── SOUL.md                     # OpenClaw agent 身份定义
└── .openclaw/                  # OpenClaw 工作区状态
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要 Python 3.10+。依赖：`biopython`、`pandas`、`openai`。

### 2. 配置 LLM API

编辑 `.env` 文件，填入 API Key。支持以下服务：

**ChatAnywhere（推荐国内用户，免费，国内直连）**

```env
LLM_API_KEY=sk-xxxxxxx
LLM_BASE_URL=https://api.chatanywhere.tech/v1
```

> 前往 [chatanywhere/GPT_API_free](https://github.com/chatanywhere/GPT_API_free) 免费领取 Key，200 次/天。

**OpenRouter（推荐海外用户，有免费模型）**

```env
LLM_API_KEY=sk-or-v1-xxxxxxx
LLM_BASE_URL=https://openrouter.ai/api/v1
```

> 注册 [openrouter.ai](https://openrouter.ai) 即可使用免费模型，无需充值。

**DeepSeek / 其他 OpenAI 兼容服务**

```env
LLM_API_KEY=sk-xxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
```

**Anthropic 中转（如 anyrouter）**

```env
LLM_API_KEY=sk-xxxxxxx
LLM_BASE_URL=https://anyrouter.top/v1
LLM_API_FORMAT=anthropic
```

### 3. 准备数据

将文件放入 `data/` 对应目录：

| `--dataset` | GenBank 参考序列 | AB1 测序文件 |
|-------------|-----------------|-------------|
| `base` | `data/gb/` | `data/ab1_files/` |
| `pro` | `data/gb_pro/` | `data/ab1_files_pro/` |
| `promax` | `data/gb_promax/` | `data/ab1_files_promax/` |

每个数据集需要：
- **GenBank 文件**（`.gb` / `.gbk`）— 质粒的理论正确序列
- **AB1 文件**（`.ab1`）— 测序公司返回的实际测序数据

### 4. 运行

```bash
# 完整分析（生信 + AI 判读）
python run.py --dataset base

# 仅生物信息学分析（不消耗 API 额度）
python run.py --dataset base --no-llm

# 指定模型
python run.py --dataset base --model gpt-4.1-mini

# 指定输出目录
python run.py --dataset pro --output-dir ./my_output
```

| 参数 | 说明 |
|------|------|
| `--dataset` | **必填**，`base` / `pro` / `promax` |
| `--model` | LLM 模型名，默认 `deepseek-v3` |
| `--no-llm` | 跳过 AI 判读阶段 |
| `--output-dir` | 输出目录，默认 `outputs/<dataset>` |

### 5. 查看结果

输出在 `outputs/<dataset>/` 目录：

| 文件 | 内容 |
|------|------|
| `evidence.txt` | 生物信息学分析汇总表 |
| `result.txt` | 最终判读（每样本一行） |
| `llm_raw.txt` | AI 原始回复（使用 LLM 时） |
| `html/` | 碱基级比对可视化（浏览器打开） |

结果格式：

```
C373-2 gene is ok
C397-a gene is wrong S334L
C379-2 gene is wrong Q131T
C789-1 gene is wrong 移码错误
```

## 通过 OpenClaw 部署使用

[OpenClaw](https://openclaw.org) 是一个 AI agent 框架，可以把 BioAgent 作为一个 Skill 注册，让 agent 通过自然语言自动调用分析管道。

### 1. 安装 OpenClaw

```bash
npm install -g openclaw
```

### 2. 启动 Gateway

```bash
openclaw gateway
```

Gateway 是 OpenClaw 的本地服务端，负责管理 agent、skill 和 LLM 调用。启动后默认监听 `localhost:18789`。

> 如需配置代理（科学上网），可编辑 `~/.openclaw/gateway.cmd`，添加：
> ```
> set HTTP_PROXY=http://127.0.0.1:你的代理端口
> set HTTPS_PROXY=http://127.0.0.1:你的代理端口
> ```

### 3. 配置 LLM Provider

OpenClaw 需要自己的 LLM 来驱动 agent（这与 `.env` 中给 skill 用的 API 是独立的两套配置）。

编辑 `~/.openclaw/openclaw.json`，在 `models.providers` 中添加 provider：

```json
{
  "models": {
    "primary": "你的provider名/模型名",
    "providers": {
      "anyrouter": {
        "baseUrl": "https://anyrouter.top",
        "apiType": "anthropic-messages",
        "apiKey": "sk-xxxxxxx",
        "models": ["claude-opus-4-6", "claude-sonnet-4-6"]
      },
      "sjtu": {
        "baseUrl": "https://models.sjtu.edu.cn/api/v1",
        "apiType": "openai-completions",
        "apiKey": "sk-xxxxxxx",
        "models": ["deepseek-v3", "deepseek-r1"]
      }
    }
  }
}
```

> **注意**：Gateway 的 LLM 是 agent 的"大脑"，用于理解指令和编排 skill；Skill 内部的 LLM（`.env` 配置）用于生物信息学判读。两者独立配置。

### 4. 注册 Skill

将 sanger_qc 技能链接到 OpenClaw 的 skills 目录：

```bash
# Linux / macOS
ln -s /你的项目路径/skills/sanger_qc ~/.openclaw/skills/sanger_qc

# Windows (管理员权限)
mklink /D "%USERPROFILE%\.openclaw\skills\sanger_qc" "D:\你的项目路径\skills\sanger_qc"
```

注册后，OpenClaw agent 会自动读取 `SKILL.md` 中的触发规则和命令模板。

### 5. 使用

通过 OpenClaw 客户端与 agent 对话，agent 会自动识别并调用 sanger_qc skill：

```
你：帮我分析一下 base 数据集的测序结果
Agent：（自动执行 python run.py --dataset base，返回判读结果）

你：只做比对，不用 AI
Agent：（自动添加 --no-llm 参数）

你：用 gpt-4.1-mini 分析 pro 数据集
Agent：（自动执行 python run.py --dataset pro --model gpt-4.1-mini）
```

### OpenClaw 架构图

```
用户 ──→ OpenClaw Client
              │
              ▼
         OpenClaw Gateway (localhost:18789)
              │
         ┌────┴────┐
         ▼         ▼
    LLM Provider   Skills
    (agent 大脑)    ├── sanger_qc/  ←── 本项目
                   ├── (未来 skill)
                   └── ...
```

## 核心算法

### 比对流程（alignment.py）

1. **GenBank 解析**：读取 `.gb` 文件，定位插入基因 CDS 区域
2. **AB1 读取 & 质量修剪**：Phred 质量过滤（Q ≥ 20），保留最长连续高质量区域
3. **环形质粒处理**：参考序列 × 2，处理跨越起点的比对
4. **双向局部比对**：正向 + 反向互补各比对一次，取最优（BioPython PairwiseAligner）
5. **突变检测**：提取 SNP、Insertion、Deletion，记录位置和碱基
6. **氨基酸翻译**：CDS 区域密码子翻译，检测 AA 变化（**质量过滤：仅报告密码子 3 碱基均 Q ≥ 30 的变异**）
7. **移码检测**：CDS 内非 3 倍数 indel → 蛋白截断
8. **多 Read 合并**：同一样本多条测序，合并覆盖度，标记冲突
9. **HTML 可视化**：暗色主题，碱基级对齐，错配红色高亮

### AI 判读规则（llm_judge.py）

系统提示词定义 10 条优先级递减的判读规则：

| 条件 | 判读 |
|------|------|
| identity ≈ 1.0, coverage > 0.55, 无移码, 无 AA 变化 | **ok** |
| 有 AA 变异 (1-5), identity > 0.95 | **wrong** + 变异列表 |
| 移码 + coverage > 0.55 | **wrong 移码错误** |
| 低覆盖 (< 0.55), 无突变 | **ok 未测通** |
| identity < 0.85, 突变 > 40 | **wrong 重叠峰，比对失败** |
| identity < 0.85, 突变 25-40 | **wrong 重叠峰** |
| AA 变异集中连续区域 | **wrong 片段缺失** |
| identity < 0.30 或 seq < 50bp | **wrong 测序失败** |
| identity 0.85-0.95, 散在突变 | **ok 生工重叠峰** |
| 双 read 冲突 | **wrong**（即使主 read 正常） |

## 免费模型推荐

### ChatAnywhere（国内直连，200 次/天）

领 Key：[chatanywhere/GPT_API_free](https://github.com/chatanywhere/GPT_API_free)

| 模型 | 备注 |
|------|------|
| `gpt-4.1` | GPT 最新旗舰 |
| `gpt-4.1-mini` | 性价比高，推荐日常使用 |
| `gpt-4.1-nano` | 最轻量，速度最快 |
| `o3` / `o4-mini` | OpenAI 推理模型 |
| `deepseek-chat` | DeepSeek V3 |

### OpenRouter（海外，有免费模型）

| 模型 | 参数量 | 备注 |
|------|--------|------|
| `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | 最强，限流较严 |
| `qwen/qwen3-coder:free` | 480B(A35B) | 中文好 |
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | 综合强 |
| `google/gemma-3-27b-it:free` | 27B | 稳定可用 |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 24B | 速度快 |

## 验证准确率

`truth/` 目录包含人工标注的真值数据。运行分析后，程序会自动与真值比对并输出准确率：

```
Accuracy (ok/wrong): 10/10 (100%)
```

| 真值文件 | 数据集 | 样本数 |
|---------|--------|--------|
| `truth/result.txt` | base | 10 |
| `truth/result_pro.txt` | pro | 7 |
| `truth/result_promax.txt` | promax | 12 |

## 各模块说明

| 文件 | 行数 | 职责 |
|------|------|------|
| `run.py` | ~178 | 入口调度：解析参数 → 调用生信分析 → 调用 LLM → 输出结果 → 比对真值 |
| `alignment.py` | ~759 | 核心算法：AB1 读取、质量修剪、双向比对、突变检测、AA 翻译、移码检测、多 read 合并、HTML 可视化 |
| `evidence.py` | ~68 | 格式化：将比对结果转为 LLM 输入文本和人类可读表格 |
| `llm_judge.py` | ~201 | LLM 客户端：66 行系统提示词 + OpenAI/Anthropic 双格式 + 指数退避重试 + 结果解析 |
