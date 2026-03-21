# BioAgent MAX — Sanger 测序 QC 智能分析平台

BioAgent MAX 是一个 Sanger 测序质控分析平台，采用**确定性规则引擎**自动判读样本状态，提供 Streamlit 可视化仪表盘和 AI 问答功能，并支持通过 Claude Code MCP 协议进行 AI 辅助交互。

## 核心特性

- **确定性规则引擎** — 10 条优先级规则，基于 identity、CDS 覆盖率、移码检测、氨基酸变异等指标，自动判读样本状态（OK / Wrong / Uncertain）
- **Streamlit 仪表盘** — 上传文件或扫描目录，查看分析结果、分布图表、碱基比对详情，管理历史记录
- **AI 问答** — 基于分析结果与 AI 模型对话，获取智能解读和建议（支持 GPT-4o 等免费模型）
- **FastAPI 后端** — RESTful API，支持外部系统集成
- **Claude Code MCP 集成** — 7 个工具函数，让 Claude 直接分析测序数据、查询结果、导出报告
- **数据互通** — Web 端和 MCP 共享同一数据库，分析结果自动同步

## 快速开始

### 1. 安装依赖

```bash
cd BioAgent_MAX
pip install -r backend/requirements.txt
```

需要 Python 3.10+。

### 2. 配置 AI 问答（可选）

在项目根目录创建 `.env` 文件：

```
LLM_API_KEY=你的API_Key
LLM_BASE_URL=https://api.chatanywhere.tech/v1
LLM_MODEL=gpt-4o
```

免费 API Key 获取：[ChatAnywhere](https://github.com/chatanywhere/GPT_API_free)

### 3. 启动

```bash
streamlit run frontend/app.py
```

浏览器打开 http://localhost:8501 即可使用。

### Docker 部署

```bash
docker-compose up --build
```

启动后访问：
- Streamlit UI: http://localhost:8501
- FastAPI API: http://localhost:8000/docs

## 使用方式

### Web 端

| 页面 | 功能 |
|------|------|
| **Analysis** | 扫描目录或上传 AB1/GB 文件，启动分析 |
| **Results** | 查看样本表格、Identity/Coverage 分布图、碱基比对详情 |
| **History** | 浏览所有历史分析批次 |
| **Settings** | 调整判读阈值参数 |
| **AI Chat** | 基于分析结果向 AI 提问 |

### Claude Code MCP

在 Claude Code 中打开本项目即可自动发现以下工具：

| 工具 | 说明 |
|------|------|
| `scan_directory` | 扫描目录，发现可分析的 AB1 和 GB 文件 |
| `analyze_directory` | 分析目录下所有样本，结果存入数据库 |
| `analyze_files` | 分析指定的 AB1 文件和 GenBank 参考序列 |
| `get_analysis_summary` | 获取分析汇总结果 |
| `get_sample_detail` | 获取单个样本详细数据 |
| `export_report` | 导出分析报告为 CSV |
| `update_thresholds` | 临时调整判读阈值 |

使用示例（在 Claude Code 对话中）：

```
帮我分析 data/batch1/gb 和 data/batch1/ab1 下的样本
C478-1 的 indel 是怎么回事
导出上次分析的报告
```

## 数据目录结构

```
data/
├── base/           ← 数据集 1
│   ├── ab1/        ← .ab1 测序文件
│   └── gb/         ← .gb 参考序列
├── pro/            ← 数据集 2
├── promax/         ← 数据集 3
├── batch1/         ← 数据集 4
└── uploads/        ← Web 上传的文件（自动创建）
```

添加新数据集：在 `data/` 下新建文件夹，放入 `ab1/` 和 `gb/` 子目录即可。

## 分享给他人（ngrok）

无需服务器，临时分享本地页面：

```bash
# 终端 1：启动 Streamlit
streamlit run frontend/app.py

# 终端 2：启动 ngrok
ngrok http 8501
```

把 ngrok 生成的公网链接发给对方即可。关掉 ngrok 链接失效。

## 配置

判读阈值在 `backend/rules_config.yaml` 中配置，修改后下次分析自动生效。也可在 Web 端 Settings 页面直接修改。

## 项目结构

```
BioAgent_MAX/
├── .mcp.json                # Claude Code MCP 配置
├── .env                     # API Key 配置（不提交到 git）
├── Dockerfile
├── docker-compose.yml
├── backend/
│   ├── main.py              # FastAPI 应用入口
│   ├── mcp_server.py        # MCP Server
│   ├── config.yaml           # 应用配置
│   ├── rules_config.yaml    # 判读阈值配置
│   ├── core/
│   │   ├── alignment.py     # 序列比对与突变检测
│   │   ├── evidence.py      # 证据格式化
│   │   └── rules.py         # 规则引擎（10 条规则）
│   ├── db/
│   │   ├── database.py      # 数据库连接
│   │   ├── models.py        # SQLAlchemy 模型
│   │   └── migrations/      # Alembic 数据库迁移
│   └── api/                 # FastAPI 路由
├── frontend/
│   ├── app.py               # Streamlit 主页
│   ├── pages/               # Analysis, Results, History, Settings, AI Chat
│   └── components/          # 图表、比对展示、样式
├── data/                    # 测序数据目录
├── docs/                    # 文档
│   └── usage_guide.md       # 操作手册
└── tests/                   # 测试
```

## License

MIT
