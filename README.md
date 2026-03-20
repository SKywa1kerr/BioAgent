# BioAgent MAX — Sanger 测序 QC 智能分析平台

BioAgent MAX 是一个 Sanger 测序质控分析平台，采用**确定性规则引擎**替代 LLM 判读，提供 Streamlit 可视化仪表盘，并支持通过 Claude Code MCP 协议进行 AI 辅助交互。

## 核心特性

- **确定性规则引擎** — 基于 identity、CDS 覆盖率、移码检测、氨基酸变异等指标，通过可配置阈值自动判读样本状态（OK / Wrong / Uncertain）
- **Streamlit 仪表盘** — 上传或扫描目录，查看分析结果、图表、比对详情，管理历史记录
- **FastAPI 后端** — RESTful API，支持外部系统集成
- **Claude Code MCP 集成** — 7 个工具函数，让 Claude 直接分析测序数据、查询结果、导出报告

## 快速开始

### 本地安装

```bash
pip install -r backend/requirements.txt
streamlit run frontend/app.py
```

需要 Python 3.10+。

### Docker 部署

```bash
docker-compose up --build
```

启动后访问：
- Streamlit UI: http://localhost:8501
- FastAPI API: http://localhost:8000/docs

## Claude Code MCP 集成

项目根目录已包含 `.mcp.json` 配置文件。在 Claude Code 中打开本项目即可自动发现以下 7 个工具：

| 工具 | 说明 |
|------|------|
| `scan_directory` | 扫描目录，发现可分析的 AB1 和 GB 文件 |
| `analyze_directory` | 分析指定目录下的所有样本，返回判读摘要 |
| `analyze_files` | 分析指定的 AB1 文件和 GenBank 参考序列 |
| `get_analysis_summary` | 获取一次分析的汇总结果 |
| `get_sample_detail` | 获取单个样本的详细分析数据 |
| `export_report` | 导出分析报告为 CSV 文本 |
| `update_thresholds` | 临时调整判读阈值 |

使用示例（在 Claude Code 对话中）：

```
请扫描 data/ 目录，找到所有测序文件
分析 data/gb 和 data/ab1_files 目录下的所有样本
导出分析 abc123 的报告
```

## 配置

判读阈值在 `backend/rules_config.yaml` 中配置：

```yaml
identity_min: 0.98        # 最低 identity
cds_coverage_min: 0.90    # 最低 CDS 覆盖率
aa_change_max: 0          # 允许的最大氨基酸变异数
```

修改后重启服务即可生效。也可通过 MCP 工具 `update_thresholds` 临时调整（仅内存生效）。

## 项目结构

```
BioAgent_MAX/
├── .mcp.json                # Claude Code MCP 配置
├── Dockerfile               # Docker 镜像定义
├── docker-compose.yml       # Docker Compose 编排
├── backend/
│   ├── main.py              # FastAPI 应用入口
│   ├── mcp_server.py        # MCP Server（Claude Code 集成）
│   ├── rules_config.yaml    # 判读阈值配置
│   ├── core/
│   │   ├── alignment.py     # 序列比对与突变检测
│   │   ├── evidence.py      # 证据格式化
│   │   └── rules.py         # 规则引擎
│   ├── db/
│   │   ├── database.py      # 数据库连接
│   │   └── models.py        # SQLAlchemy 模型
│   └── api/                 # FastAPI 路由
├── frontend/
│   ├── app.py               # Streamlit 主页
│   ├── pages/               # Streamlit 子页面
│   └── components/          # UI 组件
└── tests/                   # 测试
```

## License

MIT (placeholder)
