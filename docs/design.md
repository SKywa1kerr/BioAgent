# BioAgent MAX — Sanger 测序 QC 智能分析平台

## 设计文档

> 日期：2026-03-20
> 版本：v3（整合 spec review 修订）
> 状态：定稿

---

## 1. 项目目标

将现有的命令行 Sanger 测序 QC 管道（BioAgent_openclaw），升级为一个**可部署的 Web 应用**：

- 用户在浏览器中完成所有操作（上传文件、触发分析、查看结果）
- **确定性规则引擎**替代 LLM 判读，消除 API 依赖，100% 本地可靠运行
- Claude Code 作为**可选的 AI 增强层**，通过独立 MCP Server 调用分析工具
- 支持**本地部署**和**服务器部署**，其他用户也可以搭建自己的实例
- 判读阈值**可配置**，适应不同测序公司、批次、引物的质量差异

### 与旧项目的关系

| 项目 | 定位 | 状态 |
|------|------|------|
| BioAgent_openclaw | CLI 工具，基于 OpenClaw + LLM 判读 | 保留不动 |
| BioAgent MAX | Web 应用，确定性规则引擎 + 可选 AI | 本项目 |

核心文件 `alignment.py` 和 `evidence.py` 从旧项目**复制**后独立演进（非引用），迁移时需要做以下清理：
- 移除 `print()` 语句，改用 `logging` 模块
- 拆除嵌入式 HTML 生成代码（前端负责可视化）
- 移除硬编码的 `DATASET_MAP`，改为接受任意目录参数

---

## 2. 整体架构

```
┌──────────────────────────────────────────────────────┐
│              浏览器 (Streamlit)                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────┐  │
│  │  结果仪表盘   │  │  文件管理器    │  │ 参数设置  │  │
│  │ 表格/图表/比对│  │ 上传/扫描目录  │  │ 阈值调节  │  │
│  └──────┬───────┘  └───────┬───────┘  └────┬─────┘  │
│         └──────────────────┼───────────────┘         │
│                            │ 直接 import core/ 模块   │
└────────────────────────────┼─────────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────┐
│              后端 core 模块 (Python)                     │
│                            │                           │
│  ┌─────────────────────────┴──────────────────────┐   │
│  │              API 路由层                          │   │
│  │  POST /api/upload        文件上传               │   │
│  │  POST /api/scan          扫描目录               │   │
│  │  POST /api/analyze       触发分析               │   │
│  │  GET  /api/analyze/{id}  查询状态               │   │
│  │  GET  /api/results       历史记录               │   │
│  │  GET  /api/export/{id}   导出报告               │   │
│  │  GET  /api/config        获取当前阈值配置        │   │
│  │  PUT  /api/config        更新阈值配置            │   │
│  └─────────────────────────┬──────────────────────┘   │
│                            │                           │
│  ┌─────────────────────────┴──────────────────────┐   │
│  │           分析引擎                               │   │
│  │  alignment.py   比对/突变检测/AA翻译 (复用+清理) │   │
│  │  evidence.py    结果格式化 (复用)                │   │
│  │  rules.py       确定性规则判读 (新)              │   │
│  │  rules_config.yaml  阈值配置文件 (新)           │   │
│  └─────────────────────────┬──────────────────────┘   │
│                            │                           │
│  ┌─────────────────────────┴──────────────────────┐   │
│  │            SQLite 数据层                         │   │
│  │  分析历史 / 样本结果                              │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│          MCP Server (独立进程，可选)                      │
│  mcp_server.py                                         │
│  tools: analyze / query / export / config              │
│  共享后端的 core 模块，直接调用分析引擎                    │
└──────────────────────┬─────────────────────────────────┘
                       │ stdio (MCP 协议)
┌──────────────────────┼─────────────────────────────────┐
│         Claude Code (可选 AI 增强)                       │
│  - Claude Pro 订阅：官方通道，稳定                        │
│  - 中转站：作为备用方案                                   │
│  - 通过 MCP 工具调用分析管道、解读结果                     │
└────────────────────────────────────────────────────────┘
```

### 架构设计原则

1. **Streamlit 直接导入 core 模块**：Streamlit 前端直接 `import core.alignment`、`import core.rules`，不经 HTTP 调用 FastAPI。FastAPI 仅作为可选的外部 API 层（供 MCP Server 和第三方集成使用）
2. **Web 应用和 AI Agent 解耦**：Web Dashboard 不包含聊天面板，Claude Code CLI 本身就是最好的对话界面
3. **MCP Server 独立于 Web 后端**：作为单独进程运行，Claude Code 直连，同样直接导入 core 模块
4. **零 API 依赖**：核心分析流程（模式 A）完全本地运行，不需要任何外部 API
5. **可部署性**：支持 Docker 部署，其他用户可以在自己的服务器上搭建
6. **阈值配置安全**：`rules_config.yaml` 仅由 Streamlit UI 单进程写入；MCP Server 的 `update_thresholds` 仅在内存中临时覆盖，不写回文件

---

## 3. 两种使用模式

### 模式 A：Web 独立使用（不需要 Claude Code）

用户直接在浏览器中操作：
1. 上传 AB1 + GenBank 文件，或指定目录路径
2. （可选）在设置面板调整判读阈值
3. 点击「开始分析」
4. 后端运行生信管道 + 确定性规则判读
5. 仪表盘展示结果

**完全免费，零 API 依赖，任何人都可以部署。**

### 模式 B：Claude Code Agent 模式（AI 增强，可选）

用户在 Claude Code CLI 中用自然语言操作：
```
你："帮我分析 data/ab1_files 目录下的测序数据"
Claude Code → 调用 MCP 工具 → 运行分析 → 返回结果解读

你："C397-a 这个样本具体是什么问题？"
Claude Code → 读取详细数据 → 给出专业解读

你："把测序失败的阈值从 0.30 调到 0.25 再跑一次"
Claude Code → 更新配置 → 重新分析 → 对比差异
```

**需要 Claude Code 订阅（Claude Pro 官方通道为主，中转站为备用）。**

---

## 4. 核心模块设计

### 4.1 判读阈值配置 (rules_config.yaml)

所有判读规则的数值阈值抽离到配置文件，不硬编码在代码中。
不同测序公司、批次、引物的质量基线不同，阈值需要可调。

```yaml
# rules_config.yaml — 判读规则阈值配置
# 修改后立即生效，无需改代码

thresholds:
  # ── 测序失败 ──
  seq_failure_identity: 0.30      # identity 低于此值判定测序失败
  seq_failure_min_length: 50      # 序列长度低于此值判定测序失败

  # ── 比对质量分级 ──
  identity_high: 0.95             # 高质量比对阈值
  identity_medium_low: 0.85       # 中低质量阈值（低于此值为重叠峰区域）

  # ── CDS 覆盖度 ──
  cds_coverage_low: 0.55          # 低覆盖度阈值
  cds_coverage_deletion: 0.80     # 片段缺失上界

  # ── AA 突变数量 ──
  aa_overlap_severe: 40           # 重叠峰比对失败（突变数 > 此值）
  aa_overlap_moderate_min: 25     # 重叠峰下界
  aa_overlap_moderate_max: 40     # 重叠峰上界
  aa_mutation_max: 5              # 真实突变上限（1~此值为真实突变）
  aa_deletion_min: 5              # 片段缺失最低突变数

  # ── 生工重叠峰 ──
  synthetic_identity_min: 0.85    # 生工重叠峰 identity 下界
  synthetic_identity_max: 0.95    # 生工重叠峰 identity 上界
  synthetic_aa_min: 15            # 生工重叠峰最低 AA 变异数

  # ── 质量过滤 ──
  quality_trim_min: 20            # AB1 质量修剪最低 Phred 分数
  quality_aa_filter: 30           # AA 突变报告的最低碱基质量
```

### 4.2 规则判读引擎 (rules.py)

将 `llm_judge.py` 的 10 条判读规则转化为确定性 Python 代码，所有阈值从配置文件读取。

```python
import yaml
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent / "rules_config.yaml"

def load_thresholds(config_path: Path = DEFAULT_CONFIG) -> dict:
    """加载阈值配置。"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["thresholds"]

class SampleInput(TypedDict):
    """judge_sample 的输入类型契约。由 alignment.analyze_sample() 产出。"""
    sid: str                    # 样本 ID，如 "C373-2"
    identity: float             # 比对一致性 (0.0~1.0)
    cds_coverage: float         # CDS 覆盖度 (0.0~1.0)
    frameshift: bool            # 是否检测到移码
    aa_changes: list[str]       # AA 变异列表，如 ["S334L", "K171M"]
    aa_changes_n: int           # 质量过滤后的 AA 变异数
    seq_length: int             # 测序序列长度 (bp)
    # 以下为可选字段（多读段场景）
    other_read_issues: list[str] | None  # 其他读段发现的问题


def judge_sample(sample: SampleInput, thresholds: dict | None = None) -> dict:
    """基于规则的确定性判读，不调用任何 API。

    所有数值阈值从 thresholds 字典读取，不硬编码。
    规则按优先级从高到低排列，首个匹配即返回。
    """
    t = thresholds or load_thresholds()
    sid = sample["sid"]
    identity = sample["identity"]
    cds_cov = sample["cds_coverage"]
    frameshift = sample["frameshift"]
    aa_changes = sample["aa_changes"]
    aa_n = sample["aa_changes_n"]
    seq_len = sample["seq_length"]

    # ── 优先级 1：多读段冲突（最高优先级）──
    if sample.get("other_read_issues"):
        return {"sid": sid, "status": "wrong", "reason": "多读段冲突",
                "rule": 1, "details": sample["other_read_issues"]}

    # ── 优先级 2：测序失败 ──
    if identity < t["seq_failure_identity"] or seq_len < t["seq_failure_min_length"]:
        return {"sid": sid, "status": "wrong", "reason": "测序失败", "rule": 2}

    # ── 优先级 3：重叠峰，比对失败（identity 极低 + 大量突变）──
    if identity < t["identity_medium_low"] and aa_n > t["aa_overlap_severe"]:
        return {"sid": sid, "status": "wrong", "reason": "重叠峰，比对失败", "rule": 3}

    # ── 优先级 4：重叠峰（identity 低 + 中等突变）──
    if identity < t["identity_medium_low"] \
       and t["aa_overlap_moderate_min"] <= aa_n <= t["aa_overlap_moderate_max"]:
        return {"sid": sid, "status": "wrong", "reason": "重叠峰", "rule": 4}

    # ── 优先级 5：移码错误（不区分覆盖度，frameshift 本身已确认）──
    if frameshift:
        return {"sid": sid, "status": "wrong", "reason": "移码错误", "rule": 5}

    # ── 优先级 6：真实 AA 突变（高 identity + 少量突变）──
    if aa_changes and identity >= t["identity_high"] and 1 <= aa_n <= t["aa_mutation_max"]:
        return {"sid": sid, "status": "wrong",
                "reason": " ".join(aa_changes), "rule": 6}

    # ── 优先级 7：片段缺失（中等覆盖 + 多个突变集中）──
    if t["cds_coverage_low"] <= cds_cov <= t["cds_coverage_deletion"] \
       and aa_n >= t["aa_deletion_min"]:
        return {"sid": sid, "status": "wrong", "reason": "片段缺失", "rule": 7}

    # ── 优先级 8：生工重叠峰（散在假突变，判 ok）──
    if t["synthetic_identity_min"] <= identity <= t["synthetic_identity_max"] \
       and aa_n > t["synthetic_aa_min"]:
        return {"sid": sid, "status": "ok", "reason": "生工重叠峰", "rule": 8}

    # ── 优先级 9：低覆盖但无突变（判 ok）──
    if cds_cov < t["cds_coverage_low"] and aa_n == 0 and not frameshift:
        return {"sid": sid, "status": "ok", "reason": "未测通", "rule": 9}

    # ── 优先级 10：正常序列 ──
    if identity >= t["identity_high"] and aa_n == 0 and not frameshift:
        return {"sid": sid, "status": "ok", "reason": "", "rule": 10}

    # ── 兜底：未匹配任何规则，保守处理 ──
    return {"sid": sid, "status": "uncertain", "reason": "需人工复核",
            "rule": -1}


def judge_batch(samples: list[dict], thresholds: dict | None = None) -> list[dict]:
    """批量判读。"""
    t = thresholds or load_thresholds()
    return [judge_sample(s, t) for s in samples]
```

**相比原设计的改进：**

| 原设计问题 | 改进 |
|-----------|------|
| 阈值硬编码 (0.30, 0.85, 40 等) | 所有阈值从 `rules_config.yaml` 读取 |
| 移码错误分高/低覆盖两条规则，逻辑相同 | 合并为一条：`frameshift` 即判 wrong |
| 兜底判 `ok`，过于宽松 | 兜底判 `uncertain`，保守处理 |
| 规则 7 (identity>0.95, aa 1-5) 和规则 9 (identity 0.85-0.95) 之间有 gap | 兜底规则兜住边界情况，标记为人工复核 |
| 无规则编号追踪 | 返回 `rule` 字段，方便调试和日志 |

### 4.3 后端 (FastAPI)

```
backend/
├── main.py                    # FastAPI 应用入口
├── api/
│   ├── upload.py              # 文件上传接口
│   ├── scan.py                # 目录扫描
│   ├── analyze.py             # 分析触发与状态查询
│   ├── results.py             # 结果查询、历史记录
│   ├── export.py              # 报告导出 (PDF/CSV)
│   └── config.py              # 阈值配置 CRUD
├── core/
│   ├── alignment.py           # 复用 + 清理（见下方迁移清单）
│   ├── evidence.py            # 复用
│   └── rules.py               # 确定性规则判读引擎（新）
├── db/
│   ├── models.py              # SQLAlchemy 模型
│   └── database.py            # 数据库连接管理
├── rules_config.yaml          # 判读阈值配置
├── config.py                  # 应用配置（端口、数据目录等）
└── requirements.txt           # Python 依赖
```

#### alignment.py 迁移清单

从 `BioAgent_openclaw/skills/sanger_qc/alignment.py` 复制后，需要做以下修改：

| 修改项 | 原代码 | 改为 |
|--------|--------|------|
| print 语句 | `print(f"  Analyzing ...")` (L676, L688) | `logging.info()` 或 progress callback |
| HTML 生成 | `write_alignment_html()` 及相关函数 (L430-550) | 移除，前端负责可视化 |
| DATASET_MAP | 硬编码 base/pro/promax (L639-644) | 移除，改为接受 `gb_dir` + `ab1_dir` 参数 |
| analyze_dataset 签名 | `dataset: str` 参数 | `gb_dir: Path, ab1_dir: Path` |

#### evidence.py 在新架构中的角色

`evidence.py` 保留，主要服务于 **MCP Server 场景**：当 Claude Code 调用分析工具时，`format_evidence_for_llm()` 将结果格式化为 AI 可读的文本。Streamlit 前端不使用此模块（直接从数据库读取结构化数据渲染 UI）。

#### 文件上传约束

| 约束 | 值 |
|------|-----|
| 允许的文件类型 | `.ab1`, `.gb`, `.gbk` |
| 单文件大小上限 | 10 MB |
| 无效文件处理 | 返回 400，说明无法解析的原因 |
| 存储目录 | `config.yaml` 中的 `upload_dir`，默认 `./uploads` |
| 清理策略 | 保留最近 30 天的上传文件，可配置 |

#### 异步分析机制

`POST /api/analyze` 使用 FastAPI 的 `BackgroundTasks` 执行分析。适用于单用户/小团队场景。
限制：服务器重启时进行中的任务会丢失（状态留为 `running`），前端应对此做提示。

#### 关键接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/upload` | POST | 上传 AB1/GB 文件，返回文件 ID |
| `/api/scan` | POST | 扫描目录，返回发现的文件列表 |
| `/api/analyze` | POST | 触发分析任务（异步），返回任务 ID |
| `/api/analyze/{id}` | GET | 查询分析任务状态和结果 |
| `/api/results` | GET | 分页查询历史分析记录 |
| `/api/results/{id}` | GET | 单次分析详细结果 |
| `/api/results/{id}/samples/{sid}` | GET | 单个样本详细数据 |
| `/api/export/{id}` | GET | 导出报告（format=pdf/csv） |
| `/api/config` | GET | 获取当前判读阈值配置 |
| `/api/config` | PUT | 更新判读阈值（写回 rules_config.yaml） |

### 4.4 前端选型

**推荐 Streamlit（优先）或 Gradio，而非 React。**

| 考量 | React + TypeScript | Streamlit / Gradio |
|------|-------------------|-------------------|
| 开发语言 | 需维护 Python + TypeScript 两套 | 纯 Python |
| 开发速度 | 慢（组件开发、状态管理、构建） | 快（声明式 UI，5-10 倍提效） |
| 部署复杂度 | 需前后端分离部署 | 单进程，直接 `streamlit run` |
| 可视化 | 需自己接 Recharts/ECharts | 内置图表，直接 `st.plotly_chart()` |
| 适用场景 | 面向公众的 SaaS 产品 | 数据分析工具、内部/实验室使用 |
| 未来迁移 | — | 如需更精细 UI，后期可迁到 React |

如果选择 Streamlit，前端部分简化为：

```
frontend/
├── app.py                     # Streamlit 主入口
├── pages/
│   ├── 1_analysis.py          # 新建分析（上传/扫描）
│   ├── 2_results.py           # 分析结果仪表盘
│   ├── 3_history.py           # 历史记录
│   └── 4_settings.py          # 阈值配置面板
└── components/
    ├── alignment_viewer.py    # 碱基比对可视化组件
    └── charts.py              # 统计图表组件
```

#### 页面设计

**仪表盘 (Results)**
```
┌─────────────────────────────────────────────────┐
│  BioAgent MAX — Sanger 测序 QC 分析平台          │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │
│  │总样本│ │ OK  │ │Wrong│ │准确率│              │
│  │  29  │ │  22 │ │  7  │ │ 96% │              │
│  └─────┘ └─────┘ └─────┘ └─────┘              │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │          样本结果表格                      │    │
│  │ SID    状态    Identity  Coverage  原因   │    │
│  │ C373   ok     0.9998    0.891            │    │
│  │ C397   wrong  0.9956    0.764     S334L  │    │
│  │ C379   wrong  0.9978    0.823     Q131T  │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────┐ ┌──────────────────┐      │
│  │   质量分布图       │ │   覆盖度分布图     │      │
│  └──────────────────┘ └──────────────────┘      │
│                                                  │
│  ▼ 展开样本 C397-a                               │
│  ┌─────────────────────────────────────────┐    │
│  │ Identity: 0.9956  Coverage: 0.764        │    │
│  │ 方向: FORWARD  序列长度: 847bp             │    │
│  │ 平均质量: 42.3  移码: 无  规则: #6         │    │
│  │                                           │    │
│  │ 碱基比对:                                  │    │
│  │ REF 00334  ATCGATCG ATCGATCG ATCGATCG   │    │
│  │ MID 00334  ||||*||| |||||||| ||||||||    │    │
│  │ QRY 00334  ATCAATCG ATCGATCG ATCGATCG   │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

**设置面板 (Settings)**
```
┌─────────────────────────────────────────────────┐
│  判读参数设置                                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  测序失败                                         │
│  ├─ Identity 阈值:  [0.30 ▼]                    │
│  └─ 最短序列长度:   [50   ▼]                     │
│                                                  │
│  比对质量                                         │
│  ├─ 高质量 Identity: [0.95 ▼]                    │
│  └─ 重叠峰 Identity: [0.85 ▼]                    │
│                                                  │
│  CDS 覆盖度                                       │
│  ├─ 低覆盖阈值:     [0.55 ▼]                     │
│  └─ 片段缺失上界:   [0.80 ▼]                     │
│                                                  │
│  AA 突变数量                                      │
│  ├─ 重叠峰(严重):   [40   ▼]                     │
│  └─ 真实突变上限:   [5    ▼]                      │
│                                                  │
│  [恢复默认值]              [保存并重新分析]         │
└─────────────────────────────────────────────────┘
```

### 4.5 MCP Server（独立进程）

MCP Server **不嵌入 FastAPI 后端**，作为独立进程运行，直接导入 `core/` 模块：

```python
# mcp_server.py — 独立进程，Claude Code 通过 stdio 调用

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))  # 确保能导入 core/

from mcp.server import Server
from core.alignment import analyze_sample, build_aligner
from core.rules import judge_sample, judge_batch, load_thresholds

server = Server("bioagent")

@server.tool("analyze_directory")
async def analyze_directory(gb_dir: str, ab1_dir: str,
                            config_path: str | None = None):
    """分析指定目录下的所有 Sanger 测序样本。

    Args:
        gb_dir: GenBank 参考序列目录
        ab1_dir: AB1 测序文件目录
        config_path: 可选，自定义阈值配置文件路径
    """
    ...

@server.tool("analyze_files")
async def analyze_files(ab1_paths: list[str], gb_path: str):
    """分析指定的 AB1 文件和 GenBank 参考序列。"""
    ...

@server.tool("scan_directory")
async def scan_directory(directory: str):
    """扫描目录，发现可分析的 AB1 和 GB 文件。"""
    ...

@server.tool("get_analysis_summary")
async def get_analysis_summary(analysis_id: str):
    """获取一次分析的汇总结果。"""
    ...

@server.tool("get_sample_detail")
async def get_sample_detail(analysis_id: str, sample_id: str):
    """获取单个样本的详细分析数据。"""
    ...

@server.tool("export_report")
async def export_report(analysis_id: str, format: str = "csv"):
    """导出分析报告（csv/pdf）。"""
    ...

@server.tool("update_thresholds")
async def update_thresholds(overrides: dict):
    """临时调整判读阈值并重新分析。

    Args:
        overrides: 要覆盖的阈值键值对，如 {"seq_failure_identity": 0.25}
    """
    ...
```

#### Claude Code 配置

项目根目录 `.mcp.json`（项目级配置，推荐）：

```json
{
  "mcpServers": {
    "bioagent": {
      "command": "python",
      "args": ["backend/mcp_server.py"],
      "cwd": "."
    }
  }
}
```

或全局配置 `~/.claude/settings.json`：

```json
{
  "mcpServers": {
    "bioagent": {
      "command": "python",
      "args": ["/path/to/BioAgent_MAX/backend/mcp_server.py"]
    }
  }
}
```

### 4.6 数据库 (SQLite)

```sql
-- 分析记录
CREATE TABLE analyses (
    id          TEXT PRIMARY KEY,      -- UUID
    name        TEXT,                  -- 用户命名或自动生成
    source_type TEXT,                  -- 'upload' | 'scan'
    source_path TEXT,                  -- 目录路径（扫描模式）
    status      TEXT,                  -- 'pending' | 'running' | 'done' | 'error'
    total       INTEGER DEFAULT 0,
    ok_count    INTEGER DEFAULT 0,
    wrong_count INTEGER DEFAULT 0,
    uncertain_count INTEGER DEFAULT 0, -- 需人工复核
    config_snapshot TEXT,              -- 分析时的阈值配置快照 (JSON)
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP
);

-- 样本结果
CREATE TABLE samples (
    id              TEXT PRIMARY KEY,
    analysis_id     TEXT REFERENCES analyses(id),
    sid             TEXT,             -- 样本 ID (如 C373-2)
    clone           TEXT,
    status          TEXT,             -- 'ok' | 'wrong' | 'uncertain'
    reason          TEXT,             -- 判读原因
    rule_id         INTEGER,          -- 匹配的规则编号（调试用）
    identity        REAL,
    cds_coverage    REAL,
    frameshift      BOOLEAN,
    aa_changes      TEXT,             -- JSON 数组
    aa_changes_n    INTEGER,
    raw_aa_changes_n INTEGER,         -- 质量过滤前的 AA 变异数
    orientation     TEXT,
    seq_length      INTEGER,
    ref_length      INTEGER,
    avg_quality     REAL,
    sub_count       INTEGER,
    ins_count       INTEGER,
    del_count       INTEGER,
    -- 比对数据（用于可视化）
    ref_gapped      TEXT,
    qry_gapped      TEXT,
    quality_scores  TEXT,             -- JSON 数组
    raw_data        TEXT              -- 完整原始数据 JSON
);
```

**相比原设计新增：**
- `analyses.uncertain_count`：追踪需人工复核的样本数
- `analyses.config_snapshot`：记录分析时使用的阈值配置，支持后续对比
- `samples.rule_id`：记录匹配的规则编号
- `samples.raw_aa_changes_n`：质量过滤前的突变数，辅助调试

---

## 5. 技术栈

| 层 | 技术 | 理由 |
|----|------|------|
| 前端框架 | Streamlit（推荐）或 Gradio | 纯 Python，开发快，部署简单 |
| 后端框架 | FastAPI | 异步，自动 OpenAPI 文档，与生信代码无缝集成 |
| 数据库 | SQLite + SQLAlchemy | 零配置，单文件，本地和小型服务器部署足够 |
| MCP SDK | mcp (Python) | 官方 MCP SDK |
| 生信核心 | BioPython + pandas | 现有代码已在使用 |
| 配置管理 | PyYAML | 读写 rules_config.yaml |
| 部署 | Docker + docker-compose | 可选，方便他人一键部署 |

---

## 6. 部署方案

### 本地使用

```bash
# 安装依赖
pip install -r backend/requirements.txt

# 启动后端
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 启动前端
streamlit run frontend/app.py --server.port 8501

# （可选）启动 MCP Server 供 Claude Code 使用
# 在 .mcp.json 中配置即可，Claude Code 会自动启动
```

### Docker 部署（供他人使用）

```yaml
# docker-compose.yml
services:
  bioagent:
    build: .
    ports:
      - "8501:8501"    # Streamlit UI
      - "8000:8000"    # FastAPI API
    volumes:
      - ./data:/app/data              # 挂载数据目录
      - ./rules_config.yaml:/app/backend/rules_config.yaml  # 自定义配置
    environment:
      - DATA_DIR=/app/data
```

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
EXPOSE 8000 8501
CMD ["bash", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
```

### 配置文件（应用级）

```yaml
# config.yaml — 应用配置
server:
  host: "0.0.0.0"
  port: 8000

data:
  # 数据目录配置（替代 Windows 符号链接）
  upload_dir: "./uploads"          # 上传文件存放目录
  default_gb_dir: ""               # 默认 GenBank 目录（可选）
  default_ab1_dir: ""              # 默认 AB1 目录（可选）

database:
  url: "sqlite:///./bioagent.db"
```

---

## 7. 目录结构总览

```
BioAgent_MAX/
├── docs/
│   └── design.md                  # 本文档
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── upload.py
│   │   ├── scan.py
│   │   ├── analyze.py
│   │   ├── results.py
│   │   ├── export.py
│   │   └── config.py             # 阈值配置 API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── alignment.py          # 复用 + 清理
│   │   ├── evidence.py           # 复用（MCP Server 场景使用）
│   │   └── rules.py              # 确定性规则引擎（新）
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── database.py
│   ├── mcp_server.py             # MCP Server（独立进程）
│   ├── rules_config.yaml         # 判读阈值配置
│   ├── config.yaml               # 应用配置
│   └── requirements.txt
├── tests/
│   ├── test_rules.py             # 规则引擎单元测试
│   └── test_alignment.py         # 比对模块测试
├── frontend/
│   ├── app.py                    # Streamlit 主入口
│   ├── pages/                    # 多页面
│   └── components/               # 可视化组件
├── docker-compose.yml            # Docker 部署
├── Dockerfile
├── .mcp.json                     # Claude Code MCP 配置
├── data/                         # 示例数据（可选）
└── README.md
```

---

## 8. 实现优先级

### Phase 1：后端核心 + 规则引擎

1. FastAPI 项目骨架 + 应用配置
2. 迁移并清理 `alignment.py`（移除 print/HTML/DATASET_MAP）
3. 迁移 `evidence.py`
4. 实现 `rules.py` + `rules_config.yaml`
5. **rules.py 单元测试**：使用旧项目 truth 数据集（base/pro/promax）验证规则引擎判读准确率
6. 文件上传、目录扫描 API
7. SQLite 数据库模型
8. 分析触发 + 结果查询 API

### Phase 2：前端 Dashboard + 设置面板

1. Streamlit 项目搭建
2. 文件上传 / 目录扫描页面
3. 结果表格（排序、筛选、状态标签）
4. 样本详情展开 + 碱基比对可视化
5. 统计图表（质量分布、覆盖度分布）
6. 阈值设置面板（读写 rules_config.yaml）
7. 历史记录 + PDF/CSV 导出

### Phase 3：MCP Server + 部署

1. MCP Server 实现（独立进程）
2. `.mcp.json` 配置 + Claude Code 集成测试
3. Dockerfile + docker-compose
4. README 部署文档

---

## 9. Claude Code 使用策略

| 通道 | 场景 | 说明 |
|------|------|------|
| Claude Pro 订阅（官方） | 主力通道 | 稳定，直连 Anthropic |
| 中转站 | 备用 | 官方不可用时切换 |

Claude Code 在本项目中的角色：
- **开发阶段**：辅助编码、调试、测试
- **使用阶段**：通过 MCP Server 进行自然语言分析（可选增强）
- **边缘情况**：规则引擎判为 `uncertain` 的样本，可在 Claude Code 中读取 evidence 数据让 AI 辅助判断
