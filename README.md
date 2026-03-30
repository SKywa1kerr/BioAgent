# BioAgent — Sanger 测序质控与突变分析工具 (Electron 版)

BioAgent 是一个集成了生物信息学分析与 AI 判读的 Sanger 测序分析工具，现已迁移至 Electron 桌面端。

## 主要功能

1. **生物信息学分析** — 将 AB1 测序文件与 GenBank 参考序列进行比对，检测碱基突变、氨基酸变异和移码。支持循环质粒比对和多读段自动合并。
2. **AI 判读** — 集成大语言模型（LLM），综合比对证据自动判断样本质控结论（ok/wrong）。
3. **可视化界面** — 提供直观的序列比对查看器和峰图（Chromatogram）展示。

## 项目结构

```
BioAgent/
├── electron/             # Electron 主进程与预加载脚本
├── src/                  # React 前端源代码
├── src-python/           # Python 后端逻辑 (Sidecar)
│   └── bioagent/
│       ├── alignment.py  # 核心算法：比对、突变检测、AA 翻译
│       ├── llm_client.py # LLM 接口调用
│       ├── evidence.py   # 证据格式化
│       └── main.py       # CLI 入口
├── vite.config.ts        # Vite 配置
└── package.json          # 项目依赖与脚本
```

## 快速开始

### 1. 环境准备

- Node.js 18+
- Python 3.10+
- 安装 Python 依赖：
  ```bash
  pip install -e ./src-python
  ```

### 2. 配置 API

在根目录创建 `.env` 文件（可参考 `.env.example`），配置您的 LLM API Key：

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1  # 或其他兼容 OpenAI 的地址
```

### 3. 启动开发环境

```bash
npm install
npm run electron:dev
```

## 使用说明

1. 点击界面上的 **"Open Folder"** 按钮。
2. 选择包含 `.ab1` 文件和对应 `.gb/.gbk` 参考序列的文件夹。
   - 注意：程序会根据文件名自动匹配，例如 `C123-1.ab1` 会匹配 `C123.gb`。
3. 等待分析完成后，左侧列表将显示所有样本及其状态。
4. 点击样本可查看详细的比对情况、突变列表以及 AI 判读建议。

## 技术栈

- **Frontend**: React, TypeScript, Vite
- **Desktop**: Electron
- **Backend**: Python (Biopython, OpenAI SDK)
- **AI**: 支持 OpenRouter, DeepSeek, ChatAnywhere 等所有 OpenAI 兼容接口
