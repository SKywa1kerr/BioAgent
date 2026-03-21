# BioAgent MAX 操作手册

## 1. 启动 Web 界面

在项目根目录打开终端，运行：

```bash
cd D:\Learning\Biology\BioAgent_MAX
streamlit run frontend/app.py
```

浏览器自动打开 http://localhost:8501（首次会提示输入邮箱，直接回车跳过）。

## 2. Web 页面功能

| 页面 | 功能 |
|------|------|
| 新建分析 | 输入 GB 和 AB1 目录路径，点击开始分析 |
| 分析结果 | 查看样本表格、Identity/Coverage 分布图、展开看碱基比对 |
| 历史记录 | 浏览所有历史分析批次 |
| 参数设置 | 调整判读阈值（修改后下次分析生效） |
| AI 问答 | 选择分析记录后，向 AI 提问（需配置 .env） |

## 3. 分析数据

数据存放在 `data/` 目录下，每个数据集一个文件夹：

```
data/
├── base/       ← 数据集 base
│   ├── ab1/    ← 放 .ab1 文件
│   └── gb/     ← 放 .gb 文件
├── pro/
├── promax/
└── batch1/
```

添加新数据集：在 `data/` 下新建文件夹，里面放 `ab1/` 和 `gb/` 两个子目录，把文件拖进去即可。

分析时路径填写示例：
- GenBank 目录：`data/batch1/gb`
- AB1 目录：`data/batch1/ab1`

## 4. Claude Code 使用（MCP）

在项目目录下打开 Claude Code，直接用自然语言即可：

```
帮我分析 data/batch1/gb 和 data/batch1/ab1 下的样本
C478-1 的 indel 是怎么回事
导出上次分析的报告
```

Claude Code 和 Web 端共享数据库，两边的分析结果互通。

## 5. AI 问答配置

API 配置在项目根目录 `.env` 文件中（不会被 git 提交）：

```
LLM_API_KEY=sk-你的key
LLM_BASE_URL=https://api.chatanywhere.tech/v1
LLM_MODEL=gpt-4o-mini
```

免费 API Key 获取：
- ChatAnywhere：https://github.com/chatanywhere/GPT_API_free
- Groq：https://console.groq.com/keys

可用模型：gpt-3.5-turbo、gpt-4o-mini、gpt-4o（ChatAnywhere）

## 6. 分享给别人（ngrok）

如果别人不在你的局域网，用 ngrok 临时分享：

**终端 1** — 启动 Streamlit：
```bash
cd D:\Learning\Biology\BioAgent_MAX
streamlit run frontend/app.py
```

**终端 2** — 启动 ngrok：
```bash
ngrok http 8501
```

ngrok 会显示一个公网链接（类似 `https://xxxx.ngrok-free.app`），发给别人即可访问。关掉 ngrok 链接立刻失效。

首次使用 ngrok 需要配置 authtoken（只需一次）：
```bash
ngrok config add-authtoken 你的token
```

注册地址：https://dashboard.ngrok.com/signup（免费）

## 7. Docker 部署（可选）

如果有服务器，可以用 Docker 一键部署：

```bash
docker-compose up --build
```

访问：
- Web 界面：http://服务器IP:8501
- API 文档：http://服务器IP:8000/docs

## 8. 常用命令速查

| 操作 | 命令 |
|------|------|
| 启动 Web | `streamlit run frontend/app.py` |
| 运行测试 | `python -m pytest tests/test_rules.py -v` |
| 启动 API 服务 | `uvicorn backend.main:app --port 8000` |
| 分享页面 | `ngrok http 8501` |
| Docker 部署 | `docker-compose up --build` |
