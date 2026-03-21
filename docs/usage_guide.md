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
| 运行测试 | `python -m pytest tests/ -v` |
| 启动 API 服务 | `uvicorn backend.main:app --port 8000` |
| 分享页面 | `ngrok http 8501` |
| Docker 部署 | `docker-compose up --build` |

---

## 9. 数据库迁移（Alembic）

项目使用 Alembic 管理数据库表结构变更。**改了 `backend/db/models.py` 中的表字段后**，需要执行以下步骤：

### 场景：给表加字段 / 改字段 / 删字段

```bash
# 第 1 步：根据 models.py 的变化，自动生成迁移脚本
alembic revision --autogenerate -m "描述你改了什么"

# 第 2 步：查看生成的脚本（在 backend/db/migrations/versions/ 下）
# 确认 upgrade() 里的操作是你想要的

# 第 3 步：执行迁移，更新数据库
alembic upgrade head
```

### 示例：给 Sample 表加一个 gc_content 字段

1. 在 `backend/db/models.py` 的 Sample 类里加：
   ```python
   gc_content = Column(Float, nullable=True)
   ```

2. 生成迁移：
   ```bash
   alembic revision --autogenerate -m "add gc_content to sample"
   ```

3. 应用迁移：
   ```bash
   alembic upgrade head
   ```

搞定，老数据不会丢。

### 其他常用命令

| 操作 | 命令 |
|------|------|
| 查看当前数据库版本 | `alembic current` |
| 查看迁移历史 | `alembic history` |
| 回退一步 | `alembic downgrade -1` |
| 回退到初始状态 | `alembic downgrade base` |

### 注意事项

- **只改了代码没跑迁移** → 程序会报 `no such column` 错误
- **SQLite 限制**：不支持直接删列或改列类型，Alembic 会自动用 batch 模式处理（建新表 → 复制数据 → 删旧表 → 重命名）
- 迁移脚本是代码的一部分，要一起提交到 git

## 10. 环境变量说明

| 变量 | 用途 | 示例 |
|------|------|------|
| `LLM_API_KEY` | AI 问答的 API Key | `sk-xxx` |
| `LLM_BASE_URL` | AI 接口地址 | `https://api.chatanywhere.tech/v1` |
| `LLM_MODEL` | AI 模型名称 | `gpt-4o-mini` |
| `DATABASE_URL` | 数据库连接（可选，默认用 config.yaml） | `sqlite:///./db_data/bioagent.db` |

配置优先级：环境变量 > `backend/config.yaml` > 默认值

## 11. 项目维护备忘

| 操作 | 怎么做 |
|------|--------|
| 改判读规则 | 编辑 `backend/core/rules.py`，跑 `pytest tests/test_rules.py -v` 验证 |
| 改判读阈值 | 编辑 `backend/rules_config.yaml`，或在 Web 端 Settings 页面改 |
| 改表结构 | 改 `backend/db/models.py` → `alembic revision --autogenerate -m "xxx"` → `alembic upgrade head` |
| 加新 API | 在 `backend/api/` 下加路由文件，在 `backend/main.py` 里 include |
| 加新页面 | 在 `frontend/pages/` 下加 `N_Name.py`，Streamlit 自动发现 |
| 跑全部测试 | `python -m pytest tests/ -v` |
| 更新依赖 | 编辑 `backend/requirements.txt`，跑 `pip install -r backend/requirements.txt` |
