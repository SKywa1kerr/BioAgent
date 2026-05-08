# BioAgent UI/UX 全面升级 — Claude Desktop 暖色工作台

**Date:** 2026-05-08
**Scope:** 全面重塑视觉系统、布局结构、初始化流程，并加入多厂家模型支持。

---

## 0. 设计目标

- **专业感**：让 BioAgent 看起来像一款打磨过的桌面软件，而不是 React 模板。
- **温润感**：参考 Claude Desktop / anthropic.com 的米白暖色调与对话优先布局。
- **数据友好**：色谱图、突变表、图表这类重数据组件需要专属空间，不被塞进对话气泡里。
- **保留现有能力**：CommandPalette、ShortcutsOverlay、Toast、i18n、Onboarding、自动更新等不丢功能。

## 1. 视觉系统 (Design Tokens)

### 调性
- 整体氛围：Claude Desktop — 对话优先、留白宽裕、Sans-serif 主导。
- 配色档位：**Subtle Warm**（柔和米白 + 中性深褐 + 柿橙强调色）。

### Light Theme

| Token | Value | 用途 |
|---|---|---|
| --bg-app | #faf8f3 | 主背景 |
| --bg-sidebar | #f3f0e8 | sidebar 背景 |
| --bg-canvas | #fcfaf6 | 数据画布背景（比主背景更亮） |
| --surface | #ffffff | 卡片 / 输入框 / 高亮行 |
| --text-main | #1c1a17 | 主文本 |
| --text-muted | #5a544c | 次要文本 |
| --text-subtle | #908a80 | 占位 / 元数据 |
| --text-faint | #b1a99c | 标签 / 大写小字 |
| --border-soft | #ece6dd | 卡片描边 |
| --border-default | #e8e3d8 | 分隔线 |
| --accent | #d97757 | 柿橙强调色 |
| --accent-hover | #c0623e | hover 态 |
| --accent-soft | rgba(217,119,87,0.10) | 焦点光晕 |

### Dark Theme

| Token | Value | 用途 |
|---|---|---|
| --bg-app | #1a1815 | 主背景 |
| --bg-sidebar | #131210 | sidebar 背景 |
| --bg-canvas | #1f1c19 | 数据画布背景 |
| --surface | #211e1a | 卡片 / 输入框 |
| --text-main | #f0ece5 | 主文本 |
| --text-muted | #908a80 | 次要文本 |
| --text-subtle | #6a6357 | 占位 / 元数据 |
| --border-soft | #26231e | 卡片描边 |
| --border-default | #2e2a24 | 分隔线 |
| --accent | #d97757 | 强调色（深浅一致） |
| --accent-hover | #e08e6f | hover 态 |

### 状态色（双主题共用色相）
- --status-ok: light #4a6a30 / dark #a3c180
- --status-warn: light #8a5a18 / dark #d9a86a
- --status-error: light #a04d29 / dark #e2a07a（与 accent 同色系）

### 突变 / 序列 trace 色
- 不动现有 --color-mutation-* 与 --color-trace-* 的语义。
- 把 light 主题中的纯红/纯蓝改成略偏暖版本（实现期 fine-tune）。

### 字体栈

```
--font-ui: Inter, Inter Variable, PingFang SC, Noto Sans SC, Microsoft YaHei, system-ui, sans-serif;
--font-display: Source Serif Pro, Source Serif, PingFang SC, Noto Serif SC, Georgia, serif;
--font-mono: JetBrains Mono, JetBrains Mono Variable, SF Mono, Menlo, Consolas, monospace;
```

- UI / 数字（含 stat 数字）→ Inter
- h1 / h2 / 大标题 / 列详情头 → Source Serif Pro
- 序列 / 代码 → JetBrains Mono
- 移除 Fraunces（替换为 Source Serif Pro）

### 字号 / 行高 / 字重

| Token | Value |
|---|---|
| --font-size-xs | 11 px |
| --font-size-sm | 12 px |
| --font-size-md | 13 px |
| --font-size-lg | 15 px |
| --font-size-xl | 18 px |
| --font-size-2xl | 22 px |
| --font-size-3xl | 28 px (stat-num) |
| --line-height-tight | 1.25 |
| --line-height-normal | 1.6 |

### 圆角 / 间距 / 阴影
- 间距：保留现有 4px 步进系统。
- 圆角：保留 --radius-sm/md/lg/xl/pill。
- 阴影：light 用更柔的 alpha；dark 主要靠分层背景；不再用蓝灰阴影。

### 动效
- 维持 --duration-fast/base/slow 与 --easing-standard。
- 整体倾向克制：sidebar row hover、按钮按下、modal 出现这类基础互动有过渡；不加额外的轮播/缩放装饰。

## 2. 整体布局

### 主结构（默认态）

```
+------------------------------------------------------+
|  TitleBar  ·  BioAgent                               |
+------------+-------------------+---------------------+
|            |                   |                     |
|  SIDEBAR   |   CHAT COLUMN     |   CANVAS COLUMN     |
|  200 px    |   0.85 fr         |   1.25 fr           |
|            |                   |                     |
|  会话/数据 |   对话历史        |   数据 / 图表       |
|  /面板史   |   composer        |                     |
|            |                   |                     |
+------------+-------------------+---------------------+
| Settings (sidebar 左下角固定按钮)                    |
+------------------------------------------------------+
```

### Sidebar
- 三段式（comfortable density）：
  1. **最近会话** — 时间倒序，激活态用白色实心背景。
  2. **数据集** — 已导入的 .ab1 文件列表 + 导入占位行。
  3. **面板历史** — 分析 / 趋势 / 建议（点击切换 canvas 当前 tab）。
- 顶部 + 新分析 / + 新会话 入口。
- **左下角固定 Settings 按钮**（新增需求 #2）：齿轮图标 + 当前模型名缩写，点击打开 SettingsModal。
- 宽度：默认 200 px；Ctrl+B 收起为 56 px 图标 rail。

### Chat 列
- 顶部 h2 对话（Source Serif）。
- 消息流：用户气泡靠右暖橘背景；assistant 气泡靠左白底（dark 模式深褐底）。
- composer：圆角 12px、白底（dark 用 --surface）、阴影 1px、内嵌发送按钮（柿橙背景 + 回车图标）。
- progress 条：保留现有 chat-progress-inline，色改为柿橙渐变。

### Canvas 列
- header：crumb（uppercase 小字）+ h2（Source Serif）+ tabs（pill 样式，激活态深底白字）。
- stats 行：3 列 KPI（数字 28px Inter 600 weight，标签 9px uppercase letter-spacing 0.08em）。
- 数据卡片：白底（--surface）+ 1px --border-soft + 10px 圆角 + 极淡阴影。
- 现有 ResultsWorkbench / ChromatogramCanvas / ResultsCharts / ResultsTable 保留功能、按新 token 重写样式。

### 可拖动分隔条 + 折叠态
- chat 与 canvas 之间放 1px 分隔条，鼠标进入时显示 6×26 px 暖灰握把，cursor col-resize。
- 拖动持久化 chat 列宽（localStorage bioagent-chat-width）。
- 拖到下限阈值（< 120 px）时自动折叠 chat 为 32 px 竖条 rail，内显纵向文字 expand chat + 箭头，点击恢复上次宽度。
- 折叠态下 Canvas 占据 chat + canvas 全部空间，色谱图自动加高（trace 高度从 56 px 提升到 80 px）。
- 现有 3 段 rail 切换按钮 (wide/narrow/hidden) 废弃，替换为分隔条拖动 + Ctrl+B 折叠 sidebar。

## 3. 初始化对话框 (InitDialog) 升级（新增需求 #1）

### 现状问题
- 用 FlaskConical lucide 图标 + 蓝色 accent，不在新调色板内。
- 字段固定 3 个 (apiKey / baseUrl / model)，无 provider 概念。

### 新设计
- **顶部图标 + 应用名**：用衬线大字 BioAgent + 简洁单色 logo（实现期定）。
- **Provider 选择器**（新增需求 #3）：第一个字段是下拉，预设：
  - OpenAI
  - Anthropic
  - DeepSeek
  - SJTU（保留现有）
  - Ollama (Local)
  - Custom
- 选 Provider 时联动填充 baseURL 默认值与 model 占位提示；用户仍可手动覆盖。
- API Key、Base URL、Model 三个字段保留，按 provider 上下文显示/隐藏（Ollama 不需要 key）。
- 提交按钮：柿橙填充 + 白字。
- 整体使用新 token，圆角 14 px，阴影柔和。
- 暗色 / 亮色都成立。

### 技术细节
- 新增 src/lib/providers.ts：导出 Provider 预设数组与查询函数。
- 类型签名：
  - ProviderId 联合类型：openai | anthropic | deepseek | sjtu | ollama | custom
  - ProviderPreset 接口：id, label, defaultBaseUrl, suggestedModels[], requiresApiKey
- 内置 PROVIDERS 数组示例值：
  - openai → https://api.openai.com/v1, models: gpt-4o-mini, gpt-4o
  - anthropic → https://api.anthropic.com/v1, models: claude-sonnet-4-6, claude-haiku-4-5
  - deepseek → https://api.deepseek.com/v1, models: deepseek-chat, deepseek-coder
  - sjtu → https://models.sjtu.edu.cn/api/v1, models: deepseek-chat
  - ollama → http://localhost:11434/v1, models: llama3.2, qwen2.5（requiresApiKey: false）
  - custom → 用户填写 baseURL 和 model
- 扩展 AgentSettings 类型增加 provider: ProviderId。
- 已有的 llmBaseUrl / llmModel 仍保留（覆盖 provider 默认值）。
- loadSettings / saveSettings 兼容旧数据：缺 provider 字段时按 baseURL 推断或回落 custom。

## 4. SettingsModal 升级

- 沿用新 token，与 InitDialog 共用 Provider 选择器组件。
- 入口固定在 sidebar 左下角；Ctrl+, 仍可打开。
- 字段同 InitDialog；提供切换 Provider 后是否清空 API Key 提示。
- 保存时调用 agent.shutdown 然后 agent.initialize(next) 重新连接。

## 5. CommandPalette / ShortcutsOverlay / Toast

- 视觉重塑：用新 token 替换。
  - palette 弹窗：暖白半透明背景 + backdrop-blur（dark 模式深褐半透明）。
  - 选中项背景：--accent-soft，左侧 2 px 柿橙竖线。
- 行为不变。

## 6. 不在范围（保持现状）

- 业务面板内部布局 (AnalysisPanel / MutationTrendPanel / LabSuggestionPanel) 只做 token 替换与小处节奏修整。
- ChromatogramCanvas / ResultsCharts / ResultsTable 渲染逻辑不动，只换 trace 色与卡片外壳。
- 现有 hooks (useAgentHarness, useChromatogramViewport, useAnalysisHistory 等) 不动。
- electron 主进程、preload、agent_harness 不动。
- 高对比度 (hc) 主题：本期不动。

## 7. 文件变更清单

### 新增
- src/styles/tokens.css — 全量重写
- src/lib/providers.ts — Provider 预设
- src/components/sidebar/Sidebar.tsx + .css — 新三段式 sidebar
- src/components/workbench/Splitter.tsx — 可拖动分隔条
- src/components/InitDialog/ProviderSelect.tsx（或合并入 InitDialog）

### 重写 / 大改
- src/styles.css — 应用层布局 (chat-panel / canvas-panel → 三栏 grid)
- src/components/InitDialog.tsx + .css
- src/components/SettingsModal.tsx
- src/components/ChatPanel.tsx + 新 css module
- src/components/TitleBar.css
- src/components/CommandPalette.css
- src/components/ui/Toast.css
- src/App.tsx — 移除 rail-wide/narrow/hidden 三态状态机

### 小改
- 各 panel .css（用新 token）
- src/lib/settingsStorage.ts — 兼容 provider 字段
- package.json — 移除 @fontsource-variable/fraunces，新增 Source Serif（如需自托管）

## 8. 测试

- typecheck：npm run typecheck 通过。
- 已有 e2e (playwright)：跑通 — 主要选择器（[role=log]、按钮 aria-label）保持不变。
- 视觉回归：light / dark 两态截图，覆盖：
  - 空态（未初始化）
  - 默认态（chat + canvas 各占一半）
  - 折叠 chat 态
  - 折叠 sidebar 态
  - InitDialog（每个 provider 一遍）
  - SettingsModal
  - CommandPalette / ShortcutsOverlay
  - 至少一个非空 Analysis / Trends / Suggestions 面板

## 9. 实现顺序建议

1. **Tokens** 先行：写新 tokens.css，全应用挂上 → 视觉漂移立刻可见。
2. **Providers + InitDialog**：因为是首屏，做完先验证暖色调对路。
3. **Sidebar**（含左下 Settings 入口）。
4. **Splitter** + 三栏布局重构。
5. **ChatPanel + Composer**。
6. **Canvas / Workbench 外壳**（card 样式 + tabs）。
7. **CommandPalette / Shortcuts / Toast** 视觉同步。
8. **业务面板**逐个 token 替换。
9. **打磨 + 视觉回归**。
