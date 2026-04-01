# BioAgent Desktop — Agent 侧边栏设计规格

## 目标

在 BioAgent Desktop 的分析页右侧加入一个可展开/收起的 Agent 聊天侧边栏。用户用中文输入指令（如「帮我分析今天导入的数据」「哪些样本有突变」），Agent 调用工具执行任务并给出专业解读。

## 架构：混合模式

前端控制流程节奏（展示计划、等用户确认、逐步执行），Python 端提供工具层和 LLM 代理。

```
用户输入中文 → IPC("agent-chat") → Python agent_chat
  → LLM 返回工具调用计划
  → 前端展示计划卡片，用户点「确认执行」
  → 前端逐个调用 Python 工具 IPC
  → 每步结果实时展示
  → IPC("agent-chat") 把工具结果发给 LLM
  → LLM 返回中文总结
  → 前端渲染总结
```

## UI 设计

### 侧边栏面板

- 位置：分析页右侧，可展开/收起
- 展开宽度：360px
- 收起时：显示一个小的 Agent 图标按钮
- 展开/收起按钮在右上角

### 消息类型

| 类型 | 说明 | 样式 |
|------|------|------|
| user | 用户输入的中文指令 | 右对齐气泡 |
| text | Agent 的文本回复 | 左对齐气泡 |
| plan | 工具调用计划 | 卡片+步骤列表+「确认执行」「取消」按钮 |
| tool_status | 工具执行状态 | 紧凑行：⏳进行中 / ✅完成 / ❌失败 |
| summary | Agent 的分析总结 | 左对齐气泡，可点击样本 ID 跳转 |

### 联动

- 点击结果中的样本 ID → 左侧分析页跳转到该样本
- `run_analysis` 的结果直接更新分析页的样本列表
- 两侧共享同一份 `samples` 状态

### 美观要求

- 配色与现有 UI 一致（蓝色主色 #2a6cb6，白底灰字）
- 消息气泡有柔和圆角和阴影
- 工具状态紧凑但清晰
- 计划卡片有明显的视觉分隔
- 输入框带发送按钮，支持 Enter 发送

## 工具定义

6 个工具，每个对应一个 IPC 通道：

### run_analysis
- 功能：执行完整分析流程
- 参数：`{ab1Dir: string, genesDir?: string, plasmid?: string}`
- 返回：分析结果 JSON（样本列表）
- 复用现有 `run-analysis` IPC

### query_samples
- 功能：查询/筛选当前或历史分析的样本
- 参数：`{filter?: "ok"|"wrong"|"uncertain", sampleId?: string, analysisId?: string}`
- 返回：匹配的样本列表
- 新增 IPC `agent-query-samples`

### query_history
- 功能：查询分析历史记录
- 参数：`{limit?: number}`
- 返回：历史记录列表
- 复用现有 `get-history` IPC

### compare_datasets
- 功能：对比两次分析的差异
- 参数：`{analysisId1: string, analysisId2: string}`
- 返回：差异摘要（新增/消失的突变、状态变化）
- 新增 IPC `agent-compare`

### export_report
- 功能：导出 Excel 报告
- 参数：`{samples: Sample[], sourcePath?: string}`
- 返回：导出文件路径
- 复用现有 `export-excel` IPC

### get_sample_detail
- 功能：获取单个样本的完整信息
- 参数：`{sampleId: string}`
- 返回：完整样本数据（含突变列表、比对信息）
- 新增 IPC `agent-sample-detail`

## Python 端设计

### agent_chat.py

核心对话模块，职责：
1. 接收前端发来的消息和对话历史
2. 构造 system prompt + 工具描述 + 用户消息
3. 调用 OpenRouter LLM（通过现有 llm_client.py）
4. 解析 LLM 返回的 JSON（工具调用或文本回复）
5. 返回结构化响应给前端

**LLM 调用方式**：不依赖 function calling API，使用 prompt 内嵌工具描述 + JSON 输出格式。这样兼容任何模型（包括免费模型）。

**请求格式**：
```json
{"message": "帮我分析今天导入的数据", "history": [...], "context": {"currentSamples": [...], "currentDir": "..."}}
```

**响应格式**：
```json
// 文本回复
{"type": "text", "content": "好的，让我看看..."}

// 工具调用计划
{"type": "tool_calls", "message": "我来帮你分析数据", "calls": [
  {"tool": "run_analysis", "args": {"ab1Dir": "/path/to/data"}}
]}

// 基于工具结果的总结
{"type": "summary", "content": "分析完成，共12个样本，9个OK，2个突变..."}
```

### agent_tools.py

工具注册与执行层，职责：
1. 定义工具的元信息（名称、描述、参数schema）
2. 生成工具描述文本供 system prompt 使用
3. 提供 `query_samples`、`compare_datasets`、`get_sample_detail` 的实现（新增工具）

### System Prompt

```
你是 BioAgent，一个专业的 Sanger 测序质控助手。

你可以使用以下工具（通过返回 JSON 调用）：
1. run_analysis — 分析测序数据
2. query_samples — 查询/筛选样本
3. query_history — 查看分析历史
4. compare_datasets — 对比两次分析
5. export_report — 导出 Excel 报告
6. get_sample_detail — 查看样本详情

用户会用中文与你交流。你应该：
- 理解用户的意图，选择合适的工具
- 对分析结果给出专业的中文解读
- 标注需要关注的异常样本并解释原因
- 用简洁清晰的语言回复

当你需要调用工具时，返回以下 JSON 格式：
{"action": "tool_calls", "calls": [{"tool": "工具名", "args": {...}}]}

当你直接回复用户时，返回：
{"action": "reply", "content": "你的回复"}
```

## 前端设计

### 新增组件

**AgentPanel.tsx** — 侧边栏容器
- 展开/收起状态
- 消息列表（滚动）
- 输入框 + 发送按钮
- 管理对话历史 `ChatMessage[]`
- 处理工具调用的确认/取消/执行流程

**ChatMessage.tsx** — 单条消息渲染
- 根据 `type` 渲染不同样式
- plan 类型渲染步骤列表和按钮
- tool_status 渲染执行进度
- summary 中的样本 ID 可点击

### 类型定义

```typescript
interface ChatMessage {
  id: string;
  type: "user" | "agent" | "plan" | "tool_status" | "error";
  content?: string;
  toolCalls?: ToolCall[];
  toolName?: string;
  toolStatus?: "running" | "done" | "failed";
  toolResult?: string;
  timestamp: number;
}

interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  description?: string;
}

interface AgentResponse {
  type: "text" | "tool_calls" | "summary";
  content?: string;
  message?: string;
  calls?: ToolCall[];
}
```

### 控制流（前端 TypeScript）

```
1. 用户输入 → push user message → 发 IPC("agent-chat")
2. 收到 {type: "text"} → push agent message → 结束
3. 收到 {type: "tool_calls"} → push plan message → 等用户确认
4. 用户点「确认」→ 逐个执行工具：
   a. push tool_status(running)
   b. 调用对应 IPC
   c. 更新 tool_status(done/failed)
   d. 如果是 run_analysis → 同时更新分析页 samples
5. 全部完成 → 发 IPC("agent-chat") 带工具结果
6. 收到 {type: "summary"} → push agent message → 结束
```

## 文件清单

### 新增
- `src-python/bioagent/agent_chat.py` — Agent LLM 对话核心
- `src-python/bioagent/agent_tools.py` — 工具注册与新增工具实现
- `src/components/AgentPanel.tsx` — 侧边栏面板
- `src/components/AgentPanel.css` — 侧边栏样式
- `src/components/ChatMessage.tsx` — 消息渲染
- `src/components/ChatMessage.css` — 消息样式

### 修改
- `electron/main.js` — 新增 `agent-chat`、`agent-query-samples`、`agent-compare`、`agent-sample-detail` IPC handlers
- `src/App.tsx` — 集成 AgentPanel，管理展开/收起，传递 samples 状态和回调
- `src/App.css` — 布局调整（弹性布局支持侧边栏）
- `src/types/index.ts` — 新增 ChatMessage、ToolCall、AgentResponse 类型
- `src-python/bioagent/main.py` — 新增 `--agent-chat` 子命令入口

## LLM 配置

- 开发测试：使用 SJTU API（base_url 和 key 在设置页配置）
- 生产环境：用户在设置页填写自己的 API Key 和 Base URL
- 默认模型：通过设置页可配置，首版默认 `google/gemma-3-27b-it:free`
- 设置页已有 API Key 和 Base URL 字段，后续可加模型选择下拉框

## 错误处理

- API Key 未配置 → Agent 首条消息提示「请先在设置页配置 API Key」
- LLM 返回格式不对 → 显示「理解失败，请换个说法试试」
- 工具执行失败 → 工具状态显示 ❌ + 错误信息，不影响其他工具
- 网络超时 → 显示重试按钮

## 不在首版范围

- 对话历史持久化（刷新清空）
- 流式输出（首版等完整响应）
- 语音输入
- 多轮自动推理（每次工具调用都需要用户确认）
