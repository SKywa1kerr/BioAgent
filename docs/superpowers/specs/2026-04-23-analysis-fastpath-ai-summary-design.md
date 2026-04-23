# Analysis Fast Path + AI Summary Design

**Date:** 2026-04-23
**Scope:** 优化桌面端“分析 base/pro/promax 数据集”这类明确请求的响应速度，同时保留 AI 的助手式讲解特色。目标是让结果工作台优先出现，AI 总结在后台自动生成并补到界面中。

---

## 目标与非目标

### 目标
1. **明确分析请求直达工具**：对“分析 pro 数据集”“analyze base dataset”这类高确定性请求，跳过首轮外部 LLM 工具决策，直接调用本地 `analyze_sequences`。
2. **结果优先展示**：本地分析一完成，立即渲染 `AnalysisPanel` / `ResultsWorkbench`，不等待 AI 总结。
3. **保留 AI 助手特色**：每次分析完成后，自动生成一段偏助手讲解型总结，说明整体情况、重点样本、可能原因和建议。
4. **开放式追问继续走 AI**：像“为什么这个样本判错”“这批数据像不像引物问题”这类问题仍走现有 LLM + 工具协作流。

### 非目标
- 不重写 `core/alignment.py` 的底层比对算法。
- 不改变 MCP server / tool schema 的基础协议。
- 不做新的多页面或复杂 UI 重构。
- 不把所有自然语言都规则化；仅覆盖高确定性命令。

---

## 问题定义

当前“分析 pro 数据集”的交互链路是：

1. 前端把用户消息交给 `electron/agent_harness.mjs`
2. 外部 LLM 先判断是否调用 `analyze_sequences`
3. 本地 MCP 工具执行分析
4. agent harness 再次调用外部 LLM 生成最终回复
5. 前端为详情再补拉一次 `get_analysis_detail`

本地实测 `python run.py --dataset pro --no-llm` 约为 4.8 秒，说明“回复特别慢”的主要原因不是 `pro` 数据本体分析，而是“明确命令也要走 2 次外部 LLM + 额外 detail round-trip”的交互链路过重。

---

## 推荐方案

采用 **规则直达 + AI 讲解** 的混合模式：

- **规则层** 负责识别高确定性命令，并直接触发 MCP 工具
- **结果层** 负责优先渲染结构化分析结果
- **AI 层** 负责异步生成助手式总结与后续追问

这样 AI 仍然保留在高价值位置：解释、归纳、建议、追问，而不是浪费在“判断用户是不是想分析数据集”这一步。

---

## 用户体验

### 目标交互

当用户输入：

- `分析 pro 数据集`
- `分析 base 数据集`
- `分析 promax 数据集`
- `analyze pro dataset`

预期行为：

1. 聊天区立即显示“开始分析 `pro` 数据集”
2. 进度条直接进入工具执行阶段，而不是长时间停留在“thinking”
3. 分析完成后，`AnalysisPanel` 立即显示样本表格、工作台和详情入口
4. 面板顶部或聊天区出现“正在生成助手解读”
5. 数秒后自动追加 AI 总结

### AI 讲解风格

AI 总结不是正式报告，而是助手式说明，建议包含 4 段内容：

1. **整体概览**：这批样本整体是否健康、有没有明显问题
2. **重点样本**：优先关注哪几个样本
3. **原因解释**：结合 coverage / frameshift / aa_changes / quality 做简明解释
4. **下一步建议**：重测、复核、关注引物/模板质量等

---

## 架构

```text
User message
  -> AgentHarness fast intent matcher
     -> if explicit dataset analysis:
        -> MCP analyze_sequences
        -> emit analysis result immediately
        -> start background AI summary generation
        -> emit summary when ready
     -> else:
        -> existing LLM tool-routing flow
```

### 分层职责

- `electron/agent_harness.mjs`
  - 新增明确命令匹配
  - 负责“快路由分析”与“后台总结”编排
  - 非明确命令仍走现有 `runTurn`

- `src/hooks/useAgentHarness.ts`
  - 处理新增的“analysis summary pending / ready”事件
  - 保证主结果到达时先渲染工作台
  - AI 总结追加到聊天区，不阻塞面板

- `src/components/panels/AnalysisPanel.tsx`
  - 保持面板主职责：展示结构化结果
  - 如需要，可在顶部显示“助手解读生成中”占位文案

- Python tool layer
  - 继续复用 `analyze_sequences`
  - 避免分析完成后再做不必要的重复 detail 拉取或深拷贝链路

---

## 明确命令匹配规则

### 覆盖范围

仅覆盖高确定性命令，不尝试理解复杂自由表达。

支持的输入模式：

- `分析 base 数据集`
- `分析 pro 数据集`
- `分析 promax 数据集`
- `analyze base dataset`
- `analyze pro dataset`
- `analyze promax dataset`

可接受的轻微变体：

- 前后空格
- 中英文大小写差异
- 可选的“请”“帮我”“一下”等礼貌词

### 不覆盖的输入

- `帮我看看 pro 这批数据有什么问题`
- `分析之后顺便给我实验建议`
- `为什么 C366-3 是错的`

这些保留给现有 LLM 路由，因为需要更强的语义理解和多步工具编排。

---

## 事件流设计

### 新增事件

在 `electron/agent_harness.mjs` / `useAgentHarness.ts` 之间增加两类事件：

- `summary_pending`
  - 含义：主分析结果已到，AI 总结正在后台生成
- `summary_ready`
  - 含义：AI 助手总结生成完成，可追加到聊天区

### 顺序要求

对于快路由分析请求，事件顺序必须是：

1. `tool_call` / `tool_result` for `analyze_sequences`
2. `AnalysisPanel` 显示结构化结果
3. `summary_pending`
4. `summary_ready`

禁止把 `summary_ready` 作为主结果展示的前置条件。

---

## 结果与总结的数据来源

AI 总结应直接基于已得到的结构化分析结果，而不是重新组织完整用户对话上下文。

推荐输入给 LLM 的总结上下文：

- dataset 名称
- sample_count
- 样本级摘要（限制条数或限制字段，避免 prompt 过大）
  - sid
  - bucket / status
  - identity
  - cds_coverage
  - frameshift
  - aa_changes
  - other_read_issues

这样可以：

- 限制 token 成本
- 保证总结延迟可控
- 避免把整个原始 detail 大对象再次塞回模型

---

## 文件与职责

### 修改

| 文件 | 改动 |
|---|---|
| `electron/agent_harness.mjs` | 新增明确分析命令匹配；新增快路由执行函数；新增后台 AI 总结生成；减少不必要的 detail round-trip |
| `src/hooks/useAgentHarness.ts` | 识别 `summary_pending` / `summary_ready`；保证分析结果优先显示；控制聊天区文案 |
| `src/components/panels/AnalysisPanel.tsx` | 可选展示“助手解读生成中”状态，不阻塞主表格 |
| `tests/test_agent_harness.mjs` | 回归测试：明确分析命令不依赖首轮 LLM 决策；分析结果先于总结事件发出 |

### 可选修改

| 文件 | 原因 |
|---|---|
| `electron/main.js` | 如需补充 IPC trace 文案或调试信息 |
| `src/i18n.ts` | 新增“正在生成助手解读”等文案 |

---

## 测试策略

### 自动化

1. `tests/test_agent_harness.mjs`
   - 明确分析命令命中快路由时，不调用首轮 `client.chat.completions.create`
   - `analyze_sequences` 返回后立即发出分析结果事件
   - 总结事件在结果事件之后发出
   - 开放式请求仍走旧的 LLM 路由

2. 如前端文案或事件分支变更较大，可补对应 hook 级测试；否则先以 harness 测试为主

### 手动回归

- 输入 `分析 pro 数据集`
  - 结果工作台应比当前更早出现
  - AI 总结随后自动补上
- 输入 `为什么 C366-3 是错的`
  - 仍走 AI 路由
- 输入 `显示突变趋势`
  - 当前行为不退化

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 规则匹配过宽，误把自由表达当成快路由命令 | 只覆盖极窄模式；命中条件保守 |
| 背景 AI 总结失败，用户以为分析失败 | 主结果与总结彻底解耦；总结失败仅提示“助手解读生成失败”，不影响结果面板 |
| 总结 prompt 过大导致第二段 LLM 仍然偏慢 | 只传摘要字段，不传完整 detail / chromatogram 数据 |
| 事件时序错乱导致聊天区或面板重复更新 | 为新增事件建立明确顺序测试 |

---

## 后续优化清单

本轮不一定全部实现，但建议记录为后续 backlog：

1. **Python 侧并行分析样本**
   - `analyze_dataset` 当前完全串行；数据量更大时收益明显

2. **HTML 输出改为可选**
   - 若用户只看工作台，不必每次都写每个样本的 HTML 文件

3. **GenBank / clone 匹配缓存**
   - 减少重复解析和扫描

4. **详情对象瘦身**
   - 当前样本 detail 较大，跨进程传输和深拷贝成本可继续压缩

5. **助手总结流式输出**
   - 如果后续希望更强“对话感”，可把总结改成增量流式显示

---

## 回滚

- 若快路由带来误判，可回滚到原始全 LLM 路由
- 新逻辑尽量封装在 `AgentHarness` 内，避免扩散到多个模块
- 保持旧路径完整存在，回退成本低
