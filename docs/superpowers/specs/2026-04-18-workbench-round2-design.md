# Workbench Round 2 — 深化结果工作台

**Date:** 2026-04-18
**Scope:** 在 Round 1（App 拆分、错误边界、设置持久化、LLM 重试、虚拟化表格）基础上，把 `ResultsWorkbench` 从只读浏览升级为可筛选、可导出的分析工作台。

---

## 目标与非目标

### 目标
- 用户可按**判读状态**与**全文关键词**过滤结果；筛选状态跨会话保留。
- 用户可一键把当前（或筛选后）结果导出为 CSV、JSON、PDF（含中英文）。
- 筛选与导出同时作用于表格、汇总、图表，保持数据口径一致。
- 实施中顺手做一次 `React.memo` / `useCallback` / canvas 重绘审计。

### 非目标
- 不引入跨样本对比、详情抽屉、保存视图预设（留给后续迭代）。
- 不改动 `alignment.py` 比对逻辑或 agent harness 主流程。
- 不做分享链接 / 协作功能。

---

## 架构

```
samples (WorkbenchSample[])
        │
        ▼
useWorkbenchFilters  ──► filters (persist → localStorage)
        │
        ▼
filteredSamples ──┬──► ResultsTable  (虚拟化)
                  ├──► ResultsSummary
                  ├──► ResultsCharts
                  └──► ExportMenu (导出当前视图)
```

三层：`FilterBar`（顶部控件）→ `ResultsTable`（已虚拟化）→ `ExportMenu`（导出入口）。筛选派生数据集由 hook 统一产生，被三处消费者共享。导出逻辑独立在 `src/lib/exporters/`，不依赖 React。

---

## 组件清单

### 新建

| 文件 | 作用 |
|---|---|
| `src/components/workbench/FilterBar.tsx` | 状态 chips + 搜索框（防抖 200ms）+ 清除按钮 + 结果计数 |
| `src/hooks/useWorkbenchFilters.ts` | 管理 `{statuses: Set<Status>, query: string}`；产出 `filteredSamples`；持久化到 localStorage |
| `src/lib/exporters/csv.ts` | CSV 序列化（UTF-8 BOM，RFC 4180 转义） |
| `src/lib/exporters/json.ts` | JSON 序列化（带 `exportedAt` + `filters` 元信息） |
| `src/lib/exporters/pdf.ts` | pdfmake 报告生成（CJK 字体子集） |
| `src/lib/exporters/filename.ts` | 统一文件名格式 `bioagent-{dataset}-{YYYYMMDD-HHmm}.{ext}` |
| `src/components/workbench/ExportMenu.tsx` | 下拉菜单入口 |
| `tests/test_workbench_filters.mjs` | 筛选逻辑单测 |
| `tests/test_exporters.mjs` | 导出 smoke 测试 |

### 修改

| 文件 | 改动 |
|---|---|
| `src/components/workbench/ResultsWorkbench.tsx` | 装配 `useWorkbenchFilters`、顶部加 `FilterBar` + `ExportMenu`；向下游传 `filteredSamples` |
| `src/components/workbench/ResultsTable.tsx` | 消费 `filteredSamples`；空态 UI |
| `src/components/workbench/ResultsSummary.tsx` | 基于 `filteredSamples` 重新汇总 |
| `src/components/workbench/ResultsCharts.tsx` | 基于 `filteredSamples` 重绘 |
| `electron/preload.js` + `electron/main.js` | 新增 `showSaveDialog` IPC 通道 |

---

## 筛选规格

### 状态筛选
- 多选 chips：`ok` / `wrong` / `review` / `error`
- 默认全选。全选视为"无筛选"，不影响样本顺序。
- chip 点击切换单个状态；长按（或右键）=只保留该状态。

### 全文搜索
- 不区分大小写；跨字段匹配：`sample_id`, `gene`, `dataset`, `reason` (以及 `review_reason`/`llm_reason`/`auto_reason`), `aa_changes`。
- 输入防抖 200ms，避免频繁重新过滤。
- 搜索命中在表格中高亮关键词（仅文本字段，`<mark>` 标签）。

### 持久化
- filter 状态以 `bioagent-workbench-filters-v1` 为 key 存 localStorage。
- 读取时做 schema 校验，不兼容或异常时回退默认（全选、空搜索）。

### 空态
- 筛选后 0 条：提示"无匹配结果" + 主按钮"清除筛选"。

---

## 导出规格

### 公用
- 文件名：`bioagent-{dataset || "results"}-{YYYYMMDD-HHmm}.{ext}`
- 导出范围：**当前筛选后**的样本；`ExportMenu` 显示 `导出 X 条` 以明确口径。
- Web 端：通过 `Blob` + `URL.createObjectURL` 触发下载。
- Electron 端：若 `window.electronAPI?.invoke("showSaveDialog", ...)` 可用则走系统保存对话，否则回退 Blob 下载。

### CSV
- UTF-8 + BOM（Excel 兼容）。
- 字段（按此顺序）：`sample_id, dataset, gene, status, mutations, aa_changes, reason, q_mean, length`。
- 缺失值输出空字符串；数组字段用 `; ` 连接。
- 遵循 RFC 4180：含 `,`/`"`/换行 的字段加引号，内部 `"` 双写。

### JSON
```json
{
  "exportedAt": "2026-04-18T14:23:00+08:00",
  "filters": { "statuses": ["ok", "wrong"], "query": "" },
  "count": 42,
  "samples": [ /* 原始 WorkbenchSample[] */ ]
}
```
- 缩进 2 空格。

### PDF (pdfmake + CJK)
- 引擎：`pdfmake`（纯前端 JS，无需后端）。
- 字体：Noto Sans CJK SC Regular + Bold，预先用 `fonttools` 子集化到仅保留常用 GB2312+ASCII 字符（目标子集 < 400KB）；子集文件放在 `public/fonts/` 下，动态 fetch。
- 首次使用 PDF 时才加载字体（避免影响启动）；加载失败回退提示"PDF 不可用，请检查网络或使用 CSV/JSON"。
- 报告结构：
  1. **封面**：标题、dataset 名、导出时间、筛选条件摘要、统计（总数 / ok / wrong / review / error）
  2. **汇总表**：与 `ResultsSummary` 同口径的分布图（静态柱状图，用 pdfmake 原生绘图）
  3. **样本明细**（每样本一节）：基本信息表 + 突变列表 + aa_changes + reason；若样本有色谱图数据，用 `Canvas.toDataURL` 截取当前渲染图嵌入（320px 宽）
- 单样本模式：仅当前聚焦样本一节。
- 批量模式：按 dataset 分组；样本数 > 500 时警告 toast "建议按数据集分批导出" 并允许继续。

---

## 错误与边界
- 导出异常统一以 toast 提示，日志写 `console.error` + debug log（沿用现有通道）。
- 字体加载失败：PDF 按钮禁用并显示提示；CSV/JSON 不受影响。
- 筛选结果 0 条时 ExportMenu 禁用并提示 "无数据可导出"。
- localStorage 读写失败（隐私模式等）：回退内存态，不阻塞渲染。

---

## 性能审计（顺手做）
- `ResultsTable` 行组件用 `React.memo` 包裹，确保滚动不触发无关重渲染。
- `ResultsCharts` 的 recharts 数据 `useMemo` 缓存。
- `ChromatogramCanvas`：检查是否在 filter 变化时无效重绘，必要时加 `prevSample === sample` 短路。
- `ResultsWorkbench` 顶层 `useCallback` 包装传给子组件的回调。
- 记录审计前后 React DevTools Profiler 截图存 `docs/perf/2026-04-18-round2-before.png` / `-after.png`。

---

## 测试

### 自动化
- `tests/test_workbench_filters.mjs`：
  - 纯状态筛选（单选 / 多选 / 全选 = 无筛选）
  - 纯文本搜索（大小写、跨字段、特殊字符）
  - 组合：状态 + 搜索
  - localStorage schema 破损时回退默认
- `tests/test_exporters.mjs`：
  - CSV 转义（含逗号、引号、换行的字段）
  - CSV UTF-8 BOM 存在
  - JSON schema 与元信息正确
  - PDF docDefinition 生成不抛异常（smoke，不校验二进制）

### 手动回归
- 10k+ 样本下筛选响应 < 200ms；虚拟化滚动 60fps
- 暗色模式下 FilterBar 对比度达标
- PDF 中英文字符混排显示正确
- Electron 打包版导出保存对话正常

---

## 实施顺序（分阶段合入）

| 阶段 | 内容 | 可交付 |
|---|---|---|
| **1** | `useWorkbenchFilters` + `FilterBar` + 下游接入 | 最小可用筛选；PR #1 |
| **2** | `ExportMenu` + CSV + JSON + Electron 保存对话 | 基础导出；PR #2 |
| **3** | PDF 导出（字体子集化 → 单样本 → 批量） | 完整报告；PR #3 |
| **4** | 持久化打磨 + 空态 + 测试补齐 + 性能审计 | 收尾；PR #4 |

每个 PR 独立可部署；阶段之间最多 1 个 PR 未合并。

---

## 风险

| 风险 | 缓解 |
|---|---|
| CJK 字体子集化流程复杂 | 预先用 Python `fonttools` 离线生成；把子集文件检入仓库，避免构建期依赖 |
| pdfmake bundle 体积 | 动态 `import()` 懒加载，首次导出时才下载 chunk |
| 筛选 + 虚拟化交互 Bug | `@tanstack/react-virtual` 的 `measureElement` 在数据长度变化时需要 `measure()`，已在 hook 文档说明 |
| localStorage 跨会话脏数据 | schema 版本号 `v1`，升级时弃用旧 key |
