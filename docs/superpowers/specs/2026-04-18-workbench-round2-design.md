# Workbench Round 2 — 深化结果工作台

**Date:** 2026-04-18
**Scope:** 在已有筛选/搜索/排序基础上补齐**导出**能力、**持久化**筛选状态、统一**数据口径**（汇总/图表随筛选联动），并顺手做一次性能审计。

> **现状核对（来自代码阅读）**：`ResultsWorkbench.tsx` 已有 `statusFilter + searchQuery + sortKey` 三元状态和 `buildResultsView`（`utils.ts`）。状态 bucket 为 `ok | wrong | uncertain | untested`（无 `error`，`untested` 以 `reason === "???"` 识别）。`ResultsCharts` / `ResultsSummary` 当前接收的是**原始 samples**，与表格展示的 `visibleSamples` 不一致。

---

## 目标与非目标

### 目标
1. **导出**：CSV / JSON / PDF（pdfmake + CJK），按当前 `visibleSamples` 导出。
2. **筛选持久化**：`statusFilter` / `searchQuery` / `sortKey` 存 localStorage，跨会话恢复。
3. **数据口径统一**：`ResultsSummary` 与 `ResultsCharts` 随筛选联动（可通过顶部切换"汇总范围：全部/筛选后"，默认"筛选后"）。
4. **性能审计**：`React.memo` / `useCallback` / `useMemo` 补齐；虚拟化行组件记忆；记录 Profiler 前后数据。

### 非目标
- 不新增筛选维度（突变数阈值、质量阈值、基因等暂不做）。
- 不改 `buildResultsView` 的过滤语义或 `bucketSampleStatus` 规则。
- 不做跨样本详情抽屉、标签、保存视图预设（留给后续迭代）。
- 不触碰 `alignment.py` 与 agent harness 主流程。

---

## 架构

```
samples (原始)
   │
   ├──► useWorkbenchControls  ──► { statusFilter, searchQuery, sortKey, summaryScope }
   │        │                              (persist → localStorage v1)
   │        ▼
   └──► buildResultsView() ──► visibleSamples
                                 │
                                 ├──► ResultsTable      (已虚拟化)
                                 ├──► ResultsSummary    (summaryScope=filtered 时消费 visible)
                                 ├──► ResultsCharts     (同上)
                                 └──► ExportMenu        (导出 visibleSamples)
```

控制状态抽出为 `useWorkbenchControls` hook，原 `ResultsWorkbench` 内联的 `useState` 迁入；行为保持不变，仅新增持久化与 `summaryScope` 字段。

---

## 组件清单

### 新建

| 文件 | 作用 |
|---|---|
| `src/hooks/useWorkbenchControls.ts` | 管理三元控制状态 + `summaryScope`；读写 localStorage；提供 `reset()` |
| `src/lib/exporters/csv.ts` | RFC 4180 转义 + UTF-8 BOM |
| `src/lib/exporters/json.ts` | 结构化导出（元信息 + filters + 样本） |
| `src/lib/exporters/pdf.ts` | pdfmake 报告生成（**动态 import**） |
| `src/lib/exporters/filename.ts` | `bioagent-{dataset || "results"}-{YYYYMMDD-HHmm}.{ext}` |
| `src/lib/exporters/saveFile.ts` | Blob 下载 + Electron `showSaveDialog` fallback |
| `src/components/workbench/ExportMenu.tsx` | 下拉菜单入口（CSV/JSON/PDF） |
| `src/components/workbench/SummaryScopeToggle.tsx` | 全部 / 筛选后 切换（位于 Summary 头部） |
| `tests/test_workbench_controls.mjs` | hook 持久化与恢复测试 |
| `tests/test_exporters.mjs` | 导出 smoke + CSV 转义 |
| `public/fonts/NotoSansSC-subset.ttf` | CJK 字体子集（离线用 fonttools 预生成） |

### 修改

| 文件 | 改动 |
|---|---|
| `src/components/workbench/ResultsWorkbench.tsx` | 接入 `useWorkbenchControls`；`ExportMenu` 放在 toolbar 右侧；向 Summary/Charts 传筛选后数据 |
| `src/components/workbench/ResultsSummary.tsx` | 接受 `samples` 改为"被汇总的集合"，支持 scope 切换 |
| `src/components/workbench/ResultsCharts.tsx` | 同上 |
| `src/components/workbench/ResultsTable.tsx` | 行组件 `React.memo`；`toChromatogramData` 结果 `useMemo` |
| `electron/main.js` + `electron/preload.js` | 新增 `showSaveDialog` IPC |
| `src/i18n.ts` | 新增导出菜单、scope toggle 相关文案 |

---

## 持久化规格

- Key: `bioagent-workbench-controls-v1`
- 值: `{ statusFilter, searchQuery, sortKey, summaryScope }`
- 写：`setState` 后 debounce 300ms 写入（避免搜索每次按键都写）
- 读：挂载时读取 → schema 校验 → 非法字段回退默认
- 默认：`{ statusFilter: "all", searchQuery: "", sortKey: "status", summaryScope: "filtered" }`

---

## 导出规格

### 公用
- 文件名：`bioagent-{dataset || "results"}-{YYYYMMDD-HHmm}.{ext}`（`dataset` 取首个样本的 `dataset` 字段；若不可得则 `results`）
- 导出范围：**当前 `visibleSamples`**；`ExportMenu` 菜单上显示 `导出 X 条` 明确口径
- Electron：若 `window.electronAPI.invoke("showSaveDialog", { defaultPath, filters })` 可用则走系统对话；失败回退 Blob 下载
- 无样本时（`visibleSamples.length === 0`）菜单按钮禁用 + tooltip 提示

### CSV
- UTF-8 + BOM
- 字段顺序：`id, name, clone, status, reason, identity, cds_coverage, sub, ins, del, aa_changes, avg_quality`
  - （字段名来自 `WorkbenchSample`，缺失则回退同义：`sub_count/sub`, `ins_count/ins`, `del_count/dele`, `cds_coverage/coverage`, `avg_qry_quality/avg_quality`）
- 数组字段（`aa_changes`）用 `; ` 连接；缺失值输出空字符串
- RFC 4180：含 `,` / `"` / `\n` 的字段加引号，内部 `"` 双写

### JSON
```json
{
  "exportedAt": "2026-04-18T14:23:00+08:00",
  "filters": { "statusFilter": "wrong", "searchQuery": "", "sortKey": "status" },
  "count": 42,
  "samples": [ /* 原样 WorkbenchSample[]，不裁剪 */ ]
}
```
- 缩进 2 空格。

### PDF (pdfmake + CJK)
- **引擎**：`pdfmake`（纯前端）
- **字体**：Noto Sans SC Regular + Bold，离线用 `pyftsubset` 预生成**常用 7k 字**子集（目标 < 400KB/字重）；子集文件检入 `public/fonts/`
- **懒加载**：`pdf.ts` 用动态 `import("pdfmake/build/pdfmake")`；首次导出 PDF 时才加载并 `fetch` 字体转 base64 注入 `pdfMake.vfs`
- **报告结构**：
  1. **封面**：标题、dataset、导出时间、筛选摘要、分布统计（total/ok/wrong/uncertain/untested）
  2. **汇总柱状图**：用 pdfmake `canvas` 指令原生绘制（避免额外图表库）
  3. **样本明细**（每样本一节）：基本信息表 + 突变表（`mutations[]`）+ `aa_changes` + `reason`；样本数 > 200 时**仅**导出概要，不含明细表，并警告
- **长文本**：`reason` 超过 400 字符截断 + "..."；提示用户查看 JSON 导出获取完整内容
- **失败回退**：字体 fetch 或 pdfmake 加载失败 → toast 错误 + PDF 按钮短暂禁用 30s

---

## 错误与边界

| 场景 | 处理 |
|---|---|
| 导出时 agent 仍在运行 | 允许导出；`exportedAt` 用当时时间；不阻塞 agent |
| localStorage 不可用（隐私模式） | 回退内存态；首次挂载 `console.warn` 一次 |
| `visibleSamples` 为 0 | ExportMenu 禁用 |
| 样本数 > 500（CSV/JSON） | 正常导出，不警告 |
| 样本数 > 200（PDF 明细模式） | 切换为概要模式 + 警告 toast |
| Electron `showSaveDialog` 抛异常 | 回退到 Blob 下载路径 |

---

## 性能审计（顺手做）

- `ResultsTable`：行组件抽出为 `Row` 并 `React.memo`，prop 对比浅比较；`toChromatogramData` 结果 `useMemo`
- `ResultsCharts`：recharts 数据集 `useMemo`
- `ResultsWorkbench`：`useMemo` 已覆盖 `visibleSamples`，顶层回调 `useCallback`
- `ChromatogramCanvas`：已在 `ResultsTable` 懒加载；确认 filter 变化时不触发被折叠行的无效渲染
- 记录基准：10k 样本模拟数据下 Profiler 截图存 `docs/perf/2026-04-18-round2-before.png` / `-after.png`

---

## 测试

### 自动化（Node 测试 runner 沿用 `tests/*.mjs` 约定）
- `tests/test_workbench_controls.mjs`
  - 默认值、持久化往返、schema 不合法时回退、debounce 不丢最后一次写
- `tests/test_exporters.mjs`
  - CSV：含 `,` / `"` / `\n` 字段正确转义；BOM 存在；空数组字段输出空串
  - JSON：schema 正确；`count === samples.length`
  - PDF：`buildDocDefinition(samples, filters)` 不抛异常（不实际生成二进制）

### 手动回归
- 10k 样本（构造假数据脚本 `tests/fixtures/generate_large.mjs`）筛选响应 < 200ms；滚动 60fps
- 暗色模式 + 简体中文 下 FilterBar / ExportMenu 对比度达标
- PDF 中英文混排（样本名中文、基因名英文、reason 含 LLM 中文理由）显示正确
- Electron 打包版 `showSaveDialog` 出现在正确路径

---

## 实施顺序（分阶段合入）

| 阶段 | 内容 | 可交付 |
|---|---|---|
| **1** | `useWorkbenchControls` hook 抽取 + 持久化 + `summaryScope` 联动 Summary/Charts | PR #1（行为增强，无新 UI） |
| **2** | `ExportMenu` + CSV + JSON + Electron 保存对话 + filename 工具 | PR #2（基础导出） |
| **3** | PDF 导出（字体子集 → 动态 import → 单样本 → 批量概要） | PR #3（完整报告） |
| **4** | 性能审计 + 测试补齐 + 空态/禁用态打磨 | PR #4（收尾） |

每个 PR 独立可部署；阶段之间最多 1 个未合并的 PR。

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| CJK 字体子集化流程繁琐 | 离线用 `pyftsubset` 生成，文件检入仓库；文档写入 `docs/fonts/README.md` |
| pdfmake bundle 体积（~700KB） | 动态 `import()` 懒加载，首屏不受影响 |
| 持久化搜索串回填时触发意外筛选 | 挂载时显示 "已恢复上次筛选" toast（可关闭） |
| Summary/Charts 联动改动数据口径，破坏现有用户心智 | 提供 scope toggle，默认 "筛选后" 但可切回 "全部" |
| 字体许可 | Noto Sans 为 SIL Open Font License，允许分发 |
| 大样本 PDF 明细过长 | > 200 切换概要模式并提示 |

---

## 回滚
- PR #1 若引入回归：还原 `ResultsWorkbench.tsx` 顶部 `useState` 块即可。
- PR #2/#3 完全新增，删除新建文件即回退。
- localStorage key 采用 `-v1` 后缀，未来破坏性变更用 `-v2` 独立 key。
