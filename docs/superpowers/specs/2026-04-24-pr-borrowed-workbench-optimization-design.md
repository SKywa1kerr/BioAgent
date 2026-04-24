# PR-Borrowed Workbench Optimization Design

**Date:** 2026-04-24
**Scope:** 借鉴 `origin/archive/assistant-tool` 已合并 PR 中的实现思路，优化当前 `worktree-ultimate-bioagent` 的结果详情、序列/色谱可视化、数据归一化和前端性能。目标是吸收可复用设计，不直接合并或搬迁不相关架构。

---

## 背景

`origin/archive/assistant-tool` 已合并 PR `40cc5a2`，其中包含 `tmp/merge-desktop-agent` 的改动。该 PR 和当前分支没有共同历史，直接合并会产生大量核心文件 `add/add` 冲突，因此本轮不做 merge。

对比后可以确认：当前分支已经有更完整的 `ResultsWorkbench`、筛选、导出、drawer、命令面板和后台分析快路径。PR 中更值得借鉴的是局部实现思路：

1. gapped alignment 坐标映射和 CDS 高亮
2. CDS 起点驱动的 codon/AA 展示
3. chromatogram worker/offscreen rendering 的性能方向
4. snake_case / camelCase 字段归一化工具
5. feature-oriented 分层的边界思想

本设计采用分阶段、小范围吸收，不迁移 PR 的完整目录结构，不引入 Zustand 作为当前 workbench 的状态底座。

---

## 目标与非目标

### 目标

1. **稳定数据入口**：把样本字段兼容、mutation 归一化、chromatogram 构造、alignment view model 构造集中到纯函数层。
2. **提升详情可读性**：替换 `DetailDrawer` 中简单的 `REF/QRY pre` 展示，提供 DNA、match、CDS、mutation、AA 信息对齐的序列视图。
3. **改善色谱性能**：先通过纯绘制函数、memo/cache、降采样减少主线程压力；必要时再引入 worker/offscreen rendering。
4. **保持当前产品形态**：继续使用当前 `ResultsWorkbench` 和 drawer 交互，不做页面级重构。
5. **增加可测试边界**：坐标映射、归一化、AA/codon view model 和色谱抽样逻辑应能用小型单元测试覆盖。

### 非目标

- 不直接合并 `origin/archive/assistant-tool` 或 `origin/tmp/merge-desktop-agent`。
- 不迁移到 PR 的 `src/features/*` 全量目录结构。
- 不引入 Zustand 来替换当前 `useAgentHarness` / workbench controls。
- 不重写 Python 侧 alignment 算法。
- 不改变 MCP tool schema 或 Electron IPC 协议，除非后续性能阶段证明必须压缩 payload。

---

## 推荐方案

采用三层吸收：

1. **Phase 1: 数据入口与纯函数底座**
2. **Phase 2: 详情抽屉序列/色谱可视化**
3. **Phase 3: 色谱渲染性能优化**

顺序很重要。先让数据模型稳定，再让 UI 消费稳定 view model，最后优化渲染性能。这样每一步都能独立测试和回退。

---

## Phase 1: 数据入口

### 现状

当前 `src/components/workbench/normalize.ts` 已经处理部分字段别名：

- `refBase` / `ref_base`
- `queryBase` / `qry_base`
- `del_count` / `del`
- `samples` / `detail.samples`

但 `DetailDrawer` 仍直接判断 `traces_a`、`query_sequence`、`base_locations` 等后端字段，alignment 展示也直接读取原始字段。这会让字段兼容逻辑分散到 UI。

### 设计

在 workbench 附近新增或扩展纯函数：

- `normalizeSample(item, idx, language): WorkbenchSample`
- `normalizeMutation(item): WorkbenchMutation`
- `buildChromatogramData(sample): ChromatogramData | null`
- `buildAlignmentViewModel(sample): AlignmentViewModel | null`
- `buildCoordinateMap(refGapped, queryGapped): CoordinateMap`

`normalizeSamples` 保留为数组入口，内部调用 `normalizeSample`。

### 字段策略

当前分支后端主要输出 snake_case。前端仍以现有 `WorkbenchSample` 字段为主，不强制改成 camelCase。为兼容后续来源，归一化层可接受 camelCase 别名，但输出保持当前 workbench 约定，减少改动面。

示例兼容：

| 输出字段 | 接受输入 |
|---|---|
| `aligned_ref_g` | `aligned_ref_g`, `alignedRefG` |
| `aligned_query_g` | `aligned_query_g`, `alignedQueryG` |
| `query_sequence` | `query_sequence`, `querySequence` |
| `base_locations` | `base_locations`, `baseLocations` |
| `mixed_peaks` | `mixed_peaks`, `mixedPeaks` |
| `cds_start` | `cds_start`, `cdsStart` |
| `cds_end` | `cds_end`, `cdsEnd` |

### CoordinateMap

借鉴 PR 的 `buildCoordinateMap`，但做成无 React、无 store 依赖的纯函数：

```ts
interface CoordinateMap {
  refToGapped: number[];
  gappedToRef: Array<number | null>;
  gappedToQuery: Array<number | null>;
  queryToGapped: number[];
}
```

职责：

- gapped alignment 坐标和 ungapped ref/query 坐标互转
- CDS 区间从 reference 坐标映射到 gapped 坐标
- mutation 从 ref position 映射到可视化位置
- chromatogram base locations 和 gapped query 坐标对齐

---

## Phase 2: 详情可视化

### 现状

`DetailDrawer` 的 alignment section 当前是简单文本：

- `REF: aligned_ref_g || ref_sequence`
- `QRY: aligned_query_g || query_sequence`

这不利于检查 CDS、突变位置、AA 变化和色谱峰之间的关系。

### 设计

新增当前架构内的 `SequenceAlignmentView`，挂载在 `DetailDrawer` 的 alignment section。

建议文件：

| 文件 | 职责 |
|---|---|
| `src/components/workbench/SequenceAlignmentView.tsx` | 只负责展示 alignment view model |
| `src/components/workbench/SequenceAlignmentView.css` | 视图样式 |
| `src/components/workbench/alignmentView.ts` | 纯函数：坐标映射、match string、CDS/mutation ranges、AA row |
| `tests/test_alignment_view.mjs` | JS 单元测试 |

### 展示内容

`SequenceAlignmentView` 显示：

1. REF 行
2. match 行
3. QRY/Sanger 行
4. position ruler
5. CDS 背景区间
6. mutation 高亮
7. AA change 摘要或 AA row

第一版不需要做完整 IDE 式 genome browser。目标是让抽屉里能快速看清：

- mutation 是否落在 CDS 内
- QRY 相对 REF 的具体差异
- AA 变化和 DNA mismatch 的大致对应关系
- 当前样本是否存在 frameshift 或低覆盖导致的解释风险

### AA / codon 策略

后端已经在 `core/alignment.py` 中负责 AA 变异计算，并输出 `aa_changes` / mutation `effect` 等字段。前端第一版不重新推断所有蛋白效应，只做展示和轻量辅助：

- 使用 `cds_start` / `cds_end` 标出 CDS
- 使用 `aa_changes` 显示 AA summary
- 如需要显示 codon row，只按 CDS 起点每 3 bp 分组，并标注包含 mutation 的 codon
- 不把前端 codon row 作为判定依据

这样可以借鉴 PR 的“CDS 起点决定 reading frame”思想，但避免前端和 Python 侧出现双重生物学判定。

### UI 边界

- 保持 drawer 作为详情入口，不新增全局页面。
- 大序列使用横向滚动，不把 drawer 撑破。
- sticky gutter 可显示 `REF`、`MATCH`、`QRY`、`POS`。
- 对超长序列先保持简单渲染；如性能不足，再在 Phase 3 或后续引入 virtualization。

---

## Phase 3: 色谱性能

### 现状

`ChromatogramCanvas` 在 React component 内完成所有绘制：

- 每次 data / visible range / mutation 变化时主线程重绘
- percentile 计算、trace path 绘制、tooltip 最近点查找都在组件内
- 对较长 trace 或频繁切换样本可能卡顿

PR 中有 worker/offscreen rendering，但直接搬迁风险较高，需要处理 Vite worker 打包、Electron 环境、DPR 和 bitmap 生命周期。

### 阶段 3A: 主线程内优化

先做低风险优化：

1. 把绘制逻辑拆成 `drawChromatogram(ctx, model, options)` 纯函数。
2. 把 trace 预处理拆成 `buildChromatogramRenderModel(data, range, options)`。
3. 对可见 trace 进行降采样，保证画布宽度有限时不绘制远多于像素宽度的点。
4. 缓存 percentile / max scale 的结果，避免同一范围重复排序。
5. tooltip 最近点可用 base location 二分或预计算，避免每次 mousemove 全量扫描。

建议文件：

| 文件 | 职责 |
|---|---|
| `src/components/workbench/chromatogramRender.ts` | 纯绘制和 render model |
| `src/components/workbench/ChromatogramCanvas.tsx` | React wrapper、交互状态 |
| `tests/test_chromatogram_render.mjs` | 降采样、range、tooltip index 测试 |

### 阶段 3B: Worker 可选增强

如果 3A 后仍卡，再引入 worker：

- `src/components/workbench/chromatogram.worker.ts`
- 使用 `new Worker(new URL("./chromatogram.worker.ts", import.meta.url), { type: "module" })`
- 优先传 typed arrays 或 compact render payload
- 主线程只负责接收 bitmap 或绘制指令

Worker 版本必须通过：

- `npm run build`
- Electron dev smoke test
- 至少一个真实样本 drawer 打开/切换手测

---

## 架构取舍

### 为什么不整体搬 PR 的 feature 目录

当前分支已经形成了以 `ResultsWorkbench` 为中心的工作台架构，且包含 PR 中没有的能力：

- controls store
- export menu
- PDF/CSV/JSON 导出
- detail drawer
- command palette
- analysis fast path
- async detail hydration

整体搬 feature 目录会造成重复组件和状态模型冲突，收益低于成本。

### 为什么不引入 Zustand

PR 的 Zustand store 主要为 desktop 分支的 feature panels 服务。当前分支的核心状态来自 agent event stream 和 result payload cache。新增全局 sequencing store 会增加同步问题：

- agent payload 和 store 双写
- selected sample 和 drawer state 重复
- workbench filters 与 sequencing selection 交叉

本轮只在纯函数和局部 component 层吸收可复用能力。

### 为什么先做数据入口

如果先改 UI，字段兼容和空值处理会继续散落在组件里。先统一 normalize 和 view model，后续 UI 和 worker 都能基于稳定输入工作。

---

## 测试策略

### Unit Tests

新增或扩展 JS tests：

1. `test_workbench_normalize.mjs`
   - snake_case 和 camelCase 输入归一化为同一输出
   - mutation aliases 正确映射
   - missing chromatogram fields 返回 `null`

2. `test_alignment_view.mjs`
   - gapped/ref/query 坐标映射正确
   - CDS reference range 映射到 gapped range
   - mutation position 映射到高亮 range
   - gaps 不破坏 query/ref 坐标

3. `test_chromatogram_render.mjs`
   - downsample 不改变 range 边界
   - percentile scale 对异常峰值稳健
   - mouse x 能映射到最近 base index

### Existing Tests

每个阶段至少运行：

- `npm run test:js`
- 和该阶段相关的 Python 测试，如没有 Python 改动则不强制全量 `pytest`

完整阶段完成后运行：

- `npm run build`
- `npm run test`

### Manual Regression

至少用一个真实分析结果检查：

1. 打开 `AnalysisPanel`
2. 选择样本进入 drawer
3. 查看 alignment、AA summary、mutation table、chromatogram
4. 切换多个样本
5. 切换 light/dark theme
6. 确认 drawer 宽度拖拽、Escape 关闭仍可用

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 前端重复实现生物学判定导致和后端不一致 | 前端只做展示和坐标映射，判定仍以后端输出为准 |
| 大序列逐字符渲染导致 drawer 卡顿 | 第一版限制为详情视图；必要时后续引入 virtualization 或 canvas text rendering |
| Worker 在 Electron/Vite 打包中失败 | 先做 3A 主线程优化；worker 作为 3B 单独验证 |
| 字段归一化改变现有样本形状 | 输出保持当前 `WorkbenchSample` 约定；用测试覆盖 aliases |
| CSS 改动影响 workbench 其他区域 | 新增组件样式局限在 `SequenceAlignmentView` class namespace 下 |

---

## 回滚策略

- Phase 1 只增加纯函数和测试，回滚风险低。
- Phase 2 可通过恢复 `DetailDrawer` alignment section 回到原始 `pre` 展示。
- Phase 3A 可保留旧 `ChromatogramCanvas` 绘制路径作为 fallback，确认新路径稳定后再删除。
- Phase 3B worker 如不稳定，可回退到 3A 主线程渲染。

---

## 实施顺序

1. 扩展 normalize 和新增 alignment/chromatogram view model 纯函数。
2. 为纯函数补 JS 单元测试。
3. 新增 `SequenceAlignmentView` 并替换 drawer alignment section。
4. 拆分 `ChromatogramCanvas` 绘制逻辑，加入缓存/降采样。
5. 运行 JS 测试和 build。
6. 手测真实样本 drawer。
7. 根据性能结果决定是否进入 worker 版本。

---

## 后续 Backlog

1. alignment 视图支持跳转到下一个 mutation。
2. AA row 支持按 codon 悬停显示 ref/query codon。
3. chromatogram 和 alignment 横向滚动联动。
4. 大样本详情 payload 压缩，减少 Electron IPC 传输成本。
5. 如果多个页面都需要 sequencing state，再重新评估轻量 store，而不是提前引入全局状态。
