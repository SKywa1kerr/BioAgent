# Workbench Round 3 — 命令面板 + 首次引导 + 键盘导航

**Date:** 2026-04-18
**Scope:** 为 BioAgent 桌面端增加命令面板（Ctrl+K）、首次访问引导、键盘可达性总览，统一现有快捷键的发现性与 focus ring 质量。三个子系统相互独立、可按阶段上线。

> **前置**：Round 1（App 拆分、ErrorBoundary、LLM 重试）、Round 2（筛选持久化、CSV/JSON/PDF 导出）已落地。本轮不改 agent harness / alignment / 结果工作台数据流。

---

## 目标与非目标

### 目标
1. **命令面板**：`Ctrl+K` 打开模糊搜索面板，集中暴露导航、工作台、外观、示例 prompt、日志类动作。
2. **首次引导**：3 步轻量角落卡片（配置 → 提问 → 看结果/导出），跳过后不再显示。
3. **键盘导航**：`?` 快捷键总览；现有 `Ctrl+L` / `Ctrl+,` 统一在一张表；全局 focus-visible 审计。
4. **i18n**：所有新文案同步 zh + en。

### 非目标
- 不做 E2E 测试骨架（Playwright 留给后续）。
- 不重构 agent harness / 结果工作台数据流。
- 不引入额外快捷键库（如 mousetrap）；用原生 `KeyboardEvent`。
- 不做自定义快捷键绑定（YAGNI）。

---

## 架构

```
 ┌─────────────────────────────────────────────────────────┐
 │ App.tsx  (global keydown: Ctrl+K, ?)                    │
 │   │                                                      │
 │   ├── CommandPalette  ──► commandRegistry (module-level)│
 │   │        (portal modal)        ▲                       │
 │   │                              │                       │
 │   │    各 owner 组件 mount 时自注册命令：                  │
 │   │     - App           → nav + appearance + log + examples│
 │   │     - ResultsWorkbench → workbench (export/clear)  │
 │   │                                                      │
 │   ├── OnboardingCoach ──► useOnboarding (localStorage)  │
 │   ├── ShortcutsOverlay ──► shortcuts.ts (single source) │
 │   └── 已有面板                                            │
 └─────────────────────────────────────────────────────────┘
```

### 注册所有权原则
为避免 App 层成为跨所有状态的胖组件，**命令由拥有相关状态的组件自注册**：

- `App.tsx` 负责：导航（focus-chat / open-settings / tab 切换）、外观（theme / lang）、示例 prompt、导出调试日志
- `ResultsWorkbench.tsx` 负责：导出 CSV/JSON/PDF、clear-filters（这些命令需要访问 hook 内的 visibleSamples / filters，与 hook 共居一处最自然）

每个 owner 在 `useEffect` 里调用 `registerCommand` 返回的 unregister 函数，卸载时清理；若同 id 重复注册，后者覆盖前者（reload 场景下安全）。

---

## 组件与文件

### 新建

| 文件 | 作用 |
|---|---|
| `src/lib/commands/registry.js` | 纯 JS 注册表：`registerCommand`、`getCommands`、`filterCommands(query)`、`clearCommands` |
| `src/lib/commands/registry.d.ts` | 类型签名 |
| `src/lib/commands/fuzzy.js` | 纯函数：`fuzzyScore(text, query)`（字符顺序匹配打分，未命中返回 -1） |
| `src/lib/commands/fuzzy.d.ts` | |
| `src/lib/commands/builtin.ts` | `registerBuiltinCommands(ctx: ActionContext)` 返回 unregister 函数 |
| `src/lib/commands/shortcuts.ts` | 快捷键清单（UI 层单一来源） |
| `src/lib/exporters/runExport.ts` | 抽出 ExportMenu 的 exportAs 逻辑供命令面板复用。签名：`runExport(fmt, { samples, filters, dataset, language, onWarn })` 返回 `Promise<void>`，失败抛异常；调用者（ExportMenu 或命令注册处）负责把异常转 toast。保留现有 onWarn 回调语义 |
| `src/components/CommandPalette.tsx` | Portal 模态 + 搜索 + 键盘驱动 |
| `src/components/CommandPalette.css` | |
| `src/hooks/useOnboarding.ts` | 读写 `bioagent-onboarding-v1`，状态枚举：`active \| skipped \| done` |
| `src/components/OnboardingCoach.tsx` | 3 步右下角卡片 |
| `src/components/OnboardingCoach.css` | |
| `src/components/ShortcutsOverlay.tsx` | `?` 触发的模态 |
| `src/components/ShortcutsOverlay.css` | |
| `tests/test_command_registry.mjs` | 注册 / `when` 过滤 / fuzzy 匹配 |
| `tests/test_onboarding_store.mjs` | 状态持久化 + schema 回退 |

### 修改

| 文件 | 改动 |
|---|---|
| `src/App.tsx` | 注册 builtin commands；挂载三个新组件；新增 `Ctrl+K` / `?` 全局 keydown；构造 `ActionContext` |
| `src/i18n.ts` | 新增命令 title / 引导文案 / 快捷键说明（zh + en） |
| `src/components/ChatPanel.tsx` | 空态 CTA：新增 "打开命令面板" 按钮（触发 App 提供的 `openPalette` 回调） |
| `src/components/workbench/ExportMenu.tsx` | 抽出 `exportAs` 为 `runExport`；组件内转调用（保持行为） |

---

## 命令面板规格

### 打开 / 关闭
- 触发：`Ctrl+K` / `Cmd+K`（全局，不受焦点约束；`preventDefault()` 避免浏览器搜索）
- 关闭：`Esc`、点击遮罩、选中命令执行后自动关闭

### 搜索
- 输入框挂载后 `autoFocus`
- `fuzzyScore(title + keywords, query)` 打分，按分降序；`<= -1` 过滤
- 分数相同时保持 builtin 注册顺序
- 空查询：按分组展示全部（分组顺序：导航 → 工作台 → 外观 → 示例 → 日志）

### 键盘
- `↑ / ↓` 移动 `selectedIndex`（在可见项上循环）
- `Enter` 执行当前选中；面板立即关闭
- `Esc` 关闭
- `Tab` 不跳出面板（focus trap；自实现，不依赖第三方）

### 命令结构
```ts
export interface Command {
  id: string;                 // 唯一，稳定
  title: string;              // 已 i18n 的显示文本
  group: "nav" | "workbench" | "appearance" | "examples" | "log";
  keywords?: string[];        // 附加搜索词（如 "导出" + "csv"）
  shortcut?: string;          // 展示用，例如 "Ctrl+L"
  when?: () => boolean;       // 过滤条件
  run: () => void | Promise<void>;
}
```

### 内置命令清单

| 分组 | id | title (zh) | when | 动作 |
|---|---|---|---|---|
| nav | focus-chat | 聚焦聊天输入 | 始终 | `ctx.focusChat()` |
| nav | open-settings | 打开设置 | 始终 | `ctx.openSettings()` |
| nav | tab-analysis | 切换到 分析 | `hasAnalysis` | `ctx.setActiveTab("analysis")` |
| nav | tab-trends | 切换到 突变趋势 | `hasTrends` | `ctx.setActiveTab("trends")` |
| nav | tab-suggestions | 切换到 实验建议 | `hasSuggestions` | `ctx.setActiveTab("suggestions")` |
| workbench | export-csv | 导出当前结果为 CSV | `hasVisibleSamples` | `ctx.exportAs("csv")` |
| workbench | export-json | 导出当前结果为 JSON | `hasVisibleSamples` | `ctx.exportAs("json")` |
| workbench | export-pdf | 导出当前结果为 PDF | `hasVisibleSamples` | `ctx.exportAs("pdf")` |
| workbench | clear-filters | 清除工作台筛选 | `hasActiveFilters` | `ctx.clearWorkbenchFilters()` |
| appearance | toggle-theme | 切换浅色/深色 | 始终 | `ctx.toggleTheme()` |
| appearance | toggle-lang | 切换中文/English | 始终 | `ctx.toggleLanguage()` |
| examples | example-analyze-base | 分析 base 数据集 | 始终 | `ctx.prefillChat("分析 base 数据集")` |
| examples | example-analyze-pro | 分析 pro 数据集 | 始终 | `ctx.prefillChat("分析 pro 数据集")` |
| examples | example-trends | 显示突变趋势 | 始终 | `ctx.prefillChat("显示突变趋势")` |
| examples | example-suggestions | 给出实验建议 | 始终 | `ctx.prefillChat("给出实验建议")` |
| log | export-debug | 导出调试日志 | 始终 | `ctx.exportDebugLog()` |

`when` 回调每次渲染重新执行（成本低）；`hasXxx` 来自 App 当前状态。

---

## 引导规格

### 状态
- localStorage key: `bioagent-onboarding-v1`
- 值为 `"complete"` 或缺失（未完成）
- 进入"完成"状态有两种路径：用户走完 3 步点"完成"，或任一步骤 "跳过" / Esc / 关闭
- 两种路径都写入 `"complete"` — 用户意图明确（不想继续引导），没必要区分 skipped 与 done
- 开发者可通过 devtools `localStorage.removeItem("bioagent-onboarding-v1")` 手动重置

### UI
- 右下角浮层卡片（z-index 45，低于命令面板 50）
- 尺寸：~320px × auto，圆角 12，背景色沿用 panel
- 结构：
  - 顶部：步骤 `1/3` 胶囊 + 关闭按钮（×）
  - 标题（粗体）
  - 正文（2 行左右）
  - 底部：`跳过` 链接左、`下一步` / `完成` 按钮右
- 步骤内容：
  1. **配置智能体** — "点击左上角齿轮图标填写 API key。支持 OpenAI 兼容接口。"
  2. **输入分析请求** — "在左侧聊天框输入请求，例如：`分析 pro 数据集`。"
  3. **查看与导出结果** — "结果在右侧工作台；按 `Ctrl+K` 打开命令面板可一键导出 CSV/JSON/PDF。"

### 可达性
- 卡片带 `role="dialog"` + `aria-labelledby`；不劫持焦点（不 trap），仅在打开时把 focus 放在"下一步"按钮
- Esc 等同"跳过"

---

## 快捷键总览规格

### 触发
- `?` 键按下且当前焦点**不在**可编辑元素（input/textarea/contenteditable）时打开
- `Esc` 或点击遮罩关闭
- 不在命令面板 / 引导 / 其他模态打开时才响应

### 内容（来自 `src/lib/commands/shortcuts.ts`）
```ts
export const SHORTCUTS = [
  { key: "Ctrl+K", action: "shortcuts.palette" },
  { key: "Ctrl+L", action: "shortcuts.focusChat" },
  { key: "Ctrl+,", action: "shortcuts.settings" },
  { key: "Enter / Shift+Enter", action: "shortcuts.send" },
  { key: "?", action: "shortcuts.overlay" },
  { key: "Esc", action: "shortcuts.close" },
];
```
每条走 i18n 文案 key。

---

## Focus 审计

> 目标：键盘唯一导航也能看清当前焦点。

**统一样式**（加到 `src/index.css` 全局）：
```css
:focus-visible {
  outline: 2px solid var(--results-primary, #1f78c1);
  outline-offset: 2px;
  border-radius: 4px;
}
```

**逐组件验证**（手动回归）：
- ChatPanel：输入框、发送按钮、消息复制按钮
- SettingsModal：3 个字段 + Save / Cancel
- ResultsWorkbench：status chip、search、sort、clear、ExportMenu 触发、菜单项、summary scope toggle
- ResultsTable：expand/collapse all、行展开/折叠

若某个自定义 focus 样式与新全局冲突，本地覆盖；目标是**没有可 Tab 但看不见的元素**。

---

## 错误与边界

| 场景 | 处理 |
|---|---|
| localStorage 读写失败（隐私模式） | onboarding 视为未完成但**不重复弹出**：用 `sessionStorage` 内存 fallback；console.warn 一次 |
| `Ctrl+K` 在命令面板已打开时按下 | 关闭面板（toggle 行为） |
| `Ctrl+K` 在 SettingsModal / ConfirmationDialog / OnboardingCoach 打开时按下 | 忽略（不劫持用户正在交互的模态）。实现：App 维护 `isAnyModalOpen` 标志 |
| 命令 `run` 抛异常 | `console.error` + 通过 `ctx.showToast(message)` 显示；面板仍关闭 |
| 引导卡片出现时用户打开设置 | 引导保持当前步骤不自动推进；下次返回再显示；不阻塞设置操作 |
| 搜索为空且所有命令 `when` 均 false | 显示"暂无可用命令" |
| `?` 键按下时焦点在输入框 | 不响应（允许用户输入问号） |
| `?` 键按下时任何模态已打开 | 不响应 |

---

## 测试

### 自动化
- `tests/test_command_registry.mjs`
  - `registerCommand` 去重（同 id 覆盖）
  - `filterCommands("abc")` 返回分数降序列表
  - `when` 为 `false` 时被过滤
  - `clearCommands` 清空
- `tests/test_fuzzy.mjs`
  - 完全匹配得分最高
  - 字符乱序未命中返回 `-1`
  - 子序列命中（如 "ecv" 命中 "export csv"）
  - 大小写无关
- `tests/test_onboarding_store.mjs`
  - 初次读取返回 `active`
  - 写入 `done` / `skipped` 后再读取返回对应值
  - 非法值回退 `active`

### 手动回归
- Ctrl+K 从任意页面打开 / 关闭；焦点返回原元素
- 搜索 "导" → 命中 3 个 export 命令 + 调试日志；选中 Enter 正常执行
- 首次清除 localStorage 后启动 → 出现第 1 步引导；跳过后刷新不再出现
- `?` 弹快捷键总览；Esc 关闭
- 深色 + zh 下所有新组件对比度达标
- Tab 循环所有主要按钮均可见 focus ring

---

## 实施顺序

| 阶段 | 内容 | 可交付 |
|---|---|---|
| **P1** | registry + fuzzy + builtin 命令（不含导出）+ CommandPalette UI + Ctrl+K | PR #1：命令面板可用 |
| **P2** | runExport 抽出 + 导出 / clear-filters / example 命令接入 | PR #2：命令面板完整功能 |
| **P3** | useOnboarding + OnboardingCoach + i18n 文案 | PR #3：引导上线 |
| **P4** | shortcuts.ts + ShortcutsOverlay + focus-visible 全局 + 审计补 | PR #4：键盘可达性收尾 |

每阶段一个独立 PR。

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| `Ctrl+K` 与浏览器或扩展冲突 | Electron 环境无浏览器冲突；Web 预览通过 `preventDefault` 阻止。若用户真有扩展冲突，未来可加自定义绑定 |
| builtin 命令依赖 App 内部状态，耦合过重 | 所有状态访问走 `ActionContext` 接口；注册表不直接访问 App |
| 全局 `:focus-visible` 破坏既有 UI | 审计时逐组件目视回归；若出现破坏，在组件级覆盖 |
| 引导在多显示器 / 小窗口下错位 | 角落卡片 `position: fixed; bottom/right: 16px`；不依赖元素锚点 |
| 命令面板大量命令滚动性能 | 当前 < 20 条，不需虚拟化 |

---

## 回滚
- 任一阶段问题 → revert 对应 PR
- localStorage key 采用 `-v1` 后缀；未来破坏性改动用 `-v2` 独立 key
