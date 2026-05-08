# UI/UX 全面升级 — Claude Desktop 暖色工作台 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 BioAgent 桌面端从冷色 macOS 系统蓝改造成 Claude Desktop 风暖色工作台 — 三栏布局 + 可拖动 splitter + 暖色 token + 多 provider 模型支持。

**Architecture:** 先重写 design tokens（基础设施），再重做 first-launch 流程（provider 预设 + 暖色 InitDialog），然后从 shell 外圈到 canvas 内圈逐组件落地：Sidebar → Splitter+三栏 grid → ChatPanel → Canvas/Workbench 外壳 → 弹层（Palette/Toast/Shortcuts）→ 业务面板 token 替换 → 视觉回归打磨。每步 commit，可单独 revert。

**Tech Stack:** Electron 33 + React 18 + TypeScript 5 + Vite 5；framer-motion / lucide-react / recharts；既有 hooks（useAgentHarness 等）不动，本计划只动 UI 层（CSS + JSX 结构 + 1 个新 lib）。

**Spec reference:** docs/superpowers/specs/2026-05-08-ui-redesign-warm-workbench-design.md

---

## 文件结构

### 新增文件
- `src/styles/tokens.css` — 重写为 Subtle Warm 双主题
- `src/lib/providers.ts` — Provider 预设 + 切换字段联动逻辑
- `src/components/sidebar/Sidebar.tsx` + `Sidebar.module.css` — 新分段 sidebar
- `src/components/workbench/Splitter.tsx` + `Splitter.module.css` — chat/canvas 分隔条
- `src/components/InitDialog/ProviderSelect.tsx` — Provider 下拉（InitDialog + SettingsModal 共用）
- `tests/test_providers.mjs` — providers.ts 单元测试
- `tests/test_splitter_logic.mjs` — Splitter 状态机单元测试

### 重写 / 大改
- `src/styles.css` — app shell 改为三栏 grid，删除 rail-wide/narrow/hidden 三态
- `src/components/InitDialog.tsx` + `InitDialog.css` — 嵌 ProviderSelect，换暖色
- `src/components/SettingsModal.tsx` — 嵌 ProviderSelect，沿用 InitDialog 视觉
- `src/components/ChatPanel.tsx` + 现有 CSS module — 视觉重塑（保留 module 拆分）
- `src/components/TitleBar.css` — 暖色化
- `src/components/CommandPalette.css` — 暖色化 + 选中柿橙竖线
- `src/components/ui/Toast.css` — 暖色化
- `src/App.tsx` — 删 rail-wide/narrow/hidden 状态机，改用 sidebar + chat-width 两个独立维度

### 小改（token 替换为主）
- `src/components/panels/{Analysis,MutationTrend,LabSuggestion}Panel.tsx` 关联 css
- `src/components/workbench/*.css`（已有的）
- `src/lib/settingsStorage.ts` — 兼容 provider 字段

### 删除
- `src/lib/ui/chatRailState.ts` — 三态 rail 状态机不再使用

---

## Task 1: 重写 design tokens 为 Subtle Warm 双主题

**Files:**
- Modify: `src/styles/tokens.css`
- Reference: `docs/superpowers/specs/2026-05-08-ui-redesign-warm-workbench-design.md` §1

**Why:** 所有后续视觉工作都依赖 token，先把基础换好让全应用立刻"漂"到新色调，便于早期发现风险。

- [ ] **Step 1.1: 备份当前 tokens.css**

```bash
cp src/styles/tokens.css src/styles/tokens.css.pre-warm.bak
```
保留作为对比参考；最后任务统一删除。

- [ ] **Step 1.2: 重写 :root（light 主题）**

打开 `src/styles/tokens.css`，把 `:root { ... }` 块整体替换为下面的内容（保留 `color-scheme: light;` 和原文件里的"backwards-compat aliases"段；下面只列**值变化**，未列出的 token 名保持）：

新增/改值：
- `--bg-app: #faf8f3;`
- `--bg-sidebar: #f3f0e8;`
- `--bg-canvas: #fcfaf6;`
- `--surface: #ffffff;`
- `--text-main: #1c1a17;`
- `--text-muted: #5a544c;`
- `--text-subtle: #908a80;`
- `--text-faint: #b1a99c;`
- `--border-soft: #ece6dd;`
- `--border-default: #e8e3d8;`
- `--accent: #d97757;`
- `--accent-hover: #c0623e;`
- `--accent-soft: rgba(217, 119, 87, 0.10);`

把 `--color-bg-app / --color-text-main / --color-surface / --color-border-default / --color-accent` 等旧 token **指向**新值（用 `var(--bg-app)` 等做别名），保持向后兼容。

字体栈替换：
- `--font-display: "Source Serif Pro", "Source Serif", "PingFang SC", "Noto Serif SC", Georgia, serif;`
- `--font-ui` / `--font-mono` 保持。

新增字号 token：
- `--font-size-3xl: 28px;`

shadow 软化：
- `--shadow-sm: 0 1px 2px rgba(28, 26, 23, 0.04);`
- `--shadow-md: 0 4px 12px rgba(28, 26, 23, 0.06);`
- `--shadow-lg: 0 12px 32px rgba(28, 26, 23, 0.10);`
- `--shadow-elevated: 0 20px 48px rgba(28, 26, 23, 0.16);`

- [ ] **Step 1.3: 重写 :root[data-theme="dark"]**

新增/改值：
- `--bg-app: #1a1815;` （删除原本的 radial-gradient 复合背景，改为纯色）
- `--bg-sidebar: #131210;`
- `--bg-canvas: #1f1c19;`
- `--surface: #211e1a;`
- `--text-main: #f0ece5;`
- `--text-muted: #908a80;`
- `--text-subtle: #6a6357;`
- `--border-soft: #26231e;`
- `--border-default: #2e2a24;`
- `--accent: #d97757;`
- `--accent-hover: #e08e6f;`
- `--accent-soft: rgba(217, 119, 87, 0.16);`

shadow 用更深 alpha：
- `--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.5);`
- `--shadow-md: 0 4px 12px rgba(0, 0, 0, 0.5);`

- [ ] **Step 1.4: 删除 hc 主题中受影响的硬编码（保持 hc 行为）**

`:root[data-theme="hc"]` 段保持不动 — 高对比度主题本期不动。

- [ ] **Step 1.5: 修订 trace / mutation 颜色（light 主题略偏暖）**

light 主题下：
- `--color-trace-a: #4d8a3e;`（原 #16a34a 略偏暖绿）
- `--color-trace-t: #c44a3c;`（原 #dc2626 略偏砖红）
- `--color-trace-g: #b07021;`（原 #b45309 保留同色系）
- `--color-trace-c: #2f5fb8;`（原 #2563eb 略偏暖蓝）

dark 主题保持现值（高饱和已经合适）。

- [ ] **Step 1.6: 启动 dev server 看视觉漂移**

```bash
npm run dev
```
打开浏览器，确认：
- 整体米黄底（不再是冷灰）
- 按钮 / 链接变成柿橙
- 现有功能完全可用，无 console 报错

- [ ] **Step 1.7: typecheck**

```bash
npm run typecheck
```
预期：通过（没动 TS）。

- [ ] **Step 1.8: commit**

```bash
git add src/styles/tokens.css
git commit -m "feat(design): rewrite tokens to Subtle Warm double-theme"
```

---

## Task 2: Provider 预设模块 + 切换字段联动

**Files:**
- Create: `src/lib/providers.ts`
- Create: `tests/test_providers.mjs`
- Modify: `src/lib/settingsStorage.ts`

**Why:** InitDialog 和 SettingsModal 都要用 provider 预设；先把数据 + 联动逻辑封装好，UI 只做呈现。

- [ ] **Step 2.1: 写失败测试 — providers 静态数组**

创建 `tests/test_providers.mjs`：

```js
import test from "node:test";
import assert from "node:assert/strict";
import { PROVIDERS, getProvider, applyProviderSwitch } from "../src/lib/providers.ts";

test("PROVIDERS contains 6 presets", () => {
  const ids = PROVIDERS.map((p) => p.id);
  assert.deepEqual(ids, ["openai","anthropic","deepseek","sjtu","ollama","custom"]);
});

test("getProvider returns custom when id unknown", () => {
  assert.equal(getProvider("not-a-provider").id, "custom");
});

test("ollama preset does not require api key", () => {
  assert.equal(getProvider("ollama").requiresApiKey, false);
});

test("anthropic preset baseUrl is empty (proxy required)", () => {
  assert.equal(getProvider("anthropic").defaultBaseUrl, "");
});
```

- [ ] **Step 2.2: 跑测试确认失败**

```bash
npm run test:js -- --grep providers
```
预期：FAIL — providers.ts 不存在。

- [ ] **Step 2.3: 写 providers.ts 最小实现**

创建 `src/lib/providers.ts`：

```ts
export type ProviderId =
  | "openai" | "anthropic" | "deepseek"
  | "sjtu"   | "ollama"    | "custom";

export interface ProviderPreset {
  id: ProviderId;
  label: string;
  defaultBaseUrl: string;
  suggestedModels: readonly string[];
  requiresApiKey: boolean;
  /** 用于切换厂家时的提示文案 key */
  noteI18nKey?: string;
}

export const PROVIDERS: readonly ProviderPreset[] = [
  { id: "openai",    label: "OpenAI",
    defaultBaseUrl: "https://api.openai.com/v1",
    suggestedModels: ["gpt-4o-mini","gpt-4o"],
    requiresApiKey: true },
  { id: "anthropic", label: "Anthropic",
    defaultBaseUrl: "",
    suggestedModels: ["claude-sonnet-4-6","claude-haiku-4-5"],
    requiresApiKey: true,
    noteI18nKey: "provider.anthropic.proxyRequired" },
  { id: "deepseek",  label: "DeepSeek",
    defaultBaseUrl: "https://api.deepseek.com/v1",
    suggestedModels: ["deepseek-chat","deepseek-coder"],
    requiresApiKey: true },
  { id: "sjtu",      label: "SJTU AI Lab",
    defaultBaseUrl: "https://models.sjtu.edu.cn/api/v1",
    suggestedModels: ["deepseek-chat"],
    requiresApiKey: true },
  { id: "ollama",    label: "Ollama (Local)",
    defaultBaseUrl: "http://localhost:11434/v1",
    suggestedModels: ["llama3.2","qwen2.5"],
    requiresApiKey: false },
  { id: "custom",    label: "Custom (OpenAI-compatible)",
    defaultBaseUrl: "",
    suggestedModels: [],
    requiresApiKey: true },
] as const;

export function getProvider(id: string): ProviderPreset {
  return PROVIDERS.find((p) => p.id === id) ?? PROVIDERS[PROVIDERS.length - 1]!;
}

/** 切换 provider 时计算下一组 baseURL/model 字段值 */
export function applyProviderSwitch(args: {
  fromId: ProviderId;
  toId: ProviderId;
  currentBaseUrl: string;
  currentModel: string;
}): { baseUrl: string; model: string; baseUrlIsCustom: boolean; modelIsCustom: boolean } {
  const from = getProvider(args.fromId);
  const to = getProvider(args.toId);
  const baseUrlIsCustom = args.currentBaseUrl.trim() !== "" && args.currentBaseUrl.trim() !== from.defaultBaseUrl;
  const modelIsCustom = args.currentModel.trim() !== "" && !from.suggestedModels.includes(args.currentModel.trim());
  return {
    baseUrl: baseUrlIsCustom ? args.currentBaseUrl : to.defaultBaseUrl,
    model:   modelIsCustom   ? args.currentModel   : (to.suggestedModels[0] ?? ""),
    baseUrlIsCustom,
    modelIsCustom,
  };
}

/** 从已存在的 baseURL 推断 provider id（用于旧数据迁移） */
export function inferProviderFromBaseUrl(baseUrl: string): ProviderId {
  const u = baseUrl.trim();
  if (!u) return "custom";
  if (u.includes("openai.com")) return "openai";
  if (u.includes("anthropic.com")) return "anthropic";
  if (u.includes("deepseek.com")) return "deepseek";
  if (u.includes("sjtu.edu.cn")) return "sjtu";
  if (u.includes("localhost:11434") || u.includes("127.0.0.1:11434")) return "ollama";
  return "custom";
}
```

- [ ] **Step 2.4: 跑测试**

```bash
npm run test:js
```
预期：4 个 providers 测试 PASS。

- [ ] **Step 2.5: 增加 applyProviderSwitch 测试用例**

在 `tests/test_providers.mjs` 追加：

```js
import { applyProviderSwitch } from "../src/lib/providers.ts";

test("applyProviderSwitch overrides default baseUrl when user has not customized", () => {
  const r = applyProviderSwitch({
    fromId: "openai", toId: "deepseek",
    currentBaseUrl: "https://api.openai.com/v1",
    currentModel: "gpt-4o-mini",
  });
  assert.equal(r.baseUrl, "https://api.deepseek.com/v1");
  assert.equal(r.baseUrlIsCustom, false);
});

test("applyProviderSwitch preserves user-customized baseUrl", () => {
  const r = applyProviderSwitch({
    fromId: "openai", toId: "deepseek",
    currentBaseUrl: "https://my-proxy.example/v1",
    currentModel: "gpt-4o-mini",
  });
  assert.equal(r.baseUrl, "https://my-proxy.example/v1");
  assert.equal(r.baseUrlIsCustom, true);
});
```

- [ ] **Step 2.6: 跑测试**

```bash
npm run test:js
```
预期：6 个全 PASS。

- [ ] **Step 2.7: 扩展 AgentSettings 类型**

修改 `src/lib/settingsStorage.ts`：
- 在 `AgentSettings` interface 中加入 `provider: ProviderId;`
- 在 `loadSettings()` 里：如果 localStorage 里没有 provider 字段，调用 `inferProviderFromBaseUrl(parsed.llmBaseUrl)` 填充。
- import `ProviderId, inferProviderFromBaseUrl` from `./providers`。
- 默认值 `DEFAULT_SETTINGS`（如果文件里有同等结构）也加 `provider: "sjtu"` 兜底（保持现状的 SJTU 默认）。

- [ ] **Step 2.8: typecheck**

```bash
npm run typecheck
```
预期：PASS。如有引用 AgentSettings 字段缺失的报错，按提示在调用处补 `provider`。

- [ ] **Step 2.9: 跑全量 JS 测试**

```bash
npm run test:js
```
预期：全 PASS。

- [ ] **Step 2.10: commit**

```bash
git add src/lib/providers.ts src/lib/settingsStorage.ts tests/test_providers.mjs
git commit -m "feat(providers): add multi-provider preset module + settings storage migration"
```

---

## Task 3: ProviderSelect 组件 + InitDialog 暖色升级

**Files:**
- Create: `src/components/InitDialog/ProviderSelect.tsx`
- Modify: `src/components/InitDialog.tsx`
- Modify: `src/components/InitDialog.css`
- Modify: `src/locales/{zh,en}.json` — 加 provider 文案

**Why:** 第一屏体验，最先验证暖色调真实落地效果；同时把 multi-provider 接通。

- [ ] **Step 3.1: 加 i18n 文案**

在 `src/locales/zh.json` 顶层 `app` 块同级，新增：

```json
"provider": {
  "label": "模型厂家",
  "anthropic": {
    "proxyRequired": "Anthropic 官方端点需经 OpenAI 兼容代理（如 LiteLLM）使用。"
  },
  "switchHint": {
    "customBaseUrl": "保留你的自定义 baseURL，已切换厂家后请确认。",
    "apiKey": "切换厂家后建议确认 API Key 仍可用。"
  }
}
```
英文同结构，写在 `src/locales/en.json`：

```json
"provider": {
  "label": "Model provider",
  "anthropic": {
    "proxyRequired": "Anthropic native endpoint requires an OpenAI-compatible proxy (e.g. LiteLLM)."
  },
  "switchHint": {
    "customBaseUrl": "Kept your custom baseURL after switching provider — please verify.",
    "apiKey": "After switching provider, please confirm the API Key still applies."
  }
}
```

- [ ] **Step 3.2: 写 ProviderSelect.tsx**

创建 `src/components/InitDialog/ProviderSelect.tsx`：

```tsx
import { PROVIDERS, type ProviderId } from "../../lib/providers";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";

interface Props {
  value: ProviderId;
  onChange: (next: ProviderId) => void;
  language: AppLanguage;
  disabled?: boolean;
}

export function ProviderSelect({ value, onChange, language, disabled }: Props): JSX.Element {
  return (
    <label className="init-dialog-field">
      <span className="init-dialog-field-label">{t(language, "provider.label")}</span>
      <select
        className="init-dialog-select"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as ProviderId)}
      >
        {PROVIDERS.map((p) => (
          <option key={p.id} value={p.id}>{p.label}</option>
        ))}
      </select>
    </label>
  );
}
```

- [ ] **Step 3.3: 改写 InitDialog.tsx**

修改 `src/components/InitDialog.tsx`：

1. 删除 `import { FlaskConical } from "lucide-react";`，改用 inline SVG logo 或 lucide 中性图标（用 `import { Atom } from "lucide-react";`，size=26）
2. 加 import：
   ```ts
   import { ProviderSelect } from "./InitDialog/ProviderSelect";
   import { applyProviderSwitch, getProvider, type ProviderId } from "../lib/providers";
   ```
3. 在 `useState(initialSettings)` 之后加：
   ```ts
   const provider = getProvider(settings.provider);
   ```
4. 在表单 JSX 顶部加 ProviderSelect：
   ```tsx
   <ProviderSelect
     value={settings.provider}
     language={language}
     onChange={(nextId) => setSettings((prev) => {
       const r = applyProviderSwitch({
         fromId: prev.provider,
         toId: nextId,
         currentBaseUrl: prev.llmBaseUrl,
         currentModel: prev.llmModel,
       });
       return { ...prev, provider: nextId, llmBaseUrl: r.baseUrl, llmModel: r.model };
     })}
   />
   ```
5. 在 ApiKey 字段：当 `!provider.requiresApiKey` 时整段隐藏（return null on that field）。
6. 在 baseUrl 字段下方加提示行：
   ```tsx
   {provider.noteI18nKey ? (
     <p className="init-dialog-hint">{t(language, provider.noteI18nKey)}</p>
   ) : null}
   ```
7. `canSubmit` 改为：`(provider.requiresApiKey ? settings.llmApiKey.trim().length > 0 : true) && settings.llmBaseUrl.trim().length > 0 && !isInitializing`

- [ ] **Step 3.4: 改写 InitDialog.css 为暖色**

修改 `src/components/InitDialog.css`：

- 把所有 `rgba(...)` / 直接 hex 替换成新 token 引用：
  - `.init-dialog-scrim` 背景：`background: rgba(20, 17, 13, 0.45); backdrop-filter: blur(8px);`
  - `.init-dialog-card` 背景：`background: var(--surface); border: 1px solid var(--border-default); border-radius: 14px; box-shadow: var(--shadow-elevated);`
  - `.init-dialog-icon` 背景：`background: var(--accent-soft); color: var(--accent);`
  - `.init-dialog-title` 用 display 字体：`font-family: var(--font-display); font-weight: 400; font-size: 28px; letter-spacing: -0.02em;`
  - `.init-dialog-field input` 与 `.init-dialog-select`：`background: var(--bg-app); color: var(--text-main); border: 1px solid var(--border-default); border-radius: 10px; padding: 10px 12px; font-size: 13px;`
  - 聚焦态：`border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft);`
  - `.init-dialog-submit`：`background: var(--accent); color: #ffffff;` hover `background: var(--accent-hover);`
  - 新增 `.init-dialog-hint`：`font-size: 11px; color: var(--text-subtle); margin-top: 6px;`
- 删除原文件里所有蓝色/冷色 hex（如 #4f7cff / #6fb1ff 等）。

- [ ] **Step 3.5: typecheck**

```bash
npm run typecheck
```
预期：PASS。

- [ ] **Step 3.6: 手动验证**

```bash
npm run dev
```
- 清空 localStorage（DevTools → Application → Local Storage → Clear）触发 InitDialog
- 切换 provider 下拉，观察 baseURL 和 model 占位变化
- 切到 Anthropic 看到提示行
- 切到 Ollama，API Key 字段隐藏
- light / dark 切换两个主题都看一遍

- [ ] **Step 3.7: commit**

```bash
git add src/components/InitDialog.tsx src/components/InitDialog.css src/components/InitDialog/ProviderSelect.tsx src/locales/zh.json src/locales/en.json
git commit -m "feat(init): warm-themed InitDialog with provider preset selector"
```

---

## Task 4: SettingsModal 复用 ProviderSelect + 视觉同步

**Files:**
- Modify: `src/components/SettingsModal.tsx`

**Why:** SettingsModal 也要 provider 切换；视觉与 InitDialog 一致更省维护成本。

- [ ] **Step 4.1: 改写 SettingsModal.tsx**

修改 `src/components/SettingsModal.tsx`：

1. import：ProviderSelect、applyProviderSwitch、getProvider、ProviderId 同 Task 3。
2. 在内部 state 上加 provider，逻辑同 InitDialog。
3. JSX 顶部加 ProviderSelect。
4. ApiKey / baseUrl / model 字段按 provider 上下文显示提示与隐藏。
5. 复用 InitDialog 的 CSS 类名（.init-dialog-field、.init-dialog-select），不写第二套。
6. SettingsModal 自己的 overlay / card 类名（.settings-modal-overlay、.settings-modal）保留，但去 src/styles.css 把这两个类的样式同步到新 token：
   - .settings-modal-overlay：背景 rgba(20, 17, 13, 0.45) + blur
   - .settings-modal：surface 背景 + border-default 描边 + 14px 圆角 + shadow-elevated

- [ ] **Step 4.2: typecheck**

```bash
npm run typecheck
```

- [ ] **Step 4.3: 手动验证**

启动 dev，已初始化的状态下 Ctrl+, 打开 Settings：
- provider 下拉可切换
- 字段联动正确
- 视觉与 InitDialog 一致
- 保存触发 shutdown → reinit

- [ ] **Step 4.4: commit**

```bash
git add src/components/SettingsModal.tsx src/styles.css
git commit -m "feat(settings): SettingsModal shares ProviderSelect + warm tokens"
```

---

## Task 5: 新 Sidebar 组件（含左下 Settings 入口）

**Files:**
- Create: `src/components/sidebar/Sidebar.tsx`
- Create: `src/components/sidebar/Sidebar.module.css`
- Modify: `src/locales/{zh,en}.json` — 加 sidebar 文案

**Why:** 现在 history rail 嵌在 chat panel 内 + rail 三态切换状态机散落。这一步把它统一成单组件，含 3 段（最近会话 / 数据集 / 面板历史）+ 左下 Settings。

- [ ] **Step 5.1: 写 Sidebar.module.css**

创建 `src/components/sidebar/Sidebar.module.css`，规则要点：

- .sidebar：grid 双行（1fr + auto），bg-sidebar 背景，右侧 1px border-default
- .body：上半 1fr，overflow-y auto，padding 16px 12px
- .section：margin-bottom 18px
- .label：9px uppercase letter-spacing 0.08em，color text-faint
- .row：display flex + align-items center + gap 6px + padding 7px 8px + radius 6px + 12px font + color text-muted；hover 用 accent-soft 背景 + text-main 颜色；.active 类用 surface 背景 + 1px 0 阴影 + text-main 颜色；disabled 状态弱化为 text-faint 且 cursor not-allowed
- .meta：margin-left auto + 9px font + text-faint
- .footer：border-top 1px border-default + padding 10px 12px
- .settingsBtn：display flex + gap 8px + width 100% + padding 7px 8px + radius 6px + 12px font，hover 同 .row
- .modelLabel：margin-left auto + 10px + JetBrains Mono + ellipsis 限宽 80px


- [ ] **Step 5.2: 写 Sidebar.tsx**

创建 `src/components/sidebar/Sidebar.tsx`。Props 接口：

- language, history (id+label+ts), datasets (id+label), activeAnalysisId, activeTab (PanelType), hasAnalysisCache/hasTrendsCache/hasSuggestionsCache, modelLabel, onSelectHistory, onSelectTab, onOpenSettings

布局：

- 顶层 aside.sidebar 含 div.body + div.footer
- div.body 内三个 section.section：
  - "最近会话"：history 数组 .slice(0, 8) map 出 button.row；空数组时显示 .rowDisabled 占位
  - "数据集"：datasets map 出 button.row；空时占位
  - "面板"：硬编码三 button (analysis / trends / suggestions)，根据 hasXxxCache 设 disabled，根据 activeTab 加 .active
- div.footer 含 button.settingsBtn：lucide Settings icon (size=14) + text + span.modelLabel
- 用 lucide-react 的 Settings 图标，size=14

辅助：定义 formatRelative(ts) 把毫秒差转 "now" / "Nm" / "Nh" / "Nd"

- [ ] **Step 5.3: 加 i18n keys**

zh.json 顶层加 sidebar 段：

- sidebar.recent → "最近会话"
- sidebar.recent.empty → "暂无历史"
- sidebar.datasets → "数据集"
- sidebar.datasets.empty → "尚未导入"
- sidebar.panels → "面板"

en.json 同结构，英文：Recent / No history / Datasets / Not imported / Panels。

- [ ] **Step 5.4: typecheck**

```
npm run typecheck
```

- [ ] **Step 5.5: commit（独立组件先 commit，下一 task 接入 App）**

```
git add src/components/sidebar/ src/locales/zh.json src/locales/en.json
git commit -m "feat(sidebar): segmented sidebar with bottom-left settings entry"
```

---

## Task 6: Splitter 组件 + 状态机 hook

**Files:**
- Create: `src/components/workbench/Splitter.tsx`
- Create: `src/components/workbench/Splitter.module.css`
- Create: `src/hooks/useChatColumnWidth.ts`
- Create: `tests/test_splitter_logic.mjs`

**Why:** chat 与 canvas 之间的可拖动分隔条 + 折叠到 32px rail 是新交互；先写状态机单测，再写组件 UI。

- [ ] **Step 6.1: 写失败测试 — 状态机**

创建 `tests/test_splitter_logic.mjs`，测试纯函数 computeNextWidth(prevWidth, dx, containerWidth, canvasMin) ：
- 返回 { width, collapsed: boolean }
- 当 prevWidth + dx < 120 → 返回 { width: 32, collapsed: true }
- 当 prevWidth + dx > containerWidth - canvasMin → 夹到上限
- 当 collapsed 状态 + 任何点击恢复 → 返回 { width: rememberedWidth, collapsed: false }

四个测试用例：
1. 正常拖宽
2. 拖到 < 120 触发折叠
3. 拖超过上限被夹住
4. 折叠后恢复读 lastExpandedWidth

- [ ] **Step 6.2: 跑测试确认失败**

```
npm run test:js -- --grep splitter
```
预期：FAIL（模块不存在）。

- [ ] **Step 6.3: 写 useChatColumnWidth.ts**

导出 hook：
- 内部 state：width (number)、collapsed (boolean)、lastExpandedWidth (number)
- 持久化到 localStorage 键 `bioagent-chat-width`（值是 JSON: { width, collapsed, lastExpandedWidth }）
- 暴露 setWidth(next, containerWidth)、collapse()、expand()
- 内部用 computeNextWidth 纯函数（导出供测试）
- 默认值 width=320, collapsed=false, lastExpandedWidth=320

- [ ] **Step 6.4: 跑测试确认通过**

```
npm run test:js
```

- [ ] **Step 6.5: 写 Splitter.module.css**

样式：
- .splitter：width 1px + cursor col-resize + background border-default + position relative
- .splitter::after：absolute centered vertical 6×26 px 圆角握把 + accent-soft 背景，opacity 0 默认，hover/active 变 1
- .splitter:hover::after, .splitter[data-dragging=true]::after：opacity 1
- .collapsedRail：width 32px + cursor pointer + bg-canvas + border-right 1px + display flex column + align-items center + padding 12px 0 + gap 8px
- .collapsedRail svg + .collapsedRailLabel：vertical writing-mode + font-size 10px + color text-subtle + letter-spacing 0.1em
- .collapsedRail:hover：bg accent-soft

- [ ] **Step 6.6: 写 Splitter.tsx**

组件 props：onResize(dx)、onCollapse()、isDragging。

行为：
- 渲染 div.splitter 带 onMouseDown：标记 dragging，监听 document mousemove/mouseup，dispatch dx 到 onResize；mouseup 解绑
- aria-orientation="vertical"、role="separator"、tabindex=0
- 键盘左/右键也触发 onResize(±20)，回车触发 onCollapse

CollapsedRail 单独导出：

```
export function CollapsedRail({ onExpand, language }: { onExpand: () => void; language: AppLanguage }): JSX.Element
```
内含一个按钮 + 纵向文字 + 箭头 icon。

- [ ] **Step 6.7: typecheck**

```
npm run typecheck
```

- [ ] **Step 6.8: commit**

```
git add src/components/workbench/Splitter.tsx src/components/workbench/Splitter.module.css src/hooks/useChatColumnWidth.ts tests/test_splitter_logic.mjs
git commit -m "feat(splitter): chat/canvas resizer with auto-collapse and rail"
```

---

## Task 7: 重构 App 三栏布局，删除旧 rail 状态机

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/styles.css`
- Delete: `src/lib/ui/chatRailState.ts`
- Modify: `src/components/ChatPanel.tsx` — 移除 chat-rail-toggle 按钮 / onCycleRail prop

**Why:** 现在的 .app-shell-content 三态 grid (rail-wide/narrow/hidden) 与新设计不兼容；本任务把 shell 改成 sidebar(可折叠) + chat(可调宽 / 折叠) + canvas 三个独立维度。

- [ ] **Step 7.1: 改 styles.css 应用层布局**

把 `.app-shell-content` 的 grid-template-columns 与 rail-wide/rail-narrow/rail-hidden 三个变体替换为：

- `.app-shell-content`：display grid，grid-template-columns: var(--sidebar-w, 200px) minmax(120px, var(--chat-w, 320px)) 1px 1fr，gap 0，padding 0
- `.app-shell.sidebar-collapsed .app-shell-content`：--sidebar-w: 56px
- `.app-shell.chat-collapsed .app-shell-content`：grid-template-columns: var(--sidebar-w, 200px) 32px 1px 1fr
- 同时把 `.chat-panel` 与 `.canvas-panel` 的 border-radius 删掉（现在三栏是 edge-to-edge）；只保留 `.canvas-panel { background: var(--bg-canvas); border-left 由 splitter 提供 }`
- 删 `.chat-rail-toggle` 样式（按钮去掉）

- [ ] **Step 7.2: 改 App.tsx**

import：

```
import { Sidebar } from "./components/sidebar/Sidebar";
import { Splitter, CollapsedRail } from "./components/workbench/Splitter";
import { useChatColumnWidth } from "./hooks/useChatColumnWidth";
```

删：

- 删 `loadRailState / saveRailState / nextRailState` import 和相关 state (`railState`, `cycleRail`)
- 删 `app-shell rail-${railState}` 类名拼接
- 删传给 ChatPanel 的 onCycleRail / railLabel prop

加：

- 用 useChatColumnWidth 拿到 width / collapsed / setWidth / collapse / expand
- shellRef = useRef<HTMLDivElement>，记录容器宽度
- useState `sidebarCollapsed`（默认 false），Ctrl+B 切换
- 计算 className：`"app-shell" + (sidebarCollapsed ? " sidebar-collapsed" : "") + (collapsed ? " chat-collapsed" : "")`
- shellContent 改为四元素：Sidebar + (collapsed ? CollapsedRail : ChatPanel) + Splitter + canvas-panel

Sidebar props 来自现有 hooks：history → historyApi.items, datasets → 暂时空数组（或从 settings 拿 — 本期先空），activeAnalysisId、activeTab、hasXxxCache (panelCache 三 key 是否有值)、modelLabel = settings.llmModel，onSelectHistory = handleHistorySelect，onSelectTab = setActiveTab，onOpenSettings = setSettingsOpen(true)

- [ ] **Step 7.3: 改 ChatPanel.tsx**

- 删 props 上的 onCycleRail、railLabel、theme（theme 移到 sidebar 或 settings — 本期保留传入但不渲染按钮）
- 删 chat-rail-toggle 按钮 JSX
- panel-action-group 里的 settings / language / theme 按钮**保留**（仍可访问），但 settings 按钮可与 sidebar 左下入口冗余 — 先保留，后续 polish 阶段酌情移除

- [ ] **Step 7.4: 删除 chatRailState.ts**

```
git rm src/lib/ui/chatRailState.ts
```

- [ ] **Step 7.5: 加 Ctrl+B 全局快捷键**

App.tsx handleGlobalKeyDown 里加：

```
if (mod && e.key.toLowerCase() === "b" && !e.shiftKey) {
  e.preventDefault();
  setSidebarCollapsed((v) => !v);
  return;
}
```

- [ ] **Step 7.6: typecheck + 单测**

```
npm run typecheck
npm run test:js
```

- [ ] **Step 7.7: 手动验证**

```
npm run dev
```

- 看到三栏：sidebar / chat / canvas
- 拖 splitter 改变 chat 宽度，关掉再开 dev 看持久化
- 拖到极窄触发 rail 折叠，点击 rail 恢复
- Ctrl+B 折叠/展开 sidebar
- 旧的 chat-rail-toggle 按钮已消失

- [ ] **Step 7.8: commit**

```
git add -A src/
git commit -m "refactor(shell): three-pane grid with splitter + sidebar collapse"
```

---

## Task 8: ChatPanel 视觉重塑

**Files:**
- Modify: `src/components/ChatPanel.tsx`
- Modify: `src/components/workbench/ResultsWorkbench.module.css` 或 ChatPanel 已有的 CSS module（取决于现状 — 实施期 ls 一下）

**Why:** chat 列要从蓝色卡片样式过渡到 Claude 风对话流：用户气泡暖橘 / assistant 白底，composer 圆角 + 内嵌发送按钮。

- [ ] **Step 8.1: 检查现有 ChatPanel 的 CSS 来源**

```
grep -rn "chat-panel\|message-list\|message-user\|message-assistant\|composer" src/styles.css src/components/ChatPanel.tsx
```
弄清现在 chat 相关样式在 styles.css 还是 module 里。本步骤不写代码，只读 — 决定下面要改的文件。

- [ ] **Step 8.2: 写新样式（落在 ChatPanel.module.css，新建或扩展）**

要点：

- .chatPanel：display grid + grid-template-rows auto 1fr auto + min-height 0 + bg-app + 没有 border-radius
- .header：padding 14px 18px 10px + border-bottom 1px border-soft + display flex justify-between align-items center + font-display 14px weight 400
- .messages：padding 16px 18px + display grid + gap 14px + overflow auto
- .message：max-width 92% + padding 12px 14px + radius 14px + line-height 1.65 + font 13px
- .messageUser：背景 var(--accent-soft)，颜色 text-main，align-self flex-end
  - dark theme 下：让 accent-soft alpha 高一点（已经在 token 里）
- .messageAssistant：背景 surface + 1px border-soft + align-self flex-start
- .messageRoleName：font weight 600 + font-size 11px + color text-subtle + display block + margin-bottom 3px
- .composer：margin 14px 18px + padding 10px 14px + radius 12px + bg surface + 1px border-default + shadow-sm + display grid grid-template-columns 1fr auto + gap 8px + align-items end
- .composer textarea：bg transparent + border 0 + outline 0 + resize none + min-height 60px + font inherit + color text-main + font-size 13px
- .composer textarea::placeholder：color text-subtle
- .composer:focus-within：border-color accent + box-shadow 0 0 0 3px accent-soft
- .sendBtn：bg accent + color #fff + border 0 + radius 8px + padding 6px 14px + font 12px weight 500 + cursor pointer，hover bg accent-hover，disabled opacity 0.45

- [ ] **Step 8.3: 改 ChatPanel.tsx JSX**

- 用户消息 div 改：先一行 .messageRoleName 显示 "你" / 用户名（i18n key chat.you），然后内容
- assistant 消息：.messageRoleName 显示 "BioAgent"（i18n key chat.bot）
- 删除 panel-title-row 里的 settings/clear/lang 按钮，只保留 export-debug 与 theme（settings 已在 sidebar 左下；clear 移到 footer 小按钮或保留）
- composer 改为 grid 双列：左 textarea 右 button；button label 用图标 + "发送" 文字（lucide ArrowUp 或 SendHorizontal，size 14）
- 应用 module 类名

- [ ] **Step 8.4: typecheck**

```
npm run typecheck
```

- [ ] **Step 8.5: 手动验证 light + dark**

启动 dev：
- 发送一条消息看气泡颜色（用户暖橘半透明 / bot 白底）
- 切到 dark 模式同样
- 拖 splitter 极窄 → composer 应仍能自适应

- [ ] **Step 8.6: commit**

```
git add src/components/ChatPanel.tsx src/components/ChatPanel.module.css src/locales/
git commit -m "feat(chat): warm Claude-style chat column with rounded composer"
```

---

## Task 9: Canvas / Workbench 外壳重塑

**Files:**
- Modify: `src/components/SmartCanvas.tsx` 与/或 `src/styles.css` 中的 .canvas-panel / .panel-title / .panel-tab-bar
- Modify: `src/components/workbench/ResultsWorkbench.global.css` 与 `.module.css`（按现状决定）

**Why:** canvas header（crumb + h2 + tabs）+ stats 区 + 数据卡片要换暖色 + Source Serif。

- [ ] **Step 9.1: 找现有 panel-title / panel-tab-bar 样式**

```
grep -rn "panel-title\|panel-tab-bar\|panel-tab\|hero-grid\|hero-card\|hero-value\|hero-label" src/
```
列出修改面。

- [ ] **Step 9.2: 改 .canvas-panel / .panel-title / .panel-tab-bar**

- .canvas-panel：bg var(--bg-canvas) + border-left 1px var(--border-default) + 没有 border-radius / shadow
- .panel-title：padding 14px 20px 12px + border-bottom 1px var(--border-soft) + display flex align-items baseline gap 12px
- .panel-title 的 span (title) 改 font-display 18px weight 400 letter-spacing -0.02em
- 在 panel-title 前插入 crumb 元素：font 10px uppercase letter-spacing 0.08em color text-subtle margin-right 8px（具体加在 SmartCanvas / App 里）
- .panel-tab-bar：margin-left auto + display flex gap 2px
- .panel-tab：radius 999px padding 4px 12px font 11px color text-muted + transparent bg
- .panel-tab-active：bg text-main color bg-app（深底白字反差）

- [ ] **Step 9.3: 改 hero stats**

把 hero-grid / hero-card / hero-label / hero-value 重写：

- .hero-grid：grid 3 列 + gap 14px + margin-bottom 14px
- .hero-card：去除 background + border + padding；改为简洁 vertical layout（stat-num 上 / stat-lbl 下）
- .hero-label → .stat-lbl：font 9px uppercase letter-spacing 0.08em color text-subtle margin-top 5px
- .hero-value → .stat-num：font-family Inter（不再 Fraunces）+ size 28px + weight 600 + line-height 1 + letter-spacing -0.02em + tabular-nums

- [ ] **Step 9.4: 改 detail-card 与 audience-card**

- .detail-card：bg var(--surface) + 1px border-soft + radius 10px + padding 14px 16px + shadow-sm
- .detail-card h3：font-display 15px weight 400 + margin 0 0 10px 0
- .audience-card 空态：保留布局，h3 用 font-display 28px weight 400 letter-spacing -0.02em + color text-main，p 用 13px line 1.6 color text-muted

- [ ] **Step 9.5: 改 ChromatogramCanvas / ResultsCharts / ResultsTable 外壳**

只改 wrapper（.workbench-card / .results-card 等）的背景 / 边框 / 圆角到新 token。trace 内部 Canvas 渲染用 trace 颜色 token（已在 Task 1 改好），不改 JS 渲染逻辑。

- [ ] **Step 9.6: typecheck + 单测**

```
npm run typecheck
npm run test:js
```

- [ ] **Step 9.7: 手动验证**

跑出一次完整分析（可用 examples.analyze-base 命令），看 canvas 里：
- crumb 小字 + 衬线大标题
- 3 列 stat 数字
- 色谱卡片暖白背景
- tabs 黑底白字激活态

- [ ] **Step 9.8: commit**

```
git add -A src/
git commit -m "feat(canvas): warm-themed workbench shell with display-serif heads"
```

---

## Task 10: CommandPalette + ShortcutsOverlay + Toast 视觉同步

**Files:**
- Modify: `src/components/CommandPalette.css`
- Modify: `src/components/ShortcutsOverlay.css`
- Modify: `src/components/ui/Toast.css`
- Modify: `src/components/OnboardingCoach.css`

**Why:** 弹层视觉与新主体保持一致；行为不动。

- [ ] **Step 10.1: 改 CommandPalette.css**

- overlay：bg rgba(20, 17, 13, 0.45) + backdrop-filter blur(8px)
- panel：bg var(--surface) + 1px border-default + radius 14px + shadow-elevated
- input row：bg transparent + 1px border-soft 底
- list item：font 13px + color text-main + padding 8px 12px
- selected item：bg accent-soft + 左侧 2px 柿橙竖线（用 box-shadow inset 2px 0 0 var(--accent)）
- shortcut 显示：font-mono + 10px + color text-subtle + bg var(--bg-app) + radius 4px + padding 1px 5px

- [ ] **Step 10.2: 改 ShortcutsOverlay.css**

同 CommandPalette 风格的 panel；表格行用新 token。

- [ ] **Step 10.3: 改 Toast.css**

- toast 容器：bg var(--surface) + 1px border-default + radius 10px + shadow-md
- toast.error：左侧 3px var(--status-error) 竖线
- toast.success：左侧 3px var(--status-ok)
- toast.info：左侧 3px var(--accent)
- 文字 color text-main / 描述 color text-muted
- action button：bg transparent + 1px border-default + color accent + radius 6px + padding 4px 10px font 12px

- [ ] **Step 10.4: 改 OnboardingCoach.css**

bg var(--surface) + 1px border-default + radius 14px + shadow-md，按钮用 accent。

- [ ] **Step 10.5: typecheck + dev 验证**

```
npm run typecheck
npm run dev
```
- Ctrl+K 打开 palette 看选中柿橙竖线
- 触发一个 error toast 看左侧红条
- 关闭 onboarding 重置 localStorage 看新样式

- [ ] **Step 10.6: commit**

```
git add src/components/CommandPalette.css src/components/ShortcutsOverlay.css src/components/ui/Toast.css src/components/OnboardingCoach.css
git commit -m "feat(overlays): unify palette/shortcuts/toast/onboarding to warm tokens"
```

---

## Task 11: TitleBar + DropZone + 业务面板 token 替换

**Files:**
- Modify: `src/components/TitleBar.css`
- Modify: `src/components/DropZone.css`
- Modify: `src/components/RecentAnalysesRail.css`（如果还在用 — task 7 后可能孤立，到时按情况删除文件并清 import）
- Modify: `src/components/panels/AnalysisPanel.tsx` 关联 css
- Modify: `src/components/panels/MutationTrendPanel.tsx` 关联 css
- Modify: `src/components/panels/LabSuggestionPanel.tsx` 关联 css
- Modify: `src/components/workbench/CompareView.css`
- Modify: `src/components/workbench/DetailDrawer.css`
- Modify: `src/components/workbench/ChromatogramCanvas.css`
- Modify: `src/components/workbench/ResultsCharts.css`
- Modify: `src/components/workbench/SequenceAlignmentView.css`
- Modify: `src/components/workbench/ExportMenu.css`

**Why:** 这是"扫尾"性质的 token 替换，扫描所有 css 文件把硬编码颜色 / 字体替换成 token 引用。

- [ ] **Step 11.1: 列出所有需要扫描的 css 文件**

```
ls src/components/**/*.css 2>/dev/null
ls src/components/*.css
```

- [ ] **Step 11.2: 找硬编码颜色**

```
grep -rn "#[0-9a-fA-F]\{3,6\}" src/components/ src/styles.css | grep -v "var(--"
```
预期：找到一批硬编码 hex；挑出色相不是新主题的那些。

- [ ] **Step 11.3: 逐文件替换**

- 文本色 → var(--text-main / --text-muted / --text-subtle / --text-faint)
- 背景色 → var(--surface / --bg-app / --bg-sidebar / --bg-canvas)
- 边框 → var(--border-soft / --border-default)
- 强调色 → var(--accent) / var(--accent-soft)
- 状态色 → var(--status-ok / --status-warn / --status-error)
- 字体：font-family Source Serif 用 var(--font-display)；任何 Fraunces 引用替换成 var(--font-display)

每个 css 文件改完后 dev 看一眼。

- [ ] **Step 11.4: 删 Fraunces 字体依赖**

- package.json 移除 `"@fontsource-variable/fraunces": "^5.2.9"`
- main.tsx 或 index.html 删去 Fraunces 的 @fontsource import
- 安装替代字体（实施期决定 self-host 还是 Google Fonts CDN）

```
npm install @fontsource/source-serif-pro
```

main.tsx 加：

```
import "@fontsource/source-serif-pro/400.css";
import "@fontsource/source-serif-pro/600.css";
```

- [ ] **Step 11.5: typecheck + 全测试**

```
npm run typecheck
npm run test:js
npm run test:py
```

- [ ] **Step 11.6: commit**

```
git add -A
git commit -m "feat(polish): replace hard-coded colors with warm tokens; swap Fraunces -> Source Serif"
```

---

## Task 12: 视觉回归 + e2e + 最终打磨

**Files:**
- Create: `docs/superpowers/specs/assets/2026-05-08/` — 截图基线目录
- Modify: `playwright.config.ts` 与 `tests/e2e/*.spec.ts`（如有 selector 变化）
- Modify: 任意需要微调的 CSS

**Why:** 最后一公里 — 跑 e2e、截全套截图、对照 spec §8 检查清单。

- [ ] **Step 12.1: 跑 typecheck + 全测试套件**

```
npm run typecheck
npm run test:js
npm run test:py
```

- [ ] **Step 12.2: 跑 playwright e2e**

```
npm run test:e2e
```

如果出现 selector 失效（rail-toggle 已删 / 三态 class 已改），更新对应 spec 文件让测试基于新结构。常见点：
- `[role=log]` 仍存在
- 按钮 aria-label 与 i18n key 一致
- 窗口控件按钮（Win/Linux）保持

- [ ] **Step 12.3: 截图基线**

dev 启动后在浏览器或 Electron 窗口里手动截以下场景，存到 docs/superpowers/specs/assets/2026-05-08/：

- 01-init-dialog-light.png
- 02-init-dialog-dark.png
- 03-init-provider-anthropic.png（看 proxy 提示）
- 04-init-provider-ollama.png（看 ApiKey 隐藏）
- 05-empty-state-light.png
- 06-empty-state-dark.png
- 07-default-light.png（chat + canvas 各占一半，跑完一次分析）
- 08-default-dark.png
- 09-chat-collapsed-light.png（拖到极窄，rail 状态）
- 10-sidebar-collapsed-light.png（Ctrl+B 折叠 sidebar）
- 11-settings-modal-light.png
- 12-command-palette-light.png
- 13-shortcuts-overlay-light.png
- 14-trends-panel-light.png
- 15-suggestions-panel-light.png

- [ ] **Step 12.4: spec §8 测试检查清单逐项打勾**

打开 spec 文件 §8 测试段，逐条对照截图确认覆盖 — 缺哪张补哪张。

- [ ] **Step 12.5: 删除备份文件**

```
rm src/styles/tokens.css.pre-warm.bak
```

- [ ] **Step 12.6: 最终 commit + push（如需）**

```
git add docs/superpowers/specs/assets/2026-05-08/ src/styles/tokens.css.pre-warm.bak
git commit -m "test(visual): baseline screenshots for warm-workbench redesign"
```

---

## 总结

| Task | 主题 | 大致改动 |
|---|---|---|
| 1 | Tokens | 1 文件重写 |
| 2 | Providers 模块 | 1 新文件 + 1 改 + 1 测试 |
| 3 | InitDialog | 2 改 + 1 新组件 + i18n |
| 4 | SettingsModal | 1 改 |
| 5 | Sidebar 组件 | 2 新文件 + i18n |
| 6 | Splitter | 4 新文件 |
| 7 | App shell 重构 | 3 改 + 1 删 |
| 8 | ChatPanel | 1 改 + module css |
| 9 | Canvas 外壳 | styles.css 大段 + workbench css |
| 10 | 弹层 | 4 css |
| 11 | Token 扫尾 + 字体替换 | 多文件 + package.json |
| 12 | 视觉回归 + 打磨 | 截图 + e2e |

每 task 独立可 revert；如某 task 视觉效果不理想，回滚单个 commit 不影响其他成果。

---

## 评审反馈补丁（plan-reviewer 反馈后追加）

### T12 commit 拆分修订
取代原 12.5 和 12.6 单一 commit，改为：

- [ ] **Step 12.5: 截图 commit**

```
git add docs/superpowers/specs/assets/2026-05-08/
git commit -m "test(visual): baseline screenshots for warm-workbench redesign"
```

- [ ] **Step 12.6: 删除 tokens 备份**

```
rm src/styles/tokens.css.pre-warm.bak
git add -u src/styles/
git commit -m "chore(cleanup): remove pre-warm tokens backup"
```

### 其他实施期注意事项
- **Task 2.1（TS 文件被 .mjs 测试 import）**：动手前先 cat 一个现有 tests/test_*.mjs 看是否真能 import .ts；如不能，把测试改为先把 providers.ts 编译产物或者用 vitest（package.json 已无）—— 折中：把 providers 的纯函数额外导出到一个 `.mjs` shim（`tests/_provider_helpers.mjs`）专门给测试用，业务代码继续用 .ts。
- **Task 7.3（ChatPanel theme prop）**：选定一个：要么从 props 完全删除 onCycleRail/railLabel/theme，要么只删未用的（onCycleRail/railLabel 必删，theme 暂留 — 因为 theme toggle 按钮还在 panel-action-group 里）。本期定为：删 onCycleRail/railLabel，保留 theme 直到决定 theme 按钮搬到 sidebar 还是保留于 chat header。
- **Task 11.1（glob 兼容）**：用 `find src/components -name "*.css"` 替代 `ls **/*.css`。
- **Task 11.4（font import 位置）**：进 main.tsx 与 index.html 都看一眼，确定现有 Fraunces 引入位置后再改。
