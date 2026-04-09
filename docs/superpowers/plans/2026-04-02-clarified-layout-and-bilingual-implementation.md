# Clarified Layout And Bilingual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify the analysis workbench hierarchy, add app-level Chinese and English switching, and rebalance color usage so the UI reads as a centered scientific desktop product.

**Architecture:** Add a lightweight translation layer at the app shell, persist language in settings, and thread translated labels through the main workbench surfaces. Rework the layout so the center analysis stage is visually dominant, the left rail becomes a true navigation layer, and the right Agent panel becomes a quieter support surface. Keep Electron and Python behavior unchanged except for storing language with the existing settings payload.

**Tech Stack:** React 18, TypeScript 5, Electron 33, Vite 5, CSS

---

## File Structure

### Files to create

- `src/i18n.ts` - translation dictionaries, language type, and lookup helper

### Files to modify

- `src/types/index.ts` - add language setting type
- `src/App.tsx` - own language state, wire translated copy, add language switch, strengthen workbench hierarchy
- `src/App.css` - clarify app shell, header, left-center-right balance, and translated header controls
- `src/components/TabLayout.tsx` - support translated tab labels cleanly
- `src/components/TabLayout.css` - align tab bar with clarified product hierarchy
- `src/components/SampleList.tsx` - translate sample rail copy and make the rail read as navigation
- `src/components/SampleList.css` - rebalance left-rail structure and stronger subordinate styling
- `src/components/AgentPanel.tsx` - translate panel copy and keep support-panel tone
- `src/components/AgentPanel.css` - visually quiet the panel relative to the center stage
- `src/components/ChatMessage.tsx` - translate message metadata labels
- `src/components/HistoryPage.tsx` - translate headings, summaries, empty/loading/error states
- `src/components/HistoryPage.css` - adjust archive colors to align with revised hierarchy
- `src/components/SettingsPage.tsx` - add language control and translate settings copy
- `src/components/SettingsPage.css` - align settings visuals with revised regional color strategy

### Verification commands

- `npm.cmd run build`
- `npm.cmd run electron:dev`

---

### Task 1: Add app-level translation primitives

**Files:**
- Create: `src/i18n.ts`
- Modify: `src/types/index.ts`
- Test: `npm.cmd run build`

- [ ] **Step 1: Add the failing type and translation usage plan**

Create `src/i18n.ts` with:

```ts
export type AppLanguage = "zh" | "en";

type TranslationTree = {
  [key: string]: string | TranslationTree;
};

const translations: Record<AppLanguage, TranslationTree> = {
  zh: {
    app: {
      brandKicker: "生物分析工作台",
      languageZh: "中文",
      languageEn: "EN",
    },
  },
  en: {
    app: {
      brandKicker: "Biotech Workbench",
      languageZh: "中文",
      languageEn: "EN",
    },
  },
};

export function t(language: AppLanguage, path: string): string {
  const keys = path.split(".");
  let current: string | TranslationTree = translations[language];

  for (const key of keys) {
    if (!current || typeof current === "string" || !(key in current)) {
      return path;
    }
    current = current[key];
  }

  return typeof current === "string" ? current : path;
}
```

Append to `src/types/index.ts`:

```ts
export type AppLanguage = "zh" | "en";
```

- [ ] **Step 2: Run the build to confirm the new translation module is valid**

Run: `npm.cmd run build`
Expected: PASS, or fail only because the new `AppLanguage` export is not yet wired through consumers.

- [ ] **Step 3: Commit the translation primitives**

```bash
git add src/i18n.ts src/types/index.ts
git commit -m "feat: add lightweight bilingual translation primitives"
```

---

### Task 2: Persist language choice and wire app-shell translation

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Modify: `src/types/index.ts`
- Modify: `src/components/SettingsPage.tsx`
- Test: `npm.cmd run build`

- [ ] **Step 1: Extend app settings with language**

Append to `AppSettings` in `src/types/index.ts`:

```ts
export interface AppSettings {
  llmApiKey: string;
  llmBaseUrl: string;
  plasmid: string;
  qualityThreshold: number;
  language?: AppLanguage;
}
```

- [ ] **Step 2: Wire language state into `App.tsx`**

In `src/App.tsx`, import `useEffect`, `AppLanguage`, and `t`, then add:

```ts
const [language, setLanguage] = useState<AppLanguage>("zh");

useEffect(() => {
  (async () => {
    try {
      const result = (await invoke("load-settings")) as string | null;
      if (!result) return;
      const parsed = JSON.parse(result) as { language?: AppLanguage };
      if (parsed.language === "zh" || parsed.language === "en") {
        setLanguage(parsed.language);
      }
    } catch (error) {
      console.error("Failed to load language setting:", error);
    }
  })();
}, []);
```

Add a language toggle callback:

```ts
const handleLanguageChange = async (nextLanguage: AppLanguage) => {
  setLanguage(nextLanguage);
  try {
    const result = (await invoke("load-settings")) as string | null;
    const current = result ? JSON.parse(result) : {};
    await invoke("save-settings", JSON.stringify({ ...current, language: nextLanguage }));
  } catch (error) {
    console.error("Failed to persist language:", error);
  }
};
```

- [ ] **Step 3: Rework the top header so hierarchy and language switch are explicit**

Replace the current header block in `src/App.tsx` with:

```tsx
<header className="app-header">
  <div className="brand-block">
    <div className="brand-kicker">{t(language, "app.brandKicker")}</div>
    <div className="logo">BioAgent</div>
  </div>
  <div className="header-context">
    <div className="context-chip">
      <span className="context-label">{t(language, "app.run")}</span>
      <strong>{samples.length || 0} {t(language, "app.samples")}</strong>
    </div>
    <div className="context-chip">
      <span className="context-label">{t(language, "app.pass")}</span>
      <strong>{okCount}</strong>
    </div>
    <div className="context-chip">
      <span className="context-label">{t(language, "app.issues")}</span>
      <strong>{issueCount + pendingCount}</strong>
    </div>
  </div>
  <div className="language-switch" role="group" aria-label={t(language, "app.languageSwitch")}>
    <button
      type="button"
      className={language === "zh" ? "active" : ""}
      onClick={() => handleLanguageChange("zh")}
    >
      {t(language, "app.languageZh")}
    </button>
    <button
      type="button"
      className={language === "en" ? "active" : ""}
      onClick={() => handleLanguageChange("en")}
    >
      {t(language, "app.languageEn")}
    </button>
  </div>
</header>
```

- [ ] **Step 4: Add translated tab labels**

Replace the static `tabs` constant with:

```ts
const buildTabs = (language: AppLanguage) => [
  { id: "analysis", label: t(language, "tabs.analysis") },
  { id: "history", label: t(language, "tabs.history") },
  { id: "settings", label: t(language, "tabs.settings") },
];
```

Then render:

```tsx
<TabLayout tabs={buildTabs(language)} activeTab={activeTab} onTabChange={setActiveTab}>
```

- [ ] **Step 5: Strengthen header and top-level layout CSS**

Append to `src/App.css`:

```css
.language-switch {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px;
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.72);
}

.language-switch button {
  min-width: 56px;
  height: 34px;
  border: none;
  border-radius: 999px;
  background: transparent;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}

.language-switch button.active {
  background: linear-gradient(135deg, var(--brand-deep) 0%, #2c716c 100%);
  color: #f8fffc;
}
```

- [ ] **Step 6: Run the build to confirm language wiring compiles**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 7: Commit the app-shell bilingual wiring**

```bash
git add src/App.tsx src/App.css src/types/index.ts src/i18n.ts
git commit -m "feat: add app shell bilingual language switching"
```

---

### Task 3: Translate and rebalance the analysis workbench

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Modify: `src/components/SampleList.tsx`
- Modify: `src/components/SampleList.css`
- Test: `npm.cmd run build`

- [ ] **Step 1: Translate toolbar and analysis-state copy**

In `src/i18n.ts`, extend dictionaries with:

```ts
analysis: {
  importAb1: "导入 AB1 文件",
  importReference: "导入参考文件",
  runAnalysis: "开始分析",
  autoImport: "自动导入",
  exportExcel: "导出 Excel",
  identity: "一致性",
  coverage: "覆盖率",
  frameshift: "移码",
  mutations: "突变",
  noMutations: "未检测到突变",
  chromatogram: "色谱图谱",
  analysisError: "分析错误",
  empty: "打开包含 AB1 和参考文件的文件夹以开始分析",
}
```

Then replace the hard-coded toolbar and content strings in `src/App.tsx` with `t(language, "...")`.

- [ ] **Step 2: Make the left rail explicitly navigational**

Update `src/components/SampleList.tsx` signature:

```ts
interface SampleListProps {
  samples: Sample[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  language: AppLanguage;
}
```

Use translated copy:

```tsx
<span className="sample-list-kicker">{t(language, "sampleRail.kicker")}</span>
<h3>{t(language, "sampleRail.title")} ({samples.length})</h3>
<div className="sample-list-summary">
  <span>{okCount} {t(language, "sampleRail.pass")}</span>
  <span>{flaggedCount} {t(language, "sampleRail.issue")}</span>
  <span>{reviewCount} {t(language, "sampleRail.review")}</span>
</div>
```

Pass `language={language}` from `src/App.tsx`.

- [ ] **Step 3: Make the center stage more dominant than side surfaces**

In `src/App.css`, adjust the regional weights:

```css
.sidebar {
  width: 280px;
  background: linear-gradient(180deg, rgba(24, 61, 64, 0.96), rgba(17, 47, 49, 0.96));
}

.main-content {
  background: linear-gradient(180deg, rgba(255, 253, 249, 0.98), rgba(249, 252, 248, 0.98));
  box-shadow: 0 20px 44px rgba(18, 39, 40, 0.08);
}

.analysis-main {
  gap: 16px;
}
```

Then update `SampleList.css` text colors so it reads correctly on the darker rail.

- [ ] **Step 4: Run the build to verify the clarified workbench compiles**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit the clarified analysis hierarchy**

```bash
git add src/App.tsx src/App.css src/components/SampleList.tsx src/components/SampleList.css src/i18n.ts
git commit -m "feat: clarify analysis workbench hierarchy"
```

---

### Task 4: Translate and quiet the Agent support surface

**Files:**
- Modify: `src/components/AgentPanel.tsx`
- Modify: `src/components/AgentPanel.css`
- Modify: `src/components/ChatMessage.tsx`
- Modify: `src/i18n.ts`
- Test: `npm.cmd run build`

- [ ] **Step 1: Thread language into the Agent panel**

Update the props:

```ts
interface AgentPanelProps {
  samples: Sample[];
  selectedSampleId: string | null;
  sourcePath?: string | null;
  genesDir?: string | null;
  plasmid?: string;
  language: AppLanguage;
  onAnalysisComplete?: (nextAnalysis: AnalysisContextUpdate) => void;
}
```

Pass `language={language}` from `src/App.tsx`.

- [ ] **Step 2: Translate panel copy and metadata labels**

In `src/i18n.ts`, add:

```ts
agent: {
  title: "智能助手",
  idle: "空闲",
  running: "运行中",
  clear: "清空",
  askTitle: "询问当前分析",
  askBody: "助手可以查看当前样本、读取历史、重新运行分析并展示工具执行过程。",
  placeholder: "询问当前批次、样本详情、历史记录或导出操作",
  composerHint: "Enter 发送，Shift+Enter 换行",
  send: "发送",
  stop: "停止原因",
  tokens: "Token",
}
```

Replace the hard-coded copy in `src/components/AgentPanel.tsx` and `src/components/ChatMessage.tsx` with `t(language, "...")`.

- [ ] **Step 3: Make the Agent panel visually quieter than the center stage**

Update `src/components/AgentPanel.css`:

```css
.agent-panel {
  width: 344px;
  background:
    radial-gradient(circle at top, rgba(77, 157, 135, 0.08), transparent 34%),
    linear-gradient(180deg, #eef6f2 0%, #e7f1ec 100%);
}

.agent-panel-header {
  background: rgba(247, 252, 249, 0.82);
}
```

Keep message cards readable, but avoid stronger contrast than `.main-content`.

- [ ] **Step 4: Run the build to confirm the support surface still compiles**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit the bilingual Agent panel**

```bash
git add src/components/AgentPanel.tsx src/components/AgentPanel.css src/components/ChatMessage.tsx src/i18n.ts src/App.tsx
git commit -m "feat: localize and rebalance agent support panel"
```

---

### Task 5: Translate History and Settings surfaces

**Files:**
- Modify: `src/components/HistoryPage.tsx`
- Modify: `src/components/HistoryPage.css`
- Modify: `src/components/SettingsPage.tsx`
- Modify: `src/components/SettingsPage.css`
- Modify: `src/i18n.ts`
- Test: `npm.cmd run build`

- [ ] **Step 1: Thread language through both pages**

Update both component signatures:

```ts
interface HistoryPageProps {
  language: AppLanguage;
}

interface SettingsPageProps {
  language: AppLanguage;
  onLanguageChange: (nextLanguage: AppLanguage) => void;
}
```

Pass props from `src/App.tsx`:

```tsx
{activeTab === "history" && <HistoryPage language={language} />}
{activeTab === "settings" && (
  <SettingsPage language={language} onLanguageChange={handleLanguageChange} />
)}
```

- [ ] **Step 2: Translate archive and settings copy**

Add `history.*` and `settings.*` keys to `src/i18n.ts`, then replace hard-coded strings in both files with `t(language, "...")`.

Include a language card in `SettingsPage.tsx`:

```tsx
<div className="settings-field">
  <label htmlFor="language">{t(language, "settings.languageLabel")}</label>
  <select
    id="language"
    value={language}
    onChange={(event) => onLanguageChange(event.target.value as AppLanguage)}
  >
    <option value="zh">{t(language, "app.languageZh")}</option>
    <option value="en">{t(language, "app.languageEn")}</option>
  </select>
</div>
```

- [ ] **Step 3: Adjust regional color alignment**

Refine `HistoryPage.css` and `SettingsPage.css` so they follow the updated shell logic:

```css
.history-hero,
.settings-card {
  border-color: rgba(23, 77, 82, 0.12);
}

.summary-card-accent,
.settings-page .btn-primary {
  background: linear-gradient(135deg, var(--brand-deep) 0%, #2c716c 100%);
}
```

- [ ] **Step 4: Run the build to confirm both translated pages compile**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit the bilingual product pages**

```bash
git add src/components/HistoryPage.tsx src/components/HistoryPage.css src/components/SettingsPage.tsx src/components/SettingsPage.css src/i18n.ts src/App.tsx
git commit -m "feat: localize history and settings pages"
```

---

### Task 6: Final verification and desktop smoke test

**Files:**
- No required new files
- Test: `npm.cmd run build`, `npm.cmd run electron:dev`

- [ ] **Step 1: Run the production build**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 2: Launch the desktop app in dev mode**

Run: `npm.cmd run electron:dev`
Expected: Electron launches the BioAgent window and connects to the local dev server successfully.

- [ ] **Step 3: Perform focused manual checks**

Manual checks:

1. Default launch language is Chinese.
2. Top-right language toggle switches all major app chrome between Chinese and English.
3. Selected language remains after restarting the app.
4. Analysis page reads center-first, with a clearly subordinate left rail and quieter right Agent panel.
5. History and Settings pages remain visually aligned with the new product language.

- [ ] **Step 4: Commit final polish if needed**

```bash
git add -u
git commit -m "fix: polish clarified layout and bilingual ui"
```

---

## Self-Review

### Spec coverage

- layout hierarchy clarification: covered by Tasks 2, 3, and 4
- Chinese default and English switching: covered by Tasks 1, 2, and 5
- lightweight language toggle in the header: covered by Task 2
- richer but purposeful color system: covered by Tasks 3, 4, and 5
- shared product-language alignment across History, Settings, and Agent: covered by Tasks 4 and 5

### Placeholder scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Each task includes exact file paths and explicit verification commands.

### Type consistency

- `AppLanguage` is introduced in Task 1 and reused consistently in later tasks.
- `language` and `onLanguageChange` props use the same names across `App.tsx`, `SettingsPage.tsx`, `HistoryPage.tsx`, `SampleList.tsx`, and `AgentPanel.tsx`.
