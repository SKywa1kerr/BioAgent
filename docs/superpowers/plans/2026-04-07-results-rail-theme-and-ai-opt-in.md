# Results Rail, Theme, And AI Opt-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep sample details collapsed by default, stabilize the AI rail composer, retune dark theme styling, and make AI-assisted analysis an explicit user opt-in.

**Architecture:** Extract the AI-mode and selection defaults into small pure helpers so the default behavior is testable, then wire those helpers through `App` and `SettingsPage`. Keep the layout and dark-theme work in existing component/CSS files, with `npm.cmd run build` and targeted manual smoke checks verifying the visual changes.

**Tech Stack:** React 18, TypeScript, Electron preload/invoke bridge, CSS modules by component file, Vite build.

---

## File Map

- Modify: `src/App.tsx`
  - stop auto-selecting the first sample after a fresh analysis run
  - gate AI-assisted analysis from saved settings rather than implicit defaults
- Modify: `src/types/index.ts`
  - ensure settings types can represent explicit AI opt-in state cleanly
- Modify: `src/components/SettingsPage.tsx`
  - expose AI opt-in control and conditionally show provider fields
- Modify: `src/components/SettingsPage.css`
  - support disabled/collapsed AI provider section styling
- Modify: `src/components/AgentPanel.tsx`
  - keep current message/composer structure but ensure stable bottom composer behavior if any structural wrapper is needed
- Modify: `src/components/AgentPanel.css`
  - enforce a strict header / message list / composer rail layout and retune dark colors
- Modify: `src/components/ResultsWorkbench.css`
  - retune dark-theme tokens for cards, lists, and dossier sections
- Modify: `src/App.css`
  - align page-level dark surfaces with the new dossier palette
- Create: `src/utils/analysisPreferences.ts`
  - hold pure helpers for AI opt-in defaults and validation
- Create: `src/utils/resultSelection.ts`
  - hold pure helper for post-analysis default selection behavior
- Create: `tests/test_analysis_preferences.py`
  - regression tests for AI opt-in defaults and validation behavior
- Create: `tests/test_result_selection.py`
  - regression tests for default collapsed selection behavior

## Task 1: Add Testable AI Preference Helpers

**Files:**
- Create: `src/utils/analysisPreferences.ts`
- Create: `tests/test_analysis_preferences.py`
- Modify: `src/types/index.ts`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import re


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_analysis_preference_defaults_and_validation_contract():
    content = _read("src/utils/analysisPreferences.ts")
    assert 'export const DEFAULT_ANALYSIS_DECISION_MODE = "rules"' in content
    assert 'export function isAiReviewEnabled(' in content
    assert 'export function validateAiReviewSettings(' in content
    assert 'return { ok: false, reason: "missing_api_key" }' in content
    assert 'return { ok: false, reason: "missing_base_url" }' in content
    assert 'return { ok: false, reason: "missing_model" }' in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_preferences.py -q`
Expected: FAIL because `src/utils/analysisPreferences.ts` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```ts
import type { AppSettings } from "../types";

export const DEFAULT_ANALYSIS_DECISION_MODE: NonNullable<AppSettings["analysisDecisionMode"]> = "rules";

export function isAiReviewEnabled(settings: Pick<AppSettings, "analysisDecisionMode">) {
  return (settings.analysisDecisionMode ?? DEFAULT_ANALYSIS_DECISION_MODE) === "hybrid";
}

export function validateAiReviewSettings(
  settings: Pick<AppSettings, "analysisDecisionMode" | "llmApiKey" | "llmBaseUrl" | "llmModel">
) {
  if (!isAiReviewEnabled(settings)) {
    return { ok: true as const };
  }

  if (!settings.llmApiKey?.trim()) {
    return { ok: false as const, reason: "missing_api_key" as const };
  }
  if (!settings.llmBaseUrl?.trim()) {
    return { ok: false as const, reason: "missing_base_url" as const };
  }
  if (!settings.llmModel?.trim()) {
    return { ok: false as const, reason: "missing_model" as const };
  }

  return { ok: true as const };
}
```

```ts
export interface AppSettings {
  llmApiKey: string;
  llmBaseUrl: string;
  llmModel: string;
  plasmid: string;
  qualityThreshold: number;
  analysisDecisionMode?: "rules" | "hybrid";
  language?: AppLanguage;
  theme?: AppTheme;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_analysis_preferences.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/analysisPreferences.ts src/types/index.ts tests/test_analysis_preferences.py
git commit -m "test: add ai preference helper coverage"
```

## Task 2: Add Testable Post-Analysis Selection Helper

**Files:**
- Create: `src/utils/resultSelection.ts`
- Create: `tests/test_result_selection.py`
- Modify: `src/App.tsx`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_result_selection_defaults_to_collapsed_state():
    content = _read("src/utils/resultSelection.ts")
    assert 'export function getDefaultSelectedSampleId()' in content
    assert 'return null;' in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_result_selection.py -q`
Expected: FAIL because `src/utils/resultSelection.ts` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```ts
import type { Sample } from "../types";

export function getDefaultSelectedSampleId(_samples: Sample[]) {
  return null;
}
```

```tsx
import { getDefaultSelectedSampleId } from "./utils/resultSelection";

const nextSelectedSampleId = getDefaultSelectedSampleId(nextSamples);
setSelectedSampleId(nextSelectedSampleId);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_result_selection.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/resultSelection.ts src/App.tsx tests/test_result_selection.py
git commit -m "feat: default results list to collapsed state"
```

## Task 3: Wire Settings UI For Explicit AI Opt-In

**Files:**
- Modify: `src/components/SettingsPage.tsx`
- Modify: `src/components/SettingsPage.css`
- Modify: `src/App.tsx`
- Modify: `src/i18n.ts`
- Use: `src/utils/analysisPreferences.ts`
- Verify: `tests/test_analysis_preferences.py`

- [ ] **Step 1: Write the failing test for the default setting contract**

```python
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_settings_page_uses_rules_as_default_mode():
    content = _read("src/components/SettingsPage.tsx")
    assert 'analysisDecisionMode: "rules"' in content
    assert 'isAiReviewEnabled(settings)' in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_preferences.py tests/test_result_selection.py -q`
Expected: FAIL because `SettingsPage.tsx` still defaults to `hybrid` and does not use the helper.

- [ ] **Step 3: Write minimal implementation**

```tsx
import {
  DEFAULT_ANALYSIS_DECISION_MODE,
  isAiReviewEnabled,
  validateAiReviewSettings,
} from "../utils/analysisPreferences";

const DEFAULT_SETTINGS: AppSettings = {
  llmApiKey: "",
  llmBaseUrl: "https://models.sjtu.edu.cn/api/v1",
  llmModel: "deepseek-chat",
  plasmid: "pet22b",
  qualityThreshold: 20,
  analysisDecisionMode: DEFAULT_ANALYSIS_DECISION_MODE,
  language: undefined,
  theme: "light",
};

const aiEnabled = isAiReviewEnabled(settings);
```

```tsx
<div className="settings-field">
  <label htmlFor="analysisDecisionMode">{t(language, "settings.analysisDecisionMode")}</label>
  <select
    id="analysisDecisionMode"
    value={settings.analysisDecisionMode || DEFAULT_ANALYSIS_DECISION_MODE}
    onChange={(e) =>
      updateSetting("analysisDecisionMode", e.target.value as AppSettings["analysisDecisionMode"])
    }
  >
    <option value="rules">{t(language, "settings.analysisDecisionModeRules")}</option>
    <option value="hybrid">{t(language, "settings.analysisDecisionModeHybrid")}</option>
  </select>
</div>

<section className={`settings-card ${aiEnabled ? "" : "settings-card--muted"}`}>
  {/* provider fields */}
</section>
```

```tsx
const aiValidation = validateAiReviewSettings(settings);
if (!aiValidation.ok) {
  alert(t(language, `settings.${aiValidation.reason}`));
  return;
}
const useLLM = isAiReviewEnabled(settings);
```

```css
.settings-card--muted {
  opacity: 0.72;
}

.settings-card--muted .settings-field {
  pointer-events: none;
}
```

- [ ] **Step 4: Run tests and build**

Run: `python -m pytest tests/test_analysis_preferences.py tests/test_result_selection.py -q`
Expected: PASS

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/SettingsPage.tsx src/components/SettingsPage.css src/App.tsx src/i18n.ts src/utils/analysisPreferences.ts tests/test_analysis_preferences.py tests/test_result_selection.py
git commit -m "feat: make ai review explicit opt-in"
```

## Task 4: Stabilize Agent Rail Layout And Retune Dark Theme

**Files:**
- Modify: `src/components/AgentPanel.tsx`
- Modify: `src/components/AgentPanel.css`
- Modify: `src/App.css`
- Modify: `src/components/ResultsWorkbench.css`
- Verify: `npm.cmd run build`

- [ ] **Step 1: Write the failing test for the rail layout contract**

```python
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_agent_panel_css_has_three_part_layout_contract():
    content = _read("src/components/AgentPanel.css")
    assert '.agent-panel {' in content
    assert 'display: grid;' in content or 'grid-template-rows:' in content
    assert '.agent-message-list {' in content and 'overflow-y: auto;' in content
    assert '.agent-composer {' in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis_preferences.py tests/test_result_selection.py -q`
Expected: FAIL because the CSS contract is not grid-based yet.

- [ ] **Step 3: Write minimal implementation**

```css
.agent-panel {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  min-height: 0;
  height: 100%;
}

.agent-message-list {
  min-height: 0;
  overflow-y: auto;
}

.agent-composer {
  position: sticky;
  bottom: 0;
  z-index: 2;
}
```

```css
:root[data-theme="dark"] .agent-panel {
  border-color: rgba(122, 140, 128, 0.22);
  background:
    radial-gradient(circle at top right, rgba(132, 169, 140, 0.1), transparent 42%),
    linear-gradient(180deg, rgba(18, 26, 24, 0.96) 0%, rgba(25, 34, 31, 0.99) 100%);
}

:root[data-theme="dark"] .results-workbench,
:root[data-theme="dark"] .results-summary-panel,
:root[data-theme="dark"] .sample-detail-card {
  background: rgba(31, 41, 37, 0.92);
  border-color: rgba(122, 140, 128, 0.16);
}
```

- [ ] **Step 4: Run build and perform manual smoke checks**

Run: `npm.cmd run build`
Expected: PASS

Manual smoke checks:
- run `npm.cmd run electron:dev`
- load a dataset and confirm no sample opens automatically
- expand multiple samples and verify the AI composer remains visible and usable
- switch to dark mode and verify there are no bright white cards or sharp cyan highlights in the results workbench or AI rail

- [ ] **Step 5: Commit**

```bash
git add src/components/AgentPanel.tsx src/components/AgentPanel.css src/App.css src/components/ResultsWorkbench.css
git commit -m "feat: stabilize ai rail and retune dark theme"
```

## Self-Review

- Spec coverage checked:
  - collapsed-by-default sample details: Task 2
  - AI opt-in and user-supplied API behavior: Tasks 1 and 3
  - AI rail composer stability: Task 4
  - dark-theme retune: Task 4
- Placeholder scan checked:
  - no `TODO`, `TBD`, or task references without concrete files or commands
- Type consistency checked:
  - `analysisDecisionMode` remains `"rules" | "hybrid"`
  - helper names are consistent across tasks: `isAiReviewEnabled`, `validateAiReviewSettings`, `getDefaultSelectedSampleId`
