# Dataset Import With Advanced Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users import a batch dataset directory once, auto-detect `ab1/` and `gb/`, and still access manual AB1/GB pickers through an expandable advanced mode.

**Architecture:** Keep analysis execution unchanged and implement the feature entirely in the Electron/React shell. Add a small dataset-path resolver in the app shell, show the detected dataset summary in the toolbar, and gate the existing manual path controls behind an advanced-mode toggle. Preserve the current folder-picker IPC and reuse it for both dataset import and manual overrides.

**Tech Stack:** React, TypeScript, Electron preload IPC, Vite, pytest

---

## File Map

- Modify: `src/App.tsx`
  - Own dataset import state, auto-detection flow, advanced-mode toggle, and toolbar wiring.
- Modify: `src/App.css`
  - Restyle the analysis toolbar to support dataset summary and collapsible advanced controls.
- Modify: `src/i18n.ts`
  - Add bilingual strings for dataset import, advanced mode, and dataset validation errors.
- Modify: `src/types/index.ts`
  - Add any small UI-only type needed for dataset detection results.
- Create: `tests/test_dataset_import.py`
  - Cover dataset directory detection and error behavior as a pure function.

### Task 1: Add dataset detection tests and pure resolver

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/types/index.ts`
- Test: `tests/test_dataset_import.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))


def resolve_dataset_paths(dataset_dir: str) -> dict:
    raise NotImplementedError


def test_resolve_dataset_paths_detects_ab1_and_gb(tmp_path):
    dataset_dir = tmp_path / "base"
    (dataset_dir / "ab1").mkdir(parents=True)
    (dataset_dir / "gb").mkdir(parents=True)

    result = resolve_dataset_paths(str(dataset_dir))

    assert result == {
        "dataset_dir": str(dataset_dir),
        "dataset_name": "base",
        "ab1_dir": str(dataset_dir / "ab1"),
        "gb_dir": str(dataset_dir / "gb"),
        "missing": [],
        "valid": True,
    }


def test_resolve_dataset_paths_reports_missing_gb(tmp_path):
    dataset_dir = tmp_path / "base"
    (dataset_dir / "ab1").mkdir(parents=True)

    result = resolve_dataset_paths(str(dataset_dir))

    assert result["ab1_dir"] == str(dataset_dir / "ab1")
    assert result["gb_dir"] is None
    assert result["missing"] == ["gb"]
    assert result["valid"] is True


def test_resolve_dataset_paths_rejects_invalid_dataset(tmp_path):
    dataset_dir = tmp_path / "empty"
    dataset_dir.mkdir()

    result = resolve_dataset_paths(str(dataset_dir))

    assert result["ab1_dir"] is None
    assert result["gb_dir"] is None
    assert result["missing"] == ["ab1", "gb"]
    assert result["valid"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dataset_import.py -q`
Expected: FAIL because `resolve_dataset_paths` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path


def resolve_dataset_paths(dataset_dir: str) -> dict:
    base = Path(dataset_dir)
    ab1_dir = base / "ab1"
    gb_dir = base / "gb"

    resolved_ab1 = str(ab1_dir) if ab1_dir.exists() and ab1_dir.is_dir() else None
    resolved_gb = str(gb_dir) if gb_dir.exists() and gb_dir.is_dir() else None

    missing = []
    if resolved_ab1 is None:
        missing.append("ab1")
    if resolved_gb is None:
        missing.append("gb")

    return {
        "dataset_dir": str(base),
        "dataset_name": base.name,
        "ab1_dir": resolved_ab1,
        "gb_dir": resolved_gb,
        "missing": missing,
        "valid": bool(resolved_ab1 or resolved_gb),
    }
```

Implement the same logic in `src/App.tsx` as a small pure helper near the top-level component, and add the matching TS type in `src/types/index.ts`:

```ts
export interface DatasetImportState {
  datasetDir: string;
  datasetName: string;
  ab1Dir: string | null;
  gbDir: string | null;
  missing: Array<"ab1" | "gb">;
  valid: boolean;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_dataset_import.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_dataset_import.py src/App.tsx src/types/index.ts
git commit -m "feat: add dataset import path resolver"
```

### Task 2: Add app-shell state and dataset import flow

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Write the failing test**

Use a focused UI-state test by asserting the resolver behavior in `tests/test_dataset_import.py` for both-folder and missing-folder cases first, then add one more case:

```python
def test_resolve_dataset_paths_keeps_dataset_name_for_summary(tmp_path):
    dataset_dir = tmp_path / "promax"
    (dataset_dir / "ab1").mkdir(parents=True)
    (dataset_dir / "gb").mkdir(parents=True)

    result = resolve_dataset_paths(str(dataset_dir))

    assert result["dataset_name"] == "promax"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dataset_import.py::test_resolve_dataset_paths_keeps_dataset_name_for_summary -q`
Expected: FAIL before the new assertion support is added.

- [ ] **Step 3: Write minimal implementation**

In `src/App.tsx`, add state and handlers:

```ts
const [datasetImport, setDatasetImport] = useState<DatasetImportState | null>(null);
const [showAdvancedImport, setShowAdvancedImport] = useState(false);

const handleSelectDatasetDir = async () => {
  const folder = (await invoke("open-folder-dialog")) as string | null;
  if (!folder) return;

  const nextDataset = resolveDatasetPaths(folder);
  setDatasetImport(nextDataset);
  setAb1Dir(nextDataset.ab1Dir);
  setGenesDir(nextDataset.gbDir);

  if (!nextDataset.valid) {
    alert(t(language, "dataset.invalid"));
    return;
  }

  if (nextDataset.missing.includes("ab1")) {
    alert(t(language, "dataset.missingAb1"));
  } else if (nextDataset.missing.includes("gb")) {
    alert(t(language, "dataset.missingGb"));
  }
};
```

When manual override handlers run, keep `datasetImport` but replace the relevant field:

```ts
setDatasetImport((current) =>
  current ? { ...current, ab1Dir: folder, missing: current.missing.filter((item) => item !== "ab1") } : current
);
```

Add i18n keys in `src/i18n.ts`:

```ts
dataset: {
  importDataset: "...",
  datasetReady: "...",
  advancedMode: "...",
  datasetFolder: "...",
  ab1Folder: "...",
  gbFolder: "...",
  missingAb1: "...",
  missingGb: "...",
  invalid: "...",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_dataset_import.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/i18n.ts tests/test_dataset_import.py
git commit -m "feat: add dataset import state and validation"
```

### Task 3: Reshape the toolbar UI around dataset import

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.css`

- [ ] **Step 1: Write the failing test**

Add one more resolver safety test first to keep the work TDD-driven:

```python
def test_resolve_dataset_paths_accepts_dataset_with_only_gb(tmp_path):
    dataset_dir = tmp_path / "reference_only"
    (dataset_dir / "gb").mkdir(parents=True)

    result = resolve_dataset_paths(str(dataset_dir))

    assert result["valid"] is True
    assert result["missing"] == ["ab1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dataset_import.py::test_resolve_dataset_paths_accepts_dataset_with_only_gb -q`
Expected: FAIL before the resolver logic is updated if needed.

- [ ] **Step 3: Write minimal implementation**

Replace the current two-column toolbar layout with:

```tsx
<div className="toolbar">
  <div className="dataset-import-panel">
    <div className="dataset-import-copy">
      <span className="toolbar-kicker">{t(language, "dataset.datasetReady")}</span>
      <button className="btn-primary" onClick={handleSelectDatasetDir}>
        {t(language, "dataset.importDataset")}
      </button>
    </div>
    {datasetImport ? (
      <div className="dataset-summary">
        <span>{t(language, "dataset.datasetFolder")}: {datasetImport.datasetName}</span>
        <span>{t(language, "dataset.ab1Folder")}: {datasetImport.ab1Dir ?? "-"}</span>
        <span>{t(language, "dataset.gbFolder")}: {datasetImport.gbDir ?? "-"}</span>
      </div>
    ) : null}
  </div>

  <div className="action-buttons">
    ...
  </div>
</div>
```

Add a secondary advanced-mode trigger under the dataset summary:

```tsx
<button
  type="button"
  className="toolbar-link-button"
  onClick={() => setShowAdvancedImport((current) => !current)}
>
  {t(language, "dataset.advancedMode")}
</button>
```

In `src/App.css`, add the supporting layout:

```css
.dataset-import-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 320px;
}

.dataset-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.toolbar-link-button {
  border: 0;
  background: transparent;
  color: var(--brand-deep);
  font-size: 12px;
  font-weight: 700;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_dataset_import.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.css tests/test_dataset_import.py
git commit -m "feat: make dataset import the primary analysis entry"
```

### Task 4: Hide manual AB1/GB pickers behind advanced mode

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Modify: `src/i18n.ts`

- [ ] **Step 1: Write the failing test**

Add one final resolver regression test to lock the missing-both case:

```python
def test_resolve_dataset_paths_marks_missing_both_in_order(tmp_path):
    dataset_dir = tmp_path / "broken"
    dataset_dir.mkdir()

    result = resolve_dataset_paths(str(dataset_dir))

    assert result["missing"] == ["ab1", "gb"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dataset_import.py::test_resolve_dataset_paths_marks_missing_both_in_order -q`
Expected: FAIL if ordering or missing-state logic changes unexpectedly.

- [ ] **Step 3: Write minimal implementation**

Wrap the existing manual controls:

```tsx
{showAdvancedImport ? (
  <div className="path-selectors advanced-path-selectors">
    <button onClick={handleSelectAb1Dir} title={t(language, "analysis.importAb1")}>
      ...
    </button>
    <button onClick={handleSelectGenesDir} title={t(language, "analysis.importReference")}>
      ...
    </button>
    <select value={plasmid} onChange={(e) => setPlasmid(e.target.value)}>
      ...
    </select>
  </div>
) : (
  <div className="path-selectors advanced-path-selectors advanced-path-selectors--collapsed">
    <span>{t(language, "dataset.advancedMode")}</span>
  </div>
)}
```

And style it as subordinate in `src/App.css`:

```css
.advanced-path-selectors {
  padding-top: 8px;
  border-top: 1px dashed var(--border-soft);
}

.advanced-path-selectors--collapsed {
  display: none;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_dataset_import.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.css src/i18n.ts tests/test_dataset_import.py
git commit -m "feat: keep manual path import in advanced mode"
```

### Task 5: Verify build, regression tests, and desktop smoke

**Files:**
- Modify: `src/App.tsx` (only if final fixes are required)
- Modify: `src/App.css` (only if final fixes are required)
- Modify: `src/i18n.ts` (only if final fixes are required)
- Test: `tests/test_dataset_import.py`

- [ ] **Step 1: Run targeted tests**

Run: `python -m pytest tests\test_dataset_import.py tests\test_main.py -q`
Expected: PASS

- [ ] **Step 2: Run existing regression tests**

Run: `python -m pytest tests\test_agent_tools.py tests\test_agent_chat.py -q`
Expected: PASS

- [ ] **Step 3: Run frontend production build**

Run: `npm.cmd run build`
Expected: PASS with generated `dist` assets and no TypeScript errors.

- [ ] **Step 4: Run Electron smoke with local dataset**

Run: `npm.cmd run electron:dev`
Manual smoke:
- choose `D:\Learning\Biology\projects\BioAgent_Desktop\data\base`
- verify dataset summary auto-fills `ab1` and `gb`
- verify analysis runs without manual path selection
- verify advanced mode still exposes separate AB1/GB controls

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.css src/i18n.ts tests/test_dataset_import.py
git commit -m "feat: add dataset import flow with advanced mode"
```

## Self-Review

- Spec coverage:
  - Default dataset import: covered by Tasks 2-3
  - Advanced mode retention: covered by Task 4
  - Localized errors and labels: covered by Tasks 2 and 4
  - Verification and smoke: covered by Task 5
- Placeholder scan:
  - Removed generic “handle errors” language and replaced it with concrete alerts, keys, and verification commands.
- Type consistency:
  - Uses `DatasetImportState`, `ab1Dir`, `gbDir`, and `showAdvancedImport` consistently across tasks.
