# Real Analysis Progress And UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fake staged analysis progress with backend-driven live progress and continue polishing the results workbench presentation.

**Architecture:** Stream progress events from the Python sidecar through Electron IPC into the React app while keeping the final analysis result payload unchanged. Reuse the existing progress strip state in the frontend, then refine the workbench and panel styling around the now-live progress model.

**Tech Stack:** Electron IPC, Node child process streaming, Python sidecar, React, TypeScript, CSS

---

### Task 1: Add a failing regression test for backend progress emission

**Files:**
- Modify: `tests/test_main.py`
- Modify: `src-python/bioagent/main.py`
- Test: `python -m pytest tests/test_main.py -q`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run the test and verify it fails**
- [ ] **Step 3: Implement minimal Python-side progress reporting hooks**
- [ ] **Step 4: Run the test and verify it passes**

### Task 2: Stream progress events through Electron

**Files:**
- Modify: `electron/main.js`
- Modify: `electron/preload.js`
- Modify: `src/types/index.ts`
- Test: `npm.cmd run build`

- [ ] **Step 1: Add a missing progress subscription shape in the preload/types surface**
- [ ] **Step 2: Run build and verify it fails until the API is wired**
- [ ] **Step 3: Replace buffered analysis execution with streamed process handling and progress event forwarding**
- [ ] **Step 4: Run build and verify it passes**

### Task 3: Switch the React app from synthetic stages to real progress

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/i18n.ts`
- Test: `npm.cmd run build`

- [ ] **Step 1: Add frontend references to live progress fields that do not exist yet**
- [ ] **Step 2: Run build and verify it fails**
- [ ] **Step 3: Subscribe to backend progress, map real sample counts and stage messages into UI state, and remove timer-driven fake progress**
- [ ] **Step 4: Run build and verify it passes**

### Task 4: Continue results workbench visual polish

**Files:**
- Modify: `src/App.css`
- Modify: `src/components/ResultsWorkbench.css`
- Modify: `src/components/AgentPanel.css`
- Test: `npm.cmd run build`

- [ ] **Step 1: Tighten the progress strip and workbench styling around the live status model**
- [ ] **Step 2: Strengthen dossier-style cards and panel hierarchy without changing structure**
- [ ] **Step 3: Run build and verify it passes**

### Task 5: Final verification

**Files:**
- Modify only if fixes are required
- Test: `python -m pytest tests/test_main.py -q`
- Test: `npm.cmd run build`

- [ ] **Step 1: Re-run the targeted Python regression**
- [ ] **Step 2: Re-run the frontend build**
- [ ] **Step 3: Report remaining risks if any**
