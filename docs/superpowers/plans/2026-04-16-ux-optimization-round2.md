# UX Optimization Round 2 — Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement.

**Goal:** Enrich chat UX, stabilize backend, add sample batch operations + CSV export, panel tab navigation, and chromatogram zoom.

**Spec:** `docs/superpowers/specs/2026-04-16-ux-optimization-round1-design.md` (Round 2 section)

---

## Tasks

### Task 1: Chat improvements (timestamps, copy, clear, stable keys)
**Files:** Modify `src/components/ChatPanel.tsx`, `src/i18n.ts`, `src/styles.css`

### Task 2: MCP stderr logging
**Files:** Modify `electron/agent_harness.mjs`

### Task 3: max_tokens increase + adjustable in settings
**Files:** Modify `electron/agent_harness.mjs`, `src/lib/settingsStorage.ts`, `src/components/SettingsModal.tsx`, `src/i18n.ts`

### Task 4: Sample table — expand/collapse all + CSV export
**Files:** Modify `src/components/workbench/ResultsTable.tsx`, `src/components/workbench/ResultsWorkbench.tsx`, `src/components/workbench/ResultsWorkbench.css`, `src/i18n.ts`

### Task 5: Panel tab navigation
**Files:** Modify `src/App.tsx`, `src/hooks/useAgentHarness.ts`, `src/styles.css`, `src/i18n.ts`

### Task 6: Chromatogram zoom + mutation position highlight
**Files:** Modify `src/components/workbench/ChromatogramCanvas.tsx`, `src/components/workbench/ChromatogramCanvas.css`, `src/components/workbench/ResultsTable.tsx`

## Dependency Graph

Tasks 1-4, 6 are independent. Task 5 depends on nothing but touches App.tsx (run last or alone).
