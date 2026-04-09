# UI Redesign And Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First stabilize the Windows/Electron development and runtime path, then redesign the full BioAgent Desktop UI into a refined biotech product workbench.

**Architecture:** Treat this as two linked phases. Phase 1 removes launch/runtime friction and closes current integration gaps so the app can be verified reliably on Windows. Phase 2 introduces a new global design system and rolls it through Analysis, Agent Panel, History, and Settings in that order.

**Tech Stack:** Electron 33, React 18, TypeScript 5, Vite 5, Python 3.10+, pytest

---

## File Structure

### Files likely to modify in Phase 1

- `package.json` - fix Windows-incompatible dev script
- `electron/main.js` - only if dev boot or smoke-test issues require main-process fixes
- `src/components/AgentPanel.tsx` - final runtime/payload cleanup if smoke testing exposes another gap
- `src/App.tsx` - only if smoke testing shows parent-child analysis state drift
- `src/App.css` - only if smoke testing shows layout clipping or unusable responsive behavior

### Files likely to modify in Phase 2

- `src/styles.css` or `src/App.css` - establish global design tokens if no shared token file exists
- `src/App.tsx`
- `src/App.css`
- `src/components/AgentPanel.tsx`
- `src/components/AgentPanel.css`
- `src/components/ChatMessage.tsx`
- `src/components/ChatMessage.css`
- `src/components/SampleList.tsx`
- `src/components/SampleList.css`
- `src/components/HistoryPage.tsx`
- `src/components/HistoryPage.css`
- `src/components/SettingsPage.tsx`
- `src/components/SettingsPage.css`
- `src/components/TabLayout.tsx`
- `src/components/TabLayout.css`

---

### Task 1: Fix Windows Dev Launch Path

**Files:**
- Modify: `package.json`
- Test: `npm.cmd run electron:dev`

- [ ] **Step 1: Replace the Unix-only sleep-based dev command**

Update the `electron:dev` script in `package.json` so it works on Windows. Replace the existing `sleep 2 && electron .` pattern with a Windows-safe command, for example:

```json
"electron:dev": "concurrently -k \"vite\" \"powershell -NoProfile -Command Start-Sleep -Seconds 2; electron .\""
```

- [ ] **Step 2: Run the dev command and verify the app launches instead of failing immediately**

Run: `npm.cmd run electron:dev`
Expected: no `'sleep' is not recognized` error, Vite starts, and Electron opens the app window.

- [ ] **Step 3: Commit the script fix**

```bash
git add package.json
git commit -m "fix: make electron dev script work on windows"
```

---

### Task 2: Re-run Focused Smoke Test And Capture Remaining Stability Issues

**Files:**
- Modify only if required by findings from the smoke test
- Test: `npm.cmd run electron:dev`

- [ ] **Step 1: Start the desktop app in dev mode**

Run: `npm.cmd run electron:dev`
Expected: Electron window titled `BioAgent` stays open.

- [ ] **Step 2: Verify the current critical paths manually**

Manual checks:
1. Analysis tab opens without immediate crash
2. Right-side Agent Panel is visible
3. Typing into Agent Panel is possible
4. With no API key configured, Agent Panel returns a safe failure reply instead of crashing
5. Analysis page layout does not clip or overlap at a normal desktop size

- [ ] **Step 3: If a runtime issue appears, fix only that issue and re-run the same smoke path**

Keep the fix minimal and localized. Re-run `npm.cmd run electron:dev` and the same manual checks.

- [ ] **Step 4: Commit any stability fixes**

```bash
git add -u
git commit -m "fix: stabilize desktop agent panel smoke path"
```

---

### Task 3: Establish A Global Design System

**Files:**
- Modify: `src/App.css` or shared stylesheet used by the app shell
- Modify: any shared layout/component CSS file needed for token reuse
- Test: `npm.cmd run build`

- [ ] **Step 1: Introduce global design tokens for the new visual system**

Create CSS variables for:
- bone-white backgrounds
- white surface layers
- deep teal primary
- sea-green accent
- muted slate text
- semantic status colors
- spacing scale
- radius scale
- shadow/elevation rules

- [ ] **Step 2: Apply the tokens to the global app shell and common controls**

Update:
- app background
- top header surface
- button colors
- border tones
- text hierarchy colors

- [ ] **Step 3: Build to confirm the new token layer compiles cleanly**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 4: Commit the design-system foundation**

```bash
git add src/App.css src/styles.css src/components/TabLayout.css
git commit -m "feat: add biotech design system foundations"
```

---

### Task 4: Redesign Analysis Page And Agent Panel

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.css`
- Modify: `src/components/AgentPanel.tsx`
- Modify: `src/components/AgentPanel.css`
- Modify: `src/components/ChatMessage.tsx`
- Modify: `src/components/ChatMessage.css`
- Modify: `src/components/SampleList.tsx`
- Modify: `src/components/SampleList.css`
- Test: `npm.cmd run build`

- [ ] **Step 1: Rebuild the analysis page as a branded workbench**

Restructure the visual hierarchy so it reads as:
- left sample navigation rail
- central scientific workspace
- right intelligent assistant panel

- [ ] **Step 2: Turn the left sample area into a richer navigation surface**

Add stronger grouping for:
- run summary
- source context
- sample status scanning

- [ ] **Step 3: Refine the central analysis area**

Improve:
- section spacing
- heading hierarchy
- evidence readability
- chromatogram/mutation framing

- [ ] **Step 4: Redesign Agent Panel styling to feel product-native**

Improve:
- header hierarchy
- message card styles
- plan/tool-status trace styling
- composer visual quality

- [ ] **Step 5: Build and visually verify the analysis workbench**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 6: Commit the analysis and Agent Panel redesign**

```bash
git add src/App.tsx src/App.css src/components/AgentPanel.tsx src/components/AgentPanel.css src/components/ChatMessage.tsx src/components/ChatMessage.css src/components/SampleList.tsx src/components/SampleList.css
git commit -m "feat: redesign analysis workspace and agent panel"
```

---

### Task 5: Redesign History Page

**Files:**
- Modify: `src/components/HistoryPage.tsx`
- Modify: `src/components/HistoryPage.css`
- Test: `npm.cmd run build`

- [ ] **Step 1: Reframe the page as a results archive**

Introduce stronger top-level structure and summary framing before the list/table.

- [ ] **Step 2: Restyle records for product-quality scanning**

Improve:
- time readability
- pass-rate emphasis
- row rhythm
- empty state quality

- [ ] **Step 3: Build to confirm the page compiles cleanly**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 4: Commit the history redesign**

```bash
git add src/components/HistoryPage.tsx src/components/HistoryPage.css
git commit -m "feat: redesign history archive page"
```

---

### Task 6: Redesign Settings Page

**Files:**
- Modify: `src/components/SettingsPage.tsx`
- Modify: `src/components/SettingsPage.css`
- Test: `npm.cmd run build`

- [ ] **Step 1: Reorganize settings into product-style modules**

Use grouped sections for:
- model/API
- analysis defaults
- advanced behavior if needed

- [ ] **Step 2: Improve form styling and explanatory copy**

Each section should clearly explain its purpose and look like a polished control surface, not a raw form.

- [ ] **Step 3: Build to confirm the settings page compiles cleanly**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 4: Commit the settings redesign**

```bash
git add src/components/SettingsPage.tsx src/components/SettingsPage.css
git commit -m "feat: redesign settings control panel"
```

---

### Task 7: Final Verification And Smoke Test

**Files:**
- No required new files
- Test: focused pytest, `npm.cmd run build`, `npm.cmd run electron:dev`

- [ ] **Step 1: Run focused Python agent tests**

Run: `cd src-python && python -m pytest ../tests/test_agent_tools.py ../tests/test_agent_chat.py -q`
Expected: PASS

- [ ] **Step 2: Run the frontend build**

Run: `npm.cmd run build`
Expected: PASS

- [ ] **Step 3: Run the desktop dev app and perform a final manual smoke test**

Run: `npm.cmd run electron:dev`
Expected manual checks:
1. App launches on Windows without script errors
2. Analysis page opens with the redesigned workbench
3. Agent Panel is visible and usable
4. History page renders with the new archive design
5. Settings page renders with the new control-panel design

- [ ] **Step 4: Commit final polish if needed**

```bash
git add -u
git commit -m "fix: final polish for ui redesign and stability"
```

---

## Self-Review

### Spec coverage

- Stability first: covered in Task 1 and Task 2
- Global design system: covered in Task 3
- Analysis and Agent Panel redesign: covered in Task 4
- History redesign: covered in Task 5
- Settings redesign: covered in Task 6
- Final verification: covered in Task 7

### Placeholder scan

- No TBD/TODO placeholders remain
- Each task has concrete files and a verification command

### Type consistency

- Plan keeps current app structure intact while layering the redesign on top of the verified Agent Panel implementation
- Parent-child analysis context sync remains part of the stable path during redesign
