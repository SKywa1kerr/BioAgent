# Results Workbench Inspired By BioAgent Max

## Goal

Replace the current single-sample-centered analysis presentation with a results-first desktop workbench modeled on the BioAgent Max results page, while preserving a persistent AI chat rail on the right.

## Product Direction

The desktop app should present analysis output as a batch review workflow:

- summary first
- statistics second
- table-based scan third
- expandable sample detail last

The right rail remains dedicated to AI chat, but its role shifts from a co-equal visual panel to a context-aware assistant for the currently loaded analysis or selected sample.

## Main Layout

### Center Results Workbench

The center area becomes the dominant interface and is structured into four layers:

1. Summary cards
   - total sample count
   - `OK`
   - `Wrong`
   - `Uncertain`
   - `未测通` / not-covered state
   - average identity
   - average coverage

2. Statistics section
   - status distribution
   - identity / coverage overview
   - abnormal sample count

3. Sample table
   - one row per sample
   - sortable and scannable
   - columns:
     - sample ID
     - clone
     - status
     - reason
     - identity
     - coverage
     - mutation count

4. Expandable sample result items
   - one expandable section per sample
   - collapsed header shows:
     - sample ID
     - status
     - reason
   - expanded body shows:
     - summary metrics
     - mutation table
     - sequence alignment
     - chromatogram

### Right AI Rail

The right rail remains persistent but quieter than the center workbench:

- shows current batch context by default
- updates to selected sample context when the user clicks a table row or expands a sample
- supports quick prompts such as:
  - summarize abnormal samples
  - explain why a sample is untested
  - list review candidates
  - generate a report summary

## Status Model Change

The current `未测通` state must no longer be visually treated as ordinary `ok`.

It should become its own explicit display category in the UI:

- still sourced from backend rule output or mapped from current reason
- separated in summary cards
- separated in table filters
- visually distinct from both `OK` and `Uncertain`

This change is necessary because “not covered / untested” communicates a different analytical meaning from “passed”.

## Interaction Model

- Import and run analysis remain at the top of the analysis page
- After analysis completes, focus shifts immediately to the results workbench
- Clicking a table row:
  - highlights the row
  - scrolls the corresponding expandable detail into view or sets it active
  - updates AI chat context
- Expanding a sample detail:
  - does not hide the table
  - keeps the batch context intact

## Visual Direction

This design should follow BioAgent Max structurally, but stay consistent with the desktop app visual language:

- center content uses calm, bright workbench surfaces
- summary cards feel compact and analytical rather than decorative
- table is the main scan surface
- expanders feel like layered result dossiers
- right AI rail remains visually subordinate

## Non-Goals

- No attempt to fully clone Streamlit page chrome
- No change to backend alignment logic in this phase
- No new remote data dependencies

## Files Likely Affected

- `src/App.tsx`
- `src/App.css`
- `src/i18n.ts`
- `src/components/AgentPanel.tsx`
- `src/components/AgentPanel.css`
- potentially new result-workbench components under `src/components/`

## Verification

- `npm.cmd run build`
- existing Python tests still pass
- manual smoke using `data/base` dataset
- confirm:
  - summary cards render
  - sample table renders
  - expanders open correctly
  - AI chat remains usable on the right
