# Dataset Import With Advanced Mode

## Goal

Reduce analysis setup friction for the common case where a dataset directory already contains both `ab1/` and `gb/` subdirectories, while preserving manual control for irregular directory layouts.

## User Experience

The analysis toolbar will default to a simple import flow:

- A primary `Import Dataset` action lets the user select a batch directory such as `base`, `batch1`, `pro`, or `promax`.
- After selection, the app auto-detects:
  - `ab1/` as the AB1 source directory
  - `gb/` as the reference directory
- The UI then shows a compact summary of the detected dataset:
  - dataset folder name
  - resolved AB1 path
  - resolved GB path

An `Advanced Mode` affordance remains available in the same area:

- When collapsed, only the dataset import flow is emphasized.
- When expanded, the user can manually override AB1 and GB paths with separate pickers.
- Manual picks replace the auto-detected paths for the current session.

## Detection Rules

Given a selected dataset directory:

1. Resolve `<dataset>/ab1`
2. Resolve `<dataset>/gb`
3. Consider the dataset valid only when at least one of those exists
4. Prefer the detected subdirectories over any previous session values

Error states:

- Missing `ab1/`: show a localized error that the dataset is missing an AB1 folder
- Missing `gb/`: show a localized error that the dataset is missing a GB folder
- Missing both: show a localized error that the selected directory is not a valid dataset

The user may still enter Advanced Mode and provide missing paths manually.

## UI Changes

Files likely involved:

- `src/App.tsx`
- `src/App.css`
- `src/i18n.ts`

Changes:

- Add a primary dataset import control in the analysis toolbar
- Add a compact dataset summary card or inline summary chip group
- Add an `Advanced Mode` toggle
- Keep existing separate AB1/GB controls, but visually demote them behind the toggle
- Localize all new labels, summaries, and error messages for Chinese and English

## Data Flow

- Add dataset directory state in the app shell
- On dataset selection:
  - call existing folder picker
  - derive `ab1Dir` and `genesDir` from child folders
  - update toolbar summary state
- On advanced manual override:
  - update only the overridden path
  - keep the dataset summary visible so the user understands current source context

## Non-Goals

- No recursive “smart guessing” beyond the explicit `ab1/` and `gb/` convention
- No changes to the analysis engine invocation shape
- No persistence changes beyond what the current app already stores

## Testing

Add focused tests for dataset detection behavior:

- valid dataset with both `ab1/` and `gb/`
- dataset missing `ab1/`
- dataset missing `gb/`
- invalid dataset with neither folder

Verification after implementation:

- `npm.cmd run build`
- relevant Python/TS tests still pass
- manual smoke in Electron using `data/base` style datasets
