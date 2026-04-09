# Results Page Interaction Polish Design

## Goal

Polish the results page into a more refined dossier-style workbench by improving four areas together:

- collapse chromatogram blocks by default while making the expanded view more readable
- fix sample summary row clipping and truncation behavior
- keep the AI composer fixed at the bottom of the right rail
- add a mixed-mode analysis progress bar at the top of the workbench

This work should also reduce the perceived lag when opening sample details and improve the overall visual quality of the page without changing the core analysis pipeline.

## Product Direction

The results surface should feel like an archive workbench rather than a plain utility panel:

- summary and progress first
- scan-friendly sample index second
- layered dossier details third
- AI rail always available but visually secondary

The page should remain desktop-first, readable on narrower widths, and consistent with the existing Electron app.

## Scope

### In scope

- top progress strip below the analysis toolbar
- dossier-style refinement of the results page visuals
- controlled truncation for sample header cells
- collapsed chromatogram section inside sample details
- deferred chromatogram rendering to reduce expansion lag
- sticky AI composer in the right panel
- groundwork for future true backend-driven progress updates

### Out of scope

- changing backend analysis rules or alignment behavior
- replacing the current chat workflow
- implementing final real percent progress from the backend in this pass
- redesigning unrelated pages such as history or settings

## Main UX Changes

### 1. Mixed-mode analysis progress strip

Add a persistent progress strip directly below the analysis toolbar and above the main results content.

Phase one behavior:

- show stage-based progress immediately when analysis starts
- stages:
  - preparing input
  - scanning AB1 files
  - aligning and calling mutations
  - aggregating results
  - completed
- expose an animated determinate-like bar driven by stage weights rather than true sample counts
- show short text status near the bar

Forward-compatible behavior:

- progress state model should accept:
  - current stage
  - percent
  - processed sample count
  - total sample count
- when backend progress events exist later, the UI should swap from stage weights to true progress without changing the component structure

### 2. Results page visual upgrade

Refine the current workbench into a dossier-style layout:

- stronger section framing and clearer spacing rhythm
- calmer surfaces with more hierarchy between shell, cards, and embedded viewers
- sample list cards should feel like indexed dossier entries
- expanded sections should read as layered evidence blocks

Visual direction:

- brighter analytical surface in the main pane
- more distinct headers and card groupings
- tighter typography for scan rows
- better use of muted accents for secondary metadata
- AI rail remains lighter and less dominant than the workbench

### 3. Sample summary row clipping fix

The sample summary header row currently risks awkward clipping when text is long.

Adjustments:

- keep a stable grid structure for desktop scanning
- give the sample ID and reason columns explicit min/max behavior
- truncate long content with ellipsis rather than letting the row collapse
- preserve quick readability of:
  - sample ID
  - status
  - reason
  - identity
  - coverage
  - mutation counts
- on narrower layouts, shift from rigid grid to stacked or two-column summary layout without cutting off interactive affordances

### 4. Chromatogram collapse and readability

Chromatogram sections should be collapsed by default inside each sample dossier.

Collapsed state:

- title row: `Chromatogram`
- summary metadata:
  - base count
  - average quality if available
  - mixed peak count if available
- chevron or equivalent open/close affordance

Expanded state:

- render the chromatogram canvas only when the section is opened
- give the chart more breathing room and a clearer framed container
- keep the expanded chart horizontally scrollable if needed without compressing traces

This improves both clarity and initial render cost.

## Performance Strategy

The main lag concern is opening sample details when multiple heavy subviews are present.

Mitigations in this pass:

- defer chromatogram canvas render until its own section is opened
- avoid eager heavy content where it is not immediately visible
- preserve a single open selected sample pattern where practical
- keep scroll containers isolated so layout recalculation is smaller

This pass targets perceived responsiveness rather than full virtualization.

## AI Rail Behavior

The right panel should use a three-part structure:

- fixed header
- independently scrollable message list
- sticky composer at the bottom

Behavioral goals:

- the input area always stays visible
- long chats do not push the composer out of view
- the panel feels available but subordinate to the results workbench

## Component Boundaries

Likely affected frontend areas:

- `src/App.tsx`
- `src/App.css`
- `src/i18n.ts`
- `src/components/ResultsWorkbench.tsx`
- `src/components/ResultsWorkbench.css`
- `src/components/SampleDetailsList.tsx`
- `src/components/AgentPanel.tsx`
- `src/components/AgentPanel.css`
- optionally a small new progress component if that keeps concerns cleaner

Preferred component split:

- `ResultsProgress` for the top progress strip
- `SampleDetailsList` owns summary-row clipping and chromatogram disclosure behavior
- `AgentPanel` owns sticky composer layout

## Data and State Shape

Add a small UI-facing progress model in the frontend such as:

```ts
type AnalysisStage =
  | "idle"
  | "preparing"
  | "scanning"
  | "aligning"
  | "aggregating"
  | "completed";

interface AnalysisProgressState {
  stage: AnalysisStage;
  percent: number | null;
  processedSamples: number | null;
  totalSamples: number | null;
  message?: string;
}
```

In this pass:

- the frontend sets stage transitions around the existing run-analysis lifecycle
- `percent` can be derived from stage weights
- later backend progress can populate the same object with real values

## Error Handling

- if analysis fails, progress strip should switch to a failed state with concise copy
- if sample metadata needed for chromatogram summary is missing, hide only the missing metadata item and keep disclosure functional
- sticky composer should degrade cleanly on narrow widths without overlapping controls

## Testing and Verification

Implementation should verify:

- build passes
- targeted existing tests still pass
- manual smoke confirms:
  - progress strip appears during analysis and clears or completes correctly
  - sample rows no longer visually break on long text
  - chromatogram is collapsed by default
  - expanding sample details feels smoother than before
  - AI composer remains visible while message history scrolls

## Recommended Approach

Use a focused frontend refactor rather than a cosmetic patch:

1. add a small progress state model and top progress strip
2. refine workbench styling and sample summary grid behavior
3. lazy-render chromatograms behind a disclosure control
4. restructure the AI rail scroll behavior so the composer remains fixed

This keeps the change bounded while addressing both appearance and interaction quality.
