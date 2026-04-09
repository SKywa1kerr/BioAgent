# Results Rail, Theme, And AI Opt-In Design

## Goal

Close the remaining UX gaps on the results page by addressing four linked issues together:

- do not auto-expand the first sample detail after analysis completes
- keep the AI chat composer visible while browsing long result details
- improve the dark theme so it feels like the same dossier-style workbench rather than a generic blue-black dashboard
- make AI-assisted analysis an explicit user opt-in with user-provided API settings

The result should feel more stable in daily use, easier to promote to other users, and less dependent on a preconfigured local API environment.

## Product Direction

The page should continue to read as a dossier workbench:

- the center column is the evidence surface
- the right rail is a supporting analyst assistant
- dark mode should feel deliberate, muted, and lab-like
- AI should be available, but never assumed

This pass is about tightening behavior and defaults rather than adding new analysis capabilities.

## Scope

### In scope

- clear the default selected sample after each completed analysis run
- make the AI panel layout resilient when the main page becomes tall or heavily expanded
- retune dark theme colors across the results workbench and AI rail
- change AI-assisted analysis to explicit opt-in behavior
- hide or de-emphasize API fields when AI review is disabled
- validate that AI review cannot be used accidentally without user-supplied settings

### Out of scope

- changing mutation-calling logic or backend rules
- redesigning the history page
- replacing the current agent workflow
- adding provider presets beyond the existing compatible OpenAI-style settings shape

## Main UX Changes

### 1. Sample details stay collapsed by default

Current issue:

- after analysis, the first sample is selected and expanded automatically
- this makes the page feel busy and contributes to layout pressure on the right rail

Change:

- after each fresh analysis result is loaded, `selectedSampleId` should default to `null`
- sample details only open after an explicit user click
- if the user manually opens a sample, the current single-open behavior can remain

Expected result:

- the list reads as an index first
- first paint is calmer
- the workbench does not jump into a heavy detail state automatically

### 2. AI rail composer remains visible while browsing

Current issue:

- after repeated expansion and scrolling in the main results area, the AI composer can become effectively inaccessible
- the right rail layout is not sufficiently isolated from surrounding page height and scrolling behavior

Change:

- enforce a strict three-part panel layout:
  - fixed header
  - independently scrollable message list
  - composer pinned to the bottom of the rail
- use a sticky or absolute-bottom composer only within the rail's own scrolling context, not relative to the page
- ensure the rail container itself has a stable height contract and `min-height: 0` behavior where needed

Expected result:

- the input area always remains reachable
- message history scrolls inside the rail only
- main content expansion no longer pushes the composer out of view

### 3. Dark theme retune

Current issue:

- dark mode still has mixed visual signals:
  - some surfaces feel too blue
  - some cards still feel too bright
  - the right rail and results cards do not share one coherent palette

Change:

- shift dark mode toward a deep green-gray / slate dossier palette
- use one shared surface hierarchy:
  - shell background: deepest tone
  - panel surface: slightly raised
  - cards and disclosures: one more step lighter
  - interactive accents: restrained teal or brass, not bright cyan
- ensure charts, cards, tabs, and the agent rail use the same token family

Visual goals:

- less dashboard-like
- more laboratory archive / review workstation
- lower glare
- better distinction between background, panel, and focused card

### 4. AI review becomes explicit opt-in

Current issue:

- the current default favors hybrid analysis
- that works locally for a configured environment, but it is not appropriate for broader distribution

Change:

- set the default analysis decision mode to `rules`
- add a clear settings control for enabling AI review
- only show or emphasize `API Key`, `Base URL`, and `Model` settings when AI review is enabled
- when AI review is disabled:
  - no AI-assisted analysis requests should be made
  - the UI should present the app as fully usable in deterministic mode
- when AI review is enabled:
  - the user is responsible for providing compatible API credentials
  - missing required fields should surface clear validation before analysis begins

Expected result:

- the app works out of the box without AI
- AI becomes a user-owned enhancement path
- the product is easier to share without bundling a provider assumption

## Information Architecture

Settings should communicate a clear hierarchy:

- Analysis mode
  - Rules only
  - Rules + AI review
- AI provider settings
  - hidden, collapsed, or visually secondary until AI review is enabled

The results page should communicate:

- sample index by default
- details only on demand
- AI rail always available as a tool, not as the dominant surface

## State And Data Flow

Frontend state changes:

- default `selectedSampleId` to `null` on new analysis completion
- keep existing explicit sample selection behavior
- derive `useLLM` from the saved settings decision mode rather than a hidden default

Settings model expectations:

- `analysisDecisionMode` default becomes `rules`
- existing settings remain backward-compatible
- stored API config remains supported for users who opt in later

Validation behavior:

- if `analysisDecisionMode === "hybrid"` and required AI fields are missing, block AI execution with a user-facing message
- if `analysisDecisionMode === "rules"`, no API validation is needed

## Component Boundaries

Likely affected files:

- `src/App.tsx`
- `src/App.css`
- `src/components/AgentPanel.tsx`
- `src/components/AgentPanel.css`
- `src/components/SettingsPage.tsx`
- `src/components/SettingsPage.css`
- `src/components/ResultsWorkbench.css`
- optionally `src/i18n.ts` for clearer copy

Preferred responsibilities:

- `App` owns the default post-analysis selection behavior
- `AgentPanel` owns its own internal layout and sticky composer contract
- `SettingsPage` owns AI opt-in controls and provider-field presentation
- shared CSS tokens or root theme variables should drive the dark-theme retune

## Error Handling

- invalid or missing AI configuration should produce a clear settings-oriented message, not a generic failure
- AI-disabled mode should never warn about missing API credentials
- if legacy settings still contain API values but mode is `rules`, they should remain stored without being used

## Testing Strategy

Manual verification for this pass is important:

- run a fresh analysis and confirm no sample is open by default
- open and close multiple sample details while scrolling and confirm the AI composer remains visible
- verify dark theme across:
  - results summary cards
  - sample list cards
  - expanded detail sections
  - agent rail
  - settings page
- verify rules-only mode works without any AI credentials
- verify AI mode requires user-supplied API configuration before use

Automated checks where practical:

- settings default regression if there is a testable settings helper
- build verification after layout and i18n changes

## Risks

- sticky composer behavior can regress if the parent layout still lacks a stable height contract
- changing the default analysis mode can surprise existing local users, so the settings copy should make the mode explicit
- dark theme retuning can create contrast regressions if chart and card tokens drift apart

## Success Criteria

- after analysis, the result list stays collapsed until the user opens a sample
- the AI input remains visible and usable while browsing long results
- dark theme feels coherent and no longer mixes white or bright-blue surfaces into the workbench
- a new user can run deterministic analysis without configuring any API
- AI review is still available, but only after explicit user opt-in and self-supplied credentials
