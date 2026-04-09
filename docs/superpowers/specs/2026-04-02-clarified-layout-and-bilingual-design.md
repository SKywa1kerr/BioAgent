# BioAgent Desktop Clarified Layout And Bilingual Design

## Goal

Refine the current UI redesign so the product hierarchy becomes immediately legible, while also adding application-level Chinese and English language switching.

This is a focused second-pass design adjustment on top of the existing workbench redesign. The new pass should correct the remaining issue that the overall layout still feels visually ambiguous.

## Confirmed Priorities

The user confirmed three priorities:

1. The overall layout hierarchy is still not clear enough.
2. The app should default to Chinese and support switching between Chinese and English.
3. The color system can become richer, as long as it improves clarity rather than adding noise.

## Chosen Layout Direction

### Recommended structure

Use a three-region workbench where the center is unquestionably the main stage:

- left: sample navigation rail
- center: analysis result stage
- right: Agent assistance panel

### Why this direction

This product is fundamentally an analysis tool. The most important thing on screen is the currently selected sample result and its evidence. The UI should make that obvious at first glance.

- The left side should help users move through the run and scan sample status.
- The center should hold the scientific interpretation and evidence.
- The right side should support the user with explanation and actions without becoming the visual center.

This prevents the product from reading like an admin dashboard on the left or a chat-first AI tool on the right.

## Information Hierarchy

### Top header

The top header should become a compact product bar with clearer semantic zones:

- left: product identity and current batch context
- center or center-left: primary tab navigation
- right: language toggle and small global controls

The header should feel branded and productized, but it must remain compact enough for a desktop workflow tool.

### Left rail

The left rail should be visually distinct and clearly subordinate to the center stage.

It should include:

- current run summary
- sample counts by status
- sample list
- current selected sample marker

The rail should feel like a navigation system, not like a second content page.

### Center stage

The center must become the brightest and calmest region in the layout.

It should present content in this order:

1. selected sample summary
2. verdict and key metrics
3. sequence/evidence view
4. chromatogram
5. mutation details and interpretation

The center should read as the place where conclusions are formed.

### Right Agent panel

The right panel should remain integrated, but quieter than the center stage.

It should show:

- current context state
- user and agent messages
- tool progress trace
- response metadata

Its function is support and interpretation, not visual dominance.

## Language Switching

### Product requirement

The app should support application-level language switching between:

- Chinese
- English

### Default behavior

- default language: Chinese
- alternate language: English

### Placement

The language switch should appear in the top-right of the app header as a lightweight toggle such as:

- `中文 | EN`

It should feel like a product preference, not a large call-to-action.

### Scope

The language system should cover all application chrome and core product copy, including:

- tab labels
- page headers
- toolbar buttons
- empty states
- status/help text
- Agent panel copy
- Settings copy
- History copy

Data-bound scientific values should not be translated. Sample IDs, file paths, mutation coordinates, and similar domain data remain unchanged.

### Technical direction

Use a lightweight app-level translation layer instead of a large external framework unless implementation complexity later proves it necessary.

Recommended structure:

- one top-level language state
- two dictionaries: `zh` and `en`
- language value persisted to local settings
- shared translation helper consumed by pages and components

## Color And Visual System Adjustment

### Design principle

Use richer color, but assign it by region and purpose. The goal is to clarify layout, not decorate every surface equally.

### Regional color roles

- app shell background: warm bone-white
- left navigation rail: deeper teal/green or dark botanical surface
- center analysis stage: bright warm-white main surface
- right Agent panel: pale cyan-green support surface
- header: refined light product bar with brand accents

### Semantic color roles

- pass / ok: clear green
- issue / wrong: berry-red or softened red
- review / uncertain: amber
- processing: blue-cyan

### Hierarchy rules

- strongest contrast belongs to the center result stage
- left rail uses stronger structural color, but not stronger emphasis than the center
- right panel uses a distinct but softer supporting tint
- strong color fills should mostly appear on badges, selected states, key actions, and summary indicators

## UX Outcome

The redesigned interface should produce the following immediate read:

1. where the current result is
2. where sample switching happens
3. where AI assistance lives
4. how to switch between Chinese and English

If a first-time user cannot answer those questions in a few seconds, the redesign is still too ambiguous.

## Implementation Scope

### In scope

- rebalance analysis page layout hierarchy
- add app-level Chinese and English switching
- revise color system to make left/center/right regions more distinct
- update shared UI copy to use translations
- align History, Settings, and Agent panel with the revised hierarchy

### Out of scope

- adding new workflow features
- changing scientific data structure
- adding more tabs or new information architecture
- full localization of domain data values

## Testing Expectations

### Functional

- switching language updates all app chrome text without reload errors
- default language is Chinese on first launch
- selected language persists after restart

### Visual

- center analysis region is visibly the primary stage
- left navigation rail is clearly navigational
- right Agent panel is visually integrated but secondary
- richer colors improve comprehension instead of increasing clutter

### Responsive

- desktop remains the primary layout target
- narrower widths may stack or compress regions, but hierarchy must still remain legible

## Acceptance Criteria

This design pass is successful when:

1. the layout hierarchy is clearly understandable at first glance
2. the center analysis stage is unmistakably dominant
3. the app defaults to Chinese and supports English switching
4. the language toggle is lightweight and easy to find
5. the color system is richer without becoming noisy
6. History, Settings, and Agent surfaces all feel part of the same product language
