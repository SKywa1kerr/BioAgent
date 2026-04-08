# BioAgent Command Workbench And Assistant Redesign

## Goal

Upgrade BioAgent Desktop from a results page with a side `Agent Panel` into a command-first scientific workbench where users can issue Chinese instructions to drive analysis workflows directly.

The new product should feel like a polished desktop biology tool, not a chat wrapper. The first screen should emphasize:

- analysis results
- a prominent Chinese command entry point
- explicit execution feedback

The redesign should preserve the current Python analysis core, Electron shell, and existing result components where practical.

## Product Decision Summary

- Primary product direction: `Command-First Scientific Workbench`
- Input scope: application actions plus limited file-oriented actions
- Permission mode: low-risk actions auto-run, higher-risk actions require confirmation
- Entry style: natural-language first, with visible action plan preview
- Main page priority: `results + command entry`
- Existing `Agent Panel`: no longer primary on the analysis page
- Existing analysis engine: preserved
- Internal orchestration model: lightweight project-local `Action Registry`, not a general MCP platform
- Design goal: professional, calm, large-scale scientific UI with minimal AI surface styling

## Why This Direction

The current app already has a controlled agent loop, but it is shaped around a side chat panel with only a small fixed tool surface. That is enough for question answering, but not enough for the intended workflow of:

- importing a dataset
- setting paths and plasmid
- starting analysis from Chinese instructions
- filtering to wrong samples
- exporting the current report

The redesign moves the main interaction from "chat beside the workbench" to "commanding the workbench directly".

## Core Product Shape

The app should become a three-surface desktop product:

1. `Analysis` page
2. `Assistant` page
3. supporting pages such as `History` and `Settings`

### Analysis Page

The `Analysis` page becomes the primary workflow surface.

It contains:

- a branded but compact page header
- a large `Command Workbench` at the top
- a concise execution timeline and action preview area
- a large `Results Workbench` below

It does not contain a permanent right-side chat rail.

### Assistant Page

The current `Agent Panel` moves into a dedicated `Assistant` page or equivalent standalone surface.

Its role becomes:

- deep explanation
- follow-up questions
- agent trace review
- troubleshooting and interpretation

This keeps the analysis page focused and prevents the product from feeling like a chat-first AI shell.

## Analysis Page Information Architecture

### 1. Compact Product Header

The top header should remain present, but it should be tighter and more product-like than decorative.

It should include:

- product identity
- current tab navigation
- language and theme controls
- compact batch status indicators

The header should not dominate the page visually.

### 2. Command Workbench

This becomes the second main visual focus after results.

It should include:

- a large Chinese command input
- a short helper line with example prompts
- 3 to 5 high-frequency quick actions
- current dataset and run context summary
- current plasmid and important settings summary
- a visible action-plan preview when a command is parsed
- confirmation controls for higher-risk actions

Example prompts:

- `分析这个数据集`
- `用 pet15b 重跑`
- `只看 wrong 样本`
- `导出当前报告`

The command input should feel like a serious desktop control surface, not a chat bubble.

### 3. Execution Feedback Strip

Directly below the command area, the page should show a compact execution strip or timeline.

It should render:

- parsed intent summary
- planned actions
- current action state
- confirmation needed state
- success or failure summaries

This is the visible replacement for burying everything inside chat history.

### 4. Results Workbench

This remains the largest surface on the page.

It should combine the current BioAgent Max-inspired direction into a more coherent desktop workbench:

- batch summary cards
- sample table with strong filtering
- selected sample detail
- alignment and mutation evidence
- chromatogram view
- decision explanation

The result area should remain the first visual priority.

## Assistant Page Design

The `Assistant` page should reuse the existing `Agent Panel` logic, but its presentation should be upgraded from a narrow side rail into a full-page assistant workspace.

It should support:

- free-form questions
- explanation of why a sample is `wrong`
- inspection of recent execution traces
- help with troubleshooting
- follow-up commands when needed

This page is important, but secondary to the analysis workbench.

## Visual Direction

The visual direction should be `Scientific Workbench`, not `AI Copilot`.

### Principles

- calm and high-confidence
- large-scale layout with clear hierarchy
- restrained chrome
- minimal ornamental gradients
- no purple AI branding cues
- no generic rounded chatbot bubbles dominating the page

### Color

Use a restrained lab-software palette:

- warm off-white or mineral light background
- deep ink or graphite for primary text
- muted steel, slate, and sand neutrals
- limited accent colors for status
- red only for issue states
- green used carefully for pass states

### Typography

Avoid default startup-SaaS styling.

Use:

- a more expressive title face for headers or section labels
- a highly readable body face for operational text
- stronger typographic contrast between command surfaces and data surfaces

### Motion

Keep motion meaningful and sparse:

- command-plan reveal
- action state progression
- results refresh transition

Avoid decorative AI-style micro-animation.

## Interaction Model

### Primary Command Flow

1. User enters a Chinese command in the `Command Workbench`.
2. The app parses the request into a structured action plan.
3. The UI shows what it believes the user wants to do.
4. Low-risk actions execute immediately.
5. Higher-risk actions wait for confirmation.
6. Execution results update the results workbench directly.
7. A concise trace remains visible in the execution strip.
8. If the user wants an explanation or follow-up, they continue in the `Assistant` page.

### Low-Risk Actions

Examples:

- filter results
- select a sample
- switch result view
- inspect current history summaries

These may execute immediately.

### Higher-Risk Actions

Examples:

- run analysis
- export report
- open export folder
- change dataset paths

These should produce an explicit confirmation step.

## Action Registry

Instead of introducing a full MCP runtime, the app should add a lightweight local `Action Registry`.

Each action definition should include:

- `id`
- `label`
- `aliases`
- `description`
- `parameters`
- `risk`
- `needsConfirmation`
- `executor`

This registry is the contract between natural-language intent parsing and actual execution.

### First Action Set

- `import_dataset`
- `set_ab1_dir`
- `set_genes_dir`
- `set_plasmid`
- `run_analysis`
- `filter_results`
- `open_sample`
- `export_report`
- `open_export_folder`
- `query_history`
- `get_sample_detail`

### Why This Shape

This preserves the safety and inspectability benefits of typed tools without importing the complexity of a general plugin ecosystem.

## Command Interpreter

The app should add a dedicated `Command Interpreter` layer.

It translates Chinese instructions into action plans.

### Parsing Strategy

1. rule-based parsing for high-frequency commands
2. deterministic normalization of common entities such as plasmid names and status labels
3. LLM fallback only for ambiguous or compound requests

This prevents the product from becoming fragile or overly dependent on free-form model behavior for standard workflows.

### Output Contract

The interpreter should produce a structured plan with:

- recognized intent
- extracted parameters
- proposed ordered actions
- confidence or parsing certainty
- whether confirmation is required

## Relationship To Existing Agent Logic

The current controlled agent loop remains useful, but its role changes.

### Keep

- prompt building
- typed tool metadata pattern
- stop reasons
- visible error handling
- explanation workflows

### Move Away From

- analysis-page dependence on a side chat panel
- treating natural-language execution as just another chat turn

The current agent logic should support the `Assistant` page and fallback interpretation paths, not define the primary analysis-page interaction model.

## Architecture

The redesign should preserve the current stack:

- React + TypeScript frontend
- Electron shell and IPC
- Python analysis sidecar

### Frontend Responsibilities

The frontend should own:

- `Command Workbench`
- action plan rendering
- confirmation UI
- execution timeline
- analysis-page state transitions
- `Assistant` page presentation

### Electron Responsibilities

Electron should remain the execution boundary for local desktop actions, including:

- folder selection
- running analysis
- exporting reports
- opening folders when introduced

### Python Responsibilities

Python should continue to own:

- analysis logic
- existing explanation-oriented agent behavior
- optional intent fallback processing if needed

The Python side should not become the only place where workflow orchestration is hidden.

## Proposed File-Level Impact

### Frontend

Modify or replace:

- `src/App.tsx`
- `src/App.css`
- `src/components/TabLayout.tsx`
- `src/components/TabLayout.css`

Add:

- `src/components/CommandWorkbench.tsx`
- `src/components/CommandWorkbench.css`
- `src/components/ActionPlanCard.tsx`
- `src/components/ActionPlanCard.css`
- `src/components/ExecutionTimeline.tsx`
- `src/components/ExecutionTimeline.css`
- `src/components/AssistantPage.tsx`
- `src/components/AssistantPage.css`
- `src/utils/commandInterpreter.ts`
- `src/utils/actionRegistry.ts`

Potentially adapt:

- existing result workbench components
- `src/i18n.ts`
- `src/types/index.ts`

### Electron

Modify:

- `electron/main.js`
- `electron/preload.js`
- `src/electron.d.ts`

Potential additions:

- IPC for opening exported folders
- IPC for command execution status where needed

### Python

Keep and adapt:

- `src-python/bioagent/agent_chat.py`
- `src-python/bioagent/agent_tools.py`
- `src-python/bioagent/main.py`

Possible additions:

- `src-python/bioagent/command_intent.py`

The first implementation can keep command interpretation in the frontend if that is faster and easier to verify.

## Migration Plan

### Phase 1

- move `Agent Panel` off the analysis page
- introduce `Assistant` page
- restructure analysis page layout

### Phase 2

- add `Command Workbench`
- add action-plan preview and execution strip
- restyle the page into the new scientific workbench direction

### Phase 3

- implement `Action Registry`
- map high-frequency Chinese commands to actions
- add confirmation flow

### Phase 4

- connect action execution to current analysis flow
- sync command results into the workbench and assistant history
- polish mobile-width and smaller-window behavior

### Phase 5

- expand command coverage
- improve assistant explanation surfaces
- add more robust tests

## Testing

Testing should cover both behavior and interaction.

### Frontend

- command parsing for common Chinese instructions
- action-plan rendering
- confirmation gating
- execution timeline state transitions
- migration of assistant functionality into the new page

### Electron

- command-triggered run analysis flow
- export flow
- folder open flow

### Python

- existing agent response parsing
- existing analysis behavior
- fallback interpretation if Python participates in command parsing

### Integration

- `分析这个数据集`
- `用 pet15b 重跑`
- `只看 wrong 样本`
- `导出当前报告`
- mixed request such as `分析这个数据集，然后只看 wrong 样本`

## Non-Goals

- general bash execution
- full MCP server lifecycle
- plugin marketplace
- unrestricted system automation
- making the analysis page a chat-first product

## Acceptance Criteria

The redesign is successful when all of the following are true:

1. The `Analysis` page no longer depends on a permanent right-side chat rail.
2. The main page prominently supports Chinese command input.
3. Users can trigger analysis workflows from Chinese instructions.
4. Higher-risk actions require confirmation.
5. The results workbench remains the dominant visual surface.
6. The UI feels like a professional scientific desktop tool rather than an AI chat shell.
7. The existing Python analysis engine remains intact.
8. The former agent capability still exists through a dedicated `Assistant` page.
