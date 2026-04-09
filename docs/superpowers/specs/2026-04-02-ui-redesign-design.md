# BioAgent Desktop UI Redesign Design

## Goal

Redesign the entire BioAgent Desktop interface into a refined biotech product UI that is suitable for promotion and long-term productization.

This redesign is not a cosmetic patch. It should solve both of the current problems:

- The UI looks rough and visually inconsistent.
- The UI feels messy because hierarchy, spacing, grouping, and page structure are weak.

The target result is a product that still opens directly into a usable workbench, but looks intentional, polished, and marketable.

## Product Direction

### Chosen Direction

- Product tone: biotech brand feel
- Entry pattern: workbench with presentation quality
- Color direction: teal-green plus bone-white
- Layout strategy: hybrid layout

### Interpretation

The app should not behave like a marketing landing page, and it should not look like a generic admin dashboard. It should feel like a serious biology product that happens to be pleasant and elegant.

Users should open the app and immediately see a professional analysis workspace, but with clear brand identity, strong visual rhythm, and better emotional quality than a traditional lab tool.

## Scope

This redesign covers the entire application UI system, not only the analysis page.

### In Scope

- Global visual system
- App shell and top workbench header
- Analysis page layout and styling
- Agent Panel styling and message presentation
- History page redesign
- Settings page redesign
- Shared components such as buttons, cards, status badges, form inputs, and page sections

### Out of Scope

- New product features unrelated to UI redesign
- New navigation structure beyond what is necessary for the redesign
- Marketing-only homepage before the workbench
- Deep information architecture changes that alter the core product flow

## Core UX Principles

### 1. Direct To Work

The app still opens directly into the analysis workbench. This is a desktop tool first.

### 2. Product Quality Over Utility Styling

The interface should feel designed as a coherent product, not assembled from default control patterns.

### 3. Calm Scientific Confidence

The UI should communicate precision, trust, and clarity. It should not become playful, noisy, or startup-gimmicky.

### 4. Stronger Hierarchy

The current UI feels messy because the hierarchy is flat. The redesign should clearly separate:

- primary workspace
- secondary context
- supportive metadata
- system actions

### 5. Promotion-Ready Screens

Any major page should be good enough to appear in screenshots, demos, and product promotion without needing redesign later.

## Visual System

### Color System

The app should adopt a teal-green and bone-white system.

#### Base palette roles

- App background: warm bone-white, not plain grey
- Main surface: soft cool white
- Elevated surface: slightly brighter white with subtle contrast
- Primary brand color: deep teal-green
- Accent brand color: lighter sea-green for highlights and selected states
- Secondary ink: muted slate for supporting text

#### Status colors

Status colors should become more controlled and product-like.

- `ok`: restrained teal-green
- `wrong`: softened berry-red
- `uncertain`: amber-gold
- `processing`: cool cyan-blue

These should appear as part of a unified system rather than harsh generic red/yellow/green blocks.

### Typography

Use a two-layer typography strategy.

- Brand and major section headings: more expressive and product-identifying
- Data-heavy UI text: clean modern sans-serif optimized for legibility

If introducing a new font is impractical, the redesign must still improve typography through:

- clearer title sizes
- more deliberate weight changes
- spacing between heading, body, and metadata
- better line height and label density

### Component Language

The UI should standardize around a single component language.

#### Cards

- cleaner surfaces
- subtle borders
- modest radii
- minimal shadow weight
- layered spacing instead of heavy visual chrome

#### Buttons

- primary actions use brand emphasis
- secondary actions are quieter and flatter
- destructive or warning actions are rare and visually disciplined

#### Badges and status chips

- smaller
- tighter
- less visually loud
- used for signal, not decoration

#### Tables and lists

- less admin-dashboard styling
- more product-record styling
- stronger row rhythm and spacing

## Global Shell

### Workbench Header

The top of the app should become a real workbench header rather than a plain title strip.

It should include:

- product identity on the left
- primary page navigation in the middle or center-left
- concise context or quick actions on the right

The header should feel branded and intentional, but still compact enough for a desktop productivity tool.

### Shared Layout Behavior

The application should feel like a single environment.

That means:

- consistent page gutters
- consistent section spacing
- consistent elevation rules
- consistent header and content rhythm across tabs

## Page Designs

### Analysis Page

The analysis page becomes the main product stage.

#### Layout

Use a hybrid layout:

- left: sample navigation area
- center: primary continuous analysis workspace
- right: Agent Panel

The center should feel like the main scientific canvas. The side areas should be more card-based and supportive.

#### Left Side: Sample Navigation

This should become more than a plain sample list.

It should include:

- analysis summary
- source context
- sample list with clearer grouping and status visibility

This area should feel like a navigation rail for the current run, not a generic sidebar.

#### Center: Main Analysis Workspace

This remains the main content area for:

- sequence view
- chromatogram
- mutation detail
- evidence-level interpretation

The redesign should reduce clutter while making the content feel more premium and legible.

#### Right: Agent Panel

The Agent Panel should feel like an integrated intelligence surface, not a bolted-on chat widget.

It should visually harmonize with the workbench while staying quieter than the central analysis area.

The panel should clearly show:

- current context state
- user and assistant messages
- plan messages
- tool execution progress
- final response metadata

### History Page

The History page should become a results archive, not a plain table page.

#### Desired character

- calm archive feel
- strong time-based scanability
- better emphasis on pass rate and run quality

The page may still use a table, but it should be styled more like a product record view than an internal admin log.

### Settings Page

The Settings page should become a product control panel rather than a raw form sheet.

#### Structure

Break settings into clear modules such as:

- model and API configuration
- analysis defaults
- product behavior or advanced controls

Every module should have:

- a title
- short explanatory copy
- inputs with more elegant form styling

## Agent Panel Design

The Agent Panel deserves special handling because it is both a product feature and a visual risk area.

### Goals

- keep the panel useful and readable
- avoid making it look like a generic support chatbot
- represent plan and tool execution as a trustworthy system trace

### Structure

- top status area
- scrollable message stream
- refined message styles per message type
- bottom composer with stronger product fit

### Message Types

- `user`: clear but visually contained
- `agent`: calm assistant response card
- `plan`: subtle structured planning card
- `tool_status`: compact progress artifact

The panel should look like a scientific copilot, not a social chat app.

## Information Hierarchy Fixes

The redesign should solve the existing “乱” feeling by restructuring emphasis.

### Current problems to eliminate

- too many elements compete at once
- weak separation between primary and secondary information
- controls and results often share the same visual priority
- spacing and alignment do not create a stable reading path

### Intended improvements

- stronger block-level grouping
- more whitespace between conceptual regions
- more restrained color usage
- fewer competing strong borders and heavy fills
- more obvious visual path from navigation to evidence to assistant guidance

## Implementation Strategy

This redesign should be implemented in four steps.

### Step 1: Global Design System

Establish:

- color variables
- spacing scale
- typography hierarchy
- shared elevation and radius rules
- shared button, card, badge, and input styling

### Step 2: Analysis Page And Agent Panel

This is the first major visible phase and should deliver the strongest product transformation.

### Step 3: History Page

Redesign the archive and result-review experience to match the new system.

### Step 4: Settings Page

Bring configuration surfaces into the same polished product language.

## Why This Sequence

The analysis page is the main entry and main promotional surface. Once it looks product-grade, the app already feels transformed. History and Settings can then be aligned into the same system.

## Acceptance Criteria

The redesign is successful when all of the following are true:

1. The app no longer looks like a generic internal tool.
2. The analysis page feels like a polished biotech workbench.
3. The Agent Panel visually belongs to the product.
4. The History page feels like a result archive rather than a plain admin table.
5. The Settings page feels like a refined control surface.
6. Colors, spacing, typography, and component styling are clearly systematized.
7. Screens from the app are suitable for product promotion without another redesign pass.

## Risks To Avoid

- overdecorating the scientific workflow
- making the UI look consumer-generic instead of biotech-specific
- using too many bright colors
- relying on heavy shadows or dashboard tropes
- weakening information density to the point that the tool becomes slower to use
