# Primer Design Module — Scaffold Design

## Summary

Add a primer design module scaffold to BioAgent, inspired by Mutation_Maker. This creates the file structure, data types, and UI shell for future primer design functionality. No algorithms are implemented — just the skeleton that future work will fill in.

## Scope

- Python: new `primer_design.py` and `primer_models.py` modules with stub functions
- TypeScript: new `primer.ts` types, `PrimerDesigner.tsx` component with UI shell
- Integration: collapsible section in `App.tsx`, IPC stub in `electron/main.js`
- Out of scope: Primer3 binary, Lambda, actual algorithm logic

## Architecture

```
src-python/bioagent/
  primer_models.py    # Dataclasses: Primer, MutationTarget, PrimerResult
  primer_design.py    # Stub functions: design_sssm, design_msdm, design_pas

src/
  types/primer.ts     # TS mirrors of Python dataclasses
  components/
    PrimerDesigner.tsx # UI shell with workflow tabs + placeholder results
    PrimerDesigner.css # Styles matching existing app aesthetic

electron/
  main.js             # New 'design-primers' IPC handler (stub)
```

### Data Flow

```
PrimerDesigner.tsx
  --> IPC 'design-primers' (disabled, returns not-implemented error)
  --> primer_design.py (raises NotImplementedError)
  --> returns PrimerResult to frontend
```

## New Files

### 1. `src-python/bioagent/primer_models.py`

Dataclasses:
- `Primer`: sequence (str), tm (float), gc_content (float), position (int), direction ("forward" | "reverse"), name (str)
- `MutationTarget`: position (int), original_codon (str), target_codons (list[str]), strategy ("SSSM" | "MSDM" | "PAS")
- `PrimerResult`: primers (list[Primer]), targets (list[MutationTarget]), workflow (str), warnings (list[str])

### 2. `src-python/bioagent/primer_design.py`

Three stub functions:
- `design_sssm_primers(sequence: str, targets: list[MutationTarget]) -> PrimerResult`
- `design_msdm_primers(sequence: str, targets: list[MutationTarget]) -> PrimerResult`
- `design_pas_primers(sequence: str, targets: list[MutationTarget]) -> PrimerResult`

Each raises `NotImplementedError("Primer3 integration not yet implemented")`.

### 3. `src/types/primer.ts`

TypeScript interfaces mirroring the Python dataclasses above.

### 4. `src/components/PrimerDesigner.tsx`

UI shell:
- Workflow tabs: SSSM / MSDM / PAS (radio or tab group)
- Mutation target selector: checkboxes from current sample's mutation list
- "Design Primers" button: disabled, tooltip "Coming Soon — Primer3 integration pending"
- Results placeholder: empty table with columns (Name, Sequence, Tm, GC%, Position, Direction)
- Collapsible via a chevron header

### 5. `src/components/PrimerDesigner.css`

Consistent with existing app styles (monospace fonts, same color palette, similar border/shadow patterns).

## Integration Changes

### `src/App.tsx`

Add below the mutations table in `details-section`:
```tsx
<PrimerDesigner mutations={selectedSample.mutations} />
```

Wrapped in a collapsible section.

### `electron/main.js`

Add IPC handler:
```js
ipcMain.handle('design-primers', async () => {
  return JSON.stringify({ error: "Primer design not yet implemented" });
});
```

## Execution Plan

Since this is 100% new files with minimal integration touchpoints, it runs as a single worktree agent or sequentially. No parallelism needed.

1. Create `primer_models.py` and `primer_design.py`
2. Create `src/types/primer.ts`
3. Create `PrimerDesigner.tsx` and `PrimerDesigner.css`
4. Add IPC stub to `electron/main.js`
5. Integrate `PrimerDesigner` into `App.tsx`
6. Verify TypeScript compiles cleanly
