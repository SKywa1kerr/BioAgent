export interface MutationDiffSet {
  uniqueLeftPositions: Set<number>;
  uniqueRightPositions: Set<number>;
  sharedPositions: Set<number>;
}

export function diffMutations(
  left: ReadonlyArray<{ position?: number | null }>,
  right: ReadonlyArray<{ position?: number | null }>,
): MutationDiffSet;
