export interface PairResultEntry {
  basename: string;
  ab1: string;
  gb: string;
}

export interface PairResult {
  pairs: PairResultEntry[];
  unpairedAb1: string[];
  unpairedGb: string[];
}

export function pairAb1Gb(
  ab1Paths: readonly string[],
  gbPaths: readonly string[],
): PairResult;
