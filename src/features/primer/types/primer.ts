/** Primer design types matching Python dataclasses. */

export interface Primer {
  sequence: string;
  tm: number; // melting temperature in °C
  gc_content: number; // GC percentage
  position: number; // start position in reference sequence (0-based)
  direction: "forward" | "reverse";
  name: string; // e.g., "F1", "R2"
}

export interface MutationTarget {
  position: number; // codon position (0-based)
  original_codon: string;
  target_codons: string[]; // list of alternative codons for SSSM
  strategy: "SSSM" | "MSDM" | "PAS";
}

export interface PrimerResult {
  primers: Primer[];
  targets: MutationTarget[];
  workflow: string; // e.g., "SSSM", "MSDM", "PAS"
  warnings: string[];
}

export interface PrimerSkill {
  name: string;
  description: string;
  input_schema: Record<string, any>;
  tags: string[];
}

/** Primer alignment result from Python backend */
export interface PrimerAlignment {
  primer_sequence: string;
  start: number; // 0-based start position in reference
  end: number; // 0-based end position (exclusive)
  direction: "forward" | "reverse";
  mismatches: number[]; // positions of mismatches (0-based, relative to reference)
  match_count: number;
  alignment_score: number; // percentage match (0-100)
  tm: number; // melting temperature in Celsius
}

/** Input for primer alignment tool */
export interface PrimerAlignInput {
  name: string;
  sequence: string;
}

/** Extended alignment with display info */
export interface PrimerAlignmentDisplay extends PrimerAlignment {
  name: string; // display name (extracted from primer_sequence if format is "name:sequence")
  displaySequence: string; // sequence without name prefix
}