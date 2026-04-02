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