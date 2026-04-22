import { useMemo } from "react";
import { useAlignment } from "./useAlignment";
import { translateCodon, groupErrorsIntoRegions } from "../../../utils/sequence";
import type { ErrorRegion } from "../../../utils/sequence";

const BASE_WIDTH = 7.2;
const CODON_WIDTH = BASE_WIDTH * 3; // 3 bases per codon

export interface Codon {
  bases: [string, string, string];              // The 3 DNA bases
  gappedPositions: [number, number, number];    // Gapped positions for each base
  aminoAcid: string;                             // Translated amino acid (1 letter)
  isGap: boolean;                                // True if any base is a gap
  isMismatch: boolean;                          // True if query differs from ref
}

export interface CodonRow {
  label: string;
  codons: Codon[];
  totalWidth: number; // pixel width
}

export interface AlignmentCodonView {
  refRow: CodonRow;
  matchRow: CodonRow;  // Just "|" or " " per codon
  queryRow: CodonRow;
  refAARow: CodonRow;  // Amino acids for ref (centered over codons)
  queryAARow: CodonRow;
  cdsStart: number;    // Gapped position where CDS begins
  cdsEnd: number;       // Gapped position where CDS ends
  totalCodons: number;
  errorRegions: ErrorRegion[];
}

/**
 * Core hook for codon-aware alignment view.
 *
 * Entity relationship: 3 DNA bases (codon) → 1 amino acid
 *
 * This hook computes the codon-level view of the alignment,
 * reactive to store changes via useAlignment.
 */
export function useCodonAlignment(): AlignmentCodonView | null {
  const alignment = useAlignment();

  return useMemo(() => {
    if (!alignment || !alignment.cdsHighlight) {
      return null;
    }

    const { refGapped, queryGapped, cdsHighlight } = alignment;
    const cdsStart = cdsHighlight.start;
    const cdsEnd = cdsHighlight.end;

    // Helper: extract codon at a gapped position
    const extractCodon = (startPos: number, seq: string) => {
      const bases: string[] = [];
      const positions: number[] = [];

      for (let offset = 0; offset < 3; offset++) {
        const pos = startPos + offset;
        if (pos < 0 || pos >= seq.length || seq[pos] === "-") {
          bases.push("-");
          positions.push(-1);
        } else {
          bases.push(seq[pos]);
          positions.push(pos);
        }
      }

      return {
        bases: bases.slice(0, 3) as [string, string, string],
        positions: positions.slice(0, 3) as [number, number, number],
        isGap: bases.includes("-"),
      };
    };

    // Calculate number of codons that fit in CDS region
    const cdsLength = cdsEnd - cdsStart;
    const numCodons = Math.floor(cdsLength / 3);

    // Build codons starting from CDS start (not from position 0)
    // This ensures the reading frame starts at the correct ATG position
    const codons: Codon[] = [];

    for (let i = 0; i < numCodons; i++) {
      const codonStartPos = cdsStart + i * 3;

      const refCodon = extractCodon(codonStartPos, refGapped);
      const queryCodon = extractCodon(codonStartPos, queryGapped);

      // Check if query differs from ref at any position
      const isMismatch = refCodon.positions.some(
        (pos, idx) => pos >= 0 && queryCodon.bases[idx] !== refCodon.bases[idx]
      );

      // Translate codon to amino acid
      const refAA = refCodon.isGap ? "?" : translateCodon(refCodon.bases.join(""));

      codons.push({
        bases: refCodon.bases,
        gappedPositions: refCodon.positions,
        aminoAcid: refAA,
        isGap: refCodon.isGap,
        isMismatch,
      });
    }

    // Build rows
    const refRow: CodonRow = {
      label: "Ref",
      codons,
      totalWidth: codons.length * CODON_WIDTH,
    };

    // Match row: "|" if codon matches, " " if mismatch
    const matchRow: CodonRow = {
      label: "",
      codons: codons.map(c => ({
        bases: [" ", " ", " "] as [string, string, string],
        gappedPositions: c.gappedPositions,
        aminoAcid: c.isMismatch ? " " : "|",
        isGap: true,
        isMismatch: false,
      })),
      totalWidth: codons.length * CODON_WIDTH,
    };

    // Query row
    const queryRow: CodonRow = {
      label: "Sanger",
      codons: codons.map((_, i) => {
        const codonStartPos = cdsStart + i * 3;
        const qCodon = extractCodon(codonStartPos, queryGapped);
        return {
          bases: qCodon.bases,
          gappedPositions: qCodon.positions,
          aminoAcid: qCodon.isGap ? "?" : translateCodon(qCodon.bases.join("")),
          isGap: qCodon.isGap,
          isMismatch: codons[i].isMismatch,
        };
      }),
      totalWidth: codons.length * CODON_WIDTH,
    };

    // AA rows: amino acids centered over their codons
    // Only show amino acid at first position of codon, rest are spaces
    const refAARow: CodonRow = {
      label: "Ref AA",
      codons: codons.map(c => ({
        bases: [" ", " ", " "] as [string, string, string],
        gappedPositions: c.gappedPositions,
        aminoAcid: c.isGap ? " " : c.aminoAcid,
        isGap: true,
        isMismatch: false,
      })),
      totalWidth: codons.length * CODON_WIDTH,
    };

    const queryAARow: CodonRow = {
      label: "Query AA",
      codons: codons.map((_, i) => {
        const codonStartPos = cdsStart + i * 3;
        const qCodon = extractCodon(codonStartPos, queryGapped);
        return {
          bases: [" ", " ", " "] as [string, string, string],
          gappedPositions: qCodon.positions,
          aminoAcid: qCodon.isGap ? " " : translateCodon(qCodon.bases.join("")),
          isGap: true,
          isMismatch: false,
        };
      }),
      totalWidth: codons.length * CODON_WIDTH,
    };

    // Error regions for navigation
    const mismatchPositions = alignment.mismatchPositions.filter(
      pos => pos >= cdsStart && pos < cdsEnd
    );
    const errorRegions = groupErrorsIntoRegions(mismatchPositions, 5);

    return {
      refRow,
      matchRow,
      queryRow,
      refAARow,
      queryAARow,
      cdsStart,
      cdsEnd,
      totalCodons: codons.length,
      errorRegions,
    };
  }, [alignment]);
}