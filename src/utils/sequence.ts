const COMPLEMENT_MAP: Record<string, string> = {
  A: "T", T: "A", G: "C", C: "G",
  a: "t", t: "a", g: "c", c: "g",
  N: "N", n: "n", "-": "-",
};

export function complementBase(base: string): string {
  return COMPLEMENT_MAP[base] || "N";
}

export function complementStrand(seq: string): string {
  return seq.split("").map(complementBase).join("");
}

export function reverseComplement(seq: string): string {
  return complementStrand(seq).split("").reverse().join("");
}

const CODON_TABLE: Record<string, string> = {
  TTT: "F", TTC: "F", TTA: "L", TTG: "L",
  CTT: "L", CTC: "L", CTA: "L", CTG: "L",
  ATT: "I", ATC: "I", ATA: "I", ATG: "M",
  GTT: "V", GTC: "V", GTA: "V", GTG: "V",
  TCT: "S", TCC: "S", TCA: "S", TCG: "S",
  CCT: "P", CCC: "P", CCA: "P", CCG: "P",
  ACT: "T", ACC: "T", ACA: "T", ACG: "T",
  GCT: "A", GCC: "A", GCA: "A", GCG: "A",
  TAT: "Y", TAC: "Y", TAA: "*", TAG: "*",
  CAT: "H", CAC: "H", CAA: "Q", CAG: "Q",
  AAT: "N", AAC: "N", AAA: "K", AAG: "K",
  GAT: "D", GAC: "D", GAA: "E", GAG: "E",
  TGT: "C", TGC: "C", TGA: "*", TGG: "W",
  CGT: "R", CGC: "R", CGA: "R", CGG: "R",
  AGT: "S", AGC: "S", AGA: "R", AGG: "R",
  GGT: "G", GGC: "G", GGA: "G", GGG: "G",
};

export interface AminoAcid {
  aa: string;
  codon: string;
  position: number; // 0-based position in the sequence
}

export function translateDNA(seq: string, startPos: number = 0): AminoAcid[] {
  const result: AminoAcid[] = [];
  const upper = seq.toUpperCase();
  for (let i = startPos; i + 2 < upper.length; i += 3) {
    const codon = upper.slice(i, i + 3);
    if (codon.includes("-") || codon.includes("N")) {
      result.push({ aa: "?", codon, position: i });
    } else {
      result.push({ aa: CODON_TABLE[codon] || "?", codon, position: i });
    }
  }
  return result;
}

export interface RestrictionSite {
  name: string;
  position: number; // 0-based position in the sequence
  sequence: string;
}

// Common 6+ cutter restriction enzymes (nonredundant set)
const RESTRICTION_ENZYMES: { name: string; sequence: string }[] = [
  { name: "EcoRI", sequence: "GAATTC" },
  { name: "BamHI", sequence: "GGATCC" },
  { name: "HindIII", sequence: "AAGCTT" },
  { name: "XhoI", sequence: "CTCGAG" },
  { name: "NcoI", sequence: "CCATGG" },
  { name: "NdeI", sequence: "CATATG" },
  { name: "SalI", sequence: "GTCGAC" },
  { name: "XbaI", sequence: "TCTAGA" },
  { name: "PstI", sequence: "CTGCAG" },
  { name: "SphI", sequence: "GCATGC" },
  { name: "KpnI", sequence: "GGTACC" },
  { name: "SacI", sequence: "GAGCTC" },
  { name: "AvaI", sequence: "CYCGRG" },
  { name: "BsiWI", sequence: "CGTACG" },
  { name: "NruI", sequence: "TCGCGA" },
  { name: "BsrGI", sequence: "TGTACA" },
  { name: "StyI", sequence: "CCWWGG" },
  { name: "BlpI", sequence: "GCTNAGC" },
  { name: "CsiI", sequence: "ACCWGGT" },
  { name: "BsoBI", sequence: "CYCGRG" },
  { name: "PaeR7I", sequence: "CTCGAG" },
  { name: "SexAI", sequence: "ACCWGGT" },
];

// IUPAC ambiguity codes for regex matching
const IUPAC_MAP: Record<string, string> = {
  A: "A", T: "T", G: "G", C: "C",
  R: "[AG]", Y: "[CT]", M: "[AC]", K: "[GT]",
  S: "[GC]", W: "[AT]", H: "[ACT]", B: "[CGT]",
  V: "[ACG]", D: "[AGT]", N: "[ACGT]",
};

function iupacToRegex(seq: string): RegExp {
  const pattern = seq.split("").map((c) => IUPAC_MAP[c] || c).join("");
  return new RegExp(pattern, "gi");
}

export function findRestrictionSites(seq: string): RestrictionSite[] {
  const sites: RestrictionSite[] = [];
  const upper = seq.toUpperCase();
  const seen = new Set<string>(); // deduplicate overlapping enzymes at same position

  for (const enzyme of RESTRICTION_ENZYMES) {
    const regex = iupacToRegex(enzyme.sequence);
    let match: RegExpExecArray | null;
    while ((match = regex.exec(upper)) !== null) {
      const key = `${enzyme.name}:${match.index}`;
      if (!seen.has(key)) {
        seen.add(key);
        sites.push({
          name: enzyme.name,
          position: match.index,
          sequence: enzyme.sequence,
        });
      }
      // Move forward to find overlapping matches
      regex.lastIndex = match.index + 1;
    }
  }

  return sites.sort((a, b) => a.position - b.position);
}

// Group restriction sites that share the same position
export function groupRestrictionSites(sites: RestrictionSite[]): Map<number, string[]> {
  const groups = new Map<number, string[]>();
  for (const site of sites) {
    const existing = groups.get(site.position) || [];
    existing.push(site.name);
    groups.set(site.position, existing);
  }
  return groups;
}

export function baseColor(base: string): string {
  switch (base.toUpperCase()) {
    case "A": return "#00aa00";
    case "T": return "#aa0000";
    case "G": return "#000000";
    case "C": return "#0000aa";
    default: return "#888888";
  }
}

/**
 * Translate a gapped DNA sequence to amino acids.
 * Gaps are preserved in the output (shown as spaces).
 * Only translates complete codons (3 consecutive non-gap bases).
 */
export function translateGappedSequence(
  gappedSeq: string,
  startOffset: number = 0
): { char: string; isTranslated: boolean }[] {
  const result: { char: string; isTranslated: boolean }[] = [];
  const upper = gappedSeq.toUpperCase();

  // Track position in ungapped sequence for translation frame
  let ungappedPos = 0;

  for (let i = 0; i < upper.length; i++) {
    const base = upper[i];

    if (base === "-") {
      result.push({ char: " ", isTranslated: false });
      continue;
    }

    // Check if this base starts a complete codon
    const codonStart = ungappedPos - ((ungappedPos + startOffset) % 3);
    const codonPos = ungappedPos - codonStart;

    if (codonPos === 0) {
      // First base of codon - try to read complete codon
      let codon = "";
      let j = i;
      let ungappedCount = 0;

      while (j < upper.length && ungappedCount < 3) {
        if (upper[j] !== "-") {
          codon += upper[j];
          ungappedCount++;
        }
        j++;
      }

      if (ungappedCount === 3) {
        const aa = translateCodon(codon);
        // Mark all three positions
        result.push({ char: aa, isTranslated: true });
      } else {
        result.push({ char: " ", isTranslated: false });
      }
    } else {
      // Middle of codon - already handled above
      result.push({ char: " ", isTranslated: false });
    }

    ungappedPos++;
  }

  return result;
}

/**
 * Error region - groups consecutive errors into a single region
 */
export interface ErrorRegion {
  start: number; // gapped position
  end: number; // exclusive
  length: number;
  errorCount: number;
}

/**
 * Group consecutive error positions into regions.
 * Errors within `maxGap` bases are considered part of the same region.
 */
export function groupErrorsIntoRegions(
  errorPositions: number[],
  maxGap: number = 5
): ErrorRegion[] {
  if (errorPositions.length === 0) return [];

  const sorted = [...errorPositions].sort((a, b) => a - b);
  const regions: ErrorRegion[] = [];

  let currentStart = sorted[0];
  let currentEnd = sorted[0] + 1;
  let currentCount = 1;

  for (let i = 1; i < sorted.length; i++) {
    const pos = sorted[i];

    if (pos - currentEnd <= maxGap) {
      // Extend current region
      currentEnd = pos + 1;
      currentCount++;
    } else {
      // Start new region
      regions.push({
        start: currentStart,
        end: currentEnd,
        length: currentEnd - currentStart,
        errorCount: currentCount,
      });
      currentStart = pos;
      currentEnd = pos + 1;
      currentCount = 1;
    }
  }

  // Don't forget the last region
  regions.push({
    start: currentStart,
    end: currentEnd,
    length: currentEnd - currentStart,
    errorCount: currentCount,
  });

  return regions;
}

/**
 * Count error regions instead of individual errors
 */
export function countErrorRegions(errorPositions: number[], maxGap: number = 5): number {
  return groupErrorsIntoRegions(errorPositions, maxGap).length;
}
