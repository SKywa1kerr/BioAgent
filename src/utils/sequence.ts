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

export function translateCodon(codon: string): string {
  const upper = codon.toUpperCase();
  if (upper.length !== 3 || upper.includes("-") || upper.includes("N")) {
    return "?";
  }
  return CODON_TABLE[upper] || "?";
}

export function translateDNA(seq: string, startPos: number = 0): AminoAcid[] {
  const result: AminoAcid[] = [];
  const upper = seq.toUpperCase();
  for (let i = startPos; i + 2 < upper.length; i += 3) {
    const codon = upper.slice(i, i + 3);
    result.push({ aa: translateCodon(codon), codon, position: i });
  }
  return result;
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
