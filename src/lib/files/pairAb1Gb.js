// Pure helper for pairing dropped `.ab1` and `.gb`/`.gbk` file paths by basename.
//
// The drag-and-drop import path hands the renderer a flat list of file paths.
// Before we can call the analyze_sequences MCP tool we have to figure out
// which AB1 trace pairs with which GenBank reference. Pairing is done purely
// by stem (filename without extension), case-insensitively — directory and
// extension are not part of the match key.
//
// Output is deterministic (pairs sorted by basename) so callers can render a
// stable preview list and tests can assert on shape directly.
//
// Behaviour summary:
//   - Inputs may use forward or backslashes; we normalise before splitting.
//   - Files whose extension does not match the expected list (`.ab1` for
//     `ab1Paths`; `.gb`/`.gbk` for `gbPaths`) are pushed straight to the
//     matching `unpairedAb1`/`unpairedGb` array.
//   - When the same basename appears more than once on a side, "last write
//     wins": the most recent path becomes the active candidate and previous
//     winners are demoted to the unpaired array.
//   - AB1 candidates without a GB partner end up in `unpairedAb1`; GB
//     candidates without an AB1 partner end up in `unpairedGb`.

import { basename, extname } from "node:path";

const AB1_EXT = /^\.ab1$/i;
const GB_EXT = /^\.(gb|gbk)$/i;

function splitPath(p) {
  // Normalise backslashes to forward slashes so node's POSIX-flavoured
  // path.basename behaves identically on Windows and Linux/macOS hosts.
  const normalized = String(p).replace(/\\/g, "/");
  const file = basename(normalized);
  const ext = extname(file);
  const stem = ext ? file.slice(0, file.length - ext.length) : file;
  return { stem, ext };
}

export function pairAb1Gb(ab1Paths, gbPaths) {
  const ab1List = Array.isArray(ab1Paths) ? ab1Paths : [];
  const gbList = Array.isArray(gbPaths) ? gbPaths : [];

  const ab1Map = new Map();   // key: lower-cased stem → original path
  const ab1Stem = new Map();  // key: lower-cased stem → original-case stem
  const unpairedAb1 = [];

  for (const p of ab1List) {
    if (typeof p !== "string" || p.length === 0) continue;
    const { stem, ext } = splitPath(p);
    if (!AB1_EXT.test(ext)) {
      unpairedAb1.push(p);
      continue;
    }
    const key = stem.toLowerCase();
    if (ab1Map.has(key)) {
      // last-write-wins: the previous winner is demoted to the unpaired list.
      unpairedAb1.push(ab1Map.get(key));
    }
    ab1Map.set(key, p);
    ab1Stem.set(key, stem);
  }

  const gbMap = new Map();
  const unpairedGb = [];

  for (const p of gbList) {
    if (typeof p !== "string" || p.length === 0) continue;
    const { stem, ext } = splitPath(p);
    if (!GB_EXT.test(ext)) {
      unpairedGb.push(p);
      continue;
    }
    const key = stem.toLowerCase();
    if (gbMap.has(key)) {
      unpairedGb.push(gbMap.get(key));
    }
    gbMap.set(key, p);
  }

  const pairs = [];
  for (const [key, ab1Path] of ab1Map) {
    const gbPath = gbMap.get(key);
    if (gbPath !== undefined) {
      pairs.push({ basename: ab1Stem.get(key), ab1: ab1Path, gb: gbPath });
      gbMap.delete(key);
    } else {
      unpairedAb1.push(ab1Path);
    }
  }
  for (const remaining of gbMap.values()) {
    unpairedGb.push(remaining);
  }

  pairs.sort((a, b) => {
    const x = a.basename.toLowerCase();
    const y = b.basename.toLowerCase();
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
  });

  return { pairs, unpairedAb1, unpairedGb };
}
