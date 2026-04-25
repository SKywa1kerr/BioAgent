function padAlignment(value, length) {
  return value.padEnd(length, "-");
}

export function buildCoordinateMap(refGapped, queryGapped) {
  const length = Math.max(refGapped.length, queryGapped.length);
  const refLine = padAlignment(refGapped, length);
  const queryLine = padAlignment(queryGapped, length);
  const refToGapped = [];
  const queryToGapped = [];
  const gappedToRef = [];
  const gappedToQuery = [];
  let refPosition = 0;
  let queryPosition = 0;

  for (let index = 0; index < length; index += 1) {
    if (refLine[index] === "-") {
      gappedToRef.push(null);
    } else {
      gappedToRef.push(refPosition);
      refToGapped.push(index);
      refPosition += 1;
    }

    if (queryLine[index] === "-") {
      gappedToQuery.push(null);
    } else {
      gappedToQuery.push(queryPosition);
      queryToGapped.push(index);
      queryPosition += 1;
    }
  }

  return { refToGapped, gappedToRef, gappedToQuery, queryToGapped };
}

export function parseAaChanges(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }

  if (typeof value !== "string") return [];

  const trimmed = value.trim();
  if (!trimmed) return [];

  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch {
    // Plain strings are valid input.
  }

  return [trimmed];
}

function buildMatchLine(refLine, queryLine, matches) {
  if (matches && matches.length >= refLine.length) {
    return Array.from({ length: refLine.length }, (_, index) => (matches[index] ? "|" : " ")).join("");
  }

  return Array.from({ length: refLine.length }, (_, index) => {
    const refBase = refLine[index];
    const queryBase = queryLine[index];
    return refBase !== "-" && queryBase !== "-" && refBase === queryBase ? "|" : " ";
  }).join("");
}

function buildPositionLine(length) {
  return Array.from({ length }, (_, index) => {
    const position = index + 1;
    return position % 10 === 0 ? String(position % 10) : " ";
  }).join("");
}

function buildTickLine(length) {
  return Array.from({ length }, (_, index) => ((index + 1) % 10 === 0 ? "+" : ".")).join("");
}

function mapReferenceRange(start, end, map) {
  if (!start || !end) return null;

  const startBase = map.refToGapped[start - 1];
  const endBase = map.refToGapped[end - 1];
  if (startBase === undefined || endBase === undefined) return null;

  return { start: startBase, end: endBase + 1 };
}

function mapMutationRanges(sample, map) {
  return (sample.mutations ?? []).flatMap((mutation) => {
    if (!mutation.position) return [];

    const start = map.refToGapped[mutation.position - 1];
    if (start === undefined) return [];

    return [{
      start,
      end: start + 1,
      label: `${mutation.refBase || "-"}>${mutation.queryBase || "-"}`,
      type: mutation.type,
      effect: mutation.effect,
    }];
  });
}

export function buildAlignmentViewModel(sample) {
  const refSource = sample.aligned_ref_g || sample.ref_sequence || "";
  const querySource = sample.aligned_query_g || sample.aligned_query || sample.query_sequence || "";
  if (!refSource || !querySource) return null;

  const length = Math.max(refSource.length, querySource.length);
  const refLine = padAlignment(refSource, length);
  const queryLine = padAlignment(querySource, length);
  const coordinateMap = buildCoordinateMap(refLine, queryLine);

  return {
    sampleId: sample.id,
    refLine,
    queryLine,
    matchLine: buildMatchLine(refLine, queryLine, sample.matches),
    positionLine: buildPositionLine(length),
    tickLine: buildTickLine(length),
    coordinateMap,
    cdsRange: mapReferenceRange(sample.cds_start, sample.cds_end, coordinateMap),
    mutationRanges: mapMutationRanges(sample, coordinateMap),
    aaChanges: parseAaChanges(sample.aa_changes),
  };
}
