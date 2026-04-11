import assert from "node:assert/strict";

const { bucketSampleStatus, countSampleMutations, buildResultsView } = await import(
  "../src/utils/resultsWorkbench.ts"
);

function makeSample(overrides = {}) {
  return {
    id: "S1",
    name: "Sample 1",
    clone: "C1",
    status: "ok",
    identity: 1,
    coverage: 1,
    mutations: [],
    ref_sequence: "",
    query_sequence: "",
    aligned_query: "",
    matches: [],
    cds_start: 1,
    cds_end: 1,
    frameshift: false,
    ...overrides,
  };
}

function testBucketSampleStatusTreatsUntestedReasonAsUntested() {
  assert.equal(bucketSampleStatus(makeSample({ status: "uncertain", reason: "未测通" })), "untested");
  assert.equal(bucketSampleStatus(makeSample({ status: "wrong" })), "wrong");
  assert.equal(bucketSampleStatus(makeSample({ status: "processing" })), "uncertain");
}

function testCountSampleMutationsFallsBackToMutationArray() {
  assert.equal(
    countSampleMutations(
      makeSample({
        mutations: [{ position: 1, refBase: "A", queryBase: "T", type: "substitution" }],
      })
    ),
    1
  );
  assert.equal(countSampleMutations(makeSample({ sub_count: 2, ins_count: 1, del_count: 3 })), 6);
}

function testBuildResultsViewFiltersSearchesAndSortsSamples() {
  const samples = [
    makeSample({ id: "OK-1", clone: "C100", status: "ok", identity: 1, coverage: 0.91 }),
    makeSample({
      id: "BAD-2",
      clone: "C200",
      status: "wrong",
      identity: 0.97,
      coverage: 0.88,
      mutations: [{ position: 4, refBase: "A", queryBase: "G", type: "substitution" }],
    }),
    makeSample({ id: "WAIT-3", clone: "C300", status: "uncertain", reason: "未测通", identity: 0.99, coverage: 0.5 }),
  ];

  const filtered = buildResultsView(samples, {
    statusFilter: "wrong",
    searchQuery: "c200",
    sortKey: "sample",
  });
  assert.deepEqual(filtered.map((sample) => sample.id), ["BAD-2"]);

  const sorted = buildResultsView(samples, {
    statusFilter: "all",
    searchQuery: "",
    sortKey: "status",
  });
  assert.deepEqual(sorted.map((sample) => sample.id), ["BAD-2", "WAIT-3", "OK-1"]);

  const mutationSorted = buildResultsView(samples, {
    statusFilter: "all",
    searchQuery: "",
    sortKey: "mutations",
  });
  assert.deepEqual(mutationSorted.map((sample) => sample.id), ["BAD-2", "WAIT-3", "OK-1"]);
}

testBucketSampleStatusTreatsUntestedReasonAsUntested();
testCountSampleMutationsFallsBackToMutationArray();
testBuildResultsViewFiltersSearchesAndSortsSamples();
