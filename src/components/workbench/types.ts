export interface WorkbenchMutation {
  position?: number;
  refBase?: string;
  queryBase?: string;
  type?: string;
  effect?: string;
}

export interface ChromatogramData {
  traces: {
    A: number[];
    T: number[];
    G: number[];
    C: number[];
  };
  quality: number[];
  baseCalls: string;
  base_locations: number[];
  mixed_peaks: number[];
}

export interface WorkbenchSample {
  id: string;
  name?: string;
  clone?: string;
  status?: "ok" | "wrong" | "uncertain" | "processing" | "error";
  reason?: string;
  review_reason?: string;
  llm_reason?: string;
  auto_reason?: string;
  error?: string;
  identity?: number;
  coverage?: number;
  cds_coverage?: number;
  sub_count?: number;
  ins_count?: number;
  del_count?: number;
  sub?: number;
  ins?: number;
  dele?: number;
  aa_changes?: string[] | string;
  aa_changes_n?: number;
  avg_qry_quality?: number;
  avg_quality?: number;
  orientation?: string;
  frameshift?: boolean;
  mutations?: WorkbenchMutation[];
  ref_sequence?: string;
  query_sequence?: string;
  aligned_ref_g?: string;
  aligned_query_g?: string;
  aligned_query?: string;
  matches?: boolean[];
  cds_start?: number;
  cds_end?: number;
  traces_a?: number[];
  traces_t?: number[];
  traces_g?: number[];
  traces_c?: number[];
  quality?: number[];
  base_locations?: number[];
  mixed_peaks?: number[];
  bucket?: "ok" | "wrong" | "uncertain" | "untested";
}

export type WorkbenchStatus = "ok" | "wrong" | "uncertain" | "untested";
