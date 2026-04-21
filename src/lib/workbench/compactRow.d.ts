import type { WorkbenchSample } from "../../components/workbench/types";

export interface CompactRowView {
  aaPills: string[];
  aaOverflow: number;
  mutationTypes: string[];
}

export function compactRowView(sample: WorkbenchSample): CompactRowView;
