import type { CommandActionDefinition } from "../types";

export const actionRegistry = [
  {
    id: "import_dataset",
    labelKey: "command.actions.importDataset",
    descriptionKey: "command.actions.importDatasetDescription",
    category: "dataset",
    needsConfirmation: false,
  },
  {
    id: "set_ab1_dir",
    labelKey: "command.actions.setAb1Dir",
    descriptionKey: "command.actions.setAb1DirDescription",
    category: "dataset",
    needsConfirmation: false,
  },
  {
    id: "set_genes_dir",
    labelKey: "command.actions.setGenesDir",
    descriptionKey: "command.actions.setGenesDirDescription",
    category: "dataset",
    needsConfirmation: false,
  },
  {
    id: "set_plasmid",
    labelKey: "command.actions.setPlasmid",
    descriptionKey: "command.actions.setPlasmidDescription",
    category: "analysis",
    needsConfirmation: false,
  },
  {
    id: "run_analysis",
    labelKey: "command.actions.runAnalysis",
    descriptionKey: "command.actions.runAnalysisDescription",
    category: "analysis",
    needsConfirmation: true,
  },
  {
    id: "filter_results",
    labelKey: "command.actions.filterResults",
    descriptionKey: "command.actions.filterResultsDescription",
    category: "analysis",
    needsConfirmation: false,
  },
  {
    id: "open_sample",
    labelKey: "command.actions.openSample",
    descriptionKey: "command.actions.openSampleDescription",
    category: "navigation",
    needsConfirmation: false,
  },
  {
    id: "export_report",
    labelKey: "command.actions.exportReport",
    descriptionKey: "command.actions.exportReportDescription",
    category: "export",
    needsConfirmation: true,
  },
  {
    id: "open_export_folder",
    labelKey: "command.actions.openExportFolder",
    descriptionKey: "command.actions.openExportFolderDescription",
    category: "navigation",
    needsConfirmation: false,
  },
] as const satisfies ReadonlyArray<CommandActionDefinition>;

export type CommandActionId = CommandActionDefinition["id"];

export const actionRegistryById = Object.fromEntries(
  actionRegistry.map((action) => [action.id, action] as const)
) as Record<CommandActionId, CommandActionDefinition>;

export function getActionDefinition(actionId: CommandActionId) {
  return actionRegistryById[actionId];
}
