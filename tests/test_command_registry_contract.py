from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_command_registry_and_workbench_contract():
    registry = _read("src/utils/actionRegistry.ts")
    command_workbench = _read("src/components/CommandWorkbench.tsx")
    action_plan = _read("src/components/ActionPlanCard.tsx")
    timeline = _read("src/components/ExecutionTimeline.tsx")
    i18n = _read("src/i18n.ts")

    for action_name in [
        "import_dataset",
        "set_ab1_dir",
        "set_genes_dir",
        "set_plasmid",
        "run_analysis",
        "filter_results",
        "open_sample",
        "export_report",
        "open_export_folder",
    ]:
        assert action_name in registry

    assert "needsConfirmation" in registry
    assert "export const actionRegistry" in registry
    assert "export type CommandActionId" in registry

    assert "export function CommandWorkbench(" in command_workbench
    assert "quick prompts" in command_workbench.lower()
    assert "large command input" in command_workbench.lower()
    assert "batchSummary" in command_workbench
    assert "plasmidSummary" in command_workbench
    assert "sampleSummary" in command_workbench

    assert "export function ActionPlanCard(" in action_plan
    assert "plan summary" in action_plan.lower()
    assert "action list" in action_plan.lower()
    assert "confirm" in action_plan.lower()
    assert "cancel" in action_plan.lower()
    assert "needsConfirmation" in action_plan
    assert "command.statusRunning" in action_plan
    assert "command.statusDone" in action_plan

    assert "export function ExecutionTimeline(" in timeline
    assert "timeline-event" in timeline
    assert "status class" in timeline.lower()
    assert "timeline-events" in timeline
    assert "command.statusQueued" in timeline
    assert "command.statusFailed" in timeline

    assert "command: {" in i18n
    assert "命令工作台" in i18n
    assert "Command workbench" in i18n
    assert "quickPromptsTitle" in i18n
    assert "confirmationNeeded" in i18n
    assert "statusRunning" in i18n
    assert "statusQueued" in i18n
    assert "actions: {" in i18n


def test_open_export_folder_remains_a_confirmed_action():
    registry = _read("src/utils/actionRegistry.ts")

    assert 'id: "open_export_folder"' in registry
    assert 'needsConfirmation: true' in registry
