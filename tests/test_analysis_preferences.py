from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_analysis_preference_defaults_and_validation_contract():
    content = _read("src/utils/analysisPreferences.ts")
    assert 'export const DEFAULT_ANALYSIS_DECISION_MODE = "rules"' in content
    assert 'export function isAiReviewEnabled(' in content
    assert 'export function validateAiReviewSettings(' in content
    assert 'return { ok: false, reason: "missing_api_key" }' in content
    assert 'return { ok: false, reason: "missing_base_url" }' in content
    assert 'return { ok: false, reason: "missing_model" }' in content


def test_settings_page_uses_rules_as_default_mode():
    content = _read("src/components/SettingsPage.tsx")
    assert 'analysisDecisionMode: "rules"' in content
    assert 'isAiReviewEnabled(settings)' in content
