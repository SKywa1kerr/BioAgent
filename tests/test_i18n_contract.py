from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_i18n_declares_both_languages_and_real_en_tree():
    content = _read("src/i18n.ts")
    assert 'const translations: Record<AppLanguage, TranslationTree> = {' in content
    assert '  zh: ' in content
    assert '  en: ' in content
