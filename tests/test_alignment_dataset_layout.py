from pathlib import Path
import shutil
import uuid

import core.alignment as alignment


def _mk_temp_dir() -> Path:
    root = Path(".tmp-tests")
    root.mkdir(exist_ok=True)
    d = root / f"case-{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def test_resolve_dataset_dirs_supports_nested_layout():
    base = _mk_temp_dir()
    try:
        data_dir = base / "data"
        (data_dir / "pro" / "gb").mkdir(parents=True)
        (data_dir / "pro" / "ab1").mkdir(parents=True)

        gb_dir, ab1_dir = alignment.resolve_dataset_dirs("pro", data_dir)

        assert gb_dir == data_dir / "pro" / "gb"
        assert ab1_dir == data_dir / "pro" / "ab1"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_resolve_dataset_dirs_falls_back_to_legacy_layout():
    base = _mk_temp_dir()
    try:
        data_dir = base / "data"
        (data_dir / "gb_pro").mkdir(parents=True)
        (data_dir / "ab1_files_pro").mkdir(parents=True)

        gb_dir, ab1_dir = alignment.resolve_dataset_dirs("pro", data_dir)

        assert gb_dir == data_dir / "gb_pro"
        assert ab1_dir == data_dir / "ab1_files_pro"
    finally:
        shutil.rmtree(base, ignore_errors=True)
