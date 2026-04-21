from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import uuid

import core.alignment as alignment


def _mk_temp_dir() -> Path:
    root = Path('.tmp-tests')
    root.mkdir(exist_ok=True)
    d = root / f"case-{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def test_analyze_dataset_does_not_write_stdout(monkeypatch):
    base = _mk_temp_dir()
    try:
        data_dir = base / 'data'
        gb_dir = data_dir / 'pro' / 'gb'
        ab1_dir = data_dir / 'pro' / 'ab1'
        gb_dir.mkdir(parents=True)
        ab1_dir.mkdir(parents=True)

        gb_path = gb_dir / 'C123.gb'
        ab1_path = ab1_dir / 'x-C123-1.ab1'
        gb_path.write_text('fake', encoding='utf-8')
        ab1_path.write_text('fake', encoding='utf-8')

        monkeypatch.setattr(alignment, 'build_aligner', lambda: object())
        monkeypatch.setattr(alignment, 'find_ab1_for_clone', lambda _ab1_dir, _clone: [ab1_path])
        monkeypatch.setattr(
            alignment,
            'analyze_sample',
            lambda **kwargs: {
                'sid': 'C123-1',
                'clone': 'C123',
                'ab1': ab1_path.name,
                'gb': gb_path.name,
                'orientation': 'FORWARD',
                'identity': 0.99,
                'cds_coverage': 0.95,
                'frameshift': False,
                'aa_changes': [],
                'aa_changes_n': 0,
                'raw_aa_changes_n': 0,
                'has_indel': False,
                'sub': 0,
                'ins': 0,
                'del': 0,
                'seq_length': 100,
                'ref_length': 1000,
                'cds_start': 1,
                'cds_end': 900,
                'avg_qry_quality': 38.0,
                '_cds_positions': {},
            },
        )

        buf = StringIO()
        with redirect_stdout(buf):
            results = alignment.analyze_dataset('pro', data_dir)

        assert len(results) == 1
        assert buf.getvalue() == ''
    finally:
        shutil.rmtree(base, ignore_errors=True)
