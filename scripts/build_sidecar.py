"""Build a standalone Python sidecar exe via PyInstaller.

Output: dist-python/bioagent-sidecar/bioagent-sidecar.exe (Windows) plus its
dependency directory. electron-builder picks this up via the
`extraResources` block in package.json and ships it next to the Electron
binary.

Usage:
    pip install pyinstaller
    pip install -r requirements.txt
    python scripts/build_sidecar.py

Notes:
- We use --onedir (not --onefile) because biopython / pandas C extensions
  unpack faster from a directory and avoid the per-launch extraction cost
  that would otherwise add ~2s of MCP startup latency on first call.
- biopython needs explicit data collection because its parsers reach for
  static codon / data tables at runtime.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "src-python" / "bioagent" / "main.py"
DIST = ROOT / "dist-python"
WORK = ROOT / "build-pyinstaller"


def main() -> int:
    if not ENTRY.exists():
        print(f"[ERR] entry point missing: {ENTRY}", file=sys.stderr)
        return 1

    if DIST.exists():
        shutil.rmtree(DIST)
    if WORK.exists():
        shutil.rmtree(WORK)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "bioagent-sidecar",
        "--onedir",
        "--noconfirm",
        "--console",
        "--paths", str(ROOT / "src-python"),
        "--paths", str(ROOT),
        # Modules our entry point reaches via runtime imports.
        "--hidden-import", "core.alignment",
        "--hidden-import", "core.evidence",
        "--hidden-import", "core.llm_client",
        "--hidden-import", "bioagent.mcp_server",
        "--hidden-import", "bioagent.mcp_tools",
        "--hidden-import", "bioagent.tools_register",
        "--hidden-import", "bioagent.mutation_trends",
        "--hidden-import", "bioagent.lab_suggestions",
        "--hidden-import", "bioagent.logger",
        # biopython data tables and submodules (codon table etc).
        "--collect-data", "Bio",
        "--collect-submodules", "Bio",
        # Drop pandas's heavy optional integrations. The runtime only uses
        # DataFrame / read_csv / basic IO; we don't need torch, scipy,
        # pyarrow, matplotlib, or the jupyter stack tagging along (they
        # added 600+ MB to the first build).
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "scipy",
        "--exclude-module", "pyarrow",
        "--exclude-module", "matplotlib",
        "--exclude-module", "tkinter",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "--exclude-module", "notebook",
        "--exclude-module", "sklearn",
        "--exclude-module", "tables",
        "--exclude-module", "fastparquet",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PyQt6",
        "--exclude-module", "PySide2",
        "--exclude-module", "PySide6",
        "--distpath", str(DIST),
        "--workpath", str(WORK),
        "--specpath", str(WORK),
        str(ENTRY),
    ]

    print("Running:")
    print("  " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[ERR] PyInstaller exited with code {rc}", file=sys.stderr)
        return rc

    out_dir = DIST / "bioagent-sidecar"
    if not out_dir.exists():
        print(f"[ERR] expected output missing: {out_dir}", file=sys.stderr)
        return 1

    print(f"[OK] sidecar built at: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
