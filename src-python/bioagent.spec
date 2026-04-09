# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BioAgent Python sidecar (onedir mode)."""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all bioagent submodules
hiddenimports = collect_submodules('bioagent')
hiddenimports += ['Bio', 'Bio.SeqIO', 'Bio.Seq', 'Bio.Align', 'Bio.SeqRecord']
hiddenimports += ['sqlalchemy', 'sqlalchemy.dialects.sqlite']
hiddenimports += ['openpyxl', 'yaml']

# Data files: rules_config.yaml
datas = [
    (os.path.join('bioagent', 'rules_config.yaml'), 'bioagent'),
]

a = Analysis(
    ['bioagent/main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PIL', 'IPython', 'notebook'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='bioagent-sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='bioagent-sidecar',
)
