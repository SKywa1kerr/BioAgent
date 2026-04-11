# BioAgent Desktop

BioAgent Desktop is an Electron-based Sanger sequencing QC and mutation analysis tool. It combines local bioinformatics analysis with optional AI-assisted review.

## Core Features

- Import AB1 and reference datasets for batch analysis
- Detect base mutations, amino acid changes, and frameshifts
- Review aligned sequences and chromatogram details in a desktop UI
- Export Excel reports
- Optional AI review and assistant chat

## Privacy And API Configuration

- The repository does not need to include any real API key, token, or private base URL
- In the packaged desktop app, users must fill in their own `API Key`, `Base URL`, and `Model` in the Settings page
- These values are stored locally in the app user data directory and are not meant to be committed into Git
- Do not commit local datasets, experimental data, SQLite databases, build outputs, or release packages

## Development

Requirements:

- Node.js 18+
- Python 3.10+

Install dependencies:

```bash
npm install
pip install -e ./src-python
```

Run in development:

```bash
npm run electron:dev
```

## Packaging

Build the Python sidecar:

```bash
npm run build:python
```

Build the desktop installer:

```bash
npm run electron:build
```

If both steps succeed on Windows, the `.exe` installer will be generated under `release/`.

## Project Structure

```text
electron/      Electron main process and preload bridge
src/           React frontend
src-python/    Python analysis sidecar
tests/         Automated tests
```
