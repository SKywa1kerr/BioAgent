# BioAgent MAX Codebase Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix duplicate code, session leaks, repeated sample-saving logic, and other critical issues identified in code review.

**Architecture:** Consolidate `core/` (CLI) and `backend/core/` (web) into a single source of truth under `backend/core/`. Extract repeated sample-saving logic into a shared helper. Add context managers for session safety. Fix MCP server (already done). Improve Docker setup.

**Tech Stack:** Python 3.12, SQLAlchemy, FastAPI, Streamlit, FastMCP, BioPython

---

## File Structure

### Files to Delete
- `core/alignment.py` — forked version of `backend/core/alignment.py` (old CLI-only API, superseded)
- `core/evidence.py` — duplicate of `backend/core/evidence.py`
- `core/__init__.py` — no longer needed
- `core/llm_client.py` — moved to `backend/core/llm_client.py`

### Files to Modify
- `run.py` — update imports from `core.*` to `backend.core.*`
- `backend/core/rules.py` — add threshold caching
- `backend/db/database.py` — add session context manager
- `backend/db/models.py` — add `save_analysis_with_samples()` helper
- `backend/api/analyze.py` — use session context manager + shared save helper
- `backend/mcp_server.py` — use shared save helper (already fixed to FastMCP)
- `frontend/pages/1_Analysis.py` — use shared save helper, eliminate 2x duplication
- `Dockerfile` — improve process management
- `docker-compose.yml` — fix volume mount for SQLite

### Files to Create
- `tests/test_session_safety.py` — test session context manager
- `tests/test_save_helper.py` — test the shared save_analysis helper

---

### Task 1: Consolidate duplicate `core/` into `backend/core/`

**Files:**
- Delete: `core/alignment.py`, `core/evidence.py`, `core/__init__.py`
- Modify: `run.py`
- Keep: `core/llm_client.py` (unique file, no backend equivalent — move to `backend/core/`)

The root `core/alignment.py` has an old `analyze_dataset(dataset, data_dir)` signature with hardcoded DATASET_MAP. The `backend/core/alignment.py` has the newer `analyze_dataset(gb_dir, ab1_dir)` signature. The backend version is the correct one — `run.py` is the only consumer of the old signature.

- [ ] **Step 1: Move `core/llm_client.py` to `backend/core/llm_client.py`**

Copy `core/llm_client.py` to `backend/core/llm_client.py`. No changes needed to the file content since it uses relative `.env` path lookup via `Path(__file__).resolve().parent.parent / ".env"` — after moving one level deeper, update this to `.parent.parent.parent`.

```python
# In backend/core/llm_client.py, line 21, change:
env_path = Path(__file__).resolve().parent.parent / ".env"
# to:
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
```

- [ ] **Step 2: Update `run.py` to use `backend.core.*`**

Change imports:
```python
# Old:
from core.alignment import analyze_dataset
from core.evidence import format_evidence_for_llm, format_evidence_table
from core.llm_client import call_llm, parse_llm_result

# New:
from backend.core.alignment import analyze_dataset
from backend.core.evidence import format_evidence_for_llm, format_evidence_table
from backend.core.llm_client import call_llm, parse_llm_result
```

Also update the `analyze_dataset` call at line 61. The old signature was `analyze_dataset(dataset_name, data_dir, out_html_dir)` with hardcoded DATASET_MAP. The new signature is `analyze_dataset(gb_dir, ab1_dir)`. We need to add a small mapping in `run.py`:

**Note:** The old `core/alignment.py` supported `out_html_dir` for HTML alignment output. The backend version does not. This is intentional — HTML output was a CLI debugging feature that is replaced by the Streamlit alignment viewer in the web platform. If HTML output is needed for CLI usage later, it can be re-added as a wrapper in `run.py`.

```python
DATASET_MAP = {
    "base": {"gb": "gb", "ab1": "ab1_files"},
    "pro": {"gb": "gb_pro", "ab1": "ab1_files_pro"},
    "promax": {"gb": "gb_promax", "ab1": "ab1_files_promax"},
}
dirs = DATASET_MAP[args.dataset]
gb_dir = data_dir / dirs["gb"]
ab1_dir = data_dir / dirs["ab1"]
samples = analyze_dataset(gb_dir, ab1_dir)
```

- [ ] **Step 3: Delete the old `core/` directory**

Remove `core/alignment.py`, `core/evidence.py`, `core/__init__.py`, and the now-empty `core/llm_client.py`.

- [ ] **Step 4: Run tests to verify nothing is broken**

Run: `cd D:/Learning/Biology/BioAgent_MAX && python -m pytest tests/ -v`
Expected: All tests pass (they import from `backend.core.rules` already)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: consolidate core/ into backend/core/, remove duplicate code"
```

---

### Task 2: Add session context manager to database layer

**Files:**
- Modify: `backend/db/database.py`
- Create: `tests/test_session_safety.py`

Currently every API endpoint and frontend page manually creates sessions with `Session()` and calls `session.close()`. If an exception occurs between open and close, the session leaks.

- [ ] **Step 1: Write test for session context manager**

```python
# tests/test_session_safety.py
from backend.db.database import init_db, db_session

def test_db_session_context_manager():
    init_db()
    with db_session() as session:
        # Should be able to query
        result = session.execute(
            __import__("sqlalchemy").text("SELECT 1")
        ).scalar()
        assert result == 1

def test_db_session_rollback_on_error():
    init_db()
    try:
        with db_session() as session:
            raise ValueError("test error")
    except ValueError:
        pass
    # Session should be closed, not leaked
    with db_session() as session:
        result = session.execute(
            __import__("sqlalchemy").text("SELECT 1")
        ).scalar()
        assert result == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session_safety.py -v`
Expected: FAIL with `ImportError: cannot import name 'db_session'`

- [ ] **Step 3: Implement context manager in `backend/db/database.py`**

Add at end of file:

```python
from contextlib import contextmanager

@contextmanager
def db_session():
    """Context manager for safe session handling. Auto-commits on success, rolls back on error, always closes."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_session_safety.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/db/database.py tests/test_session_safety.py
git commit -m "feat: add db_session() context manager for safe session handling"
```

---

### Task 3: Extract shared `save_analysis_with_samples()` helper

**Files:**
- Modify: `backend/db/models.py`
- Create: `tests/test_save_helper.py`

The same 30+ line block of code for creating an Analysis + Sample records is copy-pasted in 3 places:
1. `backend/api/analyze.py` lines 31-59
2. `frontend/pages/1_Analysis.py` lines 65-105 (and again lines 163-203)
3. `backend/mcp_server.py` lines 43-83

Extract this into a single function.

- [ ] **Step 1: Write test for save helper**

```python
# tests/test_save_helper.py
import json
from backend.db.database import init_db, db_session
from backend.db.models import save_analysis_with_samples, Analysis, Sample

def test_save_analysis_with_samples():
    init_db()
    samples = [{
        "sid": "TEST-1", "clone": "TEST", "identity": 0.99,
        "cds_coverage": 1.0, "frameshift": False,
        "aa_changes": [], "aa_changes_n": 0, "raw_aa_changes_n": 0,
        "orientation": "FORWARD", "seq_length": 1000, "ref_length": 5000,
        "avg_qry_quality": 35.0, "sub": 0, "ins": 0, "del": 0,
        "ref_gapped": "ATCG", "qry_gapped": "ATCG",
        "quality_scores": [30, 35, 40, 45],
    }]
    judgments = [{"sid": "TEST-1", "status": "ok", "reason": "", "rule": 10}]
    thresholds = {"identity_high": 0.95}

    analysis_id = save_analysis_with_samples(
        samples=samples, judgments=judgments,
        thresholds=thresholds, name="Test Analysis",
        source_type="test", source_path="/tmp/test",
    )
    assert analysis_id is not None

    with db_session() as session:
        a = session.get(Analysis, analysis_id)
        assert a.total == 1
        assert a.ok_count == 1
        s = session.query(Sample).filter(Sample.analysis_id == analysis_id).first()
        assert s.sid == "TEST-1"
        assert s.status == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_save_helper.py -v`
Expected: FAIL with `ImportError: cannot import name 'save_analysis_with_samples'`

- [ ] **Step 3: Implement `save_analysis_with_samples()` in `backend/db/models.py`**

Add at end of `backend/db/models.py`:

```python
import json as _json
from datetime import datetime as _dt, timezone as _tz

def save_analysis_with_samples(
    samples: list[dict],
    judgments: list[dict],
    thresholds: dict,
    name: str = "",
    source_type: str = "scan",
    source_path: str = "",
) -> str:
    """Save analysis + samples to database. Returns analysis_id."""
    from backend.db.database import init_db, db_session

    init_db()
    ok = sum(1 for j in judgments if j["status"] == "ok")
    wrong = sum(1 for j in judgments if j["status"] == "wrong")
    uncertain = sum(1 for j in judgments if j["status"] == "uncertain")

    analysis_id = new_id()
    with db_session() as session:
        analysis = Analysis(
            id=analysis_id,
            name=name or f"分析 {_dt.now().strftime('%m-%d %H:%M')}",
            source_type=source_type, source_path=source_path,
            status="done", total=len(samples),
            ok_count=ok, wrong_count=wrong, uncertain_count=uncertain,
            config_snapshot=_json.dumps(thresholds),
            finished_at=_dt.now(_tz.utc),
        )
        session.add(analysis)
        for sd, jd in zip(samples, judgments):
            session.add(Sample(
                id=new_id(), analysis_id=analysis_id,
                sid=sd["sid"], clone=sd.get("clone", ""),
                status=jd["status"], reason=jd.get("reason", ""),
                rule_id=jd.get("rule"),
                identity=sd["identity"], cds_coverage=sd["cds_coverage"],
                frameshift=sd["frameshift"],
                aa_changes=_json.dumps(sd.get("aa_changes", [])),
                aa_changes_n=sd.get("aa_changes_n", 0),
                raw_aa_changes_n=sd.get("raw_aa_changes_n", 0),
                orientation=sd.get("orientation", ""),
                seq_length=sd.get("seq_length", 0),
                ref_length=sd.get("ref_length", 0),
                avg_quality=sd.get("avg_qry_quality"),
                sub_count=sd.get("sub", 0), ins_count=sd.get("ins", 0),
                del_count=sd.get("del", 0),
                ref_gapped=sd.get("ref_gapped", ""),
                qry_gapped=sd.get("qry_gapped", ""),
                quality_scores=_json.dumps(sd.get("quality_scores", []) or []),
                raw_data=_json.dumps(sd, default=str),
            ))
    return analysis_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_save_helper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/db/models.py tests/test_save_helper.py
git commit -m "feat: extract save_analysis_with_samples() helper to eliminate duplication"
```

---

### Task 4: Replace duplicated save logic in all consumers

**Files:**
- Modify: `backend/api/analyze.py`
- Modify: `backend/mcp_server.py`
- Modify: `frontend/pages/1_Analysis.py`

Replace the copy-pasted 30-line save blocks with calls to `save_analysis_with_samples()`.

- [ ] **Step 1: Update `backend/api/analyze.py`**

Replace `_run_analysis()` body. Key changes:
- Use `db_session()` context manager
- Use `save_analysis_with_samples()` for the save part
- Keep the status="running" update separate (it's needed before analysis starts)

```python
def _run_analysis(analysis_id: str, gb_dir: str, ab1_dir: str):
    from backend.db.database import db_session
    try:
        # Mark as running
        with db_session() as session:
            analysis = session.get(Analysis, analysis_id)
            analysis.status = "running"

        # Run analysis
        samples = analyze_dataset(Path(gb_dir), Path(ab1_dir))
        thresholds = load_thresholds()
        judgments = judge_batch(samples, thresholds)

        # Save results — delete the placeholder analysis first, save fresh
        with db_session() as session:
            analysis = session.get(Analysis, analysis_id)
            ok = sum(1 for j in judgments if j["status"] == "ok")
            wrong = sum(1 for j in judgments if j["status"] == "wrong")
            uncertain = sum(1 for j in judgments if j["status"] == "uncertain")
            analysis.status = "done"
            analysis.total = len(samples)
            analysis.ok_count = ok
            analysis.wrong_count = wrong
            analysis.uncertain_count = uncertain
            analysis.config_snapshot = json.dumps(thresholds)
            analysis.finished_at = datetime.now(timezone.utc)
            for sd, jd in zip(samples, judgments):
                session.add(Sample(
                    id=new_id(), analysis_id=analysis_id,
                    sid=sd["sid"], clone=sd.get("clone", ""),
                    status=jd["status"], reason=jd.get("reason", ""),
                    rule_id=jd.get("rule"),
                    identity=sd["identity"], cds_coverage=sd["cds_coverage"],
                    frameshift=sd["frameshift"],
                    aa_changes=json.dumps(sd.get("aa_changes", [])),
                    aa_changes_n=sd.get("aa_changes_n", 0),
                    raw_aa_changes_n=sd.get("raw_aa_changes_n", 0),
                    orientation=sd.get("orientation", ""),
                    seq_length=sd.get("seq_length", 0),
                    ref_length=sd.get("ref_length", 0),
                    avg_quality=sd.get("avg_qry_quality"),
                    sub_count=sd.get("sub", 0), ins_count=sd.get("ins", 0),
                    del_count=sd.get("del", 0),
                    ref_gapped=sd.get("ref_gapped", ""),
                    qry_gapped=sd.get("qry_gapped", ""),
                    quality_scores=json.dumps(sd.get("quality_scores", []) or []),
                    raw_data=json.dumps(sd, default=str),
                ))
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        with db_session() as session:
            analysis = session.get(Analysis, analysis_id)
            if analysis:
                analysis.status = "error"
```

Note: `_run_analysis` can't use `save_analysis_with_samples()` directly because it updates a pre-existing Analysis record rather than creating a new one. But we still benefit from `db_session()`.

Also update `trigger_analysis()` and `get_analysis_status()` to use `db_session()`.

- [ ] **Step 2: Update `backend/mcp_server.py`**

Replace the manual save block in `analyze_directory()` with:

**Important:** `mcp_server.py` uses relative imports via `sys.path.insert(0, backend/)`. Use the relative import style to match:

```python
from db.models import save_analysis_with_samples

# Replace lines 43-83 with:
analysis_id = save_analysis_with_samples(
    samples=samples, judgments=judgments,
    thresholds=thresholds,
    name=name or f"MCP 分析 {datetime.now().strftime('%m-%d %H:%M')}",
    source_type="scan", source_path=ab1_dir,
)
```

Also remove the now-unused imports at the top: `from db.database import init_db, get_session_factory` and `from db.models import Analysis, Sample, new_id` (keep only what's still needed for other tools like `get_sample_detail`).

- [ ] **Step 3: Update `frontend/pages/1_Analysis.py`**

Replace both save blocks (scan tab lines 65-105, upload tab lines 163-203) with:

```python
from backend.db.models import save_analysis_with_samples

analysis_id = save_analysis_with_samples(
    samples=samples, judgments=judgments,
    thresholds=thresholds,
    name=analysis_name or f"分析 {datetime.now().strftime('%m-%d %H:%M')}",
    source_type="scan", source_path=ab1_dir,
)
st.session_state["last_analysis_id"] = analysis_id
ok = sum(1 for j in judgments if j["status"] == "ok")
wrong = sum(1 for j in judgments if j["status"] == "wrong")
uncertain = sum(1 for j in judgments if j["status"] == "uncertain")
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add backend/api/analyze.py backend/mcp_server.py frontend/pages/1_Analysis.py
git commit -m "refactor: use db_session() context manager and shared save helper across all consumers"
```

---

### Task 5: Add threshold caching to rules engine

**Files:**
- Modify: `backend/core/rules.py`

Currently `load_thresholds()` reads YAML from disk on every call. Add a simple module-level cache with invalidation.

- [ ] **Step 1: Update `backend/core/rules.py`**

```python
_threshold_cache = None
_threshold_mtime = 0

def load_thresholds(config_path: Path = DEFAULT_CONFIG) -> dict:
    global _threshold_cache, _threshold_mtime
    try:
        mtime = config_path.stat().st_mtime
    except OSError:
        mtime = 0
    if _threshold_cache is not None and mtime == _threshold_mtime:
        return dict(_threshold_cache)  # return copy to prevent mutation
    with open(config_path, encoding="utf-8") as f:
        _threshold_cache = yaml.safe_load(f)["thresholds"]
        _threshold_mtime = mtime
    return dict(_threshold_cache)
```

- [ ] **Step 2: Run existing tests to verify caching doesn't break anything**

Run: `python -m pytest tests/test_rules.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add backend/core/rules.py
git commit -m "perf: cache threshold loading with mtime-based invalidation"
```

---

### Task 6: Use `db_session()` in all API endpoints and frontend pages

**Files:**
- Modify: `backend/api/results.py` (3 endpoints with sessions)
- Modify: `backend/api/export.py` (1 endpoint with session)
- Modify: `frontend/pages/2_Results.py`
- Modify: `frontend/pages/3_History.py`

Note: `backend/api/scan.py`, `backend/api/upload.py`, and `backend/api/config.py` do NOT use database sessions — no changes needed.

Replace all manual `Session = get_session_factory(); session = Session(); ... session.close()` patterns with `with db_session() as session:`.

Example for `backend/api/results.py`:

```python
from backend.db.database import db_session

@router.get("/results")
def list_analyses(limit: int = Query(20, le=100), offset: int = Query(0)):
    with db_session() as session:
        analyses = session.query(Analysis).order_by(Analysis.created_at.desc()).offset(offset).limit(limit).all()
        total = session.query(Analysis).count()
        return {"total": total, "items": [...]}
```

Apply same pattern to all other endpoints. For `frontend/pages/2_Results.py`, extract ORM data into plain dicts within the `with` block to avoid DetachedInstanceError (already done, just wrap in context manager).

- [ ] **Step 1: Update all API endpoints**

Apply `db_session()` to: `results.py` (3 endpoints), `export.py` (1), `analyze.py` (get endpoint).

- [ ] **Step 2: Update frontend pages**

Apply `db_session()` to: `2_Results.py`, `3_History.py`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add backend/api/ frontend/pages/
git commit -m "refactor: use db_session() context manager across all API endpoints and frontend pages"
```

---

### Task 7: Fix Docker deployment

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

Two issues:
1. `Dockerfile` runs both services with `&` — if one crashes the other doesn't know
2. `docker-compose.yml` mounts `bioagent-db:/app` which overwrites the entire app directory

- [ ] **Step 1: Fix `docker-compose.yml` volume**

```yaml
services:
  bioagent:
    build: .
    ports:
      - "8501:8501"
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./backend/rules_config.yaml:/app/backend/rules_config.yaml
      - bioagent-db:/app/db_data
    environment:
      - PYTHONPATH=/app
      - DATABASE_URL=sqlite:////app/db_data/bioagent.db

volumes:
  bioagent-db:
```

And update `backend/config.yaml` to support environment variable override, or update `database.py` to check env var first:

```python
import os
# In get_engine():
db_url = os.environ.get("DATABASE_URL") or cfg.get("database", {}).get("url", "sqlite:///./bioagent.db")
```

- [ ] **Step 2: Update Dockerfile to create db_data directory and add healthcheck**

```dockerfile
RUN mkdir -p data/uploads db_data
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/api/health || exit 1
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml backend/db/database.py
git commit -m "fix: Docker volume mount and add health check"
```

---

### Task 8: Add JSON parse error handling in frontend

**Files:**
- Modify: `frontend/pages/2_Results.py`

Line 193 does `json.loads(s["aa_changes"])` which can crash if data is corrupted.

- [ ] **Step 1: Add try-except for JSON parsing**

```python
if s["aa_changes"]:
    try:
        aa = json.loads(s["aa_changes"])
    except (json.JSONDecodeError, TypeError):
        aa = []
```

- [ ] **Step 2: Commit**

```bash
git add frontend/pages/2_Results.py
git commit -m "fix: handle corrupted JSON in sample aa_changes field"
```
