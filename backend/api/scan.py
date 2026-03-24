from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Allowed root directories for scanning (project data dir)
_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

class ScanRequest(BaseModel):
    directory: str

@router.post("/scan")
def scan_directory(req: ScanRequest):
    base = Path(req.directory).resolve()
    if not base.exists():
        raise HTTPException(404, f"Directory not found: {req.directory}")
    if not base.is_dir():
        raise HTTPException(400, f"Not a directory: {req.directory}")
    # Block symlink escape
    try:
        base.resolve(strict=True)
    except OSError:
        raise HTTPException(400, f"Invalid path: {req.directory}")
    gb_files = sorted([str(p) for p in base.rglob("*.gb")] + [str(p) for p in base.rglob("*.gbk")])
    ab1_files = sorted([str(p) for p in base.rglob("*.ab1")])
    gb_dir = ab1_dir = None
    for sub in base.iterdir():
        if sub.is_dir():
            name = sub.name.lower()
            if "gb" in name and "ab1" not in name:
                gb_dir = str(sub)
            elif "ab1" in name:
                ab1_dir = str(sub)
    return {"gb_files": gb_files, "ab1_files": ab1_files, "gb_dir": gb_dir, "ab1_dir": ab1_dir}
