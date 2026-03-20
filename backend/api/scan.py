from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ScanRequest(BaseModel):
    directory: str

@router.post("/scan")
def scan_directory(req: ScanRequest):
    base = Path(req.directory)
    if not base.exists():
        raise HTTPException(404, f"Directory not found: {req.directory}")
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
