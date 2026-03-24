from pathlib import Path
from fastapi import APIRouter, UploadFile, HTTPException
from backend.config import get_config

router = APIRouter()
ALLOWED_EXTENSIONS = {".ab1", ".gb", ".gbk"}
MAX_SIZE = 10 * 1024 * 1024


def _sanitize_filename(filename: str) -> str:
    """Strip path components to prevent directory traversal."""
    safe = Path(filename).name
    if not safe or safe.startswith("."):
        raise HTTPException(400, f"Invalid filename: {filename}")
    return safe


@router.post("/upload")
async def upload_files(files: list[UploadFile]):
    config = get_config()
    upload_dir = Path(config["data"]["upload_dir"]).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        safe_name = _sanitize_filename(f.filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        content = await f.read()
        if len(content) > MAX_SIZE:
            raise HTTPException(400, f"File too large: {safe_name}")
        dest = upload_dir / safe_name
        if not dest.resolve().is_relative_to(upload_dir):
            raise HTTPException(400, f"Invalid filename: {safe_name}")
        dest.write_bytes(content)
        saved.append({"filename": safe_name, "path": str(dest), "size": len(content)})
    return {"uploaded": saved}
