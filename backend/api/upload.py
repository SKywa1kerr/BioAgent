from pathlib import Path
from fastapi import APIRouter, UploadFile, HTTPException
from backend.config import get_config

router = APIRouter()
ALLOWED_EXTENSIONS = {".ab1", ".gb", ".gbk"}
MAX_SIZE = 10 * 1024 * 1024

@router.post("/upload")
async def upload_files(files: list[UploadFile]):
    config = get_config()
    upload_dir = Path(config["data"]["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        content = await f.read()
        if len(content) > MAX_SIZE:
            raise HTTPException(400, f"File too large: {f.filename}")
        dest = upload_dir / f.filename
        dest.write_bytes(content)
        saved.append({"filename": f.filename, "path": str(dest), "size": len(content)})
    return {"uploaded": saved}
