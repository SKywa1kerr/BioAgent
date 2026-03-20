import logging
from fastapi import FastAPI
from backend.db.database import init_db
from backend.api import analyze, scan, upload, results, config, export

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="BioAgent MAX", version="0.1.0")

app.include_router(analyze.router, prefix="/api")
app.include_router(scan.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(results.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(export.router, prefix="/api")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/api/health")
def health():
    return {"status": "ok"}
