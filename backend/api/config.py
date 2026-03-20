import yaml
from fastapi import APIRouter
from pydantic import BaseModel
from backend.core.rules import load_thresholds, DEFAULT_CONFIG

router = APIRouter()

@router.get("/config")
def get_thresholds():
    return {"thresholds": load_thresholds()}

class ThresholdUpdate(BaseModel):
    thresholds: dict

@router.put("/config")
def update_thresholds(req: ThresholdUpdate):
    current = load_thresholds()
    current.update(req.thresholds)
    with open(DEFAULT_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump({"thresholds": current}, f, allow_unicode=True, default_flow_style=False)
    return {"thresholds": current}
