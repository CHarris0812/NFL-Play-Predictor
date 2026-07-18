"""Minimal API for the React frontend: predict from saved models, retrain on demand.

Run with: uvicorn serving.app:app --reload --port 8000
"""

import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.persistence import load_xgb_model
from models.predict import predict

REPO_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="NFL Play Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Situation(BaseModel):
    down: int
    ydstogo: int
    yardline_100: int
    score_differential: int
    qtr: int
    half_seconds_remaining: int
    posteam_timeouts_remaining: int
    defteam_timeouts_remaining: int
    is_home: bool


@app.get("/status")
def status():
    return {
        "play_type_trained": (REPO_ROOT / "artifacts" / "play_type.xgb.json").exists(),
        "outcome_trained": (REPO_ROOT / "artifacts" / "outcome.xgb.json").exists(),
    }


@app.post("/predict")
def predict_situation(situation: Situation):
    features = situation.model_dump()
    features["is_home"] = int(features["is_home"])

    try:
        play_type_model = load_xgb_model("play_type")
        outcome_model = load_xgb_model("outcome")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail="No trained model found - click Train models first."
        ) from e

    return {
        "play_type": predict(play_type_model, features),
        "outcome": predict(outcome_model, features),
    }


def _run_script(script_name: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stdout[-2000:] + result.stderr[-2000:])
    return result.stdout


@app.post("/train")
def train():
    play_type_log = _run_script("train_baseline.py")
    outcome_log = _run_script("train_outcome_baseline.py")
    return {"play_type_log": play_type_log, "outcome_log": outcome_log}
