"""
Fraud-scoring API (demo deployment).

On startup it loads the `champion` model from the MLflow registry, scores every account in
the graph once (src/predict.py) and serves the results from memory. After a new model is
promoted, POST /reload picks it up without restarting the container.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import os
import threading
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from src.predict import score_accounts

GRAPH_PATH = os.getenv("GRAPH_PATH", "data/processed/graph.pt")
_state = {}
_lock = threading.Lock()


def _load():
    scores = score_accounts(GRAPH_PATH)
    with _lock:
        _state.clear()
        _state.update(scores)
    return scores["model"]


@asynccontextmanager
async def lifespan(app):
    _load()
    yield


app = FastAPI(title="GNN Fraud Detection API",
              description="Serves fraud scores from the MLflow `champion` model (T-Finance demo).",
              lifespan=lifespan)


def _label_status(i):
    if not _state["known_label"][i]:
        return "unlabeled"
    return "known_fraud" if _state["labels"][i] == 1 else "known_normal"


def _account(i):
    return {"account_id": int(i),
            "fraud_probability": round(float(_state["probs"][i]), 6),
            "alert": bool(_state["alert"][i]),
            "label_status": _label_status(i)}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _state["model"]["version"]}


@app.get("/model")
def model_info():
    """The serving model version, its alert threshold and offline metrics."""
    return {**_state["model"], "num_accounts": len(_state["probs"]),
            "num_alerts": int(_state["alert"].sum()), "scored_in_s": _state["scored_in_s"]}


@app.get("/accounts/{account_id}")
def account(account_id: int):
    if not 0 <= account_id < len(_state["probs"]):
        raise HTTPException(404, f"account_id must be in [0, {len(_state['probs']) - 1}]")
    return _account(account_id)


@app.get("/alerts")
def alerts(limit: int = Query(50, ge=1, le=1000)):
    """Highest-risk unlabeled accounts at or above the alert threshold."""
    idx = np.flatnonzero(_state["alert"])
    top = idx[np.argsort(-_state["probs"][idx])][:limit]
    return {"threshold": _state["model"]["alert_threshold"], "total_alerts": int(len(idx)),
            "alerts": [_account(i) for i in top]}


@app.post("/reload")
def reload():
    """Re-resolve the champion alias and re-score (use after a new model is promoted)."""
    old = _state["model"]["version"]
    new = _load()
    return {"previous_version": old, "serving_version": new["version"]}
