"""
api.py  —  WaferVision FastAPI Backend
=======================================
Single file that serves BOTH model_large and model_small.

Environment variables:
  WAFER_MODEL    "large" | "small"  (default: "large")
  DATABASE_URL   postgres or sqlite  (default: sqlite)

Usage:
  # Large model (primary, ResNet-50 + ViT-192):
  WAFER_MODEL=large uvicorn api:app --reload --port 8000

  # Small model (lighter, ResNet-18 + ViT-96):
  WAFER_MODEL=small uvicorn api:app --reload --port 8001

Place this file at the project root (same level as model_large/ and model_small/).
Place dashboard.html at the same level — it is served at GET /.

Endpoints:
  GET  /health   → liveness + active model name  (used by dashboard navbar)
  GET  /results  → live F1 JSON from results/from_scratch/comparison.md
  GET  /stats    → prediction counts per class
  GET  /history  → last N predictions from DB
  POST /predict  → image upload → EnsembleWaferPredictor → log → return result
  GET  /         → serves dashboard.html
"""

from __future__ import annotations

import re
import os
import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (all driven by env vars — no hardcoding)
# ─────────────────────────────────────────────────────────────────────────────

WAFER_MODEL: str = os.environ.get("WAFER_MODEL", "large").strip().lower()
if WAFER_MODEL not in ("large", "small"):
    raise ValueError(f"WAFER_MODEL must be 'large' or 'small', got: {WAFER_MODEL!r}")

_HERE         = Path(__file__).parent.resolve()          # project root
WAFER_ROOT    = _HERE / f"model_{WAFER_MODEL}"           # model_large/ or model_small/
CHECKPOINTS   = WAFER_ROOT / "checkpoints"
RESULTS_MD    = WAFER_ROOT / "results" / "from_scratch" / "comparison.md"
DASHBOARD_HTML = _HERE / "dashboard.html"

if not WAFER_ROOT.exists():
    raise RuntimeError(f"Model directory not found: {WAFER_ROOT}")

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "sqlite+aiosqlite:///./predictions.db"
)

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]

# Human-readable info shown in /health and dashboard navbar badge
MODEL_INFO = {
    "large": {"vit": "ViT-192 (4L 4H)", "resnet": "ResNet-50", "params": "~27.7M"},
    "small": {"vit": "ViT-96 (2L 2H)",  "resnet": "ResNet-18", "params": "~5.4M"},
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PARSER  (reads results/from_scratch/comparison.md off disk)
# ─────────────────────────────────────────────────────────────────────────────

def parse_comparison_report(path: Path) -> dict:
    """
    Parse results/from_scratch/comparison.md → structured JSON.

    Actual format written by evaluate_both.py:
      ## Overall (test set)
        | vit     | 0.5802 | 0.8319 |
        | resnet  | 0.7515 | 0.9557 |
        | ensemble| 0.7515 | 0.9557 |

      ## Per-class F1 (test)
        | Center | 0.4640 | 0.6338 | 0.6338 | +0.0000 ✓ |
        (cols: class | ViT | ResNet | Ensemble | delta)

      ## Val sweep
        | 1.00 | 0.9838 | 0.9947 | ←   (best row)

      ## Confusion matrix — ensemble (test)
        | **Center** | 655 | 20 | 3 | 0 | 28 | 0 | 2 | 5 | 119 |
    """
    text = path.read_text()

    # ── Per-model overall (test set) ──────────────────────────────
    def _model_row(name):
        m = re.search(r'\|\s*' + name + r'\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', text, re.IGNORECASE)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    vit_f1, vit_acc   = _model_row("vit")
    rsn_f1, rsn_acc   = _model_row("resnet")
    ens_f1, ens_acc   = _model_row("ensemble")

    result: dict = {
        "model":  WAFER_MODEL,
        # stage1 = ensemble (primary result)  stage2 = resnet (best single)
        "stage1": {"macro_f1": ens_f1, "accuracy": ens_acc, "per_class_f1": {}},
        "stage2": {"macro_f1": rsn_f1, "accuracy": rsn_acc, "per_class_f1": {}},
        "vit":    {"macro_f1": vit_f1, "accuracy": vit_acc, "per_class_f1": {}},
        "resnet": {"macro_f1": rsn_f1, "accuracy": rsn_acc, "per_class_f1": {}},
        "ensemble":{"macro_f1": ens_f1,"accuracy": ens_acc, "per_class_f1": {}},
        "val_best": {"macro_f1": None, "accuracy": None, "resnet_weight": None},
        "temperature": None,
        "gate_passed": True,
        "gate_failed_classes": [],
        "source": str(path.relative_to(_HERE)),
        "confusion_matrix": {},
        "test_distribution": {},
    }

    # ── Per-class F1  (class | ViT | ResNet | Ensemble | delta) ──
    for cls in CLASS_NAMES:
        pat = (r'\|\s*' + re.escape(cls) +
               r'\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|')
        m = re.search(pat, text)
        if m:
            vf, rf, ef = float(m.group(1)), float(m.group(2)), float(m.group(3))
            result["vit"]["per_class_f1"][cls]     = vf
            result["resnet"]["per_class_f1"][cls]  = rf
            result["ensemble"]["per_class_f1"][cls]= ef
            result["stage1"]["per_class_f1"][cls]  = ef   # ensemble → stage1
            result["stage2"]["per_class_f1"][cls]  = rf   # resnet   → stage2

    # ── Val sweep best row  (marked with ←) ──────────────────────
    val_m = re.search(r'\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*←', text)
    if val_m:
        result["val_best"] = {
            "resnet_weight": float(val_m.group(1)),
            "macro_f1":      float(val_m.group(2)),
            "accuracy":      float(val_m.group(3)),
        }

    # ── Temperature from checkpoints .npy ────────────────────────
    try:
        import numpy as _np
        for npy in ("temperature_vit.npy", "temperature_resnet.npy"):
            p = CHECKPOINTS / npy
            if p.exists():
                result["temperature"] = round(float(_np.load(str(p))[0]), 4)
                break
    except Exception:
        pass

    # ── Confusion matrix + test set distribution ──────────────────
    cm, dist = {}, {}
    for cls in CLASS_NAMES:
        pat = r'\|\s*\*\*' + re.escape(cls) + r'\*\*\s*\|((?:\s*\d+\s*\|)+)'
        m = re.search(pat, text)
        if m:
            nums = [int(x.strip()) for x in m.group(1).split('|') if x.strip().isdigit()]
            if len(nums) == len(CLASS_NAMES):
                cm[cls]   = nums
                dist[cls] = sum(nums)
    result["confusion_matrix"]  = cm
    result["test_distribution"] = dist

    return result

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

import databases
import sqlalchemy

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

predictions_table = sqlalchemy.Table(
    "predictions", metadata,
    sqlalchemy.Column("id",              sqlalchemy.Integer,  primary_key=True),
    sqlalchemy.Column("wafer_id",        sqlalchemy.String,   nullable=False),
    sqlalchemy.Column("model_variant",   sqlalchemy.String,   default=WAFER_MODEL),
    sqlalchemy.Column("predicted_class", sqlalchemy.String,   nullable=False),
    sqlalchemy.Column("class_idx",       sqlalchemy.Integer,  nullable=False),
    sqlalchemy.Column("confidence",      sqlalchemy.Float),
    sqlalchemy.Column("method",          sqlalchemy.String,   default="ensemble"),
    sqlalchemy.Column("vit_pred",        sqlalchemy.String),
    sqlalchemy.Column("resnet_pred",     sqlalchemy.String),
    sqlalchemy.Column("created_at",      sqlalchemy.DateTime, default=datetime.utcnow),
)


async def init_db() -> None:
    sync_url = DATABASE_URL.replace("+asyncpg", "").replace("+aiosqlite", "")
    engine = sqlalchemy.create_engine(sync_url)
    metadata.create_all(engine)
    await database.connect()


async def log_prediction(wafer_id: str, result: dict, method: str) -> None:
    await database.execute(predictions_table.insert().values(
        wafer_id        = wafer_id,
        model_variant   = WAFER_MODEL,
        predicted_class = result["class_name"],
        class_idx       = result["class_idx"],
        confidence      = result.get("confidence"),
        method          = method,
        vit_pred        = result.get("vit_pred"),
        resnet_pred     = result.get("resnet_pred"),
        created_at      = datetime.utcnow(),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# APP LIFESPAN  (model loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

predictor = None   # EnsembleWaferPredictor, set below


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    # Put model dir on sys.path so predict.py can import its sibling modules
    if str(WAFER_ROOT) not in sys.path:
        sys.path.insert(0, str(WAFER_ROOT))

    await init_db()

    vit_ckpt    = CHECKPOINTS / "vit_best.pth"
    resnet_ckpt = CHECKPOINTS / "resnet_best.pth"

    if vit_ckpt.exists() and resnet_ckpt.exists():
        try:
            # PyTorch 2.6 changed torch.load default to weights_only=True
            # which breaks checkpoints saved with older PyTorch.
            # Patch it before importing predict so the fix is automatic.
            import torch as _torch, functools as _functools
            if not getattr(_torch.load, "_patched_weights_only", False):
                _orig = _torch.load
                _torch.load = _functools.partial(_orig, weights_only=False)
                _torch.load._patched_weights_only = True

            from predict import EnsembleWaferPredictor
            predictor = EnsembleWaferPredictor(
                vit_ckpt    = vit_ckpt,
                resnet_ckpt = resnet_ckpt,
                ckpt_dir    = CHECKPOINTS,
            )
            info = MODEL_INFO[WAFER_MODEL]
            print(f"[startup] model_{WAFER_MODEL}: {info['vit']} + {info['resnet']}  loaded OK")
        except Exception as exc:
            print(f"[startup] Model load failed: {exc}  ->  /predict returns 503")
    else:
        missing = [f for f in ("vit_best.pth", "resnet_best.pth")
                   if not (CHECKPOINTS / f).exists()]
        print(f"[startup] Missing checkpoints: {missing}  ->  /predict unavailable")

    yield

    await database.disconnect()
    print("[shutdown] Bye.")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = f"WaferVision API [{WAFER_MODEL}]",
    description = "Semiconductor wafer defect classification — DSE 570 ASU",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Liveness check + active model info.
    Dashboard navbar fetches this to show the correct model name badge.
    """
    info = MODEL_INFO[WAFER_MODEL]
    return {
        "status":        "ok",
        "model_variant": WAFER_MODEL,              # "large" | "small"
        "vit_arch":      info["vit"],              # e.g. "ViT-192 (4L 4H)"
        "resnet_arch":   info["resnet"],           # e.g. "ResNet-50"
        "total_params":  info["params"],           # e.g. "~27.7M"
        "model_loaded":  predictor is not None,
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.get("/results")
async def get_results():
    """
    Parse results/from_scratch/comparison.md off disk → live F1 JSON.
    Frontend Performance page and KPI strip fetch this.
    Returns 404 if evaluate_both.py has not been run yet.
    """
    if not RESULTS_MD.exists():
        raise HTTPException(
            status_code = 404,
            detail      = (
                f"Results file not found: {RESULTS_MD}. "
                f"Run:  cd model_{WAFER_MODEL} && python evaluate_both.py"
            ),
        )
    try:
        return JSONResponse(parse_comparison_report(RESULTS_MD))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse error: {exc}")


@app.get("/stats")
async def get_stats():
    """
    Prediction counts per class from DB.
    Wires to the class distribution donut chart on the Overview page.
    """
    rows = await database.fetch_all(
        "SELECT predicted_class, COUNT(*) as count "
        "FROM predictions GROUP BY predicted_class"
    )
    total = await database.fetch_val("SELECT COUNT(*) FROM predictions")
    return {
        "total":    total or 0,
        "model":    WAFER_MODEL,
        "by_class": {r["predicted_class"]: r["count"] for r in rows},
    }


@app.get("/history")
async def get_history(limit: int = 20):
    """
    Last N predictions from DB.
    Wires to the Prediction Log page table.
    """
    rows = await database.fetch_all(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT :lim",
        values={"lim": limit},
    )
    records = []
    for r in rows:
        rec = dict(r)
        if isinstance(rec.get("created_at"), datetime):
            rec["created_at"] = rec["created_at"].isoformat()
        records.append(rec)
    return records


@app.post("/predict")
async def predict(
    file:   UploadFile = File(...),
    method: str        = Form(default="ensemble"),
):
    """
    Upload a wafer PNG/JPG → EnsembleWaferPredictor → log to DB → return result.

    Accepted method values:
      ensemble  (default) — ViT + ResNet weighted average
      direct              — alias for ensemble (legacy compat)
      centroid            — alias for ensemble (old method, no longer exists in predict.py)
      tta                 — ensemble + 4-rotation test-time augmentation
      vit                 — ViT model only
      resnet              — ResNet model only
    """
    if predictor is None:
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"Model not loaded. Add checkpoints to "
                f"model_{WAFER_MODEL}/checkpoints/ and restart the server."
            ),
        )

    import tempfile, uuid
    wafer_id = str(uuid.uuid4())[:12]
    suffix   = Path(file.filename).suffix if file.filename else ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        m = method.strip().lower()

        if m in ("ensemble", "direct", "centroid"):
            result      = predictor.predict_ensemble(tmp_path, use_tta=False)
            used_method = "ensemble"
        elif m == "tta":
            result      = predictor.predict_ensemble(tmp_path, use_tta=True)
            used_method = "tta"
        elif m == "vit":
            result      = predictor.predict_single(tmp_path, model_name="vit")
            used_method = "vit"
        elif m == "resnet":
            result      = predictor.predict_single(tmp_path, model_name="resnet")
            used_method = "resnet"
        else:
            raise HTTPException(
                status_code = 400,
                detail      = f"Unknown method '{method}'. Use: ensemble, tta, vit, resnet",
            )

        await log_prediction(wafer_id, result, used_method)

        # Normalise softmax_probs:
        #   predict_ensemble -> result["ensemble_probs"]  (numpy ndarray)
        #   predict_single   -> result["probs"]           (numpy ndarray)
        raw   = result.get("ensemble_probs", result.get("probs", []))
        probs = raw.tolist() if hasattr(raw, "tolist") else list(raw)

        return {
            "wafer_id":        wafer_id,
            "model_variant":   WAFER_MODEL,
            "predicted_class": result["class_name"],   # frontend key
            "class_idx":       result["class_idx"],
            "confidence":      result.get("confidence"),
            "method":          used_method,
            "softmax_probs":   probs,                  # list[float] len=9, frontend key
            "vit_pred":        result.get("vit_pred"),
            "resnet_pred":     result.get("resnet_pred"),
            "timestamp":       datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)



# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD TUNING  (reads results/threshold_tuning/ off disk)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_JSON = WAFER_ROOT / "results" / "threshold_tuning" / "chosen_thresholds.json"
SWEEP_CSV      = WAFER_ROOT / "results" / "threshold_tuning" / "sweep_results.csv"


@app.get("/threshold")
async def get_threshold():
    """
    Reads chosen_thresholds.json and sweep_results.csv off disk.
    Wires to the Threshold Tuning panel on the Performance page.
    Returns 404 if threshold_tuning/ has not been run yet.
    """
    import json, csv as csv_mod
    out: dict = {
        "chosen":    None,
        "sweep":     [],
        "source":    str(THRESHOLD_JSON.relative_to(_HERE)) if THRESHOLD_JSON.exists() else None,
    }
    if THRESHOLD_JSON.exists():
        out["chosen"] = json.loads(THRESHOLD_JSON.read_text())
    if SWEEP_CSV.exists():
        rows = []
        with open(SWEEP_CSV, newline="") as f:
            for row in csv_mod.DictReader(f):
                rows.append({k: float(v) for k, v in row.items()})
        # Sort by precision desc; return all for frontend scatter
        rows.sort(key=lambda r: r["precision"], reverse=True)
        out["sweep"] = rows
    if not out["chosen"] and not out["sweep"]:
        raise HTTPException(
            status_code=404,
            detail=f"Threshold files not found in {THRESHOLD_JSON.parent}. Run: python tune_thresholds.py",
        )
    return out

# ─────────────────────────────────────────────────────────────────────────────
# STATIC  (serves dashboard.html at root — keeps fetch() URLs relative)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    """Serve dashboard.html from the project root."""
    if not DASHBOARD_HTML.exists():
        raise HTTPException(
            status_code = 404,
            detail      = "dashboard.html not found next to api.py",
        )
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)