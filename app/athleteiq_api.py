"""
ATHLETEIQ – Main FastAPI Application (v2)
==========================================
Mounts both the original CRUD routes (kept intact) and the new
ATHLETEIQ API v1 routes used by the Next.js frontend.

Start with:
    uvicorn app.athleteiq_api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dashboard    import router as dashboard_router
from app.api.athletes     import router as athletes_router
from app.api.predictions  import router as predictions_router
from app.api.model_insights import router as model_insights_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATHLETEIQ API",
    description=(
        "Athlete Injury-Risk Prediction Platform · "
        "Transformer Encoder + Focal Loss (ROC-AUC 0.9191)"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Next.js dev server (port 3000) and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: pre-load model so the first prediction is fast
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=" * 60)
    logger.info("  ATHLETEIQ API v2  starting up …")
    logger.info("=" * 60)
    try:
        from app.inference.model_loader import load_model
        load_model()
        logger.info("  [startup] Transformer model loaded ✓")
    except Exception as exc:
        logger.warning(f"  [startup] Model could not be loaded: {exc}")
        logger.warning("  [startup] Prediction endpoints will attempt reload on first request.")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(dashboard_router)
app.include_router(athletes_router)
app.include_router(predictions_router)
app.include_router(model_insights_router)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    from app.inference.model_loader import is_loaded
    return {
        "status":       "ok",
        "model_loaded": is_loaded(),
        "version":      "2.0.0",
    }


@app.get("/")
def root() -> dict:
    return {
        "name":    "ATHLETEIQ API",
        "version": "2.0.0",
        "docs":    "/docs",
    }
