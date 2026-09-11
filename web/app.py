"""FastAPI wrapper that exposes TradingAgents as an HTTP service.

This is the deployment surface for Render (or any container host).
The original TradingAgents package is a CLI/library; this module wraps
``TradingAgentsGraph`` so it can be invoked over HTTP and bound to a port.

Endpoints
---------
GET  /                -> service info (no LLM call)
GET  /health          -> lightweight liveness probe (no LLM call)
GET  /config          -> echo the active (non-secret) provider/model config
POST /smoke           -> single LLM call to verify the NVIDIA key works (~5s)
POST /analyze         -> sync full analysis (will hit Render's 5-min timeout for slow models)
POST /analyze-async   -> enqueue a full analysis, returns job_id immediately
GET  /jobs/{job_id}   -> poll a background job's status + result

Notes
-----
- The real LLM key is read from $NVIDIA_API_KEY at request time, never logged.
- The sync /analyze endpoint will time out on Render's starter plan if the
  full multi-agent pipeline takes >5 min (Kimi-K3 with max reasoning effort
  can easily hit this). Use /analyze-async + /jobs/{id} for full runs.
- A single global ``TradingAgentsGraph`` instance is created on startup so the
  LangGraph state machine is built once and reused across requests.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load .env early so DEFAULT_CONFIG picks up the TRADINGAGENTS_* vars.
load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.llm_clients import create_llm_client  # noqa: E402

logger = logging.getLogger("tradingagents.web")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# A single graph instance shared across requests. Built once on startup.
_graph: TradingAgentsGraph | None = None
_graph_lock = threading.Lock()

# In-memory job store for /analyze-async + /jobs/{id}.
# NOTE: this is per-process; Render's starter plan runs one instance, so it's
# fine for our purposes. For multi-instance setups you'd swap this for Redis.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _build_graph() -> TradingAgentsGraph:
    """Construct the TradingAgentsGraph using DEFAULT_CONFIG (env-driven)."""
    config = DEFAULT_CONFIG.copy()
    return TradingAgentsGraph(debug=False, config=config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph once at startup; reuse across requests."""
    global _graph
    try:
        with _graph_lock:
            _graph = _build_graph()
        logger.info(
            "TradingAgentsGraph ready | provider=%s model=%s",
            DEFAULT_CONFIG.get("llm_provider"),
            DEFAULT_CONFIG.get("deep_think_llm"),
        )
    except Exception:
        logger.exception("Failed to build TradingAgentsGraph at startup")
        _graph = None
    yield
    _graph = None


app = FastAPI(
    title="TradingAgents API",
    description="Multi-agent LLM financial trading framework — NVIDIA NIM backend",
    version="0.4.1",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., examples=["NVDA", "AAPL", "TSLA"])
    date: str = Field(
        ...,
        description="Analysis date in YYYY-MM-DD format.",
        examples=["2024-05-10"],
    )


class AnalyzeResponse(BaseModel):
    ticker: str
    date: str
    decision: Any | None
    state: dict[str, Any] | None
    elapsed_seconds: float
    ok: bool
    error: str | None = None


class SmokeResponse(BaseModel):
    ok: bool
    provider: str
    model: str
    response_preview: str
    elapsed_seconds: float
    error: str | None = None


class JobAccepted(BaseModel):
    job_id: str
    ticker: str
    date: str
    status: str
    submitted_at: str


class JobStatus(BaseModel):
    job_id: str
    ticker: str
    date: str
    status: str  # queued | running | done | error
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    decision: Any | None = None
    state: dict[str, Any] | None = None
    error: str | None = None


class ConfigResponse(BaseModel):
    llm_provider: str
    deep_think_llm: str
    quick_think_llm: str
    backend_url: str | None
    output_language: str
    max_debate_rounds: int
    max_risk_discuss_rounds: int
    nvidia_api_key_configured: bool


# --------------------------------------------------------------------------- #
# Routes — info / health / config
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict[str, Any]:
    """Service info — does NOT call the LLM."""
    return {
        "service": "TradingAgents API",
        "version": "0.4.1",
        "status": "ok",
        "llm_provider": DEFAULT_CONFIG.get("llm_provider"),
        "deep_think_llm": DEFAULT_CONFIG.get("deep_think_llm"),
        "quick_think_llm": DEFAULT_CONFIG.get("quick_think_llm"),
        "endpoints": [
            "GET  /",
            "GET  /health",
            "GET  /config",
            "POST /smoke           (single LLM call, ~5s)",
            "POST /analyze         (sync; may time out on slow models)",
            "POST /analyze-async   (returns job_id immediately)",
            "GET  /jobs/{job_id}   (poll a background job)",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — Render will hit this to decide if the service is up."""
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    """Echo the active (non-secret) LLM configuration."""
    return ConfigResponse(
        llm_provider=DEFAULT_CONFIG.get("llm_provider", ""),
        deep_think_llm=DEFAULT_CONFIG.get("deep_think_llm", ""),
        quick_think_llm=DEFAULT_CONFIG.get("quick_think_llm", ""),
        backend_url=DEFAULT_CONFIG.get("backend_url"),
        output_language=DEFAULT_CONFIG.get("output_language", "English"),
        max_debate_rounds=DEFAULT_CONFIG.get("max_debate_rounds", 1),
        max_risk_discuss_rounds=DEFAULT_CONFIG.get("max_risk_discuss_rounds", 1),
        nvidia_api_key_configured=bool(os.environ.get("NVIDIA_API_KEY")),
    )


# --------------------------------------------------------------------------- #
# /smoke — single LLM call to verify NVIDIA key works (fast, ~5s)
# --------------------------------------------------------------------------- #
@app.post("/smoke", response_model=SmokeResponse)
def smoke() -> SmokeResponse:
    """Make a single LLM call to verify the NVIDIA provider + key work end-to-end.

    Does NOT run the full multi-agent pipeline; just one quick chat completion.
    Use this to verify your NVIDIA_API_KEY is valid before launching a full run.
    """
    provider = DEFAULT_CONFIG.get("llm_provider", "nvidia")
    model = DEFAULT_CONFIG.get("quick_think_llm", "moonshotai/kimi-k3")
    started = datetime.now()
    try:
        client = create_llm_client(provider=provider, model=model)
        llm = client.get_llm()
        # Tiny prompt — keeps latency low and token cost negligible.
        result = llm.invoke("Reply with exactly one word: PONG")
        text = getattr(result, "content", str(result))
        if isinstance(text, list):  # some providers return content blocks
            text = " ".join(str(b) for b in text)
        elapsed = (datetime.now() - started).total_seconds()
        return SmokeResponse(
            ok=True,
            provider=provider,
            model=model,
            response_preview=str(text)[:200],
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = (datetime.now() - started).total_seconds()
        logger.exception("Smoke test failed")
        return SmokeResponse(
            ok=False,
            provider=provider,
            model=model,
            response_preview="",
            elapsed_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


# --------------------------------------------------------------------------- #
# /analyze (sync) — kept for compatibility; will time out on slow models
# --------------------------------------------------------------------------- #
def _run_analysis(req: AnalyzeRequest) -> AnalyzeResponse:
    """Build the graph (if needed) and run one propagation."""
    global _graph
    if _graph is None:
        with _graph_lock:
            _graph = _build_graph()

    started = datetime.now()
    try:
        final_state, decision = _graph.propagate(req.ticker, req.date)
        elapsed = (datetime.now() - started).total_seconds()
        return AnalyzeResponse(
            ticker=req.ticker,
            date=req.date,
            decision=decision,
            state=final_state if isinstance(final_state, dict) else None,
            elapsed_seconds=elapsed,
            ok=True,
        )
    except Exception as exc:
        logger.exception("Analysis failed for %s %s", req.ticker, req.date)
        elapsed = (datetime.now() - started).total_seconds()
        return AnalyzeResponse(
            ticker=req.ticker,
            date=req.date,
            decision=None,
            state=None,
            elapsed_seconds=elapsed,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Run a full TradingAgents analysis synchronously.

    WARNING: this can take 5-15 min with deep-thinking models on slow providers
    and will hit Render's 5-min gateway timeout. Use /analyze-async for real runs.
    """
    if not _graph:
        raise HTTPException(
            status_code=503,
            detail="TradingAgentsGraph is not initialized. Check NVIDIA_API_KEY and server logs.",
        )
    return _run_analysis(req)


@app.post("/propagate", response_model=AnalyzeResponse)
def propagate(req: AnalyzeRequest) -> AnalyzeResponse:
    """Alias of /analyze — matches the underlying TradingAgentsGraph.propagate() name."""
    return analyze(req)


# --------------------------------------------------------------------------- #
# /analyze-async + /jobs/{id} — async job pattern (recommended)
# --------------------------------------------------------------------------- #
def _run_job(job_id: str, ticker: str, date: str) -> None:
    """Worker function executed in a background thread."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()

    global _graph
    try:
        if _graph is None:
            with _graph_lock:
                _graph = _build_graph()

        started = datetime.now()
        final_state, decision = _graph.propagate(ticker, date)
        elapsed = (datetime.now() - started).total_seconds()

        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = "done"
            job["finished_at"] = datetime.now().isoformat()
            job["elapsed_seconds"] = elapsed
            job["decision"] = decision
            job["state"] = final_state if isinstance(final_state, dict) else None
    except Exception as exc:
        logger.exception("Async job %s failed", job_id)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = "error"
            job["finished_at"] = datetime.now().isoformat()
            job["elapsed_seconds"] = (datetime.now() - datetime.fromisoformat(job["started_at"])).total_seconds()
            job["error"] = f"{type(exc).__name__}: {exc}"


@app.post("/analyze-async", response_model=JobAccepted, status_code=202)
def analyze_async(req: AnalyzeRequest) -> JobAccepted:
    """Enqueue a full TradingAgents analysis; returns a job_id immediately.

    Poll the status with GET /jobs/{job_id}. Avoids Render's 5-min request
    timeout by running the analysis in a background thread.
    """
    if not _graph:
        raise HTTPException(
            status_code=503,
            detail="TradingAgentsGraph is not initialized. Check NVIDIA_API_KEY and server logs.",
        )
    job_id = uuid.uuid4().hex[:12]
    submitted_at = datetime.now().isoformat()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "ticker": req.ticker,
            "date": req.date,
            "status": "queued",
            "submitted_at": submitted_at,
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": None,
            "decision": None,
            "state": None,
            "error": None,
        }
    # Spawn the worker; daemon=True so it dies with the process.
    threading.Thread(target=_run_job, args=(job_id, req.ticker, req.date), daemon=True).start()
    return JobAccepted(
        job_id=job_id,
        ticker=req.ticker,
        date=req.date,
        status="queued",
        submitted_at=submitted_at,
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    """Poll the status of an async analysis job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return JobStatus(**job)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("RELOAD", "0") == "1",
    )
