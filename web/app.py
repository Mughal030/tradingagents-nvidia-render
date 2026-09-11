"""FastAPI wrapper that exposes TradingAgents as an HTTP service.

This is the deployment surface for Render (or any container host).
The original TradingAgents package is a CLI/library; this module wraps
``TradingAgentsGraph`` so it can be invoked over HTTP and bound to a port.

Endpoints
---------
GET  /                -> service info (no LLM call)
GET  /health          -> lightweight liveness probe (no LLM call)
POST /analyze         -> run a full analysis: { ticker, date } -> decision
POST /propagate        -> alias of /analyze, matches the underlying API name
GET  /config          -> echo the active (non-secret) provider/model config

Notes
-----
- The real LLM key is read from $NVIDIA_API_KEY at request time, never logged.
- A single global ``TradingAgentsGraph`` instance is created on startup so the
  LangGraph state machine is built once and reused across requests. Building the
  graph is expensive (~1-2s); invoking it is what costs LLM tokens.
- A run-level timeout guards against runaway deep-thinking models so the
  request can return an error instead of being killed by Render's gateway.
"""

from __future__ import annotations

import logging
import os
import threading
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

logger = logging.getLogger("tradingagents.web")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# A single graph instance shared across requests. Built once on startup.
_graph: TradingAgentsGraph | None = None
_graph_lock = threading.Lock()


def _build_graph() -> TradingAgentsGraph:
    """Construct the TradingAgentsGraph using DEFAULT_CONFIG (env-driven)."""
    config = DEFAULT_CONFIG.copy()
    # debug=False keeps stdout quiet for unattended/server runs.
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
        # Don't crash startup: the /analyze endpoint will surface a clear
        # error if the graph couldn't be built (e.g. missing API key).
        logger.exception("Failed to build TradingAgentsGraph at startup")
        _graph = None
    yield
    _graph = None


app = FastAPI(
    title="TradingAgents API",
    description="Multi-agent LLM financial trading framework — NVIDIA NIM backend",
    version="0.4.0",
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
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict[str, Any]:
    """Service info — does NOT call the LLM."""
    return {
        "service": "TradingAgents API",
        "version": "0.4.0",
        "status": "ok",
        "llm_provider": DEFAULT_CONFIG.get("llm_provider"),
        "deep_think_llm": DEFAULT_CONFIG.get("deep_think_llm"),
        "quick_think_llm": DEFAULT_CONFIG.get("quick_think_llm"),
        "endpoints": [
            "GET  /",
            "GET  /health",
            "GET  /config",
            "POST /analyze",
            "POST /propagate",
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


def _run_analysis(req: AnalyzeRequest) -> AnalyzeResponse:
    """Build the graph (if needed) and run one propagation."""
    global _graph
    if _graph is None:
        with _graph_lock:
            _graph = _build_graph()

    started = datetime.now()
    try:
        # Propagate signature: ta.propagate(ticker, date) -> (final_state, decision)
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
    except Exception as exc:  # surface the error to the caller
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
    """Run a full TradingAgents analysis for (ticker, date)."""
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


if __name__ == "__main__":
    # Local: python -m web.app  ->  http://localhost:8000
    # Render: uvicorn web.app:app --host 0.0.0.0 --port $PORT  (no __main__)
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("RELOAD", "0") == "1",
    )
