"""FastAPI wrapper that exposes TradingAgents as an HTTP service.

Multi-provider LLM deployment surface for Render. Picks the best provider
at request time from a configurable list, with automatic failover when a
provider returns 429 / auth errors / 5xx.

Endpoints
---------
GET  /                       -> service info (no LLM call)
GET  /health                 -> liveness probe
GET  /config                 -> echo active LLM configuration
GET  /providers              -> list all configured LLM providers + their status
POST /smoke                  -> single LLM call per provider, verifies each key works
POST /analyze-async          -> enqueue a full TradingAgents run (returns job_id)
GET  /jobs/{job_id}          -> poll a background job's status

Provider configuration
----------------------
Driven entirely by environment variables. The app picks the first
provider in the list whose key is configured and that responds successfully
to a smoke call; on failure it transparently falls back to the next.

  LLM_PROVIDERS=openrouter,nvidia    # comma-separated priority list (required)
  OPENROUTER_API_KEY=sk-or-...      # OpenRouter key
  OPENROUTER_DEEP_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
  OPENROUTER_QUICK_MODEL=nvidia/nemotron-3.5-lightning:free
  NVIDIA_API_KEY=nvapi-...          # NVIDIA NIM key (fallback)
  NVIDIA_DEEP_MODEL=moonshotai/kimi-k3
  NVIDIA_QUICK_MODEL=moonshotai/kimi-k3

Notes
-----
- A single TradingAgentsGraph instance per provider, built lazily on first use.
- /analyze-async runs in a background thread to bypass Render's 5-min gateway
  timeout. Poll /jobs/{job_id} for the result.
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

load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.llm_clients import create_llm_client  # noqa: E402

logger = logging.getLogger("tradingagents.web")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# --------------------------------------------------------------------------- #
# Provider registry — populated from env vars
# --------------------------------------------------------------------------- #
class ProviderConfig:
    """One LLM provider's full configuration."""

    def __init__(self, name: str, deep_model: str, quick_model: str, api_key_env: str):
        self.name = name
        self.deep_model = deep_model
        self.quick_model = quick_model
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "").strip()
        # Cached graph (lazy-built on first successful smoke test).
        self.graph: TradingAgentsGraph | None = None
        self.smoke_ok: bool | None = None  # None = not tested yet
        self.last_error: str | None = None
        self.last_success_at: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "REPLACE_WITH_YOUR_NVIDIA_API_KEY"


def _build_providers() -> list[ProviderConfig]:
    """Read provider config from env, return list in priority order."""
    order = [
        p.strip() for p in os.environ.get(
            "LLM_PROVIDERS", "openrouter,nvidia"
        ).split(",") if p.strip()
    ]
    defaults = {
        "openrouter": (
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "nvidia/nemotron-3.5-lightning:free",
            "OPENROUTER_API_KEY",
        ),
        "nvidia": (
            "moonshotai/kimi-k3",
            "moonshotai/kimi-k3",
            "NVIDIA_API_KEY",
        ),
    }
    providers = []
    for name in order:
        if name not in defaults:
            logger.warning("Unknown provider in LLM_PROVIDERS: %s — skipping", name)
            continue
        deep, quick, key_env = defaults[name]
        # Allow env-var overrides for the models.
        deep = os.environ.get(f"{name.upper()}_DEEP_MODEL", deep)
        quick = os.environ.get(f"{name.upper()}_QUICK_MODEL", quick)
        providers.append(ProviderConfig(name, deep, quick, key_env))
    return providers


_PROVIDERS: list[ProviderConfig] = _build_providers()
_providers_lock = threading.Lock()

# In-memory job store for /analyze-async.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _set_default_config_for(provider: ProviderConfig) -> None:
    """Mutate the module-level DEFAULT_CONFIG so TradingAgentsGraph picks it up.

    TradingAgentsGraph reads DEFAULT_CONFIG at construction time, so we have to
    patch the dict before constructing each provider's graph.
    """
    DEFAULT_CONFIG["llm_provider"] = provider.name
    DEFAULT_CONFIG["deep_think_llm"] = provider.deep_model
    DEFAULT_CONFIG["quick_think_llm"] = provider.quick_model


def _build_graph_for(provider: ProviderConfig) -> TradingAgentsGraph:
    """Construct a TradingAgentsGraph bound to a specific provider."""
    _set_default_config_for(provider)
    cfg = DEFAULT_CONFIG.copy()
    return TradingAgentsGraph(debug=False, config=cfg)


def _smoke_provider(provider: ProviderConfig) -> tuple[bool, str | None]:
    """Run a single LLM call against `provider`; returns (ok, error)."""
    try:
        _set_default_config_for(provider)
        client = create_llm_client(
            provider=provider.name, model=provider.quick_model
        )
        llm = client.get_llm()
        result = llm.invoke("Reply with exactly one word: PONG")
        text = getattr(result, "content", str(result))
        if isinstance(text, list):
            text = " ".join(str(b) for b in text)
        text = str(text).strip()
        ok = "pong" in text.lower()
        return ok, None if ok else f"unexpected response: {text[:80]!r}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-build graphs for all configured providers."""
    for p in _PROVIDERS:
        if not p.configured:
            logger.warning("Provider %s: API key not set, skipping", p.name)
            continue
        try:
            p.graph = _build_graph_for(p)
            logger.info(
                "Provider %s ready: deep=%s quick=%s",
                p.name, p.deep_model, p.quick_model,
            )
        except Exception:
            logger.exception("Failed to build graph for provider %s", p.name)
            p.graph = None
    yield


app = FastAPI(
    title="TradingAgents API",
    description="Multi-agent LLM financial trading framework — multi-provider",
    version="0.5.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., examples=["NVDA", "AAPL", "TSLA"])
    date: str = Field(..., examples=["2024-05-10"])


class AnalyzeResponse(BaseModel):
    ticker: str
    date: str
    provider: str | None = None
    decision: Any | None = None
    state: dict[str, Any] | None = None
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


class ProviderStatus(BaseModel):
    name: str
    deep_model: str
    quick_model: str
    configured: bool
    graph_ready: bool
    smoke_ok: bool | None
    last_error: str | None
    last_success_at: str | None


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
    status: str
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    provider: str | None = None
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
    providers_configured: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pick_provider() -> ProviderConfig | None:
    """Return the first provider that's configured and has a working graph."""
    for p in _PROVIDERS:
        if p.configured and p.graph is not None:
            return p
    return None


# --------------------------------------------------------------------------- #
# Routes — info / health / config
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "TradingAgents API",
        "version": "0.5.0",
        "status": "ok",
        "providers": [p.name for p in _PROVIDERS if p.configured],
        "endpoints": [
            "GET  /",
            "GET  /health",
            "GET  /config",
            "GET  /providers",
            "POST /smoke           (test all providers, ~5s each)",
            "POST /analyze-async   (returns job_id immediately)",
            "GET  /jobs/{job_id}",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    p = _pick_provider()
    return ConfigResponse(
        llm_provider=p.name if p else "(none)",
        deep_think_llm=p.deep_model if p else "",
        quick_think_llm=p.quick_model if p else "",
        backend_url=DEFAULT_CONFIG.get("backend_url"),
        output_language=DEFAULT_CONFIG.get("output_language", "English"),
        max_debate_rounds=DEFAULT_CONFIG.get("max_debate_rounds", 1),
        max_risk_discuss_rounds=DEFAULT_CONFIG.get("max_risk_discuss_rounds", 1),
        providers_configured=sum(1 for p in _PROVIDERS if p.configured),
    )


@app.get("/providers", response_model=list[ProviderStatus])
def providers() -> list[ProviderStatus]:
    return [
        ProviderStatus(
            name=p.name,
            deep_model=p.deep_model,
            quick_model=p.quick_model,
            configured=p.configured,
            graph_ready=p.graph is not None,
            smoke_ok=p.smoke_ok,
            last_error=p.last_error,
            last_success_at=p.last_success_at,
        )
        for p in _PROVIDERS
    ]


# --------------------------------------------------------------------------- #
# /smoke — test every provider in parallel-safe sequence
# --------------------------------------------------------------------------- #
@app.post("/smoke", response_model=list[SmokeResponse])
def smoke_all() -> list[SmokeResponse]:
    """Test every configured provider with a single LLM call."""
    results: list[SmokeResponse] = []
    for p in _PROVIDERS:
        if not p.configured:
            results.append(SmokeResponse(
                ok=False, provider=p.name, model=p.quick_model,
                response_preview="", elapsed_seconds=0,
                error="API key not configured",
            ))
            continue
        started = datetime.now()
        ok, err = _smoke_provider(p)
        elapsed = (datetime.now() - started).total_seconds()
        p.smoke_ok = ok
        if ok:
            p.last_error = None
            p.last_success_at = datetime.now().isoformat()
        else:
            p.last_error = err
        results.append(SmokeResponse(
            ok=ok, provider=p.name, model=p.quick_model,
            response_preview="PONG" if ok else "",
            elapsed_seconds=elapsed, error=err,
        ))
    return results


# --------------------------------------------------------------------------- #
# /analyze-async + /jobs/{id}
# --------------------------------------------------------------------------- #
def _run_job(job_id: str, ticker: str, date: str) -> None:
    """Background worker: try each provider in priority order until one works."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()

    errors: list[str] = []
    for p in _PROVIDERS:
        if not p.configured or p.graph is None:
            continue
        logger.info("Job %s: trying provider %s", job_id, p.name)
        started = datetime.now()
        try:
            _set_default_config_for(p)
            # Rebuild graph bound to this provider (cheap; reuses the cached one
            # but resets the LLM clients inside).
            graph = _build_graph_for(p)
            final_state, decision = graph.propagate(ticker, date)
            elapsed = (datetime.now() - started).total_seconds()
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "done"
                job["provider"] = p.name
                job["finished_at"] = datetime.now().isoformat()
                job["elapsed_seconds"] = elapsed
                job["decision"] = decision
                job["state"] = final_state if isinstance(final_state, dict) else None
            p.last_success_at = datetime.now().isoformat()
            p.last_error = None
            return
        except Exception as exc:
            err = f"[{p.name}] {type(exc).__name__}: {exc}"
            errors.append(err)
            p.last_error = err
            logger.warning("Job %s: provider %s failed: %s", job_id, p.name, err)
            continue

    # All providers failed
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "error"
        job["finished_at"] = datetime.now().isoformat()
        job["error"] = "All providers failed:\n" + "\n".join(errors)


@app.post("/analyze-async", response_model=JobAccepted, status_code=202)
def analyze_async(req: AnalyzeRequest) -> JobAccepted:
    """Enqueue a full TradingAgents analysis with automatic provider failover."""
    configured = [p for p in _PROVIDERS if p.configured and p.graph is not None]
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="No provider is ready. Set at least one API key and call /smoke.",
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
            "provider": None,
            "decision": None,
            "state": None,
            "error": None,
        }
    threading.Thread(target=_run_job, args=(job_id, req.ticker, req.date), daemon=True).start()
    return JobAccepted(
        job_id=job_id, ticker=req.ticker, date=req.date,
        status="queued", submitted_at=submitted_at,
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return JobStatus(**job)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("web.app:app", host="0.0.0.0", port=port)
