"""FastAPI wrapper that exposes TradingAgents as an HTTP service.

Multi-provider LLM deployment surface for Render. Picks the best provider
at request time from a configurable list, with automatic failover when a
provider/model returns 429 / auth errors / 5xx / empty reply.

Endpoints
---------
GET  /                       -> service info (no LLM call)
GET  /health                 -> liveness probe
GET  /config                 -> echo active LLM configuration
GET  /providers              -> list all configured LLM providers + their status
POST /smoke                  -> single LLM call per provider, verifies each key works
POST /analyze-async          -> enqueue a full TradingAgents run (returns job_id)
GET  /jobs/{job_id}          -> poll a background job's status

Provider configuration (env vars)
----------------------------------
  LLM_PROVIDERS=unorouter,openrouter,nvidia   # priority list (comma-separated)

For each provider you can configure multiple model candidates; the app
rotates to the next candidate automatically when one rate-limits.

  UNOROUTER_API_KEY=sk-...
  UNOROUTER_BASE_URL=https://api.unorouter.com/v1
  UNOROUTER_DEEP_MODELS=nemotron-3-super-120b-a12b:free,nemotron-3-ultra-550b-a55b:free,glm-5.3:free
  UNOROUTER_QUICK_MODELS=gemini-3-flash:free,nemotron-3-super-120b-a12b:free,laguna-s-2.1:free

  OPENROUTER_API_KEY=sk-or-...
  OPENROUTER_DEEP_MODELS=nvidia/nemotron-3.5-lightning:free
  OPENROUTER_QUICK_MODELS=nvidia/nemotron-3.5-lightning:free

  NVIDIA_API_KEY=nvapi-...
  NVIDIA_DEEP_MODELS=moonshotai/kimi-k3
  NVIDIA_QUICK_MODELS=moonshotai/kimi-k3

Notes
-----
- UnoRouter itself does internal failover (across upstream providers), so
  a single call may already rotate. We layer our own model rotation on top
  so that if a specific *model* is exhausted across all upstreams, we move
  on to the next model in the candidate list.
- /analyze-async runs in a background thread to bypass Render's 5-min
  gateway timeout. Poll /jobs/{job_id} for the result.
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
# Provider registry
# --------------------------------------------------------------------------- #
# Default config for each known provider. The user can override any of these
# via env vars (see module docstring).
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "unorouter": {
        "api_key_env": "UNOROUTER_API_KEY",
        "base_url": "https://api.unorouter.com/v1",
        "deep_models": [
            "nemotron-3-super-120b-a12b:free",
            "nemotron-3-ultra-550b-a55b:free",
            "glm-5.3:free",
            "nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ],
        "quick_models": [
            "gemini-3-flash:free",
            "nemotron-3-super-120b-a12b:free",
            "laguna-s-2.1:free",
            "nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ],
        # UnoRouter is OpenAI-compatible; we use the openai_compatible provider
        # in TradingAgents and set backend_url to its base URL.
        "ta_provider": "openai_compatible",
        "ta_backend_url_env": "UNOROUTER_BASE_URL",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "deep_models": [
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ],
        "quick_models": [
            "nvidia/nemotron-3.5-lightning:free",
        ],
        "ta_provider": "openrouter",
        "ta_backend_url_env": None,  # TradingAgents has openrouter hardcoded
    },
    "nvidia": {
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "deep_models": ["moonshotai/kimi-k3"],
        "quick_models": ["moonshotai/kimi-k3"],
        "ta_provider": "nvidia",
        "ta_backend_url_env": None,
    },
}


class ProviderConfig:
    """One LLM provider's full configuration, with model candidate lists."""

    def __init__(self, name: str, spec: dict[str, Any]):
        self.name = name
        self.api_key_env = spec["api_key_env"]
        self.api_key = os.environ.get(self.api_key_env, "").strip()
        self.base_url = spec["base_url"]
        self.ta_provider = spec["ta_provider"]
        self.ta_backend_url_env = spec.get("ta_backend_url_env")
        # Deep + quick model candidate lists (env var override wins).
        env_deep = os.environ.get(f"{name.upper()}_DEEP_MODELS", "")
        if env_deep:
            self.deep_models = [m.strip() for m in env_deep.split(",") if m.strip()]
        else:
            self.deep_models = list(spec["deep_models"])
        env_quick = os.environ.get(f"{name.upper()}_QUICK_MODELS", "")
        if env_quick:
            self.quick_models = [m.strip() for m in env_quick.split(",") if m.strip()]
        else:
            self.quick_models = list(spec["quick_models"])
        # Per-provider rotation state (which model index we're trying next).
        self.deep_idx = 0
        self.quick_idx = 0
        # Cache: graph per (deep, quick) pair. Each provider keeps a fresh
        # graph for its current model combination.
        self.graph: TradingAgentsGraph | None = None
        self.current_deep_model: str | None = None
        self.current_quick_model: str | None = None
        # Smoke status
        self.smoke_ok: bool | None = None
        self.last_error: str | None = None
        self.last_success_at: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "REPLACE_WITH_YOUR_NVIDIA_API_KEY"

    def next_deep_model(self) -> str | None:
        """Return the next deep model candidate, or None if exhausted."""
        if self.deep_idx < len(self.deep_models):
            m = self.deep_models[self.deep_idx]
            self.deep_idx += 1
            return m
        return None

    def reset_rotation(self) -> None:
        """Reset both indices to 0 so the next call starts from the top model."""
        self.deep_idx = 0
        self.quick_idx = 0

    def status_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configured": self.configured,
            "base_url": self.base_url,
            "deep_models": self.deep_models,
            "quick_models": self.quick_models,
            "current_deep_model": self.current_deep_model,
            "current_quick_model": self.current_quick_model,
            "graph_ready": self.graph is not None,
            "smoke_ok": self.smoke_ok,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
        }


def _build_providers() -> list[ProviderConfig]:
    """Read LLM_PROVIDERS env var, return ProviderConfig list in priority order."""
    order_str = os.environ.get("LLM_PROVIDERS", "unorouter,openrouter,nvidia")
    order = [p.strip() for p in order_str.split(",") if p.strip()]
    providers: list[ProviderConfig] = []
    for name in order:
        spec = _PROVIDER_DEFAULTS.get(name)
        if spec is None:
            logger.warning("Unknown provider in LLM_PROVIDERS: %s — skipping", name)
            continue
        providers.append(ProviderConfig(name, spec))
    return providers


_PROVIDERS: list[ProviderConfig] = _build_providers()
_providers_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Graph construction — one graph per (provider, deep_model, quick_model) tuple
# --------------------------------------------------------------------------- #
def _apply_provider_to_config(provider: ProviderConfig, deep: str, quick: str) -> None:
    """Patch DEFAULT_CONFIG so TradingAgentsGraph picks up this provider+model."""
    DEFAULT_CONFIG["llm_provider"] = provider.ta_provider
    DEFAULT_CONFIG["deep_think_llm"] = deep
    DEFAULT_CONFIG["quick_think_llm"] = quick
    # For openai_compatible (unorouter), set backend_url; for native providers
    # the TradingAgents code already knows the endpoint.
    if provider.ta_provider == "openai_compatible":
        DEFAULT_CONFIG["backend_url"] = provider.base_url
        # The openai_compatible provider reads OPENAI_COMPATIBLE_API_KEY for the
        # API key. Set it from this provider's key env var so the SDK picks it up.
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = provider.api_key
    else:
        DEFAULT_CONFIG["backend_url"] = None


def _build_graph_for(provider: ProviderConfig, deep: str, quick: str) -> TradingAgentsGraph:
    """Build a TradingAgentsGraph bound to a specific provider + model pair."""
    _apply_provider_to_config(provider, deep, quick)
    cfg = DEFAULT_CONFIG.copy()
    return TradingAgentsGraph(debug=False, config=cfg)


def _ensure_graph(provider: ProviderConfig) -> tuple[TradingAgentsGraph, str, str] | None:
    """Make sure `provider` has a graph bound to its current model pair.

    Returns (graph, deep_model, quick_model) on success, None if no model works.
    """
    with _providers_lock:
        # Already have a graph for the current model pair?
        if (
            provider.graph is not None
            and provider.current_deep_model
            and provider.current_quick_model
        ):
            return provider.graph, provider.current_deep_model, provider.current_quick_model
        provider.reset_rotation()
        # Try each deep model until one builds a graph without error.
        while True:
            deep = provider.next_deep_model()
            if deep is None:
                return None
            quick = provider.quick_models[min(
                provider.quick_idx, len(provider.quick_models) - 1
            )]
            try:
                _apply_provider_to_config(provider, deep, quick)
                provider.graph = _build_graph_for(provider, deep, quick)
                provider.current_deep_model = deep
                provider.current_quick_model = quick
                logger.info(
                    "Built graph for %s: deep=%s quick=%s",
                    provider.name, deep, quick,
                )
                return provider.graph, deep, quick
            except Exception as exc:
                logger.warning(
                    "Failed to build graph for %s with deep=%s: %s",
                    provider.name, deep, exc,
                )
                continue


def _rotate_to_next_model(provider: ProviderConfig) -> tuple[str, str] | None:
    """Force-rotate to the next model pair after a rate-limit failure.

    Returns (deep, quick) if a next model is available, None if exhausted.
    """
    with _providers_lock:
        provider.graph = None  # force rebuild
        provider.current_deep_model = None
        provider.current_quick_model = None
        deep = provider.next_deep_model()
        if deep is None:
            return None
        quick = provider.quick_models[
            min(provider.quick_idx, len(provider.quick_models) - 1)
        ]
        try:
            _apply_provider_to_config(provider, deep, quick)
            provider.graph = _build_graph_for(provider, deep, quick)
            provider.current_deep_model = deep
            provider.current_quick_model = quick
            logger.info(
                "Rotated %s to: deep=%s quick=%s", provider.name, deep, quick
            )
            return deep, quick
        except Exception as exc:
            logger.warning("Failed rotation for %s: %s", provider.name, exc)
            return None


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic: is this error a transient rate-limit we should rotate past?"""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "429", "rate limit", "rate_limit", "too many requests",
        "busy right now", "capacity exhausted", "empty reply",
        "no endpoints found",
    ))


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
        result = _ensure_graph(p)
        if result is None:
            logger.warning("Provider %s: could not build any graph", p.name)
        else:
            logger.info("Provider %s ready", p.name)
    yield


app = FastAPI(
    title="TradingAgents API",
    description="Multi-agent LLM financial trading framework — multi-provider + multi-model failover",
    version="0.6.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Pydantic models
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
    configured: bool
    base_url: str
    deep_models: list[str]
    quick_models: list[str]
    current_deep_model: str | None
    current_quick_model: str | None
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
    deep_model: str | None = None
    quick_model: str | None = None
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
# Routes — info / health / config
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "TradingAgents API",
        "version": "0.6.0",
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
    # Pick the first configured provider that has a graph.
    chosen = None
    for p in _PROVIDERS:
        if p.configured and p.graph is not None:
            chosen = p
            break
    return ConfigResponse(
        llm_provider=chosen.ta_provider if chosen else "(none)",
        deep_think_llm=chosen.current_deep_model if chosen else "",
        quick_think_llm=chosen.current_quick_model if chosen else "",
        backend_url=DEFAULT_CONFIG.get("backend_url"),
        output_language=DEFAULT_CONFIG.get("output_language", "English"),
        max_debate_rounds=DEFAULT_CONFIG.get("max_debate_rounds", 1),
        max_risk_discuss_rounds=DEFAULT_CONFIG.get("max_risk_discuss_rounds", 1),
        providers_configured=sum(1 for p in _PROVIDERS if p.configured),
    )


@app.get("/providers", response_model=list[ProviderStatus])
def providers() -> list[ProviderStatus]:
    return [ProviderStatus(**p.status_dict()) for p in _PROVIDERS]


# --------------------------------------------------------------------------- #
# /smoke — single LLM call per provider
# --------------------------------------------------------------------------- #
def _smoke_one(provider: ProviderConfig) -> SmokeResponse:
    """Single LLM call to verify provider+key+model work end-to-end."""
    started = datetime.now()
    result = _ensure_graph(provider)
    if result is None:
        return SmokeResponse(
            ok=False, provider=provider.name, model="(none)",
            response_preview="", elapsed_seconds=0,
            error="Could not build any graph (check API key + model list)",
        )
    graph, deep, quick = result
    try:
        _apply_provider_to_config(provider, deep, quick)
        client = create_llm_client(
            provider=provider.ta_provider, model=quick,
            base_url=provider.base_url if provider.ta_provider == "openai_compatible" else None,
        )
        llm = client.get_llm()
        out = llm.invoke("Reply with exactly one word: PONG")
        text = getattr(out, "content", str(out))
        if isinstance(text, list):
            text = " ".join(str(b) for b in text)
        text = str(text).strip()
        ok = "pong" in text.lower()
        provider.smoke_ok = ok
        if ok:
            provider.last_error = None
            provider.last_success_at = datetime.now().isoformat()
        elapsed = (datetime.now() - started).total_seconds()
        return SmokeResponse(
            ok=ok, provider=provider.name, model=quick,
            response_preview=text[:80] if ok else "",
            elapsed_seconds=elapsed,
            error=None if ok else f"unexpected response: {text[:80]!r}",
        )
    except Exception as exc:
        provider.smoke_ok = False
        provider.last_error = str(exc)
        elapsed = (datetime.now() - started).total_seconds()
        # Try rotating to the next model
        _rotate_to_next_model(provider)
        return SmokeResponse(
            ok=False, provider=provider.name, model=quick,
            response_preview="", elapsed_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


@app.post("/smoke", response_model=list[SmokeResponse])
def smoke_all() -> list[SmokeResponse]:
    """Test every configured provider with a single LLM call."""
    return [_smoke_one(p) for p in _PROVIDERS if p.configured]


# --------------------------------------------------------------------------- #
# /analyze-async + /jobs/{id}
# --------------------------------------------------------------------------- #
def _run_provider(provider: ProviderConfig, ticker: str, date: str) -> tuple[bool, Any, Any, str | None]:
    """Try a provider: rotate through its model list until one works.

    Returns (ok, final_state, decision, error).
    """
    # Reset rotation so we start from the top model each job.
    provider.reset_rotation()
    last_err: str | None = None
    while True:
        result = _ensure_graph(provider)
        if result is None:
            return False, None, None, last_err or "no models available"
        graph, deep, quick = result
        logger.info(
            "Job: trying %s deep=%s quick=%s", provider.name, deep, quick
        )
        try:
            _apply_provider_to_config(provider, deep, quick)
            final_state, decision = graph.propagate(ticker, date)
            provider.last_success_at = datetime.now().isoformat()
            provider.last_error = None
            return True, final_state, decision, None
        except Exception as exc:
            err = f"[{provider.name}/{deep}] {type(exc).__name__}: {exc}"
            last_err = err
            provider.last_error = err
            logger.warning("Job: %s failed: %s", provider.name, err)
            if _is_rate_limit_error(exc):
                # Try rotating to the next model in the candidate list.
                rotated = _rotate_to_next_model(provider)
                if rotated is None:
                    return False, None, None, last_err
                # loop and retry with the new model
                continue
            # Non-rate-limit error: bail out of this provider.
            return False, None, None, last_err


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
        if not p.configured:
            continue
        started = datetime.now()
        ok, final_state, decision, err = _run_provider(p, ticker, date)
        elapsed = (datetime.now() - started).total_seconds()
        if ok:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None:
                    return
                job.update({
                    "status": "done",
                    "provider": p.name,
                    "deep_model": p.current_deep_model,
                    "quick_model": p.current_quick_model,
                    "finished_at": datetime.now().isoformat(),
                    "elapsed_seconds": elapsed,
                    "decision": decision,
                    "state": final_state if isinstance(final_state, dict) else None,
                })
            return
        else:
            errors.append(err or "unknown error")

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update({
            "status": "error",
            "finished_at": datetime.now().isoformat(),
            "error": "All providers failed:\n" + "\n".join(errors),
        })


@app.post("/analyze-async", response_model=JobAccepted, status_code=202)
def analyze_async(req: AnalyzeRequest) -> JobAccepted:
    """Enqueue a full TradingAgents analysis with multi-provider failover."""
    configured = [p for p in _PROVIDERS if p.configured]
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="No provider configured. Set at least one API key.",
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
            "deep_model": None,
            "quick_model": None,
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
