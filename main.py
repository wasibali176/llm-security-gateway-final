"""
LLM Security Gateway - Main FastAPI Application
CSC 262 Lab Final
Student: [WasibAli] | Reg: [FA24-BCS-052]
"""

import os
import sys
import time
import uuid
import yaml
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors import rule_detector, semantic_detector
from pii import presidio_custom
from policy import policy_engine
from utils.language import detect_language, is_mixed_language
from utils.logging_util import log_audit, setup_logging, read_audit_log
# ─────────────────────────────────────────────
# Load Config
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/gateway_config.yaml")

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
thresholds = config.get("thresholds", {})
setup_logging(config.get("logging", {}).get("level", "INFO"))
logger = logging.getLogger("gateway")

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="LLM-Security-Gateway",
    description="Robust Multilingual Security Gateway for LLM Applications - CSC 262 Lab Final",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    input_id: Optional[str] = None
    language_hint: Optional[str] = None

class BatchAnalyzeRequest(BaseModel):
    items: List[AnalyzeRequest]

class PolicyUpdateRequest(BaseModel):
    thresholds: dict

# ─────────────────────────────────────────────
# Core Analysis Function
# ─────────────────────────────────────────────

def analyze_text(text: str, input_id: str = None, language_hint: str = None) -> dict:
    """Run the full pipeline on a single text input."""
    global thresholds
    total_start = time.time()

    if not input_id:
        input_id = str(uuid.uuid4())[:8]

    # 1. Language Detection
    lang = language_hint or detect_language(text)
    mixed = is_mixed_language(text)

    # 2. Rule-Based Detection
    rule_result = rule_detector.detect(text)
    rule_score = rule_result["rule_score"]
    rule_codes = rule_result["reason_codes"]

    # 3. Semantic Detection
    sem_result = semantic_detector.detect(text)
    semantic_score = sem_result["semantic_score"]
    sem_codes = sem_result["reason_codes"]

    # 4. PII Detection
    pii_result = presidio_custom.analyze(
        text,
        score_threshold=thresholds.get("presidio_threshold", 0.4),
    )
    pii_entities = pii_result["pii_entities"]
    pii_score = pii_result["pii_score"]
    safe_text_pii = pii_result["safe_text"]
    composites = pii_result["composite_entities"]

    # 5. Combine reason codes
    all_codes = list(set(rule_codes + sem_codes + composites))

    # 6. Policy Decision
    policy_result = policy_engine.decide(
        rule_score=rule_score,
        semantic_score=semantic_score,
        pii_entities=pii_entities,
        pii_score=pii_score,
        reason_codes=all_codes,
        language=lang,
        is_mixed=mixed,
        thresholds=thresholds,
    )

    final_risk = policy_result["final_risk"]
    decision = policy_result["decision"]
    final_codes = policy_result["reason_codes"]

    # 7. Determine safe output
    if decision == "BLOCK":
        safe_text = None
    elif decision == "MASK":
        safe_text = safe_text_pii
    else:
        safe_text = text

    total_latency = round((time.time() - total_start) * 1000, 2)

    response = {
        "input_id": input_id,
        "language": lang,
        "is_mixed_language": mixed,
        "rule_score": rule_score,
        "semantic_score": semantic_score,
        "pii_entities": pii_entities,
        "composite_entities": composites,
        "final_risk": final_risk,
        "decision": decision,
        "safe_text": safe_text,
        "reason_codes": final_codes,
        "justification": policy_result["justification"],
        "latency_ms": total_latency,
        "component_latency": {
            "rule_ms": rule_result["latency_ms"],
            "semantic_ms": sem_result["latency_ms"],
            "pii_ms": pii_result["latency_ms"],
            "policy_ms": policy_result["latency_ms"],
        },
    }

    # 8. Audit log
    log_audit({
        "input_id": input_id,
        "language": lang,
        "is_mixed": mixed,
        "rule_score": rule_score,
        "semantic_score": semantic_score,
        "pii_count": len(pii_entities),
        "pii_types": [e["type"] for e in pii_entities],
        "composite_entities": composites,
        "final_risk": final_risk,
        "decision": decision,
        "reason_codes": final_codes,
        "latency_ms": total_latency,
    })

    logger.info(f"[{input_id}] lang={lang} rule={rule_score} sem={semantic_score} risk={final_risk} → {decision}")
    return response


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0", "service": "LLM Security Gateway"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Main analysis endpoint.
    Runs full pipeline: language detection → rule → semantic → PII → policy.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is required and must not be empty")
    try:
        result = analyze_text(req.text, req.input_id, req.language_hint)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
async def analyze_batch(req: BatchAnalyzeRequest):
    """
    Batch analysis endpoint. Analyze multiple prompts in one request.
    """
    if not req.items:
        raise HTTPException(status_code=400, detail="items list is empty")
    results = []
    for item in req.items:
        try:
            r = analyze_text(item.text, item.input_id, item.language_hint)
        except Exception as e:
            r = {"input_id": item.input_id, "error": str(e)}
        results.append(r)
    return JSONResponse(content={"results": results, "count": len(results)})


@app.get("/audit")
async def audit_log(last_n: int = 50):
    """Return last N audit log records."""
    records = read_audit_log(last_n)
    return JSONResponse(content={"records": records, "count": len(records)})


@app.get("/config")
async def get_config():
    """Return current gateway configuration (thresholds only)."""
    return JSONResponse(content={"thresholds": thresholds})


@app.post("/config/thresholds")
async def update_thresholds(req: PolicyUpdateRequest):
    """Update policy thresholds at runtime."""
    global thresholds
    thresholds.update(req.thresholds)
    return JSONResponse(content={"message": "Thresholds updated", "thresholds": thresholds})


@app.post("/llm/query")
async def llm_query(req: AnalyzeRequest):
    """
    Full LLM query endpoint.
    Analyzes the input, and if ALLOW, forwards to LLM (or mock).
    Returns gateway decision + LLM response (if allowed).
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is required")

    gateway_result = analyze_text(req.text, req.input_id, req.language_hint)

    if gateway_result["decision"] == "BLOCK":
        return JSONResponse(content={
            **gateway_result,
            "llm_response": None,
            "message": "Request blocked by security gateway.",
        })

    # Use masked text for MASK, original for ALLOW
    prompt_to_send = gateway_result["safe_text"] or req.text

    # LLM call (mock or real)
    llm_response = _call_llm(prompt_to_send)

    return JSONResponse(content={
        **gateway_result,
        "llm_response": llm_response,
    })


def _call_llm(prompt: str) -> str:
    """Call LLM or return mock response."""
    provider = config.get("llm", {}).get("provider", "mock")

    if provider == "mock":
        return f"[MOCK LLM RESPONSE] I received your prompt and I'm happy to help! (Gateway-sanitized input accepted)"

    if provider == "gemini":
        try:
            import google.generativeai as genai
            api_key = config.get("llm", {}).get("api_key") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return "[ERROR] No Gemini API key configured."
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config.get("llm", {}).get("model", "gemini-1.5-flash"))
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[LLM ERROR] {str(e)}"

    return "[ERROR] Unknown LLM provider."


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 8000)
    uvicorn.run("main:app", host=host, port=port, reload=True)
