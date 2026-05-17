"""
Policy Engine
CSC 262 Lab Final - LLM Security Gateway

Combines rule_score, semantic_score, pii_score, and secret indicators
into a final risk score and policy decision: ALLOW / MASK / BLOCK.

Risk Formula (configurable in gateway_config.yaml):
  injection_risk = max(rule_score * rule_weight, semantic_score * sem_weight)
  final_risk = injection_risk + pii_weight * has_pii + secret_weight * has_secret
  Clamped to [0.0, 1.0]
"""

import time
from typing import Optional


# Default thresholds (overridden by config)
DEFAULT_THRESHOLDS = {
    "rule_score_block": 0.6,
    "semantic_score_block": 0.7,
    "final_risk_block": 0.75,
    "final_risk_mask": 0.30,
    "pii_weight": 0.20,
    "secret_weight": 0.30,
    "rule_weight": 0.4,
    "semantic_weight": 0.6,
}


def compute_final_risk(
    rule_score: float,
    semantic_score: float,
    pii_score: float,
    has_secret: bool,
    thresholds: Optional[dict] = None,
) -> float:
    """
    Compute the final risk score.

    Formula:
        injection_risk = max(rule_score * rule_w, semantic_score * sem_w)
        final_risk = injection_risk
                   + pii_weight * min(1.0, pii_score)
                   + secret_weight * (1.0 if has_secret else 0.0)

    Clamped to [0.0, 1.0].
    """
    t = thresholds or DEFAULT_THRESHOLDS

    rule_w = t.get("rule_weight", 0.4)
    sem_w = t.get("semantic_weight", 0.6)
    pii_w = t.get("pii_weight", 0.20)
    sec_w = t.get("secret_weight", 0.30)

    injection_risk = max(rule_score * rule_w, semantic_score * sem_w)
    pii_contribution = pii_w * min(1.0, pii_score)
    secret_contribution = sec_w if has_secret else 0.0

    final_risk = injection_risk + pii_contribution + secret_contribution
    return round(min(1.0, max(0.0, final_risk)), 4)


def decide(
    rule_score: float,
    semantic_score: float,
    pii_entities: list,
    pii_score: float,
    reason_codes: list,
    language: str = "en",
    is_mixed: bool = False,
    thresholds: Optional[dict] = None,
) -> dict:
    """
    Make the final policy decision.

    Returns dict with:
      - final_risk
      - decision: "ALLOW" | "MASK" | "BLOCK"
      - reason_codes (augmented)
      - justification
    """
    start = time.time()
    t = thresholds or DEFAULT_THRESHOLDS

    # Detect if any PII entity is a secret
    secret_types = {"API_KEY", "PASSWORD", "SECRET_KEY"}
    has_secret = any(e.get("type") in secret_types for e in pii_entities)
    has_pii = len(pii_entities) > 0

    # Compute final risk
    final_risk = compute_final_risk(
        rule_score=rule_score,
        semantic_score=semantic_score,
        pii_score=pii_score,
        has_secret=has_secret,
        thresholds=t,
    )

    # Add reason codes for multilingual/mixed attacks
    augmented_codes = list(set(reason_codes))
    if is_mixed:
        augmented_codes.append("MIXED_LANGUAGE_ATTACK")
    if language not in ("en", "unknown") and any(
        c for c in augmented_codes if "INJECTION" in c or "EXTRACTION" in c
    ):
        augmented_codes.append(f"MULTILINGUAL_ATTACK_{language.upper()}")
    if has_secret:
        augmented_codes.append("SECRET_PII_DETECTED")

    # ── Decision Logic ──────────────────────────────────────────────────
    # BLOCK conditions (any one sufficient):
    block_conditions = [
        final_risk >= t.get("final_risk_block", 0.75),
        rule_score >= t.get("rule_score_block", 0.6),
        semantic_score >= t.get("semantic_score_block", 0.7),
        has_secret,
        "SYSTEM_PROMPT_EXTRACTION" in augmented_codes,
        "SECRET_EXTRACTION" in augmented_codes,
    ]

    if any(block_conditions):
        decision = "BLOCK"
        justification = _block_justification(
            rule_score, semantic_score, final_risk, has_secret, augmented_codes, t
        )
    elif has_pii and final_risk < t.get("final_risk_block", 0.75):
        decision = "MASK"
        justification = f"PII detected (score={pii_score:.2f}); prompt sanitized."
    elif final_risk >= t.get("final_risk_mask", 0.30):
        decision = "MASK"
        justification = f"Elevated risk ({final_risk:.2f}) with PII or sensitive content; masking."
    else:
        decision = "ALLOW"
        justification = f"Risk score {final_risk:.2f} is within safe thresholds."

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "final_risk": final_risk,
        "decision": decision,
        "reason_codes": augmented_codes,
        "justification": justification,
        "has_secret": has_secret,
        "latency_ms": latency_ms,
    }


def _block_justification(rule_score, semantic_score, final_risk, has_secret, codes, t):
    reasons = []
    if rule_score >= t.get("rule_score_block", 0.6):
        reasons.append(f"rule_score={rule_score:.2f} ≥ threshold {t['rule_score_block']}")
    if semantic_score >= t.get("semantic_score_block", 0.7):
        reasons.append(f"semantic_score={semantic_score:.2f} ≥ threshold {t['semantic_score_block']}")
    if final_risk >= t.get("final_risk_block", 0.75):
        reasons.append(f"final_risk={final_risk:.2f} ≥ threshold {t['final_risk_block']}")
    if has_secret:
        reasons.append("API key or secret credential detected")
    if "SYSTEM_PROMPT_EXTRACTION" in codes:
        reasons.append("system prompt extraction attempt")
    return "BLOCKED: " + "; ".join(reasons) if reasons else "BLOCKED: high injection risk"


if __name__ == "__main__":
    scenarios = [
        # (rule, semantic, pii_entities, pii_score, codes, lang, mixed)
        (0.0, 0.05, [], 0.0, [], "en", False),
        (0.0, 0.1, [{"type": "EMAIL_ADDRESS"}], 0.7, [], "en", False),
        (0.75, 0.85, [], 0.0, ["RULE_INJECTION_EN", "SYSTEM_PROMPT_EXTRACTION"], "en", False),
        (0.25, 0.78, [], 0.0, ["SEMANTIC_INJECTION", "PARAPHRASE_SIMILARITY"], "en", False),
        (0.0, 0.05, [{"type": "API_KEY"}], 0.95, ["SECRET_PII_DETECTED"], "en", False),
        (0.65, 0.8, [], 0.0, ["RULE_INJECTION_UR"], "ur", False),
    ]
    labels = ["Benign", "PII-Mask", "Direct-Block", "Paraphrase-Block", "Secret-Block", "Urdu-Block"]
    for (r, s, pii_e, pii_s, codes, lang, mixed), label in zip(scenarios, labels):
        result = decide(r, s, pii_e, pii_s, codes, lang, mixed)
        print(f"[{label}] → {result['decision']} | risk={result['final_risk']} | {result['justification'][:80]}")
