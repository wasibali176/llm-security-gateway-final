"""
Policy Engine
CSC 262 Lab Final - LLM Security Gateway
"""

import time
from typing import Optional

DEFAULT_THRESHOLDS = {
    "rule_score_block": 0.5,
    "semantic_score_block": 0.75,
    "final_risk_block": 0.65,
    "final_risk_mask": 0.35,
    "pii_weight": 0.15,
    "secret_weight": 0.30,
    "rule_weight": 0.4,
    "semantic_weight": 0.6,
}


def compute_final_risk(rule_score, semantic_score, pii_score, has_secret, thresholds=None):
    t = thresholds or DEFAULT_THRESHOLDS
    rule_w = t.get("rule_weight", 0.4)
    sem_w = t.get("semantic_weight", 0.6)
    pii_w = t.get("pii_weight", 0.15)
    sec_w = t.get("secret_weight", 0.30)
    injection_risk = max(rule_score * rule_w, semantic_score * sem_w)
    pii_contribution = pii_w * min(1.0, pii_score)
    secret_contribution = sec_w if has_secret else 0.0
    final_risk = injection_risk + pii_contribution + secret_contribution
    return round(min(1.0, max(0.0, final_risk)), 4)


def decide(rule_score, semantic_score, pii_entities, pii_score, reason_codes,
           language="en", is_mixed=False, thresholds=None):
    start = time.time()
    t = thresholds or DEFAULT_THRESHOLDS

    secret_types = {"API_KEY", "PASSWORD", "SECRET_KEY"}
    has_secret = any(e.get("type") in secret_types for e in pii_entities)
    has_pii = len(pii_entities) > 0

    final_risk = compute_final_risk(rule_score, semantic_score, pii_score, has_secret, t)

    augmented_codes = list(set(reason_codes))
    if is_mixed:
        augmented_codes.append("MIXED_LANGUAGE_ATTACK")
    if language not in ("en", "unknown"):
        augmented_codes.append(f"MULTILINGUAL_ATTACK_{language.upper()}")
    if has_secret:
        augmented_codes.append("SECRET_PII_DETECTED")

    # Korean aur Urdu mein koi bhi rule match = BLOCK
    is_multilingual_attack = language in ("ko", "ur", "ar", "hi") and rule_score > 0.0

    block_conditions = [
        final_risk >= t.get("final_risk_block", 0.65),
        rule_score >= t.get("rule_score_block", 0.5),
        semantic_score >= t.get("semantic_score_block", 0.75),
        has_secret,
        is_mixed,
        is_multilingual_attack,
        "SYSTEM_PROMPT_EXTRACTION" in augmented_codes,
        "SECRET_EXTRACTION" in augmented_codes,
        "RULE_INJECTION_KO" in augmented_codes,
        "RULE_INJECTION_UR" in augmented_codes,
        "MIXED_LANGUAGE_ATTACK" in augmented_codes,
    ]

    if any(block_conditions):
        decision = "BLOCK"
        justification = f"BLOCKED: rule={rule_score}, sem={semantic_score}, lang={language}, risk={final_risk}"
    elif has_pii:
        decision = "MASK"
        justification = f"PII detected (score={pii_score:.2f}); prompt sanitized."
    elif final_risk >= t.get("final_risk_mask", 0.35):
        decision = "MASK"
        justification = f"Elevated risk ({final_risk:.2f}); masking."
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


if __name__ == "__main__":
    scenarios = [
        (0.0, 0.05, [], 0.0, [], "en", False),
        (0.0, 0.1, [{"type": "EMAIL_ADDRESS"}], 0.7, [], "en", False),
        (0.75, 0.85, [], 0.0, ["RULE_INJECTION_EN", "SYSTEM_PROMPT_EXTRACTION"], "en", False),
        (0.25, 0.78, [], 0.0, ["SEMANTIC_INJECTION", "PARAPHRASE_SIMILARITY"], "en", False),
        (0.0, 0.05, [{"type": "API_KEY"}], 0.95, ["SECRET_PII_DETECTED"], "en", False),
        (0.25, 0.31, [], 0.0, ["RULE_INJECTION_KO"], "ko", False),
        (0.75, 0.31, [], 0.0, ["RULE_INJECTION_UR"], "ur", False),
    ]
    labels = ["Benign", "PII-Mask", "Direct-Block", "Paraphrase-Block", "Secret-Block", "Korean-Block", "Urdu-Block"]
    for (r, s, pii_e, pii_s, codes, lang, mixed), label in zip(scenarios, labels):
        result = decide(r, s, pii_e, pii_s, codes, lang, mixed)
        print(f"[{label}] → {result['decision']} | risk={result['final_risk']}")