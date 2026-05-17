"""
Tests for Policy Engine - CSC 262 Lab Final
Run: pytest tests/test_policy.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.policy.policy_engine import decide, compute_final_risk

def test_benign_allow():
    result = decide(0.0, 0.05, [], 0.0, [], "en", False)
    assert result["decision"] == "ALLOW"

def test_pii_mask():
    result = decide(0.0, 0.1, [{"type": "EMAIL_ADDRESS"}], 0.7, [], "en", False)
    assert result["decision"] == "MASK"

def test_injection_block_by_rule():
    result = decide(0.75, 0.2, [], 0.0, ["RULE_INJECTION_EN"], "en", False)
    assert result["decision"] == "BLOCK"

def test_injection_block_by_semantic():
    result = decide(0.1, 0.85, [], 0.0, ["SEMANTIC_INJECTION"], "en", False)
    assert result["decision"] == "BLOCK"

def test_secret_always_blocks():
    result = decide(0.0, 0.1, [{"type": "API_KEY"}], 0.9, [], "en", False)
    assert result["decision"] == "BLOCK"

def test_system_prompt_extraction_blocks():
    result = decide(0.3, 0.4, [], 0.0, ["SYSTEM_PROMPT_EXTRACTION"], "en", False)
    assert result["decision"] == "BLOCK"

def test_multilingual_attack_codes():
    result = decide(0.7, 0.8, [], 0.0, ["RULE_INJECTION_UR"], "ur", False)
    assert result["decision"] == "BLOCK"
    assert any("MULTILINGUAL" in c or "URDU" in c or "UR" in c for c in result["reason_codes"])

def test_mixed_language_attack():
    result = decide(0.5, 0.6, [], 0.0, ["RULE_INJECTION_EN"], "mixed", True)
    assert result["decision"] == "BLOCK"
    assert "MIXED_LANGUAGE_ATTACK" in result["reason_codes"]

def test_final_risk_computation():
    risk = compute_final_risk(0.8, 0.9, 0.0, False)
    assert 0.0 <= risk <= 1.0

def test_final_risk_clamped():
    risk = compute_final_risk(1.0, 1.0, 1.0, True)
    assert risk == 1.0
