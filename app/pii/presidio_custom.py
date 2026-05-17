"""
Presidio Custom PII Detection & Anonymization
CSC 262 Lab Final - LLM Security Gateway

Implements:
1. Custom recognizer for CNIC (Pakistani ID), Student Registration Number, API Key
2. Context-aware scoring
3. Composite entity detection
4. Confidence calibration
"""

import re
import time
from typing import List, Optional

from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    PatternRecognizer,
    Pattern,
    RecognizerResult,
    EntityRecognizer,
    AnalysisExplanation,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ─────────────────────────────────────────────
# Custom Recognizers
# ─────────────────────────────────────────────

def build_cnic_recognizer() -> PatternRecognizer:
    """Pakistani CNIC: 35202-1234567-1"""
    patterns = [
        Pattern(
            name="CNIC_FORMAT",
            regex=r"\b\d{5}-\d{7}-\d{1}\b",
            score=0.9,
        ),
        Pattern(
            name="CNIC_NO_DASH",
            regex=r"\b\d{13}\b",
            score=0.6,
        ),
    ]
    context_words = ["cnic", "شناختی کارڈ", "id card", "identity", "national id", "cnic number", "nadra"]
    return PatternRecognizer(
        supported_entity="CNIC",
        patterns=patterns,
        context=context_words,
        name="CnicRecognizer",
    )


def build_student_id_recognizer() -> PatternRecognizer:
    """FAST-style student ID: FA21-BCS-123, SP23-BSE-045, etc."""
    patterns = [
        Pattern(
            name="FAST_STUDENT_ID",
            regex=r"\b(FA|SP|FA|FS)\d{2}-[A-Z]{2,4}-\d{3,4}\b",
            score=0.95,
        ),
        Pattern(
            name="GENERIC_STUDENT_ID",
            regex=r"\b[A-Z]{1,3}\d{2}[A-Z0-9\-]{4,12}\b",
            score=0.55,
        ),
    ]
    context_words = ["student id", "registration number", "reg no", "student number", "roll number", "enrollment"]
    return PatternRecognizer(
        supported_entity="STUDENT_ID",
        patterns=patterns,
        context=context_words,
        name="StudentIdRecognizer",
    )


def build_api_key_recognizer() -> PatternRecognizer:
    """Generic API key patterns."""
    patterns = [
        Pattern(
            name="OPENAI_KEY",
            regex=r"\bsk-[A-Za-z0-9]{20,60}\b",
            score=0.98,
        ),
        Pattern(
            name="GENERIC_API_KEY_HEX",
            regex=r"\b[0-9a-fA-F]{32,64}\b",
            score=0.55,
        ),
        Pattern(
            name="BEARER_TOKEN",
            regex=r"Bearer\s+[A-Za-z0-9\-_\.]{20,}",
            score=0.9,
        ),
        Pattern(
            name="GENERIC_API_KEY_ALPHANUM",
            regex=r"\b[A-Za-z0-9_\-]{32,80}\b",
            score=0.45,
        ),
    ]
    context_words = ["api key", "api_key", "apikey", "secret", "token", "auth", "bearer", "key", "credential"]
    return PatternRecognizer(
        supported_entity="API_KEY",
        patterns=patterns,
        context=context_words,
        name="ApiKeyRecognizer",
    )


def build_pk_phone_recognizer() -> PatternRecognizer:
    """Pakistani phone number formats."""
    patterns = [
        Pattern(
            name="PK_PHONE_INTL",
            regex=r"\+92[-\s]?\d{3}[-\s]?\d{7}\b",
            score=0.9,
        ),
        Pattern(
            name="PK_PHONE_LOCAL",
            regex=r"\b0\d{3}[-\s]?\d{7}\b",
            score=0.75,
        ),
        Pattern(
            name="PK_MOBILE_03",
            regex=r"\b03\d{2}[-\s]?\d{7}\b",
            score=0.85,
        ),
    ]
    context_words = ["phone", "mobile", "cell", "contact", "number", "call", "whatsapp", "فون"]
    return PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=patterns,
        context=context_words,
        name="PkPhoneRecognizer",
    )


# ─────────────────────────────────────────────
# Analyzer & Anonymizer Engine
# ─────────────────────────────────────────────

def build_analyzer() -> AnalyzerEngine:
    """Build AnalyzerEngine with all custom recognizers."""
    # Use spacy NLP engine
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    # Add custom recognizers
    registry.add_recognizer(build_cnic_recognizer())
    registry.add_recognizer(build_student_id_recognizer())
    registry.add_recognizer(build_api_key_recognizer())
    registry.add_recognizer(build_pk_phone_recognizer())

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en"],
    )
    return analyzer


def build_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


# Singleton instances
_analyzer: Optional[AnalyzerEngine] = None
_anonymizer: Optional[AnonymizerEngine] = None


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = build_analyzer()
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = build_anonymizer()
    return _anonymizer


# ─────────────────────────────────────────────
# Context-aware score enhancement
# ─────────────────────────────────────────────

CONTEXT_BOOSTERS = {
    "EMAIL_ADDRESS": ["email", "e-mail", "mail", "ای میل"],
    "PHONE_NUMBER": ["phone", "mobile", "cell", "contact", "call", "فون", "موبائل"],
    "PERSON": ["name", "my name", "i am", "میرا نام"],
    "CNIC": ["cnic", "id card", "identity", "national id", "شناختی", "nadra"],
    "STUDENT_ID": ["student id", "registration", "reg no", "roll number", "enrollment"],
    "API_KEY": ["api key", "token", "secret", "auth", "bearer", "credential"],
}

def enhance_scores(text: str, results: List[RecognizerResult]) -> List[dict]:
    """Boost confidence when context words appear near entities."""
    text_lower = text.lower()
    enhanced = []
    for r in results:
        base_score = r.score
        boost = 0.0
        entity = r.entity_type
        if entity in CONTEXT_BOOSTERS:
            for kw in CONTEXT_BOOSTERS[entity]:
                if kw in text_lower:
                    boost += 0.1
                    break
        final_score = min(1.0, base_score + boost)
        enhanced.append({
            "type": entity,
            "text": text[r.start:r.end],
            "score": round(final_score, 3),
            "start": r.start,
            "end": r.end,
        })
    return enhanced


# ─────────────────────────────────────────────
# Composite entity detection
# ─────────────────────────────────────────────

def detect_composite_entities(entities: List[dict]) -> List[str]:
    """Detect dangerous combos like PERSON+PHONE, STUDENT_ID+EMAIL, etc."""
    types_found = {e["type"] for e in entities}
    composites = []
    if "PERSON" in types_found and "PHONE_NUMBER" in types_found:
        composites.append("COMPOSITE:PERSON+PHONE")
    if "STUDENT_ID" in types_found and "EMAIL_ADDRESS" in types_found:
        composites.append("COMPOSITE:STUDENT_ID+EMAIL")
    if "API_KEY" in types_found and ("PERSON" in types_found or "EMAIL_ADDRESS" in types_found):
        composites.append("COMPOSITE:API_KEY+IDENTITY")
    if "CNIC" in types_found and ("PERSON" in types_found or "PHONE_NUMBER" in types_found):
        composites.append("COMPOSITE:CNIC+IDENTITY")
    return composites


# ─────────────────────────────────────────────
# Anonymization with custom placeholders
# ─────────────────────────────────────────────

OPERATORS = {
    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
    "CNIC": OperatorConfig("replace", {"new_value": "<CNIC>"}),
    "API_KEY": OperatorConfig("replace", {"new_value": "<API_KEY>"}),
    "STUDENT_ID": OperatorConfig("replace", {"new_value": "<STUDENT_ID>"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
    "ORGANIZATION": OperatorConfig("replace", {"new_value": "<ORG>"}),
    "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE>"}),
    "NRP": OperatorConfig("replace", {"new_value": "<NRP>"}),
    "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
}


# ─────────────────────────────────────────────
# Main analyze function
# ─────────────────────────────────────────────

def analyze(text: str, score_threshold: float = 0.4) -> dict:
    """
    Analyze text for PII.
    Returns dict with entities, pii_score, safe_text, composites, latency_ms.
    """
    start = time.time()

    analyzer = get_analyzer()
    anonymizer = get_anonymizer()

    # Analyze
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            score_threshold=score_threshold,
        )
    except Exception as e:
        return {
            "pii_entities": [],
            "pii_score": 0.0,
            "safe_text": text,
            "composite_entities": [],
            "has_pii": False,
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }

    # Enhance scores with context
    entities = enhance_scores(text, results)

    # Detect composite entities
    composites = detect_composite_entities(entities)

    # Compute PII risk score
    if entities:
        pii_score = min(1.0, max(e["score"] for e in entities) + len(composites) * 0.1)
    else:
        pii_score = 0.0

    # Anonymize
    try:
        if results:
            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=OPERATORS,
            )
            safe_text = anonymized.text
        else:
            safe_text = text
    except Exception:
        safe_text = text

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "pii_entities": entities,
        "pii_score": round(pii_score, 4),
        "safe_text": safe_text,
        "composite_entities": composites,
        "has_pii": len(entities) > 0,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    test_cases = [
        "Explain supervised learning.",
        "My email is ali.khan@example.com please summarize this.",
        "My CNIC is 35202-1234567-1 and student ID is FA21-BCS-123.",
        "Contact me at +92-300-1234567 or sara@test.com.",
        "Here is my API key: sk-abcdefghijklmnopqrstuvwxyz123456",
        "My name is Ahmed and my phone is 0312-4567890.",
    ]
    for t in test_cases:
        r = analyze(t)
        print(f"\nText: {t}")
        print(f"  PII Score: {r['pii_score']} | Entities: {[e['type'] for e in r['pii_entities']]}")
        print(f"  Composites: {r['composite_entities']}")
        print(f"  Safe Text: {r['safe_text']}")
