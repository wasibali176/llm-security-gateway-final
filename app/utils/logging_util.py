"""
Audit Logging Utility
CSC 262 Lab Final - LLM Security Gateway
Logs every request/response as structured JSON.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
AUDIT_LOG_PATH = os.path.join(LOG_DIR, "audit.log")


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging(level: str = "INFO"):
    """Configure root logger."""
    _ensure_log_dir()
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def log_audit(record: dict):
    """
    Append a structured audit record to the audit log file.
    Each line is a JSON object.
    """
    _ensure_log_dir()
    record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_audit_log(last_n: int = 50) -> list:
    """Read last N records from audit log."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    records = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[-last_n:]


if __name__ == "__main__":
    setup_logging()
    log_audit({
        "input_id": "test_001",
        "decision": "BLOCK",
        "rule_score": 0.8,
        "semantic_score": 0.9,
        "final_risk": 0.95,
    })
    print("Logged. Last record:", read_audit_log(1))
