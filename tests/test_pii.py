"""
Tests for Presidio PII Detection - CSC 262 Lab Final
Run: pytest tests/test_pii.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pii.presidio_custom import analyze

def test_email_detected():
    result = analyze("My email is ali.khan@example.com")
    types = [e["type"] for e in result["pii_entities"]]
    assert "EMAIL_ADDRESS" in types

def test_cnic_detected():
    result = analyze("My CNIC is 35202-1234567-1 for verification.")
    types = [e["type"] for e in result["pii_entities"]]
    assert "CNIC" in types

def test_student_id_detected():
    result = analyze("My student ID is FA21-BCS-123.")
    types = [e["type"] for e in result["pii_entities"]]
    assert "STUDENT_ID" in types

def test_api_key_detected():
    result = analyze("Here is my API key: sk-abcdefghijklmnopqrstuvwxyz123456")
    types = [e["type"] for e in result["pii_entities"]]
    assert "API_KEY" in types

def test_pk_phone_detected():
    result = analyze("Call me at 0312-4567890 please.")
    types = [e["type"] for e in result["pii_entities"]]
    assert "PHONE_NUMBER" in types

def test_no_pii_benign():
    result = analyze("Explain supervised learning with an example.")
    assert len(result["pii_entities"]) == 0

def test_safe_text_masks_email():
    result = analyze("My email is ali.khan@example.com please contact me.")
    assert "ali.khan@example.com" not in result["safe_text"]
    assert "<EMAIL>" in result["safe_text"]

def test_composite_student_email():
    result = analyze("Student FA21-BCS-123 email: student@nu.edu.pk")
    composites = result["composite_entities"]
    assert any("STUDENT_ID" in c and "EMAIL" in c for c in composites)

def test_pii_score_positive_for_pii():
    result = analyze("My CNIC is 35202-1234567-1")
    assert result["pii_score"] > 0.0

def test_has_pii_flag():
    result = analyze("Contact at test@example.com")
    assert result["has_pii"] is True
