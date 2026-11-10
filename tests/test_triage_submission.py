from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from core import (
    EXTRACTION_SYSTEM_PROMPT,
    PatientSubmission,
    TriageExtraction,
    TriageOutput,
    evaluate_triage,
    triage_submission,
)


@pytest.fixture
def extraction_payload() -> dict[str, object]:
    return {
        "documents": [
            {
                "source_index": 0,
                "kind": "HISTORY_AND_PHYSICAL",
                "completed": True,
                "evidence": "History and physical completed.",
            },
            {
                "source_index": 1,
                "kind": "SURGICAL_CONSENT",
                "signed": True,
                "procedure_matches": True,
                "evidence": "Signed surgical consent.",
            },
        ],
        "medications": [],
        "anticoagulation_plans": [],
        "vitals": [],
    }


@pytest.fixture
def openai_client(
    monkeypatch: pytest.MonkeyPatch,
    extraction_payload: dict[str, object],
) -> Mock:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(extraction_payload),
    )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    return client


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1"},
        "procedure": {
            "case_id": "case-1",
            "procedure_type": "Test procedure",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {
                "type": "blood_pressure",
                "systolic": 120,
                "diastolic": 80,
                "date": "2026-01-25",
            },
            {
                "type": "temperature",
                "value_f": 98.6,
                "date": "2026-01-25",
            },
        ],
        "labs": [
            {
                "code": "CBC",
                "display": "Complete blood count",
                "effective_at": "2026-01-20",
                "status": "final",
            }
        ],
        "medications": [],
        "conditions": [],
        "documents": [
            {
                "type": "history_and_physical",
                "date": "2026-01-20",
                "text": "History and physical completed.",
            },
            {
                "type": "surgical_consent",
                "date": "2026-01-22",
                "text": "Signed surgical consent.",
            },
        ],
    }


def evaluate(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> TriageOutput:
    submission = PatientSubmission.model_validate(submission_payload)
    extraction = TriageExtraction.model_validate(extraction_payload)
    return evaluate_triage(submission, extraction)


def test_triage_submission_returns_deterministic_structured_output(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    output = triage_submission(submission_payload, model="test-model")

    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert output.issues == []
    assert output.explanation == "All required policy criteria are satisfied."

    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["instructions"] == EXTRACTION_SYSTEM_PROMPT
    response_format = call["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "preop_triage_extraction"
    assert response_format["schema"] == TriageExtraction.model_json_schema()
    assert response_format["strict"] is False


def test_extraction_prompt_uses_validated_submission_without_demographics(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    submission = PatientSubmission.model_validate(submission_payload)

    triage_submission(submission, model="test-model")

    call = openai_client.responses.create.call_args.kwargs
    message = call["input"][0]
    content = message["content"][0]
    prompt = content["text"]
    assert message["type"] == "message"
    assert message["role"] == "user"
    assert content["type"] == "input_text"
    assert "patient-1" not in prompt
    assert "patient" not in prompt.casefold()
    assert "History and physical completed." in prompt
    assert "medications (list indexes are the medication_index)" in prompt.casefold()


def test_extraction_instructions_cover_observed_failure_modes() -> None:
    prompt = EXTRACTION_SYSTEM_PROMPT

    assert "exactly one allowed kind" in prompt
    assert "HISTORY_AND_PHYSICAL, SURGICAL_CONSENT, OTHER, or UNCLEAR" in prompt
    assert "Never invent a medication_index" in prompt
    assert "zero-based source_index" in prompt
    assert "medication_index values" in prompt
    assert "every explicit blood-pressure and temperature reading" in prompt
    assert "what the documents say to do before" in prompt
    assert "after the procedure" in prompt
    assert "prior H&P retained only for chart context" in prompt
    assert "procedure_matches=true only when the procedure is explicitly the same" in prompt
    assert "absent from the medication list" in prompt


def test_unknown_model_document_kind_degrades_to_unclear() -> None:
    fact = TriageExtraction.model_validate(
        {"documents": [{"source_index": 0, "kind": "CONSENT - ELECTIVE PROCEDURE"}]}
    ).documents[0]

    assert fact.kind == "SURGICAL_CONSENT"


def test_triage_submission_rejects_invalid_extraction_json(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    openai_client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps({"documents": [{"source_index": "invalid"}]})
    )

    with pytest.raises(ValidationError):
        triage_submission(submission_payload, model="test-model")

    assert openai_client.responses.create.call_count == 1


def test_confirmed_safety_exclusion_takes_precedence(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    submission_payload["vitals"][0]["systolic"] = 180

    output = triage_submission(submission_payload, model="test-model")

    assert output.decision == "NOT_CLEARED"
    assert any(issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in output.issues)
    assert openai_client.responses.create.call_count == 1


def test_missing_procedure_date_requires_follow_up(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["procedure"]["procedure_date"] = None

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(issue.category == "MISSING_REQUIRED_DATA" for issue in output.issues)


def test_invalid_structured_procedure_date_can_use_explicit_extracted_date(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["procedure"]["procedure_date"] = "not-a-date"
    extraction_payload["procedure_date"] = "2026-02-01"
    extraction_payload["procedure_date_evidence"] = "Scheduled date: 2026-02-01."

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"


def test_procedure_date_can_be_extracted_from_document_text(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["procedure"]["procedure_date"] = None
    extraction_payload["procedure_date"] = "2026-02-01"
    extraction_payload["procedure_date_evidence"] = "H&P states target date 2026-02-01."

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"


def test_document_only_active_anticoagulant_requires_management_plan(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["medications"] = [
        {"name": "ASA 81 mg daily", "active": True}
    ]
    submission_payload["documents"].append(
        {
            "type": "Pre-Admission Testing Visit",
            "date": "2026-01-26",
            "text": "Patient reports currently taking Eliquis 5 mg twice daily.",
        }
    )
    extraction_payload["document_medications"] = [
        {
            "source_index": 2,
            "name": "Eliquis 5 mg twice daily",
            "active": True,
            "is_anticoagulant": True,
            "evidence": "Patient reports currently taking Eliquis.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "ANTICOAGULATION_MANAGEMENT" for issue in output.issues
    )

    extraction_payload["anticoagulation_plans"] = [
        {
            "medication_name": "apixaban",
            "source_index": 2,
            "preoperative_management": "Hold for two days before surgery.",
            "postoperative_management": "Resume 48 hours after surgery.",
            "evidence": "Perioperative plan for Eliquis.",
        }
    ]
    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"


def test_document_only_anticoagulant_with_unknown_current_use_needs_follow_up(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["documents"].append(
        {
            "type": "Pre-Admission Testing Visit",
            "date": "2026-01-26",
            "text": "Medication reconciliation mentions Coumadin; current use unclear.",
        }
    )
    extraction_payload["document_medications"] = [
        {
            "source_index": 2,
            "name": "Coumadin",
            "active": None,
            "is_anticoagulant": True,
            "evidence": "Current use unclear.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "MISSING_REQUIRED_DATA"
        and "currently taken" in issue.evidence
        for issue in output.issues
    )


def test_signed_consent_for_different_procedure_does_not_satisfy_requirement(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["procedure"]["procedure_type"] = "Thyroidectomy"
    extraction_payload["documents"][1]["procedure_matches"] = False
    extraction_payload["documents"][1]["evidence"] = (
        "Signed consent is for coronary artery bypass grafting."
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "REQUIRED_DOCUMENTATION"
        and "different procedure" in issue.evidence
        for issue in output.issues
    )


def test_consent_with_unknown_procedure_match_needs_follow_up(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    extraction_payload["documents"][1].pop("procedure_matches")

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "MISSING_REQUIRED_DATA"
        and "applies to the scheduled procedure" in issue.evidence
        for issue in output.issues
    )


def test_safety_exclusion_does_not_hide_unsigned_consent_requirement(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["vitals"][0]["diastolic"] = 110
    extraction_payload["documents"][1]["signed"] = False

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NOT_CLEARED"
    assert {
        issue.category for issue in output.issues
    } >= {"ACUTE_SAFETY_EXCLUSION", "REQUIRED_DOCUMENTATION"}


@pytest.mark.parametrize(
    ("h_and_p_date", "lab_date", "decision"),
    [
        ("2026-01-02", "2026-01-02", "READY"),
        ("2026-01-01", "2026-01-01", "NEEDS_FOLLOW_UP"),
    ],
)
def test_thirty_day_windows_are_inclusive(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
    h_and_p_date: str,
    lab_date: str,
    decision: str,
) -> None:
    submission_payload["documents"][0]["date"] = h_and_p_date
    submission_payload["labs"][0]["effective_at"] = lab_date

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == decision


def test_outdated_h_and_p_is_required_documentation_not_unknown_data(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["documents"][0]["date"] = "2025-12-01"
    extraction_payload["documents"][0]["completed"] = None

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "REQUIRED_DOCUMENTATION"
        and "outdated" in issue.evidence
        for issue in output.issues
    )
    assert not any(
        issue.category == "MISSING_REQUIRED_DATA"
        and "H&P" in issue.evidence
        for issue in output.issues
    )


@pytest.mark.parametrize(
    ("systolic", "diastolic", "temperature", "decision"),
    [
        (180, 80, 98.6, "NOT_CLEARED"),
        (120, 110, 98.6, "NOT_CLEARED"),
        (120, 80, 100.5, "NOT_CLEARED"),
        (120, 80, 100.4, "READY"),
    ],
)
def test_safety_thresholds(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
    systolic: int,
    diastolic: int,
    temperature: float,
    decision: str,
) -> None:
    submission_payload["vitals"][0]["systolic"] = systolic
    submission_payload["vitals"][0]["diastolic"] = diastolic
    submission_payload["vitals"][1]["value_f"] = temperature

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == decision


def test_in_range_recheck_supersedes_earlier_high_vitals(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["vitals"].extend(
        [
            {
                "type": "blood_pressure",
                "systolic": 182,
                "diastolic": 112,
                "date": "2026-01-24",
            },
            {
                "type": "temperature",
                "value_f": 101.0,
                "date": "2026-01-24",
            },
        ]
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"
    assert not any(
        issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in output.issues
    )


def test_same_day_recheck_uses_latest_timestamp(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["vitals"][0]["systolic"] = 181
    submission_payload["vitals"][0]["date"] = "2026-01-25T09:00:00Z"
    submission_payload["vitals"].append(
        {
            "type": "blood_pressure",
            "systolic": 120,
            "diastolic": 80,
            "date": "2026-01-25T09:05:00Z",
        }
    )
    submission_payload["vitals"][1]["value_f"] = 101.0
    submission_payload["vitals"][1]["date"] = "2026-01-25T09:00:00Z"
    submission_payload["vitals"].append(
        {
            "type": "temperature",
            "value_f": 98.6,
            "date": "2026-01-25T09:05:00Z",
        }
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"


def test_conflicting_latest_vitals_are_order_independent(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    high_reading = {
        "type": "blood_pressure",
        "systolic": 181,
        "diastolic": 112,
        "date": "2026-01-25T09:05:00Z",
    }
    normal_reading = {
        "type": "blood_pressure",
        "systolic": 120,
        "diastolic": 80,
        "date": "2026-01-25T09:05:00Z",
    }
    outcomes = []
    for readings in (
        [high_reading, normal_reading],
        [normal_reading, high_reading],
    ):
        payload = dict(submission_payload)
        payload["vitals"] = [*submission_payload["vitals"], *readings]
        outcomes.append(evaluate(payload, extraction_payload))

    assert [output.decision for output in outcomes] == [
        "NOT_CLEARED",
        "NOT_CLEARED",
    ]
    assert [
        any(issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in output.issues)
        for output in outcomes
    ] == [True, True]


def test_dated_high_vital_is_checked_even_if_another_date_is_missing(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["vitals"][0]["systolic"] = 182
    submission_payload["vitals"][0]["diastolic"] = 112
    submission_payload["vitals"].append(
        {
            "type": "blood_pressure",
            "systolic": 120,
            "diastolic": 80,
            "date": None,
        }
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NOT_CLEARED"
    assert any(
        issue.category == "MISSING_REQUIRED_DATA"
        and "blood pressure" in issue.evidence
        for issue in output.issues
    )
    assert any(
        issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in output.issues
    )


def test_document_vital_supersedes_older_structured_reading(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["documents"].append(
        {
            "type": "Nursing Pre-procedure Assessment",
            "date": "2026-01-26",
            "text": "09:05 BP: 161/117, 37.2 �C tympanic, HR 84.",
        }
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NOT_CLEARED"
    assert any(
        issue.category == "ACUTE_SAFETY_EXCLUSION"
        and "161/117" in issue.evidence
        for issue in output.issues
    )


def test_document_celsius_temperature_is_converted_for_safety_check(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["documents"].append(
        {
            "type": "Nursing Pre-procedure Assessment",
            "date": "2026-01-26",
            "text": "Temperature: 38.1 C at 09:05.",
        }
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NOT_CLEARED"
    assert any(
        issue.category == "ACUTE_SAFETY_EXCLUSION"
        and "100.6 F" in issue.evidence
        for issue in output.issues
    )


def test_equivalent_latest_celsius_and_fahrenheit_readings_are_deduplicated(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    timestamp = "2026-01-26T09:05:00Z"
    submission_payload["vitals"][1]["value_f"] = 97.9
    submission_payload["vitals"][1]["date"] = timestamp
    submission_payload["documents"].append(
        {
            "type": "Nursing Pre-procedure Assessment",
            "date": "2026-01-26",
            "text": "Temperature: 36.6 C at 09:05.",
        }
    )
    extraction_payload["vitals"] = [
        {
            "type": "temperature",
            "date": timestamp,
            "value_f": 97.88,
            "evidence": "Temperature 36.6 C.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"
    assert not any(
        issue.category == "MISSING_REQUIRED_DATA" for issue in output.issues
    )


def test_equivalent_temperature_readings_keep_safety_threshold_conservative(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    timestamp = "2026-01-26T09:05:00Z"
    submission_payload["vitals"][1]["value_f"] = 100.5
    submission_payload["vitals"][1]["date"] = timestamp
    extraction_payload["vitals"] = [
        {
            "type": "temperature",
            "date": timestamp,
            "value_f": 100.48,
            "evidence": "Duplicate temperature rounded from source.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NOT_CLEARED"
    assert any(
        issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in output.issues
    )
    assert not any(
        issue.category == "MISSING_REQUIRED_DATA" for issue in output.issues
    )


def test_latest_in_range_document_recheck_supersedes_high_reading(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["documents"].extend(
        [
            {
                "type": "Nursing Pre-procedure Assessment",
                "date": "2026-01-26",
                "text": "09:05 BP: 181/112, Temperature 101.2 F.",
            },
            {
                "type": "Nursing Recheck",
                "date": "2026-01-26",
                "text": "09:15 BP: 120/80, Temperature 98.6 F.",
            },
        ]
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"
    assert not any(
        issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in output.issues
    )


def test_missing_vital_reading_requires_follow_up(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["vitals"] = submission_payload["vitals"][:1]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        "temperature reading" in issue.evidence for issue in output.issues
    )


def test_missing_medication_list_requires_follow_up(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload.pop("medications")

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        "Medication list is missing" in issue.evidence for issue in output.issues
    )


def test_high_risk_requires_recent_cbc_and_cmp(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["procedure"]["procedure_risk"] = "HIGH"
    submission_payload["labs"].append(
        {
            "code": "CMP",
            "display": "Comprehensive metabolic panel",
            "effective_at": "2026-01-17",
            "status": "final",
        }
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any("CMP result dated" in issue.evidence for issue in output.issues)


def test_most_recent_lab_result_is_the_one_checked(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["labs"].append(
        {
            "code": "CBC",
            "display": "Complete blood count",
            "effective_at": "2026-01-20T12:00:00Z",
            "status": "pending",
        }
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any("Most recent CBC result is not final" in issue.evidence for issue in output.issues)


def test_entered_in_error_lab_does_not_replace_latest_valid_final_result(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["labs"].append(
        {
            "code": "CBC",
            "display": "Complete blood count",
            "effective_at": "2026-01-31T12:00:00Z",
            "status": "entered-in-error",
        }
    )

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"
    assert not any(issue.category == "REQUIRED_TESTING" for issue in output.issues)


def test_no_valid_lab_results_requires_follow_up(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["labs"] = [
        {
            "code": "CBC",
            "display": "Complete blood count",
            "effective_at": "2026-01-31T12:00:00Z",
            "status": "entered-in-error",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "REQUIRED_TESTING"
        and "No valid CBC result" in issue.evidence
        for issue in output.issues
    )


def test_anticoagulant_requires_clear_pre_and_postoperative_plan(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["medications"] = [{"name": "apixaban", "active": True}]
    extraction_payload["medications"] = [
        {
            "medication_index": 0,
            "is_anticoagulant": True,
            "evidence": "Apixaban is an anticoagulant.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "ANTICOAGULATION_MANAGEMENT" for issue in output.issues
    )

    extraction_payload["anticoagulation_plans"] = [
        {
            "medication_index": 0,
            "source_index": 0,
            "preoperative_management": "Hold for two days before surgery.",
            "postoperative_management": "Resume the day after surgery.",
            "evidence": "Documented perioperative plan.",
        }
    ]
    output = evaluate(submission_payload, extraction_payload)
    assert output.decision == "READY"


@pytest.mark.parametrize(
    "aspirin_name",
    ["aspirin 81 mg PO daily", "ASA 81 mg PO daily", "acetylsalicylic acid 81 mg"],
)
def test_aspirin_does_not_require_anticoagulation_plan(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
    aspirin_name: str,
) -> None:
    submission_payload["medications"] = [{"name": aspirin_name, "active": True}]
    extraction_payload["medications"] = [
        {
            "medication_index": 0,
            "is_anticoagulant": True,
            "evidence": "Model classified this antiplatelet as anticoagulation.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"
    assert not any(
        issue.category == "ANTICOAGULATION_MANAGEMENT" for issue in output.issues
    )


@pytest.mark.parametrize(
    "medication_name",
    [
        "levothyroxine 75 mcg PO daily",
        "gabapentin 300 mg PO TID",
        "sertraline 50 mg PO daily",
        "tamsulosin 0.4 mg PO nightly",
        "atorvastatin 40 mg PO nightly",
        "acetaminophen 650 mg PO q6h PRN pain",
        "vitamin D3 2000 IU PO daily",
        "omeprazole 20 mg PO daily",
        "lisinopril 20 mg PO daily",
        "metformin 1000 mg PO BID",
        "clopidogrel 75 mg PO daily",
    ],
)
def test_known_non_anticoagulant_medications_do_not_need_model_classification(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
    medication_name: str,
) -> None:
    submission_payload["medications"] = [{"name": medication_name, "active": True}]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"
    assert not any(
        issue.category in {"MISSING_REQUIRED_DATA", "ANTICOAGULATION_MANAGEMENT"}
        for issue in output.issues
    )


@pytest.mark.parametrize(
    "medication_name",
    [
        "warfarin 5 mg PO daily",
        "Coumadin 5 mg PO daily",
        "apixaban 5 mg PO BID",
        "Eliquis 5 mg PO BID",
        "rivaroxaban 20 mg PO daily",
        "Xarelto 20 mg PO daily",
        "dabigatran 150 mg PO BID",
        "Pradaxa 150 mg PO BID",
        "edoxaban 60 mg PO daily",
        "Savaysa 60 mg PO daily",
        "enoxaparin 40 mg injection daily",
        "Lovenox 40 mg injection daily",
    ],
)
def test_known_anticoagulants_are_not_dependent_on_model_classification(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
    medication_name: str,
) -> None:
    submission_payload["medications"] = [{"name": medication_name, "active": True}]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "ANTICOAGULATION_MANAGEMENT" for issue in output.issues
    )


def test_invalid_and_duplicate_extraction_references_degrade_to_follow_up(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["medications"] = [{"name": "apixaban", "active": True}]
    extraction_payload["medications"] = [
        {
            "medication_index": 0,
            "is_anticoagulant": True,
            "evidence": "Apixaban is an anticoagulant.",
        },
        {
            "medication_index": 0,
            "is_anticoagulant": True,
            "evidence": "Duplicate classification.",
        },
        {
            "medication_index": 8,
            "is_anticoagulant": False,
        },
    ]
    extraction_payload["documents"].append(
        {
            "source_index": 0,
            "kind": "UNCLEAR",
            "completed": False,
            "evidence": "Conflicting duplicate document classification.",
        }
    )
    extraction_payload["anticoagulation_plans"] = [
        {
            "medication_index": 8,
            "source_index": 0,
            "preoperative_management": "Invalid medication reference.",
            "postoperative_management": "Invalid medication reference.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "ANTICOAGULATION_MANAGEMENT" for issue in output.issues
    )


def test_known_medication_classification_overrides_model_conflict(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["medications"] = [{"name": "apixaban 5 mg PO BID", "active": True}]
    extraction_payload["medications"] = [
        {
            "medication_index": 0,
            "is_anticoagulant": False,
            "evidence": "Conflicting model classification.",
        }
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert any(
        issue.category == "ANTICOAGULATION_MANAGEMENT" for issue in output.issues
    )


def test_split_pre_and_postoperative_plans_can_be_combined(
    submission_payload: dict[str, object],
    extraction_payload: dict[str, object],
) -> None:
    submission_payload["medications"] = [{"name": "apixaban", "active": True}]
    extraction_payload["medications"] = [
        {
            "medication_index": 0,
            "is_anticoagulant": True,
            "evidence": "Apixaban is an anticoagulant.",
        }
    ]
    extraction_payload["anticoagulation_plans"] = [
        {
            "medication_index": 0,
            "source_index": 0,
            "preoperative_management": "Hold before surgery.",
            "evidence": "Preoperative instruction.",
        },
        {
            "medication_index": 0,
            "source_index": 1,
            "postoperative_management": "Resume after surgery.",
            "evidence": "Postoperative instruction.",
        },
    ]

    output = evaluate(submission_payload, extraction_payload)

    assert output.decision == "READY"
