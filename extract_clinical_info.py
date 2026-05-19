"""
Extract structured epilepsy-related information from synthetic clinical notes.

This script has two modes:

1. rule-based:
   Uses simple keyword rules. It does not require an LLM.

2. llm:
   Uses a local Ollama model to extract the same fields.

Important:
- The notes are synthetic.
- This is a portfolio demo, not a clinical decision-support tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from ollama import Client, ResponseError
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Allowed labels
# ---------------------------------------------------------------------

DiagnosisLabel = Literal[
    "focal epilepsy",
    "temporal lobe epilepsy",
    "generalised epilepsy",
    "functional non-epileptic seizures",
    "epilepsy",
    "unclear",
]

SeizureTypeLabel = Literal[
    "focal impaired awareness seizures",
    "focal seizures",
    "nocturnal seizures",
    "generalised seizures",
    "functional non-epileptic seizures",
    "unclear",
]

TreatmentResponseLabel = Literal[
    "ongoing seizures despite treatment",
    "drug-resistant or persistent seizures despite treatment",
    "controlled on treatment",
    "partial improvement",
    "unclear",
]

ClinicalOutcomeLabel = Literal[
    "persistent seizures",
    "seizure-free or controlled",
    "reduced seizure frequency",
    "unclear",
]

InvestigationLabel = Literal["EEG", "MRI", "ECG", "video-EEG"]


class ClinicalExtraction(BaseModel):
    """Expected structured output for one clinical note."""

    note_id: str = Field(description="Unique note ID")
    diagnosis: DiagnosisLabel = Field(description="Main diagnosis only")
    seizure_type: SeizureTypeLabel = Field(description="Seizure/event type only")
    medications: list[str] = Field(description="Anti-seizure medication names only")
    treatment_response: TreatmentResponseLabel = Field(description="Standard treatment response label")
    comorbidities: list[str] = Field(description="Comorbidities mentioned in the note")
    investigations: list[InvestigationLabel] = Field(description="Investigation names only")
    clinical_outcome: ClinicalOutcomeLabel = Field(description="Standard clinical outcome label")
    evidence_quote: str = Field(description="Exact sentence from the note supporting the extraction")
    confidence: Literal["low", "medium", "high"] = Field(description="Extraction confidence")


# ---------------------------------------------------------------------
# Simple dictionaries used by the rule-based baseline
# ---------------------------------------------------------------------

MEDICATIONS = [
    "levetiracetam",
    "lamotrigine",
    "valproate",
    "carbamazepine",
    "lacosamide",
    "clobazam",
    "topiramate",
    "ethosuximide",
    "oxcarbazepine",
    "phenytoin",
    "brivaracetam",
]

COMORBIDITIES = [
    "anxiety",
    "depression",
    "migraine",
    "sleep disturbance",
    "memory complaints",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def find_terms(text: str, terms: list[str]) -> list[str]:
    """Return terms found in the note."""
    text_lower = text.lower()
    return [term for term in terms if term.lower() in text_lower]


def find_investigations(text: str) -> list[str]:
    """Return investigation names only, without findings."""
    text_lower = text.lower()
    investigations = []

    if "video-eeg" in text_lower or "video eeg" in text_lower:
        investigations.append("video-EEG")
    elif "eeg" in text_lower:
        investigations.append("EEG")

    if "mri" in text_lower:
        investigations.append("MRI")

    if "ecg" in text_lower:
        investigations.append("ECG")

    return investigations


def find_evidence_sentence(text: str) -> str:
    """Return one useful sentence from the note as evidence."""
    keywords = [
        "drug-resistant",
        "failed",
        "continues",
        "ongoing",
        "controlled",
        "no seizures",
        "no events",
        "reduced",
        "persist",
        "seizure frequency",
    ]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            return sentence

    return sentences[0] if sentences else "unclear"


# ---------------------------------------------------------------------
# Rule-based extraction
# ---------------------------------------------------------------------

def rule_based_extract(note_id: str, clinical_note: str) -> ClinicalExtraction:
    """Extract fields using simple keyword matching."""
    text = clinical_note.lower()

    # Diagnosis
    if "functional non-epileptic" in text or "non-epileptic seizures" in text:
        diagnosis = "functional non-epileptic seizures"
    elif "temporal lobe" in text:
        diagnosis = "temporal lobe epilepsy"
    elif "generalised epilepsy" in text or "generalized epilepsy" in text or "childhood-onset generalised" in text:
        diagnosis = "generalised epilepsy"
    elif "focal epilepsy" in text:
        diagnosis = "focal epilepsy"
    elif "epilepsy" in text or "epileptic seizures" in text:
        diagnosis = "epilepsy"
    else:
        diagnosis = "unclear"

    # Seizure type
    if "focal impaired awareness" in text:
        seizure_type = "focal impaired awareness seizures"
    elif "functional non-epileptic" in text or "non-epileptic seizures" in text:
        seizure_type = "functional non-epileptic seizures"
    elif "nocturnal seizures" in text:
        seizure_type = "nocturnal seizures"
    elif "focal seizures" in text or "focal epilepsy" in text:
        seizure_type = "focal seizures"
    elif "generalised" in text or "generalized" in text:
        seizure_type = "generalised seizures"
    else:
        seizure_type = "unclear"

    # Response and outcome
    if "drug-resistant" in text or "failed trials" in text or "failed" in text or "despite multiple" in text:
        treatment_response = "drug-resistant or persistent seizures despite treatment"
        clinical_outcome = "persistent seizures"
        confidence = "high"
    elif "continues" in text or "ongoing" in text or "persist" in text:
        treatment_response = "ongoing seizures despite treatment"
        clinical_outcome = "persistent seizures"
        confidence = "high"
    elif "controlled" in text or "no seizures" in text or "no events" in text:
        treatment_response = "controlled on treatment"
        clinical_outcome = "seizure-free or controlled"
        confidence = "high"
    elif "reduced" in text:
        treatment_response = "partial improvement"
        clinical_outcome = "reduced seizure frequency"
        confidence = "medium"
    else:
        treatment_response = "unclear"
        clinical_outcome = "unclear"
        confidence = "medium"

    return ClinicalExtraction(
        note_id=note_id,
        diagnosis=diagnosis,
        seizure_type=seizure_type,
        medications=find_terms(clinical_note, MEDICATIONS),
        treatment_response=treatment_response,
        comorbidities=find_terms(clinical_note, COMORBIDITIES),
        investigations=find_investigations(clinical_note),
        clinical_outcome=clinical_outcome,
        evidence_quote=find_evidence_sentence(clinical_note),
        confidence=confidence,
    )


# ---------------------------------------------------------------------
# Ollama LLM extraction
# ---------------------------------------------------------------------

def llm_extract(
    note_id: str,
    clinical_note: str,
    model: str = "llama3.1",
    host: str = "http://localhost:11434",
) -> ClinicalExtraction | None:
    """Extract fields using a local Ollama model."""
    client = Client(host=host)

    prompt = f"""
You are extracting structured information from a synthetic clinical note.

Return valid JSON only.
Use only the allowed labels from the schema.
Do not invent new labels.
Do not combine diagnosis and seizure type.
For investigations, return names only: EEG, MRI, ECG, or video-EEG.
The evidence_quote must be one exact sentence copied from the note.

Note ID:
{note_id}

Synthetic clinical note:
{clinical_note}
""".strip()

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=ClinicalExtraction.model_json_schema(),
            options={"temperature": 0},
        )

        content = response["message"]["content"]
        return ClinicalExtraction.model_validate_json(content)

    except ResponseError as error:
        logger.error("Ollama error for note %s: %s", note_id, error)
        return None

    except ValidationError as error:
        logger.warning("The LLM returned invalid JSON for note %s: %s", note_id, error)
        return None

    except Exception as error:
        logger.warning("LLM extraction failed for note %s: %s", note_id, error)
        return None


# ---------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------

def run_extraction(
    input_path: Path,
    output_path: Path,
    backend: str,
    model: str,
    host: str,
) -> list[dict]:
    """Run extraction for all notes in the CSV file."""
    df = pd.read_csv(input_path)

    if not {"note_id", "clinical_note"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'note_id' and 'clinical_note' columns.")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting ({backend})"):
        note_id = str(row["note_id"])
        clinical_note = str(row["clinical_note"])

        if backend == "rule-based":
            extraction = rule_based_extract(note_id, clinical_note)
        elif backend == "llm":
            extraction = llm_extract(note_id, clinical_note, model=model, host=host)
            if extraction is None:
                continue
        else:
            raise ValueError("backend must be either 'rule-based' or 'llm'.")

        results.append(extraction.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured fields from synthetic epilepsy clinical notes."
    )

    parser.add_argument("--input", type=Path, default=Path("data/synthetic_clinical_notes.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/example_extractions.json"))
    parser.add_argument("--backend", choices=["rule-based", "llm"], default="rule-based")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.1"))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    args = parser.parse_args()

    run_extraction(
        input_path=args.input,
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        host=args.host,
    )


if __name__ == "__main__":
    main()
