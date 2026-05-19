# Clinical Text Extraction Demo with Local Ollama LLMs

A lightweight Python project for extracting structured epilepsy-related information from free-text clinical notes.

This repository demonstrates how a **local open-source LLM running through Ollama** can convert synthetic clinical notes into a strict JSON schema. It also includes a simple rule-based baseline for comparison and offline testing.

> **Important:** The data used in this project is fully synthetic and is included only for demonstration purposes.

---

## Project purpose

Clinical information in Electronic Health Records is often stored as free text. This makes it difficult to use directly for downstream analysis or prediction modelling.

This project shows a small example of how free-text epilepsy notes can be converted into structured variables such as:

- diagnosis
- seizure type
- anti-seizure medications
- treatment response
- comorbidities
- investigations
- clinical outcome
- evidence quote
- extraction confidence

For example, a note such as:

```text
Patient with focal epilepsy. The patient has tried levetiracetam and lamotrigine but continues to report monthly seizures. Past history includes anxiety.
```

can be converted into:

```json
{
  "diagnosis": "focal epilepsy",
  "seizure_type": "focal seizures",
  "medications": ["levetiracetam", "lamotrigine"],
  "treatment_response": "ongoing seizures despite treatment",
  "comorbidities": ["anxiety"],
  "clinical_outcome": "persistent seizures",
  "confidence": "high"
}
```

---

## Why this project is useful

This demo is relevant to clinical AI and health data science because it shows:

1. use of local LLMs for clinical text extraction;
2. conversion of unstructured notes into structured JSON;
3. schema validation using Pydantic;
4. reproducible Python code;
5. safe use of synthetic data only;
6. a simple baseline for comparison with the LLM method.

---

## Repository structure

```text
llm-epilepsy-clinical-text-extraction/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── synthetic_clinical_notes.csv
├── notebooks/
│   └── demo_llm_extraction.ipynb
├── src/
│   └── extract_clinical_info.py
└── outputs/
    └── example_extractions.json
```

---

## Method overview

The project has two extraction modes.

### 1. Rule-based baseline

This mode does not use an LLM. It uses simple keyword rules.

```bash
python src/extract_clinical_info.py --backend rule-based
```

This is included so the project can run immediately without installing or running an LLM.

### 2. Ollama LLM extraction

This mode uses a local LLM through Ollama.

```bash
python src/extract_clinical_info.py --backend llm --model llama3.1
```

The LLM is asked to return structured JSON following a predefined Pydantic schema.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-epilepsy-clinical-text-extraction.git
cd llm-epilepsy-clinical-text-extraction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Ollama setup

Install Ollama from the official Ollama website.

Then pull the Llama model:

```bash
ollama pull llama3.1
```

Check that the model works:

```bash
ollama run llama3.1
```

You can test it with:

```text
Return only JSON: {"status": "ok"}
```

---

## Running the project

### Run the rule-based baseline

```bash
python src/extract_clinical_info.py --backend rule-based
```

### Run the Ollama LLM extractor

```bash
python src/extract_clinical_info.py --backend llm --model llama3.1
```

### Use another Ollama model

For example:

```bash
python src/extract_clinical_info.py --backend llm --model llama3.2
```

or:

```bash
python src/extract_clinical_info.py --backend llm --model mistral
```

---

## Input data

The input file should be a CSV file with these columns:

```text
note_id, clinical_note
```

Default input path:

```text
data/synthetic_clinical_notes.csv
```

---

## Output

The extracted results are saved as JSON.

Default output path:

```text
outputs/example_extractions.json
```

Each extracted note follows this schema:

```json
{
  "note_id": "N001",
  "diagnosis": "focal epilepsy",
  "seizure_type": "focal impaired awareness seizures",
  "medications": ["levetiracetam", "lamotrigine"],
  "treatment_response": "ongoing seizures despite treatment",
  "comorbidities": ["anxiety"],
  "investigations": ["EEG", "MRI"],
  "clinical_outcome": "persistent seizures",
  "evidence_quote": "The patient has tried levetiracetam and lamotrigine but continues to report monthly seizures.",
  "confidence": "high"
}
```

---

## Jupyter notebook

A notebook walkthrough is available at:

```text
notebooks/demo_llm_extraction.ipynb
```

It shows how to:

- load the synthetic notes;
- run the rule-based baseline;
- run the Ollama LLM extractor;
- view the extracted fields as a table.

---

## Ethical note

This project is for educational and portfolio purposes only.

It is **not** a clinical decision-support system and should not be used for diagnosis, treatment planning, triage, or patient management.

Do not upload real patient records, identifiable health data, or confidential NHS/clinical data to this repository.

---

## Possible future improvements

- Add evaluation against manually created labels.
- Compare multiple Ollama models.
- Add more synthetic notes with diverse epilepsy presentations.
- Add named entity recognition evaluation.
- Add longitudinal synthetic patient timelines.
- Add a small Streamlit interface.
- Use extracted variables for a simple prediction-modelling demonstration.

---

## License

This project is released under the MIT License.
