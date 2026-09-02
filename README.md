# DataMaskingETL
A proof-of-concept, local-first data pipeline that sanitizes personal info locally, uses an LLM to enrich and classify the masked context, and safely reassembles the text for GDPR-compliant financial analytics.

## The Business Value
Financial institutions want to leverage LLMs for text analytics (e.g., customer support logs, transaction notes), but sending raw data to external APIs is a major security risk. Traditional Regex scripts are often too rigid—they delete valuable business context (like software names or corporate branches) by mistaking them for personal data. 

This pipeline solves the problem by separating **local data security** from **cloud AI intelligence**.

## Architecture Overview
1. **Extract & Local Masking:** A local engine scans the text and replaces capitalized entity candidates with placeholders (e.g., `<TOKEN_1>`). The true values are locked in a local memory vault.
2. **Cloud Context Analysis:** Only the masked "skeleton" string is sent to the OpenAI API. The LLM analyzes the grammar and returns a structured JSON verdict on which tokens are sensitive (people/locations) and which are safe (products/companies).
3. **Local Reassembly:** The pipeline rebuilds the text locally. Sensitive tokens become `[REDACTED]`, while safe tokens are restored from the local vault.

## Key Engineering Features
* **Zero Data Leakage:** The local vault (`MaskedDocument`) is designed to prevent original PII from ever crossing the network boundary or leaking into error logs.
* **Fail-Closed Design:** If the API fails, times out, or misses a token, the system defaults to full redaction. It sacrifices utility, never confidentiality.
* **Deterministic Offline QA Suite:** Includes a rule-based `HeuristicClassifier` that mimics the LLM. This allows CI/CD pipelines to run automated tests locally without spending API credits or suffering from network latency.

## Next Steps: Enterprise Integration
Currently, the pipeline operates via CLI. To integrate this into a production environment, the immediate next step is **Database Integration**. By utilizing libraries like SQLAlchemy, this pipeline can be connected to a relational database to extract batches of raw logs, process them through the ETL flow, and load the sanitized text into a secure Data Warehouse schema for internal analytics teams.

## Usage

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```
**2. Run the offline QA Suite (No API key needed):**
Executes the test suite using the deterministic local heuristic classifier.

```bash
python main.py
```

**3. Run with Live OpenAI API:**
Requires OPENAI_API_KEY to be set in your environment variables.

```bash
python main.py --live --text "Text to be masked"
```
