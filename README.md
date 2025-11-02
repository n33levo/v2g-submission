# V2G Technical Assessment Implementation

This repository contains my submission for the **Agentic Variant-to-Gene (V2G) Framework Research Internship** technical assessment. The work is organized along the three required parts and focuses on reasoning clarity, AI/ML thinking, and reproducible execution.

---

## 1. Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Optional (for Part 2 summaries):

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## 2. Deliverables Overview

| Part | File | Purpose | Key Design Decisions |
|------|------|---------|----------------------|
| Part 1 | `part1_variant_annotation.py` | Complete variant annotation pipeline using the provided `sample_variants.csv` | Rate-limited Ensembl queries, validation before expensive calls, MyVariant.info enrichment, dependency-injected components |
| Part 2 | `part2_agentic_workflow.py` | Conceptual AI agent workflow with pseudocode integrations for GTEx, gnomAD, UniProt | Tool abstraction layer, adaptive strategy planner, confidence scoring, self-verification, optional OpenAI synthesis |
| Part 3 | `part3_scientific_reasoning.md` | Structured responses to the scientific reasoning prompts | Focus on AI-assisted validation, ML-driven experimentation, and causal limitations |
| Architecture | `architecture.md` | Mermaid diagram and narrative for the Part 2 workflow | Shows end-to-end reasoning flow |

Running instructions:

```bash
python part1_variant_annotation.py    # Generates variant_summary.json 
python part2_agentic_workflow.py      # Demonstrates the agentic workflow 
```

`part3_scientific_reasoning.md` is a prose document and does not require execution.

---

## 3. Reasoning & Workflow Structure

### Part 1 – Core Programming & Data Reasoning
- **Why**: Validating inputs before annotation prevents wasteful API calls and surfaces data quality issues early.
- **How**: A `VariantAnnotator` orchestrates validation, Ensembl annotation, and MyVariant.info enrichment; results are saved to JSON for reproducibility.
- **AI/Biology Link**: Uses real biomedical APIs while applying software engineering safeguards (rate limiting, retries, logging).

### Part 2 – AI/ML Agentic Workflow
- **Why**: Demonstrates understanding of modular reasoning agents where evidence comes from heterogeneous biological APIs.
- **How**: Pseudocode tools expose consistent `construct_query` / `parse_response` / `calculate_confidence` interfaces; a `V2GReasoningAgent` plans, collects evidence, scores confidence, verifies, and (optionally) synthesises a report via OpenAI.
- **AI/Biology Link**: Shows how gene expression, population genetics, and protein function evidence are combined programmatically to prioritise variant–gene hypotheses.

### Part 3 – Scientific Reasoning
- **Why**: Bridges computational proposals with experimental validation and acknowledges AI limits (e.g., correlation vs. causation).
- **How**: Provides computational+experimental validation routes, outlines how agents assist researchers, and discusses limitations with mitigation tactics.

---

## 4. Thought Process Summary

- Start with data hygiene (Part 1) to avoid propagating bad inputs into downstream reasoning.
- Design agents (Part 2) around extensibility and transparency: abstract tools, explicit reasoning chains, quantitative confidence.
- Keep scientific communication (Part 3) grounded in AI/ML framing while acknowledging biological realities.
- Separate architecture documentation (`architecture.md`) for quick visual reference without cluttering the workflow code.

---

## 5. Reproducibility Notes

- External services (Ensembl, MyVariant.info) are queried live; reruns will refresh annotations.
- OpenAI integration in Part 2 is optional. Without a key the script produces deterministic summaries, demonstrating fallback design.
- Outputs such as `variant_summary.json` are not committed; regenerate by re-running Part 1.

---

For any review, start with the architecture diagram, inspect each part-specific file, and run the scripts as needed to reproduce the results. This structure keeps the submission faithful to the assessment brief while highlighting both software engineering rigour and AI/biology reasoning.
