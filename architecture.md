# V2G Agentic Workflow Architecture

```mermaid
graph TD
    A[Variant Input<br/>(variant_id, gene_symbol)] --> B[Strategy Planner]
    B --> C{Context Detected?}
    C -->|Cardiac/HCM| D[Cardiac-focused Strategy]
    C -->|Other/Unknown| E[Comprehensive Strategy]

    D --> F[Tool Orchestrator]
    E --> F

    F --> G[GTEx Tool<br/>(eQTL evidence)]
    F --> H[gnomAD Tool<br/>(population constraint)]
    F --> I[UniProt Tool<br/>(protein function)]

    G --> J[Evidence Aggregator]
    H --> J
    I --> J

    J --> K[Confidence Scoring<br/>(diversity, strength, coverage)]
    K --> L[Self-Verification<br/>(consistency, warnings)]
    L --> M{Quality Gate}

    M -->|Pass| N[LLM Summary Generator]
    M -->|Warnings| O[Flag for manual review]
```

## Component Overview

| Component | Responsibility | Notes |
|-----------|----------------|-------|
| Strategy Planner | Selects analysis strategies using disease context and gene priors | Enables adaptive reasoning (cardiac vs. general workflows) |
| Tool Orchestrator | Coordinates calls to biological data tools | Extensible via `BiologicalAPITool` abstraction |
| Evidence Aggregator | Normalises responses into unified `Evidence` objects | Supports consistent downstream scoring |
| Confidence Scoring | Weighs evidence diversity, strength, and coverage | Prevents single-source domination |
| Self-Verification | Checks for conflicts and low-confidence signals | Produces recommendations and warnings |
| LLM Summary Generator | Optional synthesis step when API key provided | Falls back to rule-based summary otherwise |

## Key Design Patterns & Concepts

1. **Tool Abstraction Pattern**  
   Concrete GTEx/gnomAD/UniProt tools implement `construct_query → parse_response → calculate_confidence`, allowing new sources to slot in without touching orchestration logic.

2. **Evidence-Based Reasoning**  
   Confidence scores are computed per source and combined with adaptive weighting; reasoning chains expose how each source influenced the final score.

3. **Agentic Workflow**  
   PLAN → QUERY → AGGREGATE → VERIFY → SYNTHESIZE mirrors modern reasoning-agent loops while keeping each phase auditable.

4. **Self-Verification Hooks**  
   The quality gate flags low average confidence, missing modalities, or duplicate sources before generating a final report.

5. **Adaptive Strategy Selection**  
   Disease context and known gene priors influence tool priority and weighting so cardiac variants receive different treatment than novel findings.

## Accuracy Assurance Strategies

| Strategy | Purpose | Implementation Hook |
|----------|---------|---------------------|
| Confidence quantification | Avoids hand-wavy decisions | Source-specific scoring functions |
| Multi-source validation | Rewards concordant evidence | Diversity bonus in scoring pipeline |
| Quality control checkpoints | Blocks weak or inconsistent findings | Self-verification module |
| Transparent reasoning chains | Enables human auditability | `reasoning_chain` log returned to caller |

## Appendix: ASCII Sketch

```
Input (Variant + Gene + Context)
              |
              v
      +--------------------------+
      |   V2G Reasoning Agent    |
      |   (Central Orchestrator) |
      +------------+-------------+
                   |
                   v
      +---------------------------+
      | Biological Data Tools     |
      | (GTEx / gnomAD / UniProt) |
      +------------+--------------+
                   |
                   v
      +---------------------------+
      | Evidence Aggregator       |
      | + Confidence Scoring      |
      +------------+--------------+
                   |
                   v
      +---------------------------+
      | Self-Verification Layer   |
      +------------+--------------+
                   |
          +--------+---------+
          |                  |
          v                  v
  Synthesised Report   Warnings for Review
```

Refer to `part2_agentic_workflow.py` for the step-by-step conceptual implementation that aligns with this architecture.
