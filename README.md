# Supplementary Materials: What Survives, What Matters


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-verified-success.svg)]()

This repository contains the complete supplementary materials, datasets, raw model completions, attack corpora, and reproduction code for:

> **What Survives, What Matters**  


All results, tables, and figures can be reproduced **offline in seconds on standard CPU hardware** without requiring API keys or external GPU access.

---

## Directory Structure

```
Supplimentary-Materials/
├── README.md                           # Main documentation & reproduction guide
├── MANIFEST.md                         # Detailed file manifest with row counts & sizes
├── LICENSE                             # MIT License
├── CITATION.cff                        # Citation File Format metadata
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # Pip dependencies
│
├── configs/                            # Configuration files
│   ├── retrieval.yaml                  # Retrieval & fusion pipeline configuration
│   ├── detectors.yaml                  # Detectors D0-D6 & calibration parameters
│   ├── behavior.yaml                   # Generation & downstream replay setup
│   ├── models.yaml                     # Model definitions (Qwen, Phi, GPT-5-mini)
│   └── environment.yaml                # Reproduction environment specification
│
├── prompts/                            # System & evaluation prompts
│   ├── system_plain.txt                # Plain undefended system prompt
│   ├── system_hardened.txt             # Hardened prompt with untrusted data directive
│   ├── system_warning.txt              # Warning system prompt
│   ├── qa_prompt_template.txt          # Standard RAG QA template
│   ├── tool_selection_prompt.txt       # Tool calling prompt
│   ├── prompt_templates.json           # Machine-readable prompt repository
│   ├── tool_selection_schema.json      # Tool inventory and schemas
│   └── output_parsing.md               # Canary parsing documentation
│
├── data/                               # Evaluation datasets and manifests
│   ├── query_manifest.csv              # 300 SciFact target queries
│   ├── calibration_manifest.csv        # 1,500 clean documents for detector calibration
│   ├── fpr_eval_manifest.csv           # 1,500 held-out clean documents for FPR evaluation
│   ├── attack_corpus/                  # Full attack corpus
│   │   ├── attack_documents.parquet    # A0, A1, A2, A3, A4, A6 documents (1,800 rows)
│   │   ├── a5_documents.parquet        # Adaptive A5 documents (1,260 rows)
│   │   └── A0-A6/                      # Individual CSV exports per attack level
│   └── records/                        # Raw experimental records & completions
│       ├── qwen_per_episode.parquet    # 35,200 Qwen2.5-1.5B/3B episode records
│       ├── phi_per_episode.parquet     # 10,400 Phi-3.5-mini episode records
│       ├── gpt5_raw_responses.jsonl    # 600 raw unedited GPT-5-mini completions & tokens
│       ├── gpt5_episode_results.csv    # 14,400 evaluated GPT-5-mini episodes across detectors
│       ├── gpt5_audit_log.json         # Safety filter audit log
│       ├── detector_scores.csv         # Candidate survival matrix across detectors & FPRs
│       ├── detector_scores.parquet     # Parquet version of detector scores
│       ├── detector_decisions.parquet  # 133,380 per-query exposure & compliance indicators
│       ├── retrieval_per_query.parquet # 78,267 per-query retrieval scores on BEIR
│       ├── a5_optimisation_records.csv # Optimization trajectories for adaptive attacker
│       ├── sensitivity_analysis.csv    # GPT-5-mini sensitivity analysis across treatments
│       └── qwen_truncation_audit.csv   # Context length audit confirming 0% prompt truncation
│
├── src/                                # Reference modular Python packages
│   ├── attack_generation/              # A0-A6 attack generator implementations
│   ├── detector/                       # D0-D6 detectors and quantile calibration
│   ├── scoring/                        # EM, token-F1, canary detection, tool selection
│   ├── analysis/                       # Exposure-compliance decomposition & McNemar test
│   ├── reproduction/                   # Automated table rebuilder
│   └── plotting/                       # Publication plotting routines (6.75in textwidth)
│
├── results/                            # Manuscript tables and figures
│   ├── tables/                         # CSV versions of Tables 1-14, 20, 21, 25
│   └── figures/                        # High-resolution publication figures (PNG & PDF)
│
└── examples/                           # Standalone reproduction scripts
    ├── reproduce_main.py               # Recomputes core reversal tables in seconds
    ├── reproduce_sensitivity.py        # Audits GPT-5 responses & computes Table 25
    ├── reproduce_figures.py            # Validates and renders Figures 2-7
    └── reproduce_all.py                # Full automated test and verification suite
```

---



