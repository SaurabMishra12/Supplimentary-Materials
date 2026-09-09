# Supplementary Materials: What Survives, What Matters

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-verified-success.svg)]()

This repository contains the complete supplementary materials, datasets, raw model completions, attack corpora, and reproduction code for:

> **What Survives, What Matters: The False Decoupling of Retrieval and Behavior in Adversarial RAG**  
> *Under review at Transactions on Machine Learning Research (TMLR)*

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

## Quickstart

### 1. Environment Setup

Clone the repository and install dependencies using Conda or virtualenv:

```bash
git clone https://anonymous.4open.science/r/cognisync-supplementary
cd Supplimentary-Materials

# Using conda:
conda env create -f environment.yml
conda activate cognisync

# Or using pip:
pip install -r requirements.txt
```

### 2. Fast Offline Reproduction

#### Recomputing Core Tables (Tables 2, 3, 4, 14, 20, 21)
Rebuilds the primary tables from raw evaluation records and verifies numerical equivalence:
```bash
python examples/reproduce_main.py
```

#### Reproducing GPT-5-mini Sensitivity Analysis (Table 25)
Audits the 18 Azure content-filter blocked / empty completions and recomputes compliance across:
1. **Non-compliant** (baseline conservative assignment, $N=300$)
2. **Missing** (complete cases only, $N=282$)
3. **Compliant** (upper-bound assignment, $N=300$)

```bash
python examples/reproduce_sensitivity.py
```

#### Validating Figures (Figures 2 through 7)
Verifies all publication figure assets (including Figure 7 with the corrected legend label `unconditional QA compliance`):
```bash
python examples/reproduce_figures.py
```

---

## Key Experimental Findings

1. **Exposure–Behavior Reversal**:
   - Attacks designed to minimize retrieval-detector visibility (e.g., $A_3$ semantic camouflage) achieve high candidate exposure ($P(E) = 0.55$) but suffer low downstream behavioral compliance ($P(C|E) = 0.04$ on GPT-5-mini).
   - In contrast, length-matched payload attacks ($A_4$) achieve lower exposure ($P(E) = 0.23$) but dramatically higher behavioral compliance ($P(C|E) = 0.27$ on GPT-5-mini).
   - End-to-end vulnerability is dominated by downstream compliance, completely reversing the ranking inferred from retrieval-layer metrics alone ($p < 10^{-13}$).

2. **Cross-Model Replication**:
   - The ranking reversal holds across **Qwen2.5-1.5B**, **Qwen2.5-3B**, **Phi-3.5-mini**, and **GPT-5-mini**.

3. **Robustness to Missing Data**:
   - Treating platform-filtered episodes as non-compliant, dropping them, or treating them as compliant all preserve the reversal with $p < 4 \times 10^{-14}$.

---

## License & Citation

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{anonymous2026whatsurvives,
  title={What Survives, What Matters: The False Decoupling of Retrieval and Behavior in Adversarial RAG},
  author={Anonymous},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://anonymous.4open.science/r/cognisync-supplementary}
}
```
