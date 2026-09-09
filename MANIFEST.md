# Supplementary Materials Manifest

This repository contains the complete replication materials, raw model completions, detector calibration splits, attack corpora, reproduction scripts, and manuscript tables/figures for:

> **What Survives, What Matters: The False Decoupling of Retrieval and Behavior in Adversarial RAG**

Every table and figure in the manuscript can be recomputed deterministically offline in seconds without requiring GPU hardware or API keys.

---

## 1. Top-Level Metadata

| Path | Format | Description |
|---|---|---|
| `README.md` | Markdown | Comprehensive repository guide, quickstart instructions, reproduction commands |
| `MANIFEST.md` | Markdown | Detailed file inventory with row counts, byte sizes, and manuscript mappings |
| `LICENSE` | Text | MIT Open Source License |
| `CITATION.cff` | YAML | Citation Metadata in standard CFF format |
| `environment.yml` | YAML | Pinned Conda environment specification |
| `requirements.txt` | Text | Pinned pip dependencies |

---

## 2. Configuration Files (`configs/`)

| Path | Format | Description | Backs |
|---|---|---|---|
| `configs/retrieval.yaml` | YAML | Retrieval pipeline config (MiniLM, BM25, convex fusion grid, reranker) | Section 4, Table 2 |
| `configs/detectors.yaml` | YAML | Specifications for detectors D0-D6, training corpora, calibration quantiles | Section 5, Tables 3, 12 |
| `configs/behavior.yaml` | YAML | Downstream LLM generation settings, canary definitions, position settings | Section 6, Table 4 |
| `configs/models.yaml` | YAML | Specifications for Qwen2.5 (1.5B, 3B), Phi-3.5-mini, and GPT-5-mini | Sections 6-7, Tables 20-21 |
| `configs/environment.yaml` | YAML | Standard conda environment spec for offline replication | Section 7 |

---

## 3. Prompts (`prompts/`)

| Path | Format | Description |
|---|---|---|
| `prompts/system_plain.txt` | Text | Standard undefended RAG system prompt |
| `prompts/system_hardened.txt` | Text | Hardened system prompt with untrusted data directive |
| `prompts/system_warning.txt` | Text | Warning variant used in position sensitivity study |
| `prompts/qa_prompt_template.txt` | Text | Template combining retrieved context documents and question |
| `prompts/tool_selection_prompt.txt` | Text | Instruction prompt for the agent tool selection study |
| `prompts/prompt_templates.json` | JSON | Machine-readable index of all system and evaluation prompts |
| `prompts/tool_selection_schema.json` | JSON | Tool definitions and JSON schema for tool-use experiments |
| `prompts/output_parsing.md` | Markdown | Documentation of canary parsing and regular expressions |

---

## 4. Datasets and Evaluation Records (`data/`)

### Manifests & Corpora

| Path | Rows | Size | Description | Backs |
|---|---:|---:|---|---|
| `data/query_manifest.csv` | 300 | 25 KB | The 300 SciFact target queries used for injection and retrieval | Sections 4-5 |
| `data/calibration_manifest.csv` | 1,500 | 1.8 MB | Clean SciFact documents used to calibrate detector thresholds at target FPRs | Section 5 |
| `data/fpr_eval_manifest.csv` | 1,500 | 1.8 MB | Held-out clean documents used to measure realised FPR | Section 5 |
| `data/attack_corpus/attack_documents.parquet` | 1,800 | 1.1 MB | Generated attack documents for A0, A1, A2, A3, A4, A6 across 300 queries | Section 5 |
| `data/attack_corpus/a5_documents.parquet` | 1,260 | 450 KB | Optimized A5 documents from greedy score-guided search | Section 5, Table 13 |
| `data/attack_corpus/A0-A6/*.csv` | 300-1260 | ~100-500 KB | Per-attack CSV exports for easy inspection | Section 5 |

### Episode Evaluation Records (`data/records/`)

| Path | Rows | Size | Description | Backs |
|---|---:|---:|---|---|
| `data/records/qwen_per_episode.parquet` | 35,200 | 597 KB | Complete episode completions and scores for Qwen2.5-1.5B/3B | Tables 4, 5, 6 |
| `data/records/phi_per_episode.parquet` | 10,400 | 280 KB | Complete episode completions and scores for Phi-3.5-mini | Table 20 |
| `data/records/gpt5_raw_responses.jsonl` | 600 | 414 KB | Raw unedited API completions, token counts, and latency for GPT-5-mini | Table 21 |
| `data/records/gpt5_episode_results.csv` | 14,400 | 5.8 MB | Episode-level evaluation for GPT-5-mini across all 8 detectors and 3 FPRs | Tables 21, 25 |
| `data/records/gpt5_audit_log.json` | - | 120 B | Audit metadata for Azure safety filters and empty reasoning calls | Table 25 |
| `data/records/sensitivity_analysis.csv` | 72 | 11 KB | GPT-5-mini sensitivity evaluation across all treatments and detectors | Table 25 |
| `data/records/qwen_truncation_audit.csv` | 88 | 9 KB | Sequence length audit across all 35,200 episodes proving 0% truncation | Section 6 |
| `data/records/detector_scores.csv` | 168 | 12 KB | Defended/clean nDCG, ASR, and utility cost per attack, detector, and FPR | Table 3 |
| `data/records/detector_scores.parquet` | 168 | 9 KB | Parquet format of detector scores matrix | Table 3 |
| `data/records/detector_decisions.parquet` | 133,380 | 215 KB | Exposure (E) and compliance (C) indicators for every query and detector | Tables 5, 6 |
| `data/records/retrieval_per_query.parquet` | 78,267 | 260 KB | Per-query retrieval scores across all 6 BEIR collections | Table 2 |
| `data/records/a5_optimisation_records.csv` | 1,260 | 66 KB | Query counts, detector calls, and evasion outcomes for A5 search | Table 13 |
| `data/records/detector_calibration.csv` | 21 | 1 KB | Realised vs target FPRs for all detectors | Table 12 |

---

## 5. Source Reference Modules (`src/`)

- `src/attack_generation/`: Reference implementation of the A0-A6 attack ladder.
- `src/detector/`: Implementations of heuristic, embedding, transformer, and ensemble detectors.
- `src/scoring/`: QA exact match, token-level F1, canary regex matching, and tool selection metrics.
- `src/analysis/`: Exposure-compliance decomposition, McNemar tests, and sensitivity analysis.
- `src/reproduction/`: Automated rebuilder for manuscript tables.
- `src/plotting/`: Publication styling and figure generation routines.

---

## 6. Results (`results/`)

### Tables (`results/tables/`)
- `table1_attack_ladder.csv`: Attack capabilities, knowledge, and perturbation spaces.
- `table2_full_corpus_retrieval.csv`: Full-corpus BEIR retrieval performance under matched budgets.
- `table3_selected_maximum_bound.csv`: Candidate survival rates across detectors and target FPRs.
- `table4_undefended_behaviour.csv`: Undefended Qwen2.5 behavioral ASR and QA utility.
- `table5_exposure_compliance_endtoend.csv`: Exposure-compliance decomposition.
- `table6_detector_conditioned_matrix.csv`: End-to-end compromise under detector-conditioned replay.
- `table14_latency.csv`: Latency profiles across detectors and models.
- `table20_phi_behavioural.csv`: Phi-3.5-mini behavioral replication.
- `table21_gpt5_replication.csv`: GPT-5-mini behavioral replication.
- `table25_gpt5_sensitivity.csv`: GPT-5-mini sensitivity analysis across treatment regimes.

### Figures (`results/figures/`)
- `figure3.png` / `figure3.pdf`: Fusion alpha headroom and flatness curves.
- `figure4.png` / `figure4.pdf`: Exposure vs behavioral ordering across detector operating points.
- `figure5.png` / `figure5.pdf`: Evaluation suite summary and flat-query distribution.
- `figure6.png` / `figure6.pdf`: Three failure modes of candidate survival.
- `figure7.png` / `figure7.pdf`: Conversion rate and unconditional QA compliance (verified updated legend).

---

## 7. Reproduction Scripts (`examples/`)

| Script | Purpose | Execution Time |
|---|---|---|
| `examples/reproduce_main.py` | Recomputes Tables 2, 3, 4, 14, 20, 21 from raw records | ~2 seconds (CPU) |
| `examples/reproduce_sensitivity.py` | Audits raw completions and reproduces Table 25 sensitivity analysis | ~1 second (CPU) |
| `examples/reproduce_figures.py` | Verifies and validates Figures 2-7 publication assets | ~1 second (CPU) |
| `examples/reproduce_all.py` | Comprehensive verification suite across all tables | ~3 seconds (CPU) |
