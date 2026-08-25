# Manifest

Every file below comes from the run that produced the manuscript. Row counts are
as shipped. `examples/reproduce_all.py` rebuilds `results/` from `data/` and checks
each value against the printed one.

## Records and corpora

| File | Rows | Size | Contents | Backs |
|---|---:|---:|---|---|
| `data/records/retrieval_per_query.parquet` | 78,267 | 260 KB | per-query nDCG@10, recall@100 and MRR@10 for every first stage, collection and reranking budget, plus the per-query learned, override and oracle fusion weights | Tables 2 and 10, Figure 3(c) |
| `data/records/retrieval_summary.csv` | 126 | 10 KB | the same records aggregated per collection, system and budget | Tables 2 and 10 |
| `data/records/corpus_stats.csv` | 6 | 0 KB | document and query counts per collection | Table 10 and the 272,117 / 3,727 totals |
| `data/records/alpha_curves_fiqa.parquet` | 648 | 79 KB | per-query nDCG@10 over the 21-point fusion grid, the six fusion features, and the RRF score, for FiQA | Table 8, Figure 3(a,b) |
| `data/records/alpha_curves_scifact.parquet` | 300 | 40 KB | the same for SciFact | Table 8, Figure 3(a,b) |
| `data/records/alpha_policy_comparison.csv` | 8 | 0 KB | fusion policies on the two-collection tuning split with paired bootstrap intervals | Table 8 |
| `data/records/alpha_predictability.csv` | 3 | 0 KB | cross-validated fit of the oracle weight for three model families | Table 9 (top) |
| `data/records/alpha_feature_mi.csv` | 6 | 0 KB | mutual information between each feature and the oracle weight | Table 9 (bottom) |
| `data/records/alpha_headroom.csv` | 2 | 0 KB | per-collection headroom and flatness summary | Figure 3 |
| `data/records/candidate_survival_per_query.parquet` | 45,720 | 15 KB | per-target-query survival outcome and defended/clean nDCG@10 for every attack, detector and target FPR | Tables 3 and 12, Figure 4 |
| `data/records/candidate_survival_matrix.csv` | 168 | 12 KB | the same aggregated to the attack x detector x target matrix | Tables 3 and 12 |
| `data/records/detector_calibration.csv` | 21 | 1 KB | realised false-positive rate on the held-out clean split for each detector and target | Tables 3 and 12 |
| `data/records/detector_separability.csv` | 42 | 2 KB | threshold-free ROC-AUC of each detector against each attack configuration | Table 11 |
| `data/records/a5_optimisation_records.csv` | 1,260 | 65 KB | one row per A5 payload: final detection score, detector queries consumed, and whether it ended below the threshold | Table 13 |
| `data/records/behavioural_summary.csv` | 88 | 7 KB | compliance, exact match and token-F1 per model, prompt, configuration and context position, including the clean episode arm | Table 4, Figure 6, the Section 6 position and prompt claims |
| `data/records/tool_selection_episodes.parquet` | 400 | 4 KB | one row per tool-selection episode: the tool chosen and whether it was the attacker's | Table 4 (tool rows), Figure 6(b) |
| `data/records/detector_conditioned_indicators.parquet` | 133,380 | 35 KB | one row per replay episode with the exposure indicator E, the compliance indicator C, and their conjunction, for every attack, detector, target, model, prompt and position | Tables 5 and 6 |
| `data/records/latency_profile.csv` | 16 | 2 KB | end-to-end and per-stage latency with nDCG@10 for each collection, depth and reranking budget | Table 14, Figure 7 |
| `data/attack_corpus/attack_documents.parquet` | 1,800 | 1.1 MB | the generated adversarial documents for A0-A4 and A6, one row per configuration and target query; the poison column holds a list of chunks, three for A6 and one otherwise | the corpus every Section 5 and 6 measurement was computed on |
| `data/attack_corpus/a5_documents.parquet` | 1,260 | 153 KB | the A5 documents, re-optimised separately against each detector and operating point, with their final score, query count and evasion flag | Table 13 |
| `data/sample_cases.json` | 15 | 31 KB | 15 cases followed from target query through insertion, filtering and replay | readable entry point; rebuilt by examples/make_sample_cases.py |

## Regenerated outputs

`results/` and `figures/` are build products. Delete them and run

```bash
python examples/reproduce_all.py && python examples/make_figures.py
```

to rebuild the directory in full. `examples/protocol_constants.csv` is the source
for `results/table7_protocol_constants.csv` and is copied through unchanged.

| File | Contents |
|---|---|
| `results/figure3_data.csv` | the alpha sweep per collection and the per-query range distribution (Figure 3a,b) |
| `results/figure4_data.csv` | mean, maximum and lowest cell per detector over A1-A6 (Figure 4) |
| `results/figure6_data.csv` | entry against compliance, and conversion by task (Figure 6) |
| `results/figure7_data.csv` | quality against measured latency by reranking budget (Figure 7) |
| `results/paper_values.csv` | every value transcribed from the manuscript, with the table and cell it appears in |
| `results/table10_per_collection.csv` | per-collection nDCG@10 with and without top-100 reranking (Table 10) |
| `results/table11_detector_separability.csv` | ROC-AUC of each detector against each attack (Table 11) |
| `results/table12_operating_points.csv` | realised FPR, clean nDCG@10 cost and maximum ASR at each target (Table 12) |
| `results/table13_adaptive_attacker_cost.csv` | evasion rate and detector-query cost of the adaptive attacker (Table 13) |
| `results/table14_latency.csv` | end-to-end and per-stage latency at first-stage depth 1000 (Table 14) |
| `results/table1_attack_ladder.csv` | the attack ladder as a partial order over attacker knowledge (Table 1) |
| `results/table2_full_corpus_retrieval.csv` | macro-averaged nDCG@10 at three budgets and query-weighted differences against dense with bootstrap intervals (Table 2) |
| `results/table3_candidate_survival_full_matrix.csv` | candidate-survival ASR, cell counts and clean-retrieval cost for every attack x detector x target (Table 3, and the body of Table 12) |
| `results/table3_selected_maximum_bound.csv` | the maximal cell and its simultaneous one-sided bound (Remark 3, Proposition 3) |
| `results/table4_tool_selection.csv` | attacker-tool selection rate per model and inventory condition (Table 4, lower block) |
| `results/table4_undefended_behaviour.csv` | candidate survival and undefended compliance at context position 0 (Table 4) |
| `results/table5_exposure_compliance_endtoend.csv` | exposure, survivor compliance and end-to-end compromise for the DistilBERT filter (Table 5) |
| `results/table6_detector_conditioned_matrix.csv` | the same quantities with counts and intervals for every attack-detector pair (Table 6) |
| `results/table7_protocol_constants.csv` | every protocol constant needed to reproduce the matrix without reading code (Table 7) |
| `results/table8_fusion_policies.csv` | fusion-weight policies on the two-collection tuning split (Table 8) |
| `results/table9_alpha_predictability.csv` | cross-validated prediction of the oracle weight (Table 9, top) |
| `results/table9_feature_mutual_information.csv` | mutual information per feature (Table 9, bottom) |
| `results/verification_report.csv` | printed value, regenerated value, tolerance, source and status for all 604 checks |

| File | Contents |
|---|---|
| `figures/figure3.png` | Figure 3, redrawn from `results/` by `examples/make_figures.py` |
| `figures/figure4.png` | Figure 4, redrawn from `results/` by `examples/make_figures.py` |
| `figures/figure5.png` | Figure 5, redrawn from `results/` by `examples/make_figures.py` |
| `figures/figure6.png` | Figure 6, redrawn from `results/` by `examples/make_figures.py` |
| `figures/figure7.png` | Figure 7, redrawn from `results/` by `examples/make_figures.py` |

## Not included

The attack-generation implementation (including the A5 detector-query optimisation
loop), experiment orchestration, dataset-construction machinery, debugging utilities,
unreleased attack variants, and the raw model generations from the behavioural
harnesses. See the README for what stands in their place.

