# What Survives, What Matters — reproducibility package

Supplementary material for *What Survives, What Matters: Matched Evaluation of
Retrieval, Filtering, and Downstream Behavior*.

This package provides a **minimal reference implementation** of the evaluation
protocol together with the **experimental records** behind every reported number.
It is sufficient to inspect the protocol, audit the reported measurements, and
regenerate every table and figure in the paper. It does not include the complete
internal experiment-generation pipeline.

Everything here is derived from the same run that produced the manuscript. One
command rebuilds every table and figure dataset from the released records and
checks each regenerated value against the value printed in the paper:

```bash
pip install -r requirements.txt
python examples/reproduce_all.py
```

```
604 values checked against the manuscript: 604 PASS, 0 FAIL
```

## What is here

| Layer | Directory | What it answers |
|---|---|---|
| Specification | `configs/`, `prompts/` | What was run, with which models, constants and prompts |
| Reference implementation | `examples/` | How each measurement is defined and computed |
| Records | `data/` | The per-query and per-episode outcomes the numbers come from |
| Results | `results/`, `figures/` | The tables and figures, regenerated from those records |

## Layout

```
configs/          retrieval.yaml, detector.yaml, behaviour.yaml
prompts/          both system prompts, the QA and tool-selection harnesses,
                  the A0 templates, and the output-parsing rules
examples/         reproduce_all.py     rebuild every table, check against the paper
                  retrieval_demo.py    query -> dense -> BM25 -> fusion -> rerank -> filter -> metric
                  filter_demo.py       calibration -> threshold -> survival -> realised FPR -> intervals
                  replay_demo.py       detector decision -> survivors -> P(C|E,D) -> P(C|D)
                  make_figures.py      redraw Figures 3-7 from results/
                  make_sample_cases.py rebuild data/sample_cases.json
data/records/     per-query and per-episode records (see MANIFEST.md)
data/attack_corpus/  the 1,800 generated adversarial documents and the 1,260
                  detector-specific A5 documents with their optimisation records
data/sample_cases.json  15 cases followed through the whole pipeline
results/          table1-table14, figure3/4/6/7 datasets, verification_report.csv
figures/          figure3.png - figure7.png, redrawn from results/
```

## The three measurements, and why they are separate

The paper follows one evaluation path through three stages, and the package is
organised the same way.

**Retrieval quality under matched reranking budgets** (`retrieval_demo.py`).
Systems are compared only at equal cross-encoder forward passes. The fusion-weight
oracle is a hindsight bound *within the evaluated scalar convex family on a
21-point grid* — not a bound on any per-query fusion policy.

**Candidate survival at a stated operating point** (`filter_demo.py`). An attack
success rate is not interpretable without the false-positive rate the detector was
actually running at, so every realised rate is reported beside its target. The
mean and the maximum over the attack suite rank the detectors differently; both
are reported, and the maximum carries a simultaneous confidence statement rather
than the per-cell interval, which is not valid for a selected maximum.

**Detector-conditioned behavioural replay** (`replay_demo.py`). Candidate survival
is `P(E | D)`, an exposure quantity. Compliance among survivors, `P(C | E, D)`, is
measured by replaying exactly the episodes the detector let through. End-to-end
compromise is `P(C | D) = P(C | E, D) · P(E | D)`, which holds by construction in
this candidate-insertion design because the directive can reach the model only
through the inserted payload. Filtering changes both how much adversarial content
reaches the model and which adversarial content does; only the first is visible to
a candidate-survival metric.



The algorithmic specification of every attack configuration is in
`configs/detector.yaml`, `results/table1_attack_ladder.csv` and
`results/table7_protocol_constants.csv`, and the resulting documents are in
`data/attack_corpus/`, which is what the reported measurements were computed on.
A5's records carry the final detection score, the number of detector queries
consumed, and whether the document ended below the evaluated threshold, so the
optimisation can be audited rather than taken on trust.

## Reproduction notes

`reproduce_all.py` recomputes each table from the per-query and per-episode records
wherever the released records make that possible; the `source` column of
`results/verification_report.csv` marks which values are recomputed and which are
released aggregates. Two families of quantities cannot be recomputed from what is
released here and are carried as aggregates:

- **realised false-positive rates**, which require the clean calibration and
  evaluation scores for each detector;
- **behavioural compliance rates**, which require the model generations.

Every other value in the report — the retrieval tables, the candidate-survival
matrix, the operating points, the adaptive-attacker cost, the exposure and
conditional-compliance tables, the bootstrap and binomial intervals — is
recomputed from the records in `data/`.

Random seeds are 42 throughout. The bootstrap is a paired bootstrap over per-query
scores with 10,000 resamples; interval endpoints reproduce to within 6e-4 across
NumPy versions.

## Scope

All numerical findings are scoped to the evaluated retrieval stack, attack suite,
models and behavioural harnesses: one dense retriever paired with one
cross-encoder, one English security corpus, two compact instruction-tuned models,
a single-turn question-answering harness, and a single-step tool-selection harness.
They do not extend to other retrievers, rerankers, fusion families, languages,
model scales, or multi-step agents with persistent state.

## Dual use

The released corpora carry a single inert directive that requests a fixed token.
They were built against detectors implemented and evaluated by the authors, on a
public research corpus, and nothing here was directed at a live system or a third
party. The defensive value is that a filter can be audited across the same named
attack configurations and stated operating points, and that the detector-conditioned
replay shows what exposure reduction does and does not change downstream.
