# Supplementary Materials



This package provides a **minimal reference implementation** of the evaluation protocol together with the **experimental records underlying the reported results**. It is sufficient to inspect the protocol, audit the released experimental artifacts, and regenerate the experiment-derived tables and figures to the extent supported by the released records. It does not include the complete internal experiment-generation pipeline.

The package includes the **full realized A0-A6 attack corpus** and the **full A5-optimized corpus** used in the reported experiments. The internal machinery used to generate or optimize these documents is not included. The reference implementation therefore operates on the released corpora and records rather than regenerating them.

Everything released here is derived from the same experimental run that produced the manuscript. One command verifies the reported numerical values against the released records:

```bash
pip install -r requirements.txt
python examples/reproduce_all.py
```

```text
604 reported values verified against the manuscript:
604 PASS, 0 FAIL
```

## What is here

| Layer                    | Directory              | What it answers                                                                                                  |
| ------------------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Specification            | `configs/`, `prompts/` | What was run, with which models, constants, thresholds, and prompts                                              |
| Reference implementation | `examples/`            | How the reported measurements and analyses are defined and computed                                              |
| Records                  | `data/`                | The released per-query and per-episode outcomes and realized attack corpora underlying the reported measurements |
| Results                  | `results/`, `figures/` | The experiment-derived tables and figures regenerated or verified from the released records                      |

## Artifact boundary

The package releases the **realized experimental artifacts** used in the reported evaluation, including the full A0-A6 attack corpus and the full A5-optimized corpus, together with the records required to analyze their measured outcomes.

The package does **not** release the internal code used to construct, mutate, or optimize new attack documents. In particular, the A5 documents are released as realized experimental artifacts, together with their recorded optimization outcomes, rather than with the internal optimization machinery that generated them.

This boundary is intentional: the released package supports inspection and reproduction of the reported evaluation while avoiding disclosure of the complete internal experiment-generation pipeline.

## Layout

```text
configs/
    retrieval.yaml
    detector.yaml
    behaviour.yaml

prompts/
    system prompts
    QA and tool-selection harnesses
    A0 templates
    output-parsing rules

examples/
    reproduce_all.py
        verify the reported values against the released records

    retrieval_demo.py
        query -> dense -> BM25 -> fusion -> rerank -> metric

    filter_demo.py
        calibration -> threshold -> survival -> realised FPR -> intervals

    replay_demo.py
        detector decision -> survivors -> P(C|E,D) -> P(C|D)

    make_figures.py
        redraw experiment-derived Figures 3-7 from results/

    make_sample_cases.py
        rebuild data/sample_cases.json

data/
    records/
        per-query and per-episode records
        see MANIFEST.md

    attack_corpus/
        nb3_attack_documents.parquet
            1,800 realized adversarial documents
            covering A0-A6 × 300 queries

        nb3_a5_documents.parquet
            1,260 A5-optimized documents
            with recorded detection scores and detector-query counts

    sample_cases.json
        15 cases followed through the whole evaluation pipeline

results/
    table1-table14
    figure3/4/6/7 datasets
    verification_report.csv

figures/
    figure3.png - figure7.png
    redrawn from the released result datasets
```

## The three measurements, and why they are separate

The paper follows one evaluation path through three stages, and the package is organized the same way.

**Retrieval quality under matched reranking budgets** (`retrieval_demo.py`).

Systems are compared only at equal cross-encoder forward passes. The fusion-weight oracle is a hindsight bound **within the evaluated scalar convex family on a 21-point grid**, not a bound on arbitrary per-query fusion policies.

**Candidate survival at a stated operating point** (`filter_demo.py`).

An attack success rate is not interpretable without the false-positive rate at which the detector was actually operating, so every realized rate is reported beside its target. The mean and the maximum over the evaluated attack suite rank the detectors differently; both are reported. The maximum uses a simultaneous confidence statement rather than the per-cell interval, because the selected maximum is a different statistical quantity.

**Detector-conditioned behavioural replay** (`replay_demo.py`).

Candidate survival is `P(E | D)`, an exposure quantity. Compliance among survivors, `P(C | E, D)`, is measured by replaying the episodes that the detector allows through. End-to-end compromise is

```text
P(C | D) = P(C | E, D) · P(E | D)
```

in this candidate-insertion design because the directive can reach the model only through the inserted payload. Filtering therefore changes both how much adversarial content reaches the model and which adversarial content survives; only the first effect is visible to a candidate-survival metric.

The algorithmic specification of every attack configuration is given in:

```text
configs/detector.yaml
results/table1_attack_ladder.csv
results/table7_protocol_constants.csv
```

The resulting realized attack documents used for the reported measurements are provided in:

```text
data/attack_corpus/
```

A5 records include the final detection score, detector-query count, and final threshold outcome. These records allow the **reported optimization outcomes and detector-query budget to be independently inspected**. They do not expose the internal edit-generation or detector-query optimization implementation.

## Reproduction notes

`reproduce_all.py` recomputes each table from the released per-query and per-episode records wherever those records contain the underlying observations. The `source` column in:

```text
results/verification_report.csv
```

identifies whether each reported value is:

* recomputed directly from released records;
* derived from released records; or
* carried as a released aggregate where the underlying raw inputs are not part of the package.

This distinction is intentional and makes the provenance of every reported value explicit.

Two families of quantities cannot be independently recomputed from the released package and are therefore provided as recorded aggregates:

* **realized false-positive rates**, which require the underlying clean calibration and evaluation scores for each detector;
* **behavioural compliance rates**, which require the original model generations.

Every other reported value that is supported by the released records, including the retrieval tables, candidate-survival matrix, operating-point summaries, adaptive-attacker cost, exposure quantities, conditional-compliance quantities, and statistical intervals, is recomputed from `data/`.

Random seeds are 42 throughout.

The bootstrap is a paired bootstrap over per-query scores with 10,000 resamples. Interval endpoints reproduce to within `6e-4` across tested NumPy versions.

## Verification output

The verification script checks the reported numerical values against the manuscript and records the provenance of each check in:

```text
results/verification_report.csv
```

A successful run produces:

```text
604 reported values verified against the manuscript:
604 PASS, 0 FAIL
```

A `PASS` indicates that the value in the manuscript is consistent with the corresponding released or recomputed value under the verification procedure. The verification report distinguishes direct recomputation from checks against released aggregates.

## Scope

All numerical findings are scoped to the evaluated retrieval stack, attack suite, models, and behavioural harnesses: one dense retriever paired with one cross-encoder, one English security corpus, two compact instruction-tuned models, a single-turn question-answering harness, and a single-step tool-selection harness.

The released artifacts do not establish claims about other retrievers, rerankers, fusion families, languages, model scales, or multi-step agents with persistent state.

## Dual use

The released corpora carry a single inert directive that requests a fixed token. They were built against detectors implemented and evaluated by the authors, using a public research corpus. Nothing in the released package was directed at a live system or a third party.

The package does **not** include code for automated generation, mutation, or detector-specific optimization of new attack documents.

The defensive value of the release is to make the evaluated filters auditable against the same named attack configurations and stated operating points, while allowing the detector-conditioned replay analysis to be independently inspected. In particular, the released artifacts make it possible to examine what exposure reduction does and does not change in downstream behaviour without providing the machinery for automatically generating new optimized attack corpora.

## Reproducibility principle

The package is designed around a simple separation:

```text
released realized artifacts
        +
released evaluation specification
        +
reference evaluation code
        ↓
verification and reconstruction of reported measurements
```

The goal is to make the **reported scientific claims reproducible and auditable**.
