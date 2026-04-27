\# Experiments



Reproducible experiments for the Hallucination Fingerprints paper.

Run each script independently. Results match paper figures exactly.



\## Experiment 1 — Relation Dropout

`python experiments/01\_relation\_dropout.py`

Tests Relation Dropout hypothesis on our 806K toy model.

Produces: Table 1 in paper



\## Experiment 2 — GPT-2 Validation  

`python experiments/02\_gpt2\_validation.py`

Validates findings on GPT-2 124M across 35 prompts and 5 categories.

Produces: Table 2 in paper



\## Experiment 3 — Layer Suppression Analysis

`python experiments/03\_layer\_suppression.py`

Layer-by-layer analysis showing Last-Layer Suppression in GPT-2.

Produces: Figure 1 in paper



\## Requirements

pip install torch transformers

