# Hallucination Fingerprints

Identifying consistent internal activation patterns that precede
hallucinations in transformer language models.

**Author:** Nikhil Upadhyay  
**Status:** Paper complete — arXiv submission pending  
**Goal:** arXiv preprint + open source detection tool  

---

## The Core Question

LLMs don't hallucinate randomly. When a model confidently says
the wrong thing, something specific is happening inside — in the
attention maps and FFN activations — before the wrong token is generated.

This project finds, names, and builds a detector for those patterns.

---

## Results (20,000 prompts, RTX 4060 GPU)

| Hallucination Type | Count | % |
|-------------------|-------|---|
| Correct | 954 | 4.8% |
| Type 1 — Relation Dropout | 2,946 | 14.7% |
| Type 2a — Last-Layer Suppression | 2,481 | 12.4% |
| Type 2b — Knowledge Gap | 13,619 | 68.1% |

Average peak factual layer: **11.1** | Average suppression layer: **12.0**

---

## Two Named Findings

### 1. Relation Dropout (small models)
When a model hallucinates on factual recall tasks, attention to
the *relation token* (e.g. "capital") drops below 0.05 in the
final transformer block — even when the *entity token*
(e.g. "germany") is strongly attended to.

| Prompt | Predicted | Correct | Relation Attn | Result |
|--------|-----------|---------|---------------|--------|
| the capital of france is | the | paris | 0.037 ⚠️ | ✗ Hallucination |
| the capital of germany is | paris | berlin | 0.028 ⚠️ | ✗ Hallucination |
| the capital of italy is | in | rome | 0.033 ⚠️ | ✗ Hallucination |
| the capital of japan is | the | tokyo | 0.033 ⚠️ | ✗ Hallucination |
| the capital of spain is | madrid | madrid | 0.082 ✓ | ✓ Correct |

### 2. Last-Layer Suppression (GPT-2)
Factual knowledge emerges strongly in blocks 10–11 of GPT-2,
then is systematically suppressed by block 12 via structural
pattern interference.

| Prompt | Peak Layer | Peak Prob | Suppression |
|--------|-----------|-----------|-------------|
| capital of France is | Block 10 | 0.182 | Block 12 |
| capital of Germany is | Block 11 | 0.347 | Block 12 |
| capital of Japan is | Block 11 | 0.461 | Block 12 |
| Berlin Wall fell in | Block 10 | 0.128 | Block 12 ✓ survived |

Block 12 suppresses in **every single case** across 20,000 prompts.

---

## Hallucination Taxonomy

| Type | Description | Signal | Example |
|------|-------------|--------|---------|
| Type 1 — Dropout | Fact not learned | Relation Attn < 0.05 | Our 806K model |
| Type 2a — Suppression | Fact known but overridden | Correct in top-10, rank 2+ | GPT-2 on France |
| Type 2b — Gap | Fact not in training data | Correct not in top-10 | GPT-2 on authors |

---

## Project Structure

hallucination-fingerprints/
├── src/
│   ├── transformer.py      # Full transformer built from scratch
│   ├── tokenizer.py        # BPE tokenizer
│   ├── data.py             # Data pipeline
│   ├── train.py            # Training loop
│   ├── fingerprint.py      # Hallucination inspector
│   ├── gpt2_inspect.py     # GPT-2 validation
│   ├── large_scale_test.py # 35 prompt experiment
│   └── layer_analysis.py   # Layer suppression analysis
├── hallscan/
│   ├── init.py         # Public API
│   ├── scanner.py          # Core detection engine
│   └── report.py           # Structured results
├── experiments/
│   ├── 01_relation_dropout.py
│   ├── 02_gpt2_validation.py
│   └── 03_layer_suppression.py
├── paper/
│   └── hallucination_fingerprints.pdf  # Full paper
├── large_scale_gpu.py      # 20k GPU experiment
└── results/
└── large_scale_results.json

---

## HallScan — Detection Tool

```python
from hallscan import scan

result = scan(
    "The capital of France is",
    relation_word="capital",
    correct_answer="Paris"
)
print(result)
# Type: TYPE2A_LAST_LAYER_SUPPRESSION
# Risk: 85%
# Suppression at block: 12
# Factual peak at block: 10
```

---

## HallBench Dataset

20,000 labeled hallucination examples on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("Trazemag/hallbench")
```

→ [huggingface.co/datasets/Trazemag/hallbench](https://huggingface.co/datasets/Trazemag/hallbench)

---

## Roadmap

- [x] Build transformer from scratch (806K params)
- [x] Train on factual corpus, observe hallucinations
- [x] Identify and name Relation Dropout
- [x] Validate on GPT-2 (124M params)
- [x] Layer-by-layer suppression analysis
- [x] Identify and name Last-Layer Suppression
- [x] Build HallScan detection tool
- [x] Run 20,000 prompts on RTX 4060 GPU
- [x] Publish HallBench dataset (20,000 examples)
- [x] Write 6-page academic paper (LaTeX)
- [ ] arXiv submission (cs.CL)
- [ ] PyPI packaging (pip install hallscan)

---

## Paper

Full paper available in `paper/hallucination_fingerprints.pdf`

Cite as:Upadhyay, N. (2026). Hallucination Fingerprints: Consistent
Failure Patterns in Large Language Models.
arXiv preprint (submission pending).

---

## Built With

Python · PyTorch · HuggingFace Transformers · RTX 4060 GPU  
From scratch — no pre-trained models used in core experiments.

---

*Independent research. 7 weeks of work. arXiv cs.CL submission pending.*
