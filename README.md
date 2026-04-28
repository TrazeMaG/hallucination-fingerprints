Here's the full updated README first:
markdown# Hallucination Fingerprints

> Identifying consistent internal activation patterns that precede
> hallucinations in transformer language models.

**Author:** Nikhil Upadhyay  
**Institution:** Dublin Business School, Dublin, Ireland  
**Status:** Paper complete — arXiv submission pending  
**Paper:** `paper/hallucination_fingerprints.pdf`

---

## Install

```bash
pip install hallscan
```

```python
from hallscan import scan

result = scan(
    "The capital of France is",
    relation_word="capital",
    correct_answer="Paris"
)
print(result)
# Type:       TYPE2A_LAST_LAYER_SUPPRESSION
# Risk:       85%
# Suppression at block: 12
# Factual peak at block: 10
```

---

## What This Project Found

### Finding 1 — Relation Dropout
In small transformer models, hallucination correlates with
collapse of attention to the semantic relation token in the
final block.

| Prompt | Predicted | Correct | Relation Attn | Result |
|--------|-----------|---------|---------------|--------|
| the capital of france is | the | paris | 0.037 ⚠️ | ✗ |
| the capital of germany is | paris | berlin | 0.028 ⚠️ | ✗ |
| the capital of italy is | in | rome | 0.033 ⚠️ | ✗ |
| the capital of japan is | the | tokyo | 0.033 ⚠️ | ✗ |
| the capital of spain is | madrid | madrid | 0.082 ✓ | ✓ |

### Finding 2 — Last-Layer Suppression
In GPT-2, factual knowledge emerges strongly in blocks 10–11
then is systematically suppressed by block 12.

| Prompt | Peak Layer | Peak Prob | Suppression |
|--------|-----------|-----------|-------------|
| capital of France is | Block 10 | 0.182 | Block 12 |
| capital of Germany is | Block 11 | 0.347 | Block 12 |
| capital of Japan is | Block 11 | 0.461 | Block 12 |
| Berlin Wall fell in | Block 10 | 0.128 | Block 12 ✓ survived |

Block 12 suppresses in **every single case** across 20,000 prompts.
Average peak factual layer: **11.1** | Average suppression layer: **12.0**

---

## Hallucination Taxonomy

| Type | Description | Signal | Example |
|------|-------------|--------|---------|
| Type 1 — Relation Dropout | Fact not learned | Relation Attn < 0.05 | Our 806K model |
| Type 2a — Last-Layer Suppression | Fact known, overridden | Correct in top-10 | GPT-2 on France |
| Type 2b — Knowledge Gap | Fact not in training data | Correct not in top-10 | GPT-2 on authors |

---

## Large Scale Results (20,000 prompts, RTX 4060)

| Hallucination Type | Count | % |
|-------------------|-------|---|
| Correct | 954 | 4.8% |
| Type 1 — Relation Dropout | 2,946 | 14.7% |
| Type 2a — Last-Layer Suppression | 2,481 | 12.4% |
| Type 2b — Knowledge Gap | 13,619 | 68.1% |

---

## Dataset

20,000 labeled hallucination examples on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("Trazemag/hallbench")
```

→ [huggingface.co/datasets/Trazemag/hallbench](https://huggingface.co/datasets/Trazemag/hallbench)

---

## Project Structure
hallucination-fingerprints/
├── src/
│   ├── transformer.py       # 806K transformer built from scratch
│   ├── tokenizer.py         # BPE tokenizer
│   ├── data.py              # Training data pipeline
│   ├── train.py             # Training loop
│   ├── fingerprint.py       # Hallucination inspector
│   ├── gpt2_inspect.py      # GPT-2 validation
│   ├── large_scale_test.py  # 35-prompt experiment
│   └── layer_analysis.py    # Layer suppression analysis
├── hallscan/
│   ├── init.py          # pip install hallscan
│   ├── scanner.py           # Core detection engine
│   └── report.py            # Structured results
├── experiments/
│   ├── 01_relation_dropout.py    # Table 1
│   ├── 02_gpt2_validation.py     # Table 2
│   └── 03_layer_suppression.py   # Figure 1
├── paper/
│   └── hallucination_fingerprints.pdf
├── large_scale_gpu.py       # 20k GPU experiment
└── results/
└── large_scale_results.json

---

## Reproduce

```bash
git clone https://github.com/TrazeMaG/hallucination-fingerprints
pip install torch transformers
python experiments/01_relation_dropout.py
python experiments/02_gpt2_validation.py
python experiments/03_layer_suppression.py
```

---

## Roadmap

- [x] Build transformer from scratch (806K params)
- [x] Train on factual corpus, observe hallucinations
- [x] Identify and name Relation Dropout
- [x] Validate on GPT-2 (124M params)
- [x] Layer-by-layer suppression analysis
- [x] Identify and name Last-Layer Suppression
- [x] Run 20,000 prompts on RTX 4060 GPU
- [x] Build HallScan (pip install hallscan)
- [x] Publish HallBench (20,000 labeled examples)
- [x] Write 6-page academic paper (LaTeX/PDF)
- [ ] arXiv submission (cs.CL)

---

## Citation

```bibtex
@article{upadhyay2026hallucination,
  title={Hallucination Fingerprints: Consistent Failure Patterns
         in Large Language Models},
  author={Upadhyay, Nikhil},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/TrazeMaG/hallucination-fingerprints}
}
```

---

## Built With

Python · PyTorch · HuggingFace Transformers · NVIDIA RTX 4060  
Independent research — 7 weeks — Dublin, Ireland

---

*arXiv submission pending — cs.CL*