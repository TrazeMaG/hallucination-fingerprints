# Hallucination Fingerprints

> Identifying consistent internal activation patterns that precede
> hallucinations in transformer language models.

**Author:** Nikhil Upadhyay
**Institution:** Independent Researcher, Dublin, Ireland
**Preprint:** [doi.org/10.5281/zenodo.19934537](https://doi.org/10.5281/zenodo.19934537)
**Status:** Published — arXiv submission pending

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
| the capital of france is | the | paris | 0.037 | ✗ |
| the capital of germany is | paris | berlin | 0.028 | ✗ |
| the capital of italy is | in | rome | 0.033 | ✗ |
| the capital of japan is | the | tokyo | 0.033 | ✗ |
| the capital of spain is | madrid | madrid | 0.082 | ✓ |

### Finding 2 — Last-Layer Suppression
In GPT-2, factual knowledge emerges strongly in blocks 10-11
then is systematically suppressed by block 12.

| Prompt | Peak Layer | Peak Prob | Suppression |
|--------|-----------|-----------|-------------|
| capital of France is | Block 10 | 0.182 | Block 12 |
| capital of Germany is | Block 11 | 0.347 | Block 12 |
| capital of Japan is | Block 11 | 0.461 | Block 12 |
| Berlin Wall fell in | Block 10 | 0.128 | Block 12 (survived) |

Block 12 suppresses in every single case across 20,000 prompts.
Average peak factual layer: 11.1 | Average suppression layer: 12.0

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

20,000 labeled hallucination examples on HuggingFace.
Downloaded 29+ times within 48 hours of release with zero promotion.

```python
from datasets import load_dataset
ds = load_dataset("Trazemag/hallbench")
```

[huggingface.co/datasets/Trazemag/hallbench](https://huggingface.co/datasets/Trazemag/hallbench)

---

## Published On

| Platform | Status | Link |
|----------|--------|------|
| Zenodo | Published | [doi.org/10.5281/zenodo.19934537](https://doi.org/10.5281/zenodo.19934537) |
| Academia.edu | Published | [academia.edu](https://www.academia.edu) |
| OSF MetaArXiv | Published | [osf.io](https://osf.io) |
| arXiv cs.CL | Endorsement pending | — |
| GitHub | Live | [TrazeMaG/hallucination-fingerprints](https://github.com/TrazeMaG/hallucination-fingerprints) |
| PyPI | Live | [pip install hallscan](https://pypi.org/project/hallscan/) |
| HuggingFace | Live | [Trazemag/hallbench](https://huggingface.co/datasets/Trazemag/hallbench) |

---

## Project Structure
```
hallucination-fingerprints/
├── src/
│   ├── transformer.py          # 806K transformer built from scratch
│   ├── tokenizer.py            # BPE tokenizer
│   ├── data.py                 # Training data pipeline
│   ├── train.py                # Training loop
│   ├── fingerprint.py          # Hallucination inspector
│   ├── gpt2_inspect.py         # GPT-2 validation
│   ├── large_scale_test.py     # 35-prompt experiment
│   └── layer_analysis.py       # Layer suppression analysis
├── hallscan/
│   ├── __init__.py             # pip install hallscan
│   ├── scanner.py              # Core detection engine
│   └── report.py               # Structured results
├── experiments/
│   ├── 01_relation_dropout.py  # Table 1 — reproducible
│   ├── 02_gpt2_validation.py   # Table 2 — reproducible
│   └── 03_layer_suppression.py # Figure 1 — reproducible
├── paper/
│   ├── hallucination_fingerprints.pdf
│   ├── figure1_layer_suppression.png
│   ├── figure2_relation_dropout.png
│   └── figure3_taxonomy_results.png
├── generate_figures.py         # Generates all 3 figures
├── large_scale_gpu.py          # 20k GPU experiment
└── results/
    └── large_scale_results.json
```

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
- [x] Write 10-page academic paper with 3 figures (LaTeX/PDF)
- [x] Publish preprint on Zenodo (DOI issued)
- [x] Publish on Academia.edu
- [x] Publish on OSF MetaArXiv
- [ ] arXiv submission (cs.CL) — endorsement pending

---

## Citation

```bibtex
@article{upadhyay2026hallucination,
  title={Hallucination Fingerprints: Consistent Failure
         Patterns in Large Language Models},
  author={Upadhyay, Nikhil},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.19934537},
  url={https://doi.org/10.5281/zenodo.19934537}
}
```

---

## Built With

Python · PyTorch · HuggingFace Transformers · NVIDIA RTX 4060
Independent research · Dublin, Ireland · April 2026

---

*Preprint: doi.org/10.5281/zenodo.19934537*
*arXiv submission pending — cs.CL*
