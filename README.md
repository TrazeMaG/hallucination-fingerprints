\# Hallucination Fingerprints



> Identifying consistent internal activation patterns that precede 

> hallucinations in transformer language models.



\*\*Author:\*\* Nikhil Upadhyay 

\*\*Status:\*\* Active Research — Week 1 of 10  

\*\*Goal:\*\* arXiv preprint + open source detection tool



\---



\## The Core Question



LLMs don't hallucinate randomly. When a model confidently says 

the wrong thing, something specific is happening inside — in the 

attention maps and FFN activations — before the wrong token is generated.



This project finds, names, and builds a detector for those patterns.



\---



\## Key Finding (Week 1)



We trained a 806K parameter transformer from scratch and observed 

a consistent pattern we call \*\*Relation Dropout\*\*:



> When a model hallucinates on factual recall tasks, attention to 

> the \*relation token\* (e.g. "capital") drops below 0.05 in the 

> final transformer block — even when the \*entity token\* 

> (e.g. "germany") is strongly attended to.



| Prompt | Predicted | Correct | Relation Attn | Result |

|--------|-----------|---------|---------------|--------|

| the capital of france is | the | paris | 0.037 ⚠️ | ✗ Hallucination |

| the capital of germany is | paris | berlin | 0.028 ⚠️ | ✗ Hallucination |

| the capital of italy is | in | rome | 0.033 ⚠️ | ✗ Hallucination |

| the capital of japan is | the | tokyo | 0.033 ⚠️ | ✗ Hallucination |

| the capital of spain is | madrid | madrid | 0.082 ✓ | ✓ Correct |



4/5 hallucination cases showed Relation Dropout.  

The single correct prediction maintained relation attention above 0.08.



\---



\## Project Structure



hallucination-fingerprints/

├── src/

│   ├── transformer.py   # Full transformer built from scratch

│   ├── tokenizer.py     # BPE tokenizer

│   ├── data.py          # Data pipeline

│   ├── train.py         # Training loop

│   └── fingerprint.py   # Hallucination inspector

├── research/

│   └── gap-analysis.md  # Research notes and findings

└── experiments/         # Coming soon



\---



\## Roadmap



\- \[x] Build transformer from scratch

\- \[x] Train on factual corpus

\- \[x] Observe and measure hallucinations

\- \[x] Identify Relation Dropout pattern

\- \[ ] Validate across larger models (GPT-2)

\- \[ ] Build HallScan detection tool

\- \[ ] Publish HallBench dataset

\- \[ ] arXiv preprint



\---



\## Built With



Python · PyTorch · From scratch (no pre-trained models)



\---



\*This is independent research conducted as part of a 10-week 

project targeting publication on arXiv cs.CL.\*

