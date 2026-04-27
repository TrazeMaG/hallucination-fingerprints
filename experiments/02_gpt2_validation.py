"""
Experiment 2: GPT-2 Validation
================================
Validates hallucination taxonomy on GPT-2 (124M parameters).
Tests 35 prompts across 5 fact categories.
Classifies each hallucination as Type 1, 2a, or 2b.

Finding: GPT-2 shows zero Type 1 (Relation Dropout).
All hallucinations are Type 2 (Suppression or Knowledge Gap).

Authors: Nikhil Upadhyay
Date: April 2026
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=" * 60)
print("EXPERIMENT 2: GPT-2 VALIDATION")
print("35 prompts | 5 categories | 3 hallucination types")
print("=" * 60)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
model.eval()
print("GPT-2 loaded\n")

def inspect(prompt, correct, relation_word):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    top10_probs = torch.softmax(logits, dim=-1)
    top10_vals, top10_idx = top10_probs.topk(10)
    top10 = [(tokenizer.decode([idx.item()]).strip(), val.item())
             for idx, val in zip(top10_idx, top10_vals)]
    predicted = top10[0][0]
    is_correct = correct.lower() in predicted.lower()
    correct_rank = None
    for rank, (token, prob) in enumerate(top10, 1):
        if correct.lower() in token.lower():
            correct_rank = rank
            break
    relation_pos = None
    for i, tok in enumerate(tokens):
        if relation_word.lower() in tok.lower():
            relation_pos = i
            break
    final_attn = outputs.attentions[11]
    last_pos = len(tokens) - 1
    relation_attn = 0
    if relation_pos is not None:
        for h in range(12):
            relation_attn += final_attn[0, h, last_pos, relation_pos].item()
        relation_attn /= 12
    if is_correct:
        h_type = "CORRECT"
    elif correct_rank:
        h_type = f"TYPE2A (rank {correct_rank})"
    elif relation_attn < 0.05:
        h_type = "TYPE1"
    else:
        h_type = "TYPE2B"
    return predicted, is_correct, correct_rank, relation_attn, h_type

test_cases = [
    # CAPITALS
    ("The capital of France is",       "Paris",      "capital"),
    ("The capital of Germany is",      "Berlin",     "capital"),
    ("The capital of Italy is",        "Rome",       "capital"),
    ("The capital of Japan is",        "Tokyo",      "capital"),
    ("The capital of Spain is",        "Madrid",     "capital"),
    ("The capital of Portugal is",     "Lisbon",     "capital"),
    ("The capital of Poland is",       "Warsaw",     "capital"),
    ("The capital of Greece is",       "Athens",     "capital"),
    ("The capital of Egypt is",        "Cairo",      "capital"),
    ("The capital of Russia is",       "Moscow",     "capital"),
    ("The capital of Canada is",       "Ottawa",     "capital"),
    ("The capital of Mexico is",       "Mexico",     "capital"),
    ("The capital of India is",        "New",        "capital"),
    ("The capital of Argentina is",    "Buenos",     "capital"),
    ("The capital of Turkey is",       "Ankara",     "capital"),
    # INVENTORS
    ("The telephone was invented by",  "Bell",       "invented"),
    ("The lightbulb was invented by",  "Edison",     "invented"),
    ("The theory of gravity by",       "Newton",     "theory"),
    ("The theory of relativity by",    "Einstein",   "theory"),
    ("The printing press invented by", "Gutenberg",  "invented"),
    # AUTHORS
    ("Hamlet was written by",          "Shakespeare","written"),
    ("The Odyssey was written by",     "Homer",      "written"),
    ("1984 was written by",            "Orwell",     "written"),
    ("Don Quixote was written by",     "Cervantes",  "written"),
    ("The Iliad was written by",       "Homer",      "written"),
    # SCIENCE
    ("Water is made of hydrogen and",  "oxygen",     "made"),
    ("The Earth orbits the",           "Sun",        "orbits"),
    ("The speed of light is",          "299",        "speed"),
    ("The human body has",             "206",        "body"),
    ("DNA was discovered by",          "Watson",     "discovered"),
    # HISTORY
    ("The Berlin Wall fell in",        "1989",       "fell"),
    ("World War 2 ended in",           "1945",       "ended"),
    ("World War 1 ended in",           "1918",       "ended"),
    ("The French Revolution began in", "1789",       "began"),
    ("Neil Armstrong landed on",       "Moon",       "landed"),
]

categories = {
    'CAPITALS': test_cases[:15],
    'INVENTORS': test_cases[15:20],
    'AUTHORS': test_cases[20:25],
    'SCIENCE': test_cases[25:30],
    'HISTORY': test_cases[30:],
}

all_results = []
print(f"{'Prompt':<38} {'Pred':<10} {'Type'}")
print("-" * 65)

for prompt, correct, relation in test_cases:
    predicted, is_correct, rank, rel_attn, h_type = inspect(
        prompt, correct, relation
    )
    status = "✓" if is_correct else "✗"
    print(f"{status} {prompt[:36]:<36} {predicted[:8]:<10} {h_type}")
    all_results.append({
        'is_correct': is_correct, 'type': h_type
    })

print("\n" + "=" * 60)
print("RESULTS BY CATEGORY")
print("=" * 60)

idx = 0
for cat, cases in categories.items():
    cat_results = all_results[idx:idx+len(cases)]
    idx += len(cases)
    correct = sum(1 for r in cat_results if r['is_correct'])
    t1 = sum(1 for r in cat_results if 'TYPE1' in r['type'])
    t2a = sum(1 for r in cat_results if 'TYPE2A' in r['type'])
    t2b = sum(1 for r in cat_results if 'TYPE2B' in r['type'])
    print(f"\n{cat}: {correct}/{len(cases)} correct")
    print(f"  Type 1  (Relation Dropout): {t1}")
    print(f"  Type 2a (Suppression):      {t2a}")
    print(f"  Type 2b (Knowledge Gap):    {t2b}")

total = len(all_results)
correct_total = sum(1 for r in all_results if r['is_correct'])
t1 = sum(1 for r in all_results if 'TYPE1' in r['type'])
t2a = sum(1 for r in all_results if 'TYPE2A' in r['type'])
t2b = sum(1 for r in all_results if 'TYPE2B' in r['type'])

print(f"\n{'=' * 60}")
print(f"OVERALL: {correct_total}/{total} correct ({correct_total/total*100:.0f}%)")
print(f"Type 1  — Relation Dropout: {t1}")
print(f"Type 2a — Suppression:      {t2a}")
print(f"Type 2b — Knowledge Gap:    {t2b}")
print(f"\nThis experiment produces TABLE 2 in the paper.")