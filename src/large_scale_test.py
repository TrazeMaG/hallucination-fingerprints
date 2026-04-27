import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=" * 60)
print("LARGE SCALE HALLUCINATION TEST — 100+ PROMPTS")
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

    # Check if correct answer is anywhere in top 10
    correct_rank = None
    for rank, (token, prob) in enumerate(top10, 1):
        if correct.lower() in token.lower():
            correct_rank = rank
            break

    # Relation attention in final block
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

    # Classify hallucination type
    if is_correct:
        h_type = "CORRECT"
    elif correct_rank and correct_rank <= 10:
        h_type = f"TYPE2a (rank {correct_rank})"
    elif relation_attn < 0.05:
        h_type = "TYPE1 (dropout)"
    else:
        h_type = "TYPE2b (gap)"

    return predicted, is_correct, correct_rank, relation_attn, h_type

# ─── TEST CASES ───────────────────────────────────────────────────

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
    ("The capital of Argentina is",    "Buenos",     "capital"),
    ("The capital of Canada is",       "Ottawa",     "capital"),
    ("The capital of Mexico is",       "Mexico",     "capital"),
    ("The capital of India is",        "New",        "capital"),
    ("The capital of Russia is",       "Moscow",     "capital"),
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

    # SCIENCE FACTS
    ("The speed of light is",          "299",        "speed"),
    ("Water is made of hydrogen and",  "oxygen",     "made"),
    ("The human body has",             "206",        "body"),
    ("DNA was discovered by",          "Watson",     "discovered"),
    ("The Earth orbits the",           "Sun",        "orbits"),

    # HISTORY
    ("World War 2 ended in",           "1945",       "ended"),
    ("World War 1 ended in",           "1918",       "ended"),
    ("The French Revolution began in", "1789",       "began"),
    ("The Berlin Wall fell in",        "1989",       "fell"),
    ("Neil Armstrong landed on",       "Moon",       "landed"),
]

# ─── RUN ALL TESTS ───────────────────────────────────────────────

print(f"Running {len(test_cases)} prompts across 5 fact categories...\n")

results = []
categories = {
    'CAPITALS': [], 'INVENTORS': [], 'AUTHORS': [],
    'SCIENCE': [], 'HISTORY': []
}

cat_map = {
    0: 'CAPITALS', 15: 'INVENTORS', 20: 'AUTHORS',
    25: 'SCIENCE', 30: 'HISTORY'
}

current_cat = 'CAPITALS'
for i, (prompt, correct, relation) in enumerate(test_cases):
    if i in cat_map:
        current_cat = cat_map[i]

    predicted, is_correct, rank, rel_attn, h_type = inspect(
        prompt, correct, relation
    )
    result = {
        'prompt': prompt, 'predicted': predicted,
        'correct': correct, 'is_correct': is_correct,
        'rank': rank, 'rel_attn': rel_attn,
        'type': h_type, 'category': current_cat
    }
    results.append(result)
    categories[current_cat].append(result)

    status = "✓" if is_correct else "✗"
    print(f"{status} {prompt[:45]:<45} → '{predicted[:10]:<10}' | {h_type}")

# ─── ANALYSIS ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS BY CATEGORY")
print("=" * 60)

for cat, cat_results in categories.items():
    if not cat_results:
        continue
    correct = sum(1 for r in cat_results if r['is_correct'])
    total = len(cat_results)
    type1 = sum(1 for r in cat_results if 'TYPE1' in r['type'])
    type2a = sum(1 for r in cat_results if 'TYPE2a' in r['type'])
    type2b = sum(1 for r in cat_results if 'TYPE2b' in r['type'])
    print(f"\n{cat}: {correct}/{total} correct")
    print(f"  Type 1 (dropout):    {type1}")
    print(f"  Type 2a (suppressed):{type2a}")
    print(f"  Type 2b (gap):       {type2b}")

print("\n" + "=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)

total = len(results)
correct_total = sum(1 for r in results if r['is_correct'])
type1_total = sum(1 for r in results if 'TYPE1' in r['type'])
type2a_total = sum(1 for r in results if 'TYPE2a' in r['type'])
type2b_total = sum(1 for r in results if 'TYPE2b' in r['type'])

print(f"Total prompts:        {total}")
print(f"Correct:              {correct_total} ({correct_total/total*100:.1f}%)")
print(f"Hallucinations:       {total-correct_total} ({(total-correct_total)/total*100:.1f}%)")
print(f"\nHallucination types:")
print(f"  Type 1  — Ignorance/Dropout: {type1_total}")
print(f"  Type 2a — Suppression:       {type2a_total}")
print(f"  Type 2b — Knowledge gap:     {type2b_total}")
print(f"\nAvg relation attention (correct):       "
      f"{sum(r['rel_attn'] for r in results if r['is_correct'])/(correct_total+1e-8):.4f}")
print(f"Avg relation attention (hallucinated):  "
      f"{sum(r['rel_attn'] for r in results if not r['is_correct'])/((total-correct_total)+1e-8):.4f}")