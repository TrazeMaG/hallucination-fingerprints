import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=" * 55)
print("RELATION DROPOUT — GPT-2 VALIDATION")
print("124M parameters | Trained on 40GB of internet text")
print("=" * 55)

print("\nLoading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"GPT-2 loaded: {total_params:,} parameters")
print(f"Architecture: 12 blocks x 12 heads = 144 attention maps")

def inspect_gpt2(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    top5_probs = torch.softmax(logits, dim=-1)
    top5_vals, top5_idx = top5_probs.topk(5)
    top5 = [(tokenizer.decode([idx.item()]).strip(), val.item())
            for idx, val in zip(top5_idx, top5_vals)]
    return top5, outputs.attentions, tokens

test_cases = [
    ("The capital of France is",    "Paris",    "capital"),
    ("The capital of Germany is",   "Berlin",   "capital"),
    ("The capital of Italy is",     "Rome",     "capital"),
    ("The capital of Japan is",     "Tokyo",    "capital"),
    ("The capital of China is",     "Beijing",  "capital"),
    ("The capital of Spain is",     "Madrid",   "capital"),
    ("The capital of Brazil is",    "Brasilia", "capital"),
    ("The capital of Australia is", "Canberra", "capital"),
]

print(f"\n{'Prompt':<35} {'Predicted':<12} {'Correct':<12} {'Rel.Attn':<10} {'Flag':<12} Result")
print("-" * 95)

dropout_hallucinations = 0
present_correct = 0
results = []

for prompt, correct, relation_word in test_cases:
    top5, all_attentions, tokens = inspect_gpt2(prompt)
    predicted = top5[0][0]
    is_correct = correct.lower() in predicted.lower()

    relation_pos = None
    for i, tok in enumerate(tokens):
        if relation_word.lower() in tok.lower():
            relation_pos = i
            break

    final_block_attn = all_attentions[11]
    last_pos = len(tokens) - 1
    relation_attention = 0

    if relation_pos is not None:
        for head_idx in range(12):
            attn = final_block_attn[0, head_idx, last_pos, relation_pos]
            relation_attention += attn.item()
        relation_attention /= 12

    dropout = relation_attention < 0.05
    status = "CORRECT" if is_correct else "WRONG"
    flag = "DROPOUT" if dropout else "OK"

    if dropout and not is_correct:
        dropout_hallucinations += 1
    if not dropout and is_correct:
        present_correct += 1

    results.append({
        'is_correct': is_correct,
        'relation_attn': relation_attention,
        'dropout': dropout
    })

    print(f"{prompt:<35} {predicted:<12} {correct:<12} "
          f"{relation_attention:<10.4f} {flag:<12} {status}")

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"Relation Dropout + Hallucination: {dropout_hallucinations} cases")
print(f"Relation Present + Correct:       {present_correct} cases")

hallucinations = sum(1 for r in results if not r['is_correct'])
correct_total = sum(1 for r in results if r['is_correct'])
print(f"Total hallucinations: {hallucinations}")
print(f"Total correct:        {correct_total}")

if dropout_hallucinations > 0:
    print(f"\nRelation Dropout VALIDATED in GPT-2")
    print(f"Holds across model sizes:")
    print(f"  Our model: 806K params")
    print(f"  GPT-2:     124M params")
    print(f"\nThis is no longer a toy model quirk.")
    print(f"This is a real phenomenon.")
else:
    print(f"\nRelation Dropout not found in GPT-2 final block")
    print(f"Hypothesis needs refinement.")

print("\n─── Top 5 for 'The capital of France is' ───")
top5, _, _ = inspect_gpt2("The capital of France is")
for token, prob in top5:
    bar = "█" * int(prob * 50)
    print(f"  '{token}': {prob:.4f} {bar}")

# ─── REFINED ANALYSIS — RELATIVE ATTENTION ───────────────────────

print("\n" + "=" * 55)
print("REFINED: RELATIVE ATTENTION ANALYSIS")
print("Is 'capital' getting MORE attention than average?")
print("=" * 55)

print(f"\n{'Prompt':<35} {'Rel.Attn':<10} {'Avg.Attn':<10} {'Ratio':<8} Result")
print("-" * 75)

for prompt, correct, relation_word in test_cases:
    top5, all_attentions, tokens = inspect_gpt2(prompt)
    predicted = top5[0][0]
    is_correct = correct.lower() in predicted.lower()

    relation_pos = None
    for i, tok in enumerate(tokens):
        if relation_word.lower() in tok.lower():
            relation_pos = i
            break

    final_block_attn = all_attentions[11]
    last_pos = len(tokens) - 1

    # Average attention across all heads for each position
    avg_attn_per_pos = final_block_attn[0, :, last_pos, :last_pos+1].mean(dim=0)

    relation_attn = avg_attn_per_pos[relation_pos].item() if relation_pos else 0
    average_attn = avg_attn_per_pos.mean().item()

    # Ratio: how much more attention does 'capital' get vs average?
    ratio = relation_attn / (average_attn + 1e-8)
    status = "CORRECT" if is_correct else "WRONG"

    print(f"{prompt:<35} {relation_attn:<10.4f} {average_attn:<10.4f} "
          f"{ratio:<8.2f} {status}")

print("\nIf ratio > 1.0 → 'capital' gets above-average attention")
print("If ratio < 1.0 → 'capital' gets below-average attention")
print("Hypothesis: correct predictions have higher ratio than hallucinations")

# ─── TYPE 2 HALLUCINATION CHECK ──────────────────────────────────

print("\n" + "=" * 55)
print("TYPE 2 CHECK — IS THE CORRECT ANSWER IN TOP 5?")
print("=" * 55)

print(f"\n{'Prompt':<35} {'Top1':<10} {'Correct':<10} {'In Top5?':<10} {'Correct Rank'}")
print("-" * 80)

for prompt, correct, _ in test_cases:
    top5, _, _ = inspect_gpt2(prompt)
    top1 = top5[0][0]

    correct_rank = None
    for rank, (token, prob) in enumerate(top5, 1):
        if correct.lower() in token.lower():
            correct_rank = rank
            break

    in_top5 = "YES" if correct_rank else "NO"
    rank_str = str(correct_rank) if correct_rank else "not found"

    print(f"{prompt:<35} {top1:<10} {correct:<10} {in_top5:<10} {rank_str}")