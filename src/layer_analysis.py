import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("=" * 60)
print("LAYER-BY-LAYER SUPPRESSION ANALYSIS")
print("Where does 'the' beat 'Paris' in GPT-2?")
print("=" * 60)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
model.eval()

# We need access to hidden states at each layer
# output_hidden_states=True gives us the residual stream at every block
model2 = GPT2LMHeadModel.from_pretrained(
    'gpt2',
    output_attentions=True,
    output_hidden_states=True
)
model2.eval()
print("Models loaded\n")

# ─── HELPER: DECODE HIDDEN STATE AT EACH LAYER ───────────────────

def get_layer_predictions(prompt, target_correct, target_wrong):
    """
    At each layer, project the hidden state to vocabulary
    and see what the model would predict if it stopped there.
    This shows us layer by layer how the prediction evolves.
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model2(input_ids)

    # hidden_states = tuple of 13 tensors (embedding + 12 layers)
    # each shape: [1, seq_len, 768]
    hidden_states = outputs.hidden_states
    last_pos = input_ids.shape[1] - 1

    # GPT-2's final layer norm and unembedding matrix
    ln_f = model2.transformer.ln_f
    lm_head = model2.lm_head

    # Get token IDs for the words we care about
    correct_ids = tokenizer.encode(" " + target_correct)
    wrong_ids = tokenizer.encode(" " + target_wrong)
    correct_id = correct_ids[0] if correct_ids else None
    wrong_id = wrong_ids[0] if wrong_ids else None

    print(f"\nPrompt: '{prompt}'")
    print(f"Tracking: '{target_correct}' (correct) vs '{target_wrong}' (suppressor)")
    print(f"\n{'Layer':<8} {'Top Prediction':<20} {target_correct+' prob':<15} "
          f"{target_wrong+' prob':<15} {'Winner'}")
    print("-" * 70)

    crossover_layer = None

    for layer_idx, hidden in enumerate(hidden_states):
        # Project hidden state at this layer to vocabulary
        normed = ln_f(hidden[0, last_pos, :])
        logits = lm_head(normed)
        probs = torch.softmax(logits, dim=-1)

        # Top prediction at this layer
        top_idx = logits.argmax().item()
        top_token = tokenizer.decode([top_idx]).strip()
        top_prob = probs[top_idx].item()

        # Probabilities of correct and wrong tokens
        correct_prob = probs[correct_id].item() if correct_id else 0
        wrong_prob = probs[wrong_id].item() if wrong_id else 0

        winner = target_correct if correct_prob > wrong_prob else target_wrong

        # Detect crossover — where wrong overtakes correct
        if crossover_layer is None and wrong_prob > correct_prob and layer_idx > 0:
            crossover_layer = layer_idx
            marker = " ← SUPPRESSION STARTS HERE"
        else:
            marker = ""

        layer_name = "Embed" if layer_idx == 0 else f"Block {layer_idx}"
        print(f"{layer_name:<8} {top_token:<20} {correct_prob:<15.4f} "
              f"{wrong_prob:<15.4f} {winner}{marker}")

    return crossover_layer

# ─── RUN ANALYSIS ON KEY PROMPTS ─────────────────────────────────

prompts_to_analyze = [
    ("The capital of France is", "Paris", "the"),
    ("The capital of Germany is", "Berlin", "the"),
    ("The capital of Japan is", "Tokyo", "the"),
    ("Hamlet was written by", "Shakespeare", "the"),
    ("The Berlin Wall fell in", "1989", "the"),  # correct case
]

crossover_layers = []

for prompt, correct, wrong in prompts_to_analyze:
    layer = get_layer_predictions(prompt, correct, wrong)
    if layer:
        crossover_layers.append(layer)
        print(f"\nSuppression crossover at Block {layer}")
    else:
        print(f"\nNo crossover found — correct answer won throughout")
    print("\n" + "=" * 60)

# ─── SUMMARY ─────────────────────────────────────────────────────

if crossover_layers:
    avg_crossover = sum(crossover_layers) / len(crossover_layers)
    print(f"\nSUMMARY")
    print(f"Suppression crossover layers: {crossover_layers}")
    print(f"Average crossover layer: {avg_crossover:.1f}")
    print(f"\nInterpretation:")
    print(f"Blocks 1-{int(avg_crossover)-1}: factual processing dominates")
    print(f"Block {int(avg_crossover)}: structural pattern takes over")
    print(f"Blocks {int(avg_crossover)+1}-12: wrong answer amplified")
    print(f"\nThis is the suppression mechanism in GPT-2.")