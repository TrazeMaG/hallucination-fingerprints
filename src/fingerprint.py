import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer import Hallucinations_Transformer
from tokenizer import BPETokenizer

# ─── LOAD THE SAVED MODEL ────────────────────────────────────────

print("=" * 50)
print("HALLUCINATION FINGERPRINT INSPECTOR")
print("=" * 50)

# Load checkpoint
checkpoint = torch.load(
    '../checkpoints/model_v1.pt',
    map_location='cpu',
    weights_only=False
)

# Rebuild tokenizer from saved state
tokenizer = BPETokenizer(vocab_size=500)
tokenizer.merges = checkpoint['tokenizer_merges']
tokenizer.vocab = checkpoint['tokenizer_vocab']
tokenizer.idx_to_token = checkpoint['tokenizer_idx']

print(f"Tokenizer loaded: {len(tokenizer.vocab)} tokens")

# Rebuild model
model = Hallucinations_Transformer(
    vocab_size=len(tokenizer.vocab),
    d_model=128,
    num_heads=4,
    num_blocks=4,
    max_seq_len=64
)

# Load trained weights
model.load_state_dict(checkpoint['model_state'])
model.eval()  # switch to evaluation mode — no training

print(f"Model loaded successfully")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─── HELPER FUNCTION ─────────────────────────────────────────────

def get_prediction_and_attention(prompt, seq_len=32):
    """Run the model and capture everything happening inside"""
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    encoded = encoded[:seq_len]
    padded = encoded + [0] * (seq_len - len(encoded))
    input_tensor = torch.tensor([padded])
    
    # Run model — capture attention weights from all blocks
    with torch.no_grad():
        logits, all_weights = model(input_tensor)
    
    # Get top 5 predictions for next token
    last_pos = len(encoded) - 1
    next_logits = logits[0, last_pos, :]
    top5_probs = torch.softmax(next_logits, dim=-1)
    top5_vals, top5_idx = top5_probs.topk(5)
    
    top5 = [(tokenizer.idx_to_token.get(idx.item(), '?'), 
             val.item()) for idx, val in zip(top5_idx, top5_vals)]
    
    return top5, all_weights, encoded, last_pos

# ─── INSPECT THE HALLUCINATION ───────────────────────────────────

print("\n─── What does the model predict? ───")
prompt = "the capital of france is"
top5, all_weights, encoded, last_pos = get_prediction_and_attention(prompt)

print(f"\nPrompt: '{prompt}'")
print(f"\nTop 5 predictions:")
for token, prob in top5:
    bar = "█" * int(prob * 40)
    print(f"  '{token}': {prob:.4f} {bar}")

# ─── READ THE 48 ATTENTION MAPS ──────────────────────────────────

print(f"\n─── Attention Maps at the Hallucination Moment ───")
print(f"We have {len(all_weights)} blocks × 4 heads = "
      f"{len(all_weights) * 4} attention maps")
print(f"Reading attention at position {last_pos} "
      f"(the last real token — where prediction happens)\n")

token_labels = [tokenizer.idx_to_token.get(t, '?') 
                for t in encoded[:last_pos+1]]

for block_idx, weights in enumerate(all_weights):
    # weights shape: [1, 4, seq_len, seq_len]
    print(f"Block {block_idx + 1}:")
    
    for head_idx in range(4):
        # Attention from last position to all previous positions
        attn = weights[0, head_idx, last_pos, :last_pos+1]
        
        # Find which token got most attention
        max_idx = attn.argmax().item()
        max_token = token_labels[max_idx] if max_idx < len(token_labels) else '?'
        max_val = attn[max_idx].item()
        
        # Show attention distribution
        attn_str = ' '.join([f"{v:.2f}" for v in attn.tolist()])
        
        print(f"  Head {head_idx+1}: "
              f"most attention → '{max_token}' ({max_val:.3f}) | "
              f"[{attn_str}]")
    print()

# ─── THE KEY QUESTION ────────────────────────────────────────────

print("─── The Fingerprint Question ───")
print(f"The model predicted: '{top5[0][0]}' with {top5[0][1]:.2%} confidence")
print(f"The correct answer is: 'paris'")
print(f"\nLook at the attention maps above.")
print(f"Which heads are looking at 'france'?")
print(f"Which heads are looking at 'capital'?")
print(f"Which heads are ignoring the relevant words?")
print(f"\nThe pattern you see here — this is the hallucination fingerprint.")

# ─── FIND A CORRECT PREDICTION ───────────────────────────────────

print("\n─── Testing Multiple Prompts ───")
test_prompts = [
    "the capital of germany is",
    "the capital of italy is",
    "the capital of japan is",
    "the capital of china is",
    "the capital of spain is",
    "paris is a beautiful",
]

for prompt in test_prompts:
    top5, _, _, _ = get_prediction_and_attention(prompt)
    top_token = top5[0][0].replace('</w>', '')
    top_prob = top5[0][1]
    print(f"'{prompt}' → '{top_token}' ({top_prob:.2%})")

# ─── COMPARE HALLUCINATION VS CORRECT ────────────────────────────

print("\n" + "=" * 50)
print("HALLUCINATION vs CORRECT — ATTENTION COMPARISON")
print("=" * 50)

comparison_prompts = [
    ("the capital of germany is", "berlin", "HALLUCINATION"),
    ("the capital of spain is",   "madrid", "CORRECT"),
]

for prompt, correct_answer, label in comparison_prompts:
    top5, all_weights, encoded, last_pos = get_prediction_and_attention(prompt)
    predicted = top5[0][0].replace('</w>', '')
    confidence = top5[0][1]
    
    print(f"\n{'─'*50}")
    print(f"[{label}]")
    print(f"Prompt:    '{prompt}'")
    print(f"Predicted: '{predicted}' ({confidence:.2%})")
    print(f"Correct:   '{correct_answer}'")
    print(f"\nBlock 4 attention (final layer — where prediction happens):")
    
    token_labels = [tokenizer.idx_to_token.get(t, '?') 
                    for t in encoded[:last_pos+1]]
    
    # Focus on Block 4 only — the final decision layer
    block4_weights = all_weights[3]  # index 3 = Block 4
    
    # Track attention to meaningful vs meaningless words
    meaningful_words = [correct_answer, 
                       prompt.split()[1]]  # 'capital'
    meaningless_words = ['the</w>', 'of</w>', 'is</w>']
    
    meaningful_attention = 0
    meaningless_attention = 0
    
    for head_idx in range(4):
        attn = block4_weights[0, head_idx, last_pos, :last_pos+1]
        max_idx = attn.argmax().item()
        max_token = token_labels[max_idx] if max_idx < len(token_labels) else '?'
        max_val = attn[max_idx].item()
        
        # Is this head looking at something meaningful?
        is_meaningful = any(m in max_token for m in 
                           [correct_answer, 'capital', 'germany', 
                            'spain', 'france', 'italy', 'japan'])
        flag = "✓ RELEVANT" if is_meaningful else "✗ IRRELEVANT"
        
        print(f"  Head {head_idx+1}: → '{max_token}' "
              f"({max_val:.3f}) {flag}")
        
        if is_meaningful:
            meaningful_attention += max_val
        else:
            meaningless_attention += max_val
    
    print(f"\n  Attention to RELEVANT words:   {meaningful_attention:.3f}")
    print(f"  Attention to IRRELEVANT words: {meaningless_attention:.3f}")
    
    ratio = meaningful_attention / (meaningless_attention + 1e-8)
    print(f"  Relevant/Irrelevant ratio:     {ratio:.2f}x")
    print(f"\n  FINGERPRINT SCORE: {'LOW ⚠️  (hallucination risk)' if ratio < 1 else 'HIGH ✓ (likely correct)'}")


# ─── RELATION DROPOUT TEST ───────────────────────────────────────

print("\n" + "=" * 50)
print("RELATION DROPOUT — SYSTEMATIC TEST")
print("=" * 50)

test_cases = [
    ("the capital of france is",  "paris",  "capital"),
    ("the capital of germany is", "berlin", "capital"),
    ("the capital of italy is",   "rome",   "capital"),
    ("the capital of japan is",   "tokyo",  "capital"),
    ("the capital of china is",   "beijing","capital"),
    ("the capital of spain is",   "madrid", "capital"),
]

print(f"\n{'Prompt':<35} {'Predicted':<12} {'Correct':<10} "
      f"{'Relation Attn':>13} {'Result'}")
print("-" * 85)

relation_dropout_hallucinations = 0
relation_present_correct = 0

for prompt, correct, relation_word in test_cases:
    top5, all_weights, encoded, last_pos = get_prediction_and_attention(prompt)
    predicted = top5[0][0].replace('</w>', '')
    confidence = top5[0][1]
    is_correct = predicted == correct
    
    # Measure attention to relation word in Block 4
    token_labels = [tokenizer.idx_to_token.get(t, '?') 
                    for t in encoded[:last_pos+1]]
    
    block4 = all_weights[3]
    relation_attention = 0
    
    for head_idx in range(4):
        attn = block4[0, head_idx, last_pos, :last_pos+1]
        for tok_idx, label in enumerate(token_labels):
            if relation_word in label:
                relation_attention += attn[tok_idx].item()
    
    # Normalize by number of heads
    relation_attention /= 4
    
    dropout = relation_attention < 0.05
    
    if dropout and not is_correct:
        relation_dropout_hallucinations += 1
    if not dropout and is_correct:
        relation_present_correct += 1
    
    status = "✓ CORRECT" if is_correct else "✗ HALLUCINATION"
    dropout_flag = "⚠ DROPOUT" if dropout else "OK"
    
    print(f"{prompt:<35} {predicted:<12} {correct:<10} "
          f"{relation_attention:>13.4f} "
          f"{dropout_flag:<12} {status}")

print(f"\nRelation Dropout → Hallucination: "
      f"{relation_dropout_hallucinations} cases")
print(f"Relation Present → Correct:       "
      f"{relation_present_correct} cases")
print(f"\nHypothesis {'SUPPORTED ✓' if relation_dropout_hallucinations > 0 else 'NOT YET SUPPORTED'}")