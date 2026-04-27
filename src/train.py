import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer import Hallucinations_Transformer
from data import prepare_data

# ─── SETTINGS ────────────────────────────────────────────────────

# Model settings
VOCAB_SIZE   = 500
D_MODEL      = 128    # smaller = faster training
NUM_HEADS    = 4
NUM_BLOCKS   = 4
MAX_SEQ_LEN  = 64

# Training settings
SEQ_LEN      = 32
BATCH_SIZE   = 8
EPOCHS       = 10
LEARNING_RATE = 3e-4  # 0.0003 — the most common LR for transformers

# ─── DATA ────────────────────────────────────────────────────────

text = """
the capital of france is paris the capital of germany is berlin
the capital of italy is rome the capital of spain is madrid
the capital of japan is tokyo the capital of china is beijing
france is a country in europe germany is a country in europe
italy is a country in europe spain is a country in europe
paris is a beautiful city berlin is a historic city
rome is an ancient city madrid is a vibrant city
the president of france visited paris the king of spain lives in madrid
the prime minister of italy visited rome the chancellor of germany is in berlin
france and germany are neighbours italy and france share a border
the eiffel tower is in paris the brandenburg gate is in berlin
the colosseum is in rome the sagrada familia is in barcelona
tokyo is the capital of japan beijing is the capital of china
the emperor of japan lives in tokyo the president of china is in beijing
""" * 50

# ─── SETUP ───────────────────────────────────────────────────────

print("=" * 50)
print("HALLUCINATION FINGERPRINTS — TRAINING RUN")
print("=" * 50)

# Prepare data
tokenizer, dataloader = prepare_data(
    text,
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE
)

# Build model
model = Hallucinations_Transformer(
    vocab_size=len(tokenizer.vocab),
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    max_seq_len=MAX_SEQ_LEN
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# Loss function — measures how wrong the prediction is
# CrossEntropyLoss is standard for next token prediction
loss_fn = nn.CrossEntropyLoss()

# Optimiser — the thing that nudges parameters after each mistake
# Adam is the most common optimiser — handles learning rate automatically
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ─── TRAINING LOOP ───────────────────────────────────────────────

print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Each epoch = {len(dataloader)} batches")
print("-" * 50)

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # inputs:  [batch_size, seq_len] — the question
        # targets: [batch_size, seq_len] — the correct answer

        # ── Forward pass ──
        # Run inputs through the model
        logits, _ = model(inputs)
        # logits shape: [batch_size, seq_len, vocab_size]

        # ── Calculate loss ──
        # Reshape for loss function
        # logits:  [batch_size × seq_len, vocab_size]
        # targets: [batch_size × seq_len]
        loss = loss_fn(
            logits.view(-1, len(tokenizer.vocab)),
            targets.view(-1)
        )

        # ── Backward pass ──
        # Zero out old gradients first
        optimizer.zero_grad()

        # Calculate gradients — which parameters caused the mistake?
        loss.backward()

        # Clip gradients — prevents them exploding to huge numbers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Nudge parameters in the right direction
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # Average loss for this epoch
    avg_loss = total_loss / num_batches

    # ── Test what the model predicts ──
    model.eval()
    with torch.no_grad():
        test_input = "the capital of france is"
        encoded = tokenizer.encode(test_input)
        encoded = encoded[:SEQ_LEN]
        # Pad if needed
        padded = encoded + [0] * (SEQ_LEN - len(encoded))
        input_tensor = torch.tensor([padded])

        logits, _ = model(input_tensor)

        # Get prediction for the last real token
        last_pos = len(encoded) - 1
        next_token_logits = logits[0, last_pos, :]
        predicted_idx = next_token_logits.argmax().item()
        predicted_token = tokenizer.idx_to_token.get(predicted_idx, '?')

    model.train()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"'the capital of france is ___' → '{predicted_token}'")

print("-" * 50)
print("Training complete!")
print("\nIf loss went down and the model predicts 'paris'")
print("or something close — it is learning.")

# Save the model
torch.save({
    'model_state': model.state_dict(),
    'tokenizer_merges': tokenizer.merges,
    'tokenizer_vocab': tokenizer.vocab,
    'tokenizer_idx': tokenizer.idx_to_token,
}, 'checkpoints/model_v1.pt')
print("\nModel saved to checkpoints/model_v1.pt")