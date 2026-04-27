import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transformer import Hallucinations_Transformer
    from data import prepare_data
except ModuleNotFoundError:
    from src.transformer import Hallucinations_Transformer
    from src.data import prepare_data

# ─── SETTINGS ────────────────────────────────────────────────────

VOCAB_SIZE    = 500
D_MODEL       = 128
NUM_HEADS     = 4
NUM_BLOCKS    = 4
MAX_SEQ_LEN   = 64
SEQ_LEN       = 32
BATCH_SIZE    = 8
EPOCHS        = 10
LEARNING_RATE = 3e-4

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

tokenizer, dataloader = prepare_data(
    text,
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE
)

model = Hallucinations_Transformer(
    vocab_size=len(tokenizer.vocab),
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    max_seq_len=MAX_SEQ_LEN
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ─── TRAINING LOOP ───────────────────────────────────────────────

print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Each epoch = {len(dataloader)} batches")
print("-" * 50)

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        logits, _ = model(inputs)
        loss = loss_fn(
            logits.view(-1, len(tokenizer.vocab)),
            targets.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    model.eval()
    with torch.no_grad():
        test_input = "the capital of france is"
        encoded = tokenizer.encode(test_input)
        encoded = encoded[:SEQ_LEN]
        padded = encoded + [0] * (SEQ_LEN - len(encoded))
        input_tensor = torch.tensor([padded])
        logits, _ = model(input_tensor)
        last_pos = len(encoded) - 1
        predicted_idx = logits[0, last_pos, :].argmax().item()
        predicted_token = tokenizer.idx_to_token.get(predicted_idx, '?')
    model.train()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"'the capital of france is ___' → '{predicted_token}'")

print("-" * 50)
print("Training complete!")

os.makedirs('checkpoints', exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'tokenizer_merges': tokenizer.merges,
    'tokenizer_vocab': tokenizer.vocab,
    'tokenizer_idx': tokenizer.idx_to_token,
}, 'checkpoints/model_v1.pt')
print("\nModel saved to checkpoints/model_v1.pt")