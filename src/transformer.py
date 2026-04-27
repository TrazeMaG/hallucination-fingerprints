import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─── MULTI-HEAD ATTENTION ─────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, seq_len, d_model = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        Q = self.split_heads(self.W_Q(x))
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch, seq_len, d_model)
        return self.W_O(attended), weights


# ─── FEED FORWARD NETWORK ─────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * 4)
        self.compress = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.compress(F.relu(self.expand(x)))


# ─── TRANSFORMER BLOCK ────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attended, weights = self.attention(x)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x, weights


# ─── POSITIONAL ENCODING ──────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()

        # Create a matrix of shape [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)

        # Position indices: 0, 1, 2, ..., max_seq_len
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # Sine and cosine waves of different frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )

        # Even dimensions get sine, odd dimensions get cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer — not a learned parameter, just stored
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to word embeddings
        # x shape: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


# ─── FULL TRANSFORMER MODEL ───────────────────────────────────────

class Hallucinations_Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_blocks, max_seq_len):
        super().__init__()

        # Convert word indices to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Add position information
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Stack of transformer blocks — this is the deep part
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) 
            for _ in range(num_blocks)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Project from d_model back to vocab_size
        # This produces a score for every word in vocabulary
        # Highest score = predicted next word
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: [batch, seq_len] — indices of words

        # Step 1: word indices → vectors
        x = self.embedding(x)                    # [batch, seq_len, d_model]

        # Step 2: add position information
        x = self.pos_encoding(x)                 # [batch, seq_len, d_model]

        # Step 3: pass through all transformer blocks
        all_weights = []
        for block in self.blocks:
            x, weights = block(x)
            all_weights.append(weights)          # save attention maps

        # Step 4: final normalisation
        x = self.norm(x)                         # [batch, seq_len, d_model]

        # Step 5: project to vocabulary scores
        logits = self.output_projection(x)       # [batch, seq_len, vocab_size]

        # Return logits and all attention weights
        # all_weights is what we'll read for hallucination fingerprints
        return logits, all_weights


# ─── TEST THE FULL MODEL ──────────────────────────────────────────

if __name__ == "__main__":
    # Model settings
    VOCAB_SIZE = 10000    # we know 10,000 words
    D_MODEL = 256         # numbers per word (smaller for speed)
    NUM_HEADS = 8         # attention heads
    NUM_BLOCKS = 6        # transformer blocks stacked
    MAX_SEQ_LEN = 512     # maximum sentence length

    # Build the model
    model = Hallucinations_Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_seq_len=MAX_SEQ_LEN
    )

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built successfully")
    print(f"Total parameters: {total_params:,}")
    print(f"That's {total_params/1e6:.1f} million numbers to learn\n")

    # Fake input — batch of 2 sentences, each 10 words long
    # Numbers represent word indices (like "France" = word number 4521)
    fake_input = torch.randint(0, VOCAB_SIZE, (2, 10))
    print(f"Input shape: {fake_input.shape}")
    print(f"Sample input (word indices): {fake_input[0]}\n")

    # Run forward pass
    logits, all_weights = model(fake_input)

    print(f"Output logits shape: {logits.shape}")
    print(f"  = {logits.shape[0]} sentences")
    print(f"  × {logits.shape[1]} words")
    print(f"  × {logits.shape[2]} vocabulary scores per word\n")

    print(f"Number of attention maps saved: {len(all_weights)}")
    print(f"Each map shape: {all_weights[0].shape}")
    print(f"  = {all_weights[0].shape[1]} heads")
    print(f"  × {all_weights[0].shape[2]}×{all_weights[0].shape[3]} attention grid\n")

    # The predicted next word for each position
    predicted_indices = logits.argmax(dim=-1)
    print(f"Predicted word indices: {predicted_indices[0]}")
    print(f"\nModel is untrained so predictions are random.")
    print(f"After training these will be meaningful.")
    print(f"\n── This is what we will read for hallucination fingerprints ──")
    print(f"all_weights has {len(all_weights)} blocks × {all_weights[0].shape[1]} heads")
    print(f"= {len(all_weights) * all_weights[0].shape[1]} attention maps per sentence")
    print(f"We will scan all of these looking for the fingerprint.")