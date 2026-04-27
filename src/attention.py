import torch
import torch.nn.functional as F

# Let's build attention step by step
# We have a sentence: 5 words, each with 4 numbers
# (using 4 instead of 512 so we can actually read the output)

sentence = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # word 1: "The"
    [0.0, 2.0, 0.0, 2.0],  # word 2: "capital"
    [1.0, 1.0, 1.0, 1.0],  # word 3: "of"
    [0.0, 0.0, 2.0, 2.0],  # word 4: "France"
    [1.0, 2.0, 0.0, 1.0],  # word 5: "is"
])

print("Sentence shape:", sentence.shape)
# Should be [5, 4] - 5 words, 4 numbers each

# STEP 1: every word asks every other word "how similar are you to me?"
# We do this by multiplying the sentence with itself (transposed)
# Don't worry about why yet - just run it and look at the output

similarity = torch.matmul(sentence, sentence.T)
print("\nSimilarity scores (every word vs every word):")
print(similarity)
print("Shape:", similarity.shape)
# Should be [5, 5] - each word scored against each word

# STEP 2: convert raw scores into probabilities using Softmax
# Softmax squashes all numbers so each row adds up to 1.0
# High scores get close to 1, low scores get close to 0

attention_weights = F.softmax(similarity, dim=-1)
print("\nAttention weights (each row adds up to 1.0):")
print(attention_weights.round(decimals=2))

# Verify each row adds to 1
print("\nEach row sum (should all be 1.0):")
print(attention_weights.sum(dim=-1))

# STEP 3: each word rewrites itself by mixing in
# information from words it's paying attention to
#
# attention_weights = how much each word cares about each other word
# sentence = the actual information each word contains
#
# new meaning = attention_weights × sentence
# i.e. blend all words together weighted by attention scores

new_meaning = torch.matmul(attention_weights, sentence)
print("\nOriginal sentence (before attention):")
print(sentence)
print("\nAfter attention (each word updated with context):")
print(new_meaning)
print("\nShape:", new_meaning.shape)

# Let's look at how much "France" changed
print("\n--- How much did France change? ---")
print("Before:", sentence[3])
print("After: ", new_meaning[3].round(decimals=2))

import torch.nn as nn

# d_model = how many numbers per word (we use 4 in our toy example)
# In real models this is 512 or 768 or 4096
d_model = 4

# These are the three learned matrices
# They start random - in a real model they get learned during training
torch.manual_seed(42)  # so we all get the same random numbers
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)

print("W_Q shape:", W_Q.shape)  # [4, 4]

# Transform our sentence through each matrix
Q = torch.matmul(sentence, W_Q)  # what is each word looking for?
K = torch.matmul(sentence, W_K)  # what does each word contain?
V = torch.matmul(sentence, W_V)  # what will each word share?

print("Q shape:", Q.shape)  # still [5, 4] - same shape, different meaning
print("K shape:", K.shape)
print("V shape:", V.shape)

# Now compute attention using Q and K instead of raw sentence
import math
d_k = d_model  # size of each query/key vector

scores = torch.matmul(Q, K.T) / math.sqrt(d_k)  # scale by √d_k
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)

print("\nReal attention output shape:", output.shape)
print("\nAttention weights (real):")
print(weights.round(decimals=2))

# ─── MULTI-HEAD ATTENTION ───────────────────────────────────────

# We'll use 2 heads (our toy example only has 4 dims, so 2 heads of 2 dims each)
# Real models: 8 heads, 64 dims each = 512 total
num_heads = 2
d_model = 4
d_head = d_model // num_heads  # 2 dims per head

print("\n─── Multi-Head Attention ───")
print(f"Total dims: {d_model}, Heads: {num_heads}, Dims per head: {d_head}")

torch.manual_seed(42)

# Each head has its own W_Q, W_K, W_V
# Shape: [d_model, d_head] - projects from full size down to head size
heads_output = []

for i in range(num_heads):
    # Each head gets its own learned matrices
    W_Q_i = torch.randn(d_model, d_head)
    W_K_i = torch.randn(d_model, d_head)
    W_V_i = torch.randn(d_model, d_head)

    # Project sentence into this head's Q, K, V
    Q_i = torch.matmul(sentence, W_Q_i)  # [5, 2]
    K_i = torch.matmul(sentence, W_K_i)  # [5, 2]
    V_i = torch.matmul(sentence, W_V_i)  # [5, 2]

    # Compute attention for this head
    scores_i = torch.matmul(Q_i, K_i.T) / math.sqrt(d_head)
    weights_i = F.softmax(scores_i, dim=-1)
    output_i = torch.matmul(weights_i, V_i)  # [5, 2]

    heads_output.append(output_i)
    print(f"\nHead {i+1} attention weights:")
    print(weights_i.round(decimals=2))

# Concatenate all head outputs along the last dimension
# [5, 2] + [5, 2] → [5, 4]
multi_head_output = torch.cat(heads_output, dim=-1)
print("\nAfter concatenating all heads:")
print("Shape:", multi_head_output.shape)  # back to [5, 4]

# Final linear projection - mixes information across heads
W_O = torch.randn(d_model, d_model)
final_output = torch.matmul(multi_head_output, W_O)
print("After final projection shape:", final_output.shape)  # [5, 4]
print("\nFinal output:")
print(final_output.round(decimals=2))

# ─── FEED FORWARD NETWORK ────────────────────────────────────────

print("\n─── Feed Forward Network ───")

# In our toy example: d_model=4, we expand to 4*4=16 then back to 4
# In real models: 512 → 2048 → 512
d_model = 4
d_ff = d_model * 4  # expand to 4x size

# Two learned weight matrices
torch.manual_seed(0)
W1 = torch.randn(d_model, d_ff)   # expand: [4, 16]
W2 = torch.randn(d_ff, d_model)   # compress: [16, 4]
b1 = torch.randn(d_ff)            # bias 1
b2 = torch.randn(d_model)         # bias 2

print(f"W1 shape: {W1.shape}  (expand {d_model} → {d_ff})")
print(f"W2 shape: {W2.shape}  (compress {d_ff} → {d_model})")

# Run FFN on our sentence (use the multi_head_output from before)
# Step 1: expand
hidden = torch.matmul(multi_head_output, W1) + b1
print(f"\nAfter expansion shape: {hidden.shape}")  # [5, 16]

# Step 2: ReLU - set all negative numbers to 0
# Negative numbers = "this pattern is not relevant here"
# Positive numbers = "this pattern matters"
hidden_activated = torch.relu(hidden)
print(f"After ReLU shape: {hidden_activated.shape}")  # still [5, 16]
print(f"\nBefore ReLU (first word, first 6 numbers): {hidden[0,:6].round(decimals=2)}")
print(f"After  ReLU (first word, first 6 numbers): {hidden_activated[0,:6].round(decimals=2)}")

# Step 3: compress back down
ffn_output = torch.matmul(hidden_activated, W2) + b2
print(f"\nAfter compression shape: {ffn_output.shape}")  # [5, 4]
print("\nFFN output:")
print(ffn_output.round(decimals=2))

# ─── LAYER NORM ──────────────────────────────────────────────────

print("\n─── Layer Norm ───")

# PyTorch has this built in - normalizes the last dimension
layer_norm = torch.nn.LayerNorm(d_model)

# Apply to FFN output
normed_output = layer_norm(ffn_output)

print("Before Layer Norm (first word):", ffn_output[0].round(decimals=2))
print("After  Layer Norm (first word):", normed_output[0].round(decimals=2))

print("\nBefore - mean:", ffn_output[0].mean().round(decimals=2).item(),
      "std:", ffn_output[0].std().round(decimals=2).item())
print("After  - mean:", normed_output[0].mean().round(decimals=2).item(),
      "std:", normed_output[0].std().round(decimals=2).item())

print("\nFinal shape:", normed_output.shape)