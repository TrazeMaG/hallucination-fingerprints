import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import BPETokenizer
import os

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.seq_len = seq_len
        
        # Encode the entire text into one long list of numbers
        print("Encoding text...")
        self.tokens = tokenizer.encode(text)
        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Sequence length: {seq_len}")
        print(f"Total training examples: {len(self):,}")

    def __len__(self):
        # How many training examples can we make?
        # Each example is seq_len tokens
        # We need one extra token for the target
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        # Input: tokens at positions idx to idx+seq_len
        # Target: tokens at positions idx+1 to idx+seq_len+1
        # (shifted by one — the model predicts the next token)
        
        input_tokens = self.tokens[idx : idx + self.seq_len]
        target_tokens = self.tokens[idx+1 : idx + self.seq_len + 1]
        
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long)
        )


def prepare_data(text, vocab_size=1000, seq_len=32, batch_size=16):
    """Full pipeline: text → tokenizer → dataset → dataloader"""
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    tokenizer.build_vocab(text)
    
    # Create dataset
    dataset = TextDataset(text, tokenizer, seq_len)
    
    # Create dataloader — shuffles and batches automatically
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    print(f"\nDataloader ready")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Each batch: input {batch_size}×{seq_len}, "
          f"target {batch_size}×{seq_len}")
    
    return tokenizer, dataloader


# ─── TEST IT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sample text — we'll replace this with Wikipedia later
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
    """ * 50  # repeat to get more training data

    tokenizer, dataloader = prepare_data(
        text,
        vocab_size=500,
        seq_len=32,
        batch_size=4
    )

    # Look at one batch
    print("\n─── Sample Batch ───")
    inputs, targets = next(iter(dataloader))
    print(f"Input shape:  {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"\nFirst example:")
    print(f"Input tokens:  {inputs[0].tolist()}")
    print(f"Target tokens: {targets[0].tolist()}")
    print(f"\nDecoded input:  '{tokenizer.decode(inputs[0].tolist())}'")
    print(f"Decoded target: '{tokenizer.decode(targets[0].tolist())}'")
    print(f"\nNotice: target is input shifted by one position")
    print(f"Model learns: given these tokens, predict the next one")