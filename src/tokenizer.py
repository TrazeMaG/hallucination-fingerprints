from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.idx_to_token = {}

    def get_pairs(self, vocab):
        """Count all adjacent pairs across all words"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_pair(self, pair, vocab):
        """Merge a specific pair everywhere it appears"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    def train(self, text):
        """Learn merge rules from text"""
        print(f"Training BPE tokenizer...")
        print(f"Target vocab size: {self.vocab_size}")

        words = text.lower().split()
        vocab = Counter()
        for word in words:
            vocab[' '.join(list(word)) + ' </w>'] += 1

        print(f"Starting vocab size: {len(vocab)} unique words")
        print(f"Sample before training: {list(vocab.items())[:3]}\n")

        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            pairs = self.get_pairs(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]

            if best_count < 2:
                break

            vocab = self.merge_pair(best_pair, vocab)
            self.merges[best_pair] = ''.join(best_pair)

            if i < 5:
                print(f"Merge {i+1}: '{best_pair[0]}' + '{best_pair[1]}' "
                      f"→ '{''.join(best_pair)}' (appeared {best_count} times)")

        print(f"\nLearned {len(self.merges)} merge rules")

    def tokenize(self, text):
        """Convert text to tokens using learned merges"""
        tokens = []
        for word in text.lower().split():
            word_tokens = list(word) + ['</w>']
            word_str = ' '.join(word_tokens)

            for pair, merged in self.merges.items():
                bigram = ' '.join(pair)
                word_str = word_str.replace(bigram, merged)

            tokens.extend(word_str.split())
        return tokens

    def build_vocab(self, text):
        """Assign a number to every token"""
        all_tokens = self.tokenize(text)
        unique_tokens = sorted(set(all_tokens))

        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }

        for token in unique_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)} tokens")

    def encode(self, text):
        """Text → list of numbers"""
        tokens = self.tokenize(text)
        return [self.vocab.get(t, 1) for t in tokens]

    def decode(self, indices):
        """List of numbers → text"""
        tokens = [self.idx_to_token.get(i, '<UNK>') for i in indices]
        text = ' '.join(tokens)
        text = text.replace('</w> ', ' ').replace('</w>', '')
        return text.strip()


# ─── TEST IT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    text = """
    the capital of france is paris the capital of germany is berlin
    the capital of italy is rome the capital of spain is madrid
    the capital of japan is tokyo the capital of china is beijing
    france is a country in europe germany is a country in europe
    paris is a beautiful city berlin is a historic city
    the president of france visited paris the king of spain lives in madrid
    """

    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(text)

    # Test tokenization
    print("\n─── Tokenization Examples ───")
    test_sentences = [
        "the capital of france is paris",
        "france is beautiful",
        "the president visited berlin",
    ]

    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        print(f"\nInput:  '{sentence}'")
        print(f"Tokens: {tokens}")
        print(f"Count:  {len(tokens)} tokens")

    # Build vocabulary and test encode/decode
    print("\n─── Encode & Decode ───")
    tokenizer.build_vocab(text)

    test = "the capital of france is paris"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)

    print(f"Original: '{test}'")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  '{decoded}'")