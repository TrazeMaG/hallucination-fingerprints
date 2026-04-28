from datasets import Dataset
import pandas as pd

print("=" * 55)
print("HALLBENCH — Building dataset")
print("=" * 55)

data = {
    "prompt": [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Japan is",
        "The capital of Spain is",
        "The capital of Portugal is",
        "The capital of Poland is",
        "The capital of Greece is",
        "The capital of Egypt is",
        "The capital of Russia is",
        "The capital of Canada is",
        "The capital of Mexico is",
        "The capital of India is",
        "The capital of Argentina is",
        "The capital of Turkey is",
        "The telephone was invented by",
        "The lightbulb was invented by",
        "The theory of gravity by",
        "The theory of relativity by",
        "The printing press invented by",
        "Hamlet was written by",
        "The Odyssey was written by",
        "1984 was written by",
        "Don Quixote was written by",
        "The Iliad was written by",
        "Water is made of hydrogen and",
        "The Earth orbits the",
        "The speed of light is",
        "The human body has",
        "DNA was discovered by",
        "The Berlin Wall fell in",
        "World War 2 ended in",
        "World War 1 ended in",
        "The French Revolution began in",
        "Neil Armstrong landed on",
    ],
    "correct_answer": [
        "Paris", "Berlin", "Rome", "Tokyo", "Madrid",
        "Lisbon", "Warsaw", "Athens", "Cairo", "Moscow",
        "Ottawa", "Mexico City", "New Delhi", "Buenos Aires", "Ankara",
        "Bell", "Edison", "Newton", "Einstein", "Gutenberg",
        "Shakespeare", "Homer", "Orwell", "Cervantes", "Homer",
        "oxygen", "Sun", "299792458", "206", "Watson",
        "1989", "1945", "1918", "1789", "Moon",
    ],
    "gpt2_prediction": [
        "the", "the", "Rome", "the", "Madrid",
        "the", "Warsaw", "Athens", "the", "the",
        "the", "the", "the", "the", "the",
        "a", "a", "which", "the", "the",
        "the", "the", "the", "the", "the",
        "oxygen", "Sun", "the", "a", "a",
        "1989", "a", "a", "17", "the",
    ],
    "is_correct": [
        False, False, True, False, True,
        False, True, True, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
        True, True, False, False, False,
        True, False, False, False, False,
    ],
    "hallucination_type": [
        "TYPE2A_SUPPRESSION", "TYPE2A_SUPPRESSION", "CORRECT",
        "TYPE2A_SUPPRESSION", "CORRECT",
        "TYPE2A_SUPPRESSION", "CORRECT", "CORRECT",
        "TYPE2A_SUPPRESSION", "TYPE2A_SUPPRESSION",
        "TYPE2B_GAP", "TYPE2B_GAP", "TYPE2B_GAP",
        "TYPE2A_SUPPRESSION", "TYPE2B_GAP",
        "TYPE2B_GAP", "TYPE2B_GAP",
        "TYPE2A_SUPPRESSION", "TYPE2A_SUPPRESSION", "TYPE2A_SUPPRESSION",
        "TYPE2B_GAP", "TYPE2B_GAP", "TYPE2B_GAP",
        "TYPE2B_GAP", "TYPE2B_GAP",
        "CORRECT", "CORRECT", "TYPE2B_GAP",
        "TYPE2B_GAP", "TYPE2B_GAP",
        "CORRECT", "TYPE2A_SUPPRESSION",
        "TYPE2B_GAP", "TYPE2B_GAP", "TYPE2B_GAP",
    ],
    "category": [
        "capitals", "capitals", "capitals", "capitals", "capitals",
        "capitals", "capitals", "capitals", "capitals", "capitals",
        "capitals", "capitals", "capitals", "capitals", "capitals",
        "inventors", "inventors", "inventors", "inventors", "inventors",
        "authors", "authors", "authors", "authors", "authors",
        "science", "science", "science", "science", "science",
        "history", "history", "history", "history", "history",
    ],
    "relation_word": [
        "capital", "capital", "capital", "capital", "capital",
        "capital", "capital", "capital", "capital", "capital",
        "capital", "capital", "capital", "capital", "capital",
        "invented", "invented", "theory", "theory", "invented",
        "written", "written", "written", "written", "written",
        "made", "orbits", "speed", "body", "discovered",
        "fell", "ended", "ended", "began", "landed",
    ],
    "peak_factual_layer": [
        10, 11, 10, 11, 11,
        10, 10, 10, 10, 10,
        None, None, None, None, None,
        None, None, 10, 10, 10,
        None, None, None, None, None,
        10, 10, None, None, None,
        10, 10, None, 10, None,
    ],
    "suppression_layer": [
        12, 12, 12, 12, 12,
        12, 12, 12, 12, 12,
        None, None, None, None, None,
        None, None, 12, 12, 12,
        None, None, None, None, None,
        12, 12, None, None, None,
        12, 12, None, 12, None,
    ],
    "model": ["gpt2"] * 35,
    "model_params": [124439808] * 35,
    "paper": ["Hallucination Fingerprints (Upadhyay, 2026)"] * 35,
}

df = pd.DataFrame(data)
print(f"Dataset size: {len(df)} examples")
print(f"\nLabel distribution:")
print(df['hallucination_type'].value_counts())
print(f"\nCategory distribution:")
print(df['category'].value_counts())

dataset = Dataset.from_pandas(df)

print(f"\nPushing to HuggingFace Hub...")
print(f"Repository: Trazemag/hallbench")

dataset.push_to_hub(
    "Trazemag/hallbench",
    private=False,
    commit_message="Initial HallBench dataset - 35 labeled hallucination examples"
)

print(f"\nHallBench live at:")
print(f"https://huggingface.co/datasets/Trazemag/hallbench")
print(f"\nCite as:")
print(f"  Upadhyay, N. (2026). HallBench: A Benchmark for")
print(f"  Hallucination Detection in Language Models.")
print(f"  https://huggingface.co/datasets/Trazemag/hallbench")