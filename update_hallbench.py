import json
import pandas as pd
from datasets import Dataset

print("=" * 55)
print("HALLBENCH UPDATE — 20,000 examples")
print("=" * 55)

# Load results from GPU experiment
with open('results/large_scale_results.json', 'r') as f:
    results = json.load(f)

print(f"Loaded {len(results):,} results")

# Convert to dataset format
df = pd.DataFrame(results)

# Clean up column names to match original schema
df = df.rename(columns={
    'correct': 'correct_answer',
    'peak_layer': 'peak_factual_layer',
})

# Add metadata columns
df['model'] = 'gpt2'
df['model_params'] = 124439808
df['paper'] = 'Hallucination Fingerprints (Upadhyay, 2026)'

print(f"\nDataset columns: {list(df.columns)}")
print(f"\nLabel distribution:")
print(df['hallucination_type'].value_counts())
print(f"\nCategory distribution:")
print(df['category'].value_counts())
print(f"\nAccuracy: {df['is_correct'].mean()*100:.1f}%")

# Push to HuggingFace
dataset = Dataset.from_pandas(df)

print(f"\nPushing 20,000 examples to HuggingFace...")
dataset.push_to_hub(
    "Trazemag/hallbench",
    private=False,
    commit_message="Update: 20,000 examples from large scale GPU experiment"
)

print(f"\nHallBench updated at:")
print(f"https://huggingface.co/datasets/Trazemag/hallbench")
print(f"\n20,000 labeled hallucination examples now public.")
print(f"Other researchers can load with:")
print(f"  from datasets import load_dataset")
print(f"  ds = load_dataset('Trazemag/hallbench')")