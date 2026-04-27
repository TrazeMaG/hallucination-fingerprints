from hallscan import scan

print("=" * 55)
print("HALLSCAN v0.1.0 — DEMO")
print("by Nikhil Upadhyay")
print("=" * 55)

# Test 1 — Type 2a suppression
result1 = scan(
    "The capital of France is",
    relation_word="capital",
    correct_answer="Paris"
)
print(result1)

# Test 2 — Correct prediction
result2 = scan(
    "The Berlin Wall fell in",
    correct_answer="1989"
)
print(result2)

# Test 3 — No correct answer provided (blind scan)
result3 = scan(
    "The capital of Germany is",
    relation_word="capital"
)
print(result3)
print(f"Top 5 predictions:")
for token, prob in result3.top10[:5]:
    bar = "█" * int(prob * 40)
    print(f"  '{token}': {prob:.4f} {bar}")