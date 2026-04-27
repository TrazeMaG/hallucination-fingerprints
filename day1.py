import torch

# A tensor with a single number 
a = torch.tensor(5.0)
print("single number:", a)
print("Shape:", a.shape)

# A 1D tensor  -  like a list
b = torch.tensor([1.0, 2.0, 3.0])
print("\n1D tensor:", b)
print("Shape:", b.shape)

# A 2D tensor  -  like a grid 
c = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
print("\n2D tensor:", c)
print("Shape:", c.shape)
print("Rows:", c.shape[0], "Columns:", c.shape[1])

# This is what a word looks like inside an AI model
# "France" gets represented as 512 random numbers
# (in a real model, these numbers are learned, not random)
word = torch.randn(512)
print("\nWhat 'France' looks like to the model:")
print("Shape:", word.shape)
print("First 5 numbers:", word[:5])

# A whole sentence of 5 words, each with 512 numbers
sentence = torch.randn(5, 512)
print("\nWhat a 5-word sentence looks like:")
print("Shape:", sentence.shape)
print("That's", sentence.shape[0], "words,", sentence.shape[1], "numbers per word")