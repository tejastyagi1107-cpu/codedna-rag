# explore_embeddings.py
# Learn how sentence embeddings work using a small model

# Step 1: Check if sentence-transformers is installed, install if not
import importlib
import subprocess
import sys

if importlib.util.find_spec("sentence_transformers") is None:
    print("sentence-transformers not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    print("Installation complete!\n")
else:
    print("sentence-transformers is already installed.\n")

# Step 2 & 3: Import and load the model
from sentence_transformers import SentenceTransformer, util

print("Loading model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

# Step 4: Define 3 sentences
sentences = [
    "The cat sat on the mat",       # sentence 1
    "A feline rested on the rug",   # sentence 2
    "I love eating pizza",          # sentence 3
]

# Step 5: Encode all 3 sentences into vectors
print("Encoding sentences...")
embeddings = model.encode(sentences)

# Step 6: Print the shape/length of the first vector
print(f"\nShape of the first embedding vector: {embeddings[0].shape}")
print(f"  (That's {len(embeddings[0])} numbers per sentence)\n")

# Step 7: Print the first 5 numbers of the first vector
print("First 5 numbers of sentence 1's vector (raw embedding values):")
print(f"  {embeddings[0][:5]}\n")

# Step 8: util is already imported above from sentence_transformers

# Step 9: Cosine similarity between sentence 1 and sentence 2
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])

# Step 10: Cosine similarity between sentence 1 and sentence 3
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

# Step 11: Print both similarity scores
print("--- Cosine Similarity Scores ---")
print(f"Similarity between sentence 1 and 2 (cat/feline): {sim_1_2.item():.4f}")
print(f"Similarity between sentence 1 and 3 (cat/pizza):  {sim_1_3.item():.4f}")
print()
print("Interpretation:")
print("  Score close to 1.0 → sentences are very similar in meaning")
print("  Score close to 0.0 → sentences are very different in meaning")
