# model.py
# Shared embedding model — loaded ONCE here, imported everywhere else.
# This avoids reloading weights every time a module that needs the model is imported.

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
