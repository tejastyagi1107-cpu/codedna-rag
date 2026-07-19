# vector_store.py
# Takes embedded chunks and stores them in a local ChromaDB vector database

# Step 1: Check if chromadb is installed, install if not
import importlib
import subprocess
import sys

if importlib.util.find_spec("chromadb") is None:
    print("chromadb not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb"])
    print("Installation complete!\n")

# Step 2: Import chromadb
import chromadb

# Import SentenceTransformer for encoding query strings
# Import the shared model — loaded once in core.model, reused here
from core.model import model

# Import our chunker and embedder
from core.ast_chunker import extract_chunks
from core.embedder import embed_chunks


# Step 5: Define the store_chunks function
def store_chunks(chunks: list[dict]) -> None:
    """
    Takes a list of embedded chunk dicts and stores them in a local ChromaDB collection.

    Each chunk must already have an 'embedding' field (from embed_chunks).
    Stores:
      - ids:        unique string per chunk
      - embeddings: the vector (as a plain list)
      - documents:  the raw source code string
      - metadatas:  name, type, filename, start_line, end_line
    """
    # Step 6a: Create a persistent client — data is saved to the "chroma_db" folder
    client = chromadb.PersistentClient(path="chroma_db")

    # Step 6b: Get an existing collection or create it fresh
    collection = client.get_or_create_collection(name="code_chunks")

    # Step 6c: Loop through chunks and add each one to the collection
    for chunk in chunks:
        # Build a unique ID for this chunk
        unique_id = f"{chunk['filename']}_{chunk['name']}_{chunk['start_line']}"

        collection.add(
            ids=[unique_id],

            # Convert numpy array → plain Python list (ChromaDB requires this)
            embeddings=[chunk["embedding"].tolist()],

            # The raw source code is stored as the "document"
            documents=[chunk["source_code"]],

            # Metadata: lightweight fields for filtering/display
            # NOTE: we intentionally exclude 'embedding' and 'source_code' here
            #       since they're already stored above
            metadatas=[{
                "name":       chunk["name"],
                "type":       chunk["type"],
                "filename":   chunk["filename"],
                "start_line": chunk["start_line"],
                "end_line":   chunk["end_line"],
            }]
        )


def query_chunks(question: str, n_results: int = 5) -> list[dict]:
    """
    Searches ChromaDB for the code chunks most semantically similar to the question.

    Args:
        question:  A natural language question or description to search for
        n_results: How many top results to return (default: 3)

    Returns:
        A list of dicts, each containing:
          - 'document': the raw source code of the matching chunk
          - 'metadata': dict with name, type, filename, start_line, end_line
          - 'distance': how close the match is (lower = more similar)
    """
    # Connect to the same persistent database
    client = chromadb.PersistentClient(path="chroma_db")

    # Use get_collection (not get_or_create) — it must already exist from store_chunks
    collection = client.get_collection(name="code_chunks")

    # Encode the question into a vector using the same model that encoded the chunks
    # (must use the same model so the vector spaces are compatible)
    question_vector = model.encode(question)

    # Query ChromaDB for the n most similar chunks
    # query_embeddings expects a LIST of vectors (one per query), so we wrap in [[ ... ]]
    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=n_results,
    )

    # collection.query() returns nested lists — one list per query.
    # We only sent one query, so we grab index [0] to get a flat list of results.
    documents = results["documents"][0]   # list of source code strings
    metadatas = results["metadatas"][0]   # list of metadata dicts
    distances = results["distances"][0]   # list of distance floats

    # Zip them together into a clean list of dicts, one per result
    return [
        {
            "document": doc,
            "metadata": meta,
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


# Test block — only runs when this file is executed directly
if __name__ == "__main__":
    results = query_chunks("how do I add two numbers")
    for r in results:
        print(f"Name:     {r['metadata']['name']}")
        print(f"Distance: {r['distance']:.4f}")
        print(f"Code:\n{r['document']}")
        print("---")
