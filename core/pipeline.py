# pipeline.py
# Ties everything together: walk a folder, extract chunks, embed them, and store them

# Step 1: Import os so we can walk the folder
import os

# Step 2: Import extract_chunks — turns a single .py file into a list of chunk dicts
from core.ast_chunker import extract_chunks

# Step 3: Import embed_chunks — adds an 'embedding' vector to each chunk
from core.embedder import embed_chunks

# Step 4: Import store_chunks and query_chunks from our vector store
from core.vector_store import store_chunks, query_chunks


# Step 5: Define the main pipeline function
def index_folder(folder_path: str) -> int:
    """
    Walks through every .py file in a folder (and all its subfolders),
    extracts code chunks from each file, embeds them all, stores them
    in ChromaDB, and returns the total number of chunks indexed.

    Args:
        folder_path: Path to the root folder to scan (e.g. "src/myproject")

    Returns:
        The total number of chunks indexed (an integer)
    """

    # Step 6a: Collect all chunks from every .py file in the folder
    all_chunks = []

    # os.walk() yields (current_folder, list_of_subfolders, list_of_files)
    # for every folder it visits, starting at folder_path
    for current_folder, subfolders, files in os.walk(folder_path):
        for filename in files:
            # Only process Python files
            if filename.endswith(".py"):
                # Build the full path to this file
                file_path = os.path.join(current_folder, filename)

                # Extract all functions and classes from this file
                chunks = extract_chunks(file_path)

                # Add them to our running list
                all_chunks.extend(chunks)

    # Step 6b: Embed all chunks in one go (one model.encode call per chunk internally)
    embedded_chunks = embed_chunks(all_chunks)

    # Step 6c: Store all embedded chunks in ChromaDB
    store_chunks(embedded_chunks)

    # Step 6d: Return how many chunks were indexed
    return len(embedded_chunks)


# Step 7: Test block — only runs when you execute this file directly
if __name__ == "__main__":
    total = index_folder("tests/sample_codebases/flask/src/flask")
    print(f"Indexed {total} chunks from the folder")

    results = query_chunks("how does session handling work")
    for r in results:
        print(f"Name: {r['metadata']['name']}, File: {r['metadata']['filename']}, Distance: {r['distance']}")
