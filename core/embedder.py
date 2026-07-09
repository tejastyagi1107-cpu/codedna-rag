# embedder.py
# Takes AST chunks and turns their source code into embedding vectors

# Step 1: Import extract_chunks from our chunker
from core.ast_chunker import extract_chunks

# Step 2 & 3: Import the shared model from core.model
# The model is loaded once there and reused everywhere — no double loading
from core.model import model


# Step 4: Define the embed_chunks function
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Takes a list of chunk dicts (from extract_chunks) and adds an
    'embedding' key to each one containing the vector for its source code.

    Args:
        chunks: list of dicts, each with at least a 'source_code' field

    Returns:
        The same list of dicts, each now also containing an 'embedding' field
    """
    # Step 5: Loop through each chunk and encode its source code
    for chunk in chunks:
        # Encode the source code string into a 384-dim vector
        vector = model.encode(chunk["source_code"])

        # Add the vector back into the same chunk dict
        chunk["embedding"] = vector

    # Return the updated list (each chunk now has all original keys + "embedding")
    return chunks


# Step 6: Test block — only runs when this file is executed directly
if __name__ == "__main__":
    # Extract chunks from our sample file
    chunks = extract_chunks("tests/sample.py")

    # Embed all chunks
    embedded_chunks = embed_chunks(chunks)

    # Print results for each chunk
    for chunk in embedded_chunks:
        print(f"Name:            {chunk['name']}")
        print(f"Type:            {chunk['type']}")
        print(f"Embedding shape: {chunk['embedding'].shape}")
        print("---")
