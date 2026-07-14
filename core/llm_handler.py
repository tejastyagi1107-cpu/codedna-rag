# llm_handler.py
# Takes a question + relevant code chunks and uses a local LLM (via Ollama) to generate an answer

# Step 1: Import ollama — the Python client for locally-running LLMs
import ollama


# Step 2: Define the generate_answer function
def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Generates a natural language answer to a question using relevant code chunks as context.

    Args:
        question: A natural language question about the codebase
        chunks:   A list of result dicts from query_chunks(), each containing:
                    - 'document':  the raw source code string
                    - 'metadata':  dict with 'name', 'filename', etc.
                    - 'distance':  similarity score (not used here)

    Returns:
        A string containing the LLM's answer
    """

    # Step 3a: Build a single "context" string from all the chunks
    # Each chunk gets a header showing which function/class it is and where it lives,
    # followed by its source code, then a "---" divider
    context_parts = []
    for chunk in chunks:
        name     = chunk["metadata"]["name"]
        filename = chunk["metadata"]["filename"]
        code     = chunk["document"]

        # Format each chunk clearly so the LLM can read it easily
        context_parts.append(f"Function: {name} (from {filename})\n{code}\n---")

    # Join all the formatted chunks into one big context string
    context = "\n".join(context_parts)

    # Step 3b: Build the full prompt
    # We instruct the LLM to ONLY use the provided code — no hallucinating
    prompt = f"""You are a helpful assistant answering questions about a codebase. \
Use ONLY the following code to answer the question. If the code doesn't contain \
enough information to answer, say so honestly instead of guessing.

Code context:
{context}

Question: {question}

Answer:"""

    # Step 3c: Call Ollama with the llama3 model
    # messages is a list of dicts — like a chat history. We only send one user message.
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    # Step 3d: Extract and return just the text of the reply
    return response['message']['content']


# Step 4: Test block — only runs when you execute this file directly
if __name__ == "__main__":
    from core.vector_store import query_chunks

    question = "how does session handling work"

    # Retrieve the top matching code chunks from ChromaDB
    chunks = query_chunks(question)

    # Ask the LLM to answer based on those chunks
    answer = generate_answer(question, chunks)

    print("Question:", question)
    print("\nAnswer:", answer)
