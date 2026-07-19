# llm_handler.py
# Takes a question + relevant code chunks and uses an LLM to generate an answer.
# Automatically uses Groq (cloud) if GROQ_API_KEY is set, otherwise falls back to Ollama (local).

# Step 1: Standard library imports
import os

# Step 2: Import ollama — the Python client for locally-running LLMs
import ollama

# Step 3: Import Groq — fast cloud inference via the Groq API
from groq import Groq


# Step 4: Define the generate_answer function
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

    Backend selection:
        - If the GROQ_API_KEY environment variable is set → use Groq API (llama-3.3-70b-versatile)
        - Otherwise → fall back to local Ollama (llama3)
    """

    # Step 5a: Build a single "context" string from all the chunks
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

    # Step 5b: Build the full prompt
    # We instruct the LLM to ONLY use the provided code — no hallucinating
    prompt = f"""You are a helpful assistant answering questions about a codebase. \
Use ONLY the following code to answer the question. If the code doesn't contain \
enough information to answer, say so honestly instead of guessing.

Code context:
{context}

Question: {question}

Answer:"""

    # Step 5c: Choose backend based on whether GROQ_API_KEY is set
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if groq_api_key:
        # ── Groq path (cloud) ───────────────────────────────────────────────
        # Fast inference via Groq API; model: llama-3.3-70b-versatile
        client = Groq(api_key=groq_api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the text reply from the Groq response object
        return response.choices[0].message.content

    else:
        # ── Ollama path (local fallback) ────────────────────────────────────
        # Requires Ollama running locally with the llama3 model pulled
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract and return just the text of the reply
        return response['message']['content']


# Step 6: Test block — only runs when you execute this file directly
if __name__ == "__main__":
    from core.vector_store import query_chunks

    question = "how does session handling work"

    # Retrieve the top matching code chunks from ChromaDB
    chunks = query_chunks(question)

    # Ask the LLM to answer based on those chunks
    answer = generate_answer(question, chunks)

    print("Question:", question)
    print("\nAnswer:", answer)

