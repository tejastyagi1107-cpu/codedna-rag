# eval/test_questions.py
# Runs 5 questions through the full RAG pipeline and prints the results

# Step 1: Import query_chunks — retrieves relevant code chunks from ChromaDB
from core.vector_store import query_chunks

# Step 2: Import generate_answer — sends chunks + question to the LLM
from core.llm_handler import generate_answer


# Step 3: Define the 5 test questions about the Flask codebase
questions = [
    "how does session handling work",
    "how does routing work in this codebase",
    "what does the Flask class do",
    "how are cookies set in responses",
    "how does error handling work",
]


# Step 4: Loop through each question and run the full RAG pipeline
for question in questions:

    # Retrieve the top 5 most relevant code chunks from ChromaDB
    chunks = query_chunks(question, n_results=5)

    # Generate a grounded answer using the LLM
    answer = generate_answer(question, chunks)

    # Print the question
    print(f"QUESTION: {question}")
    print()

    # Print just the names of the 5 retrieved chunks (not the full code)
    print("Retrieved chunks:")
    for chunk in chunks:
        print(f"  - {chunk['metadata']['name']} (from {chunk['metadata']['filename']})")
    print()

    # Print the LLM's answer
    print(f"Answer:\n{answer}")

    # Print a clear separator between questions
    print("=" * 60)
    print()
