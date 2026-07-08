import sys
sys.path.insert(0, ".")

from core.ast_chunker import extract_chunks


def naive_chunk(file_path: str, chunk_size: int = 500) -> list[str]:
    with open(file_path, "r") as f:
        source_code = f.read()

    chunks = []
    for i in range(0, len(source_code), chunk_size):
        chunks.append(source_code[i:i + chunk_size])

    return chunks


if __name__ == "__main__":
    file_path = "tests/sample_codebases/flask/src/flask/app.py"

    naive_chunks = naive_chunk(file_path)
    ast_chunks = extract_chunks(file_path)

    print("=" * 60)
    print("CHUNKING COMPARISON: Naive vs AST-based")
    print("=" * 60)
    print(f"Total naive chunks (500 chars each): {len(naive_chunks)}")
    print(f"Total AST chunks:                    {len(ast_chunks)}")
    print()

    # Find a function chunk that gets split by naive chunking
    split_example = None

    for chunk in ast_chunks:
        if chunk["type"] != "function":
            continue

        func_source = chunk["source_code"]
        func_start = func_source[:40]   # first 40 chars to locate it in naive chunks
        func_end = func_source[-40:]    # last 40 chars to locate where it ends

        start_chunk_idx = None
        end_chunk_idx = None

        for idx, naive in enumerate(naive_chunks):
            if func_start in naive and start_chunk_idx is None:
                start_chunk_idx = idx
            if func_end in naive:
                end_chunk_idx = idx

        if start_chunk_idx is not None and end_chunk_idx is not None:
            if start_chunk_idx != end_chunk_idx:
                split_example = {
                    "chunk": chunk,
                    "start_chunk_idx": start_chunk_idx,
                    "end_chunk_idx": end_chunk_idx,
                }
                break

    if split_example is None:
        print("No function found that is split across naive chunks (all functions are short).")
    else:
        func = split_example["chunk"]
        start_idx = split_example["start_chunk_idx"]
        end_idx = split_example["end_chunk_idx"]

        print("-" * 60)
        print(f"EXAMPLE FUNCTION: '{func['name']}'")
        print(f"Lines {func['start_line']} to {func['end_line']} in {func['filename']}")
        print("-" * 60)

        print("\n>>> FULL SOURCE CODE (as seen by AST chunker):\n")
        print(func["source_code"])

        print()
        print("=" * 60)
        print("COMPARISON RESULT")
        print("=" * 60)

        func_char_count = len(func["source_code"])
        print(f"AST chunking result for '{func['name']}':")
        print(f"  Complete, uncut, {func_char_count} characters — extracted as one clean unit.")

        print()
        print(f"Naive chunking result for the same function:")
        print(f"  Split across chunk #{start_idx} and chunk #{end_idx} — cut mid-function.")
        print(f"  Chunk #{start_idx} starts the function but does not contain its end.")
        print(f"  Chunk #{end_idx} contains the end but is missing the beginning context.")

        print()
        print(f"  [Chunk #{start_idx} tail — last 120 chars]:")
        print(f"  ...{naive_chunks[start_idx][-120:]!r}")
        print()
        print(f"  [Chunk #{end_idx} head — first 120 chars]:")
        print(f"  {naive_chunks[end_idx][:120]!r}...")
