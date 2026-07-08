import sys
sys.path.insert(0, ".")

from core.ast_chunker import extract_chunks

chunks = extract_chunks("tests/sample_codebases/flask/src/flask/app.py")

print(f"Total chunks found: {len(chunks)}")

function_chunks = [c for c in chunks if c["type"] == "function"]
class_chunks = [c for c in chunks if c["type"] == "class"]

print(f"Functions: {len(function_chunks)}")
print(f"Classes: {len(class_chunks)}")

print("\nFirst 5 chunk names:")
for chunk in chunks[:5]:
    print(f"  - {chunk['name']}")
