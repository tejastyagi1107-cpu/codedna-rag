import ast


def extract_chunks(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        source_code = f.read()

    if not source_code.strip():
        return []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    # Collect method nodes inside classes so we can skip them as top-level functions
    method_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_nodes.add(id(item))

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and id(node) not in method_nodes:
            chunks.append({
                "name": node.name,
                "type": "function",
                "source_code": ast.get_source_segment(source_code, node),
                "filename": file_path,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            })
        elif isinstance(node, ast.ClassDef):
            chunks.append({
                "name": node.name,
                "type": "class",
                "source_code": ast.get_source_segment(source_code, node),
                "filename": file_path,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            })

    return chunks


if __name__ == "__main__":
    chunks = extract_chunks("tests/sample.py")
    for chunk in chunks:
        print(chunk)
        print("---")
