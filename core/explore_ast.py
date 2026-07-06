import ast

with open("tests/sample.py", "r") as f:
    source_code = f.read()

tree = ast.parse(source_code)
print(ast.dump(tree, indent=4))
