import sys
sys.path.insert(0, ".")

import unittest
import tempfile
import os

from core.ast_chunker import extract_chunks


class TestExtractChunks(unittest.TestCase):

    def test_extracts_correct_number_of_chunks(self):
        chunks = extract_chunks("tests/sample.py")
        self.assertEqual(len(chunks), 4, "Expected 4 chunks: 3 functions + 1 class")

    def test_no_chunk_has_empty_source_code(self):
        chunks = extract_chunks("tests/sample.py")
        for chunk in chunks:
            self.assertTrue(chunk["source_code"], f"Empty source_code for chunk: {chunk['name']}")

    def test_chunk_has_all_required_keys(self):
        chunks = extract_chunks("tests/sample.py")
        required_keys = {"name", "type", "source_code", "filename", "start_line", "end_line"}
        for chunk in chunks:
            self.assertEqual(set(chunk.keys()), required_keys, f"Missing keys in chunk: {chunk['name']}")

    def test_empty_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            result = extract_chunks(temp_path)
            self.assertEqual(result, [], "Expected empty list for an empty file")
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
