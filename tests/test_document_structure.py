import sys
import unittest

sys.path.insert(0, "backend")

from document_structure import DocumentStructureParser, SemanticChunker
from document_loader import DocumentLoader


class DocumentStructureTests(unittest.TestCase):
    def test_parser_recognizes_headings_lists_code_tables_and_images(self):
        text = """
# Title

Intro paragraph.

- item one
- item two

| a | b |
|---|---|
| 1 | 2 |

```python
print("hello")
```

![caption](image.png)
"""
        elements = DocumentStructureParser().parse(text)
        types = [element.element_type for element in elements]

        self.assertIn("heading", types)
        self.assertIn("paragraph", types)
        self.assertIn("list", types)
        self.assertIn("table", types)
        self.assertIn("code", types)
        self.assertIn("image", types)

    def test_semantic_chunker_keeps_code_and_attaches_image_metadata(self):
        elements = DocumentStructureParser().parse(
            """
# Title

Intro paragraph.

```python
print("hello")
```

![caption](image.png)
"""
        )
        chunks = SemanticChunker(coarse_min_tokens=1, coarse_max_tokens=50, semantic_min_tokens=1, semantic_max_tokens=50).chunk(elements)

        self.assertTrue(chunks)
        self.assertTrue(any("print(\"hello\")" in chunk["text"] for chunk in chunks))
        self.assertTrue(any(chunk["metadata"]["semantic_chunking"] for chunk in chunks))

    def test_fallback_chunks_are_available(self):
        chunker = SemanticChunker()
        chunks = chunker.fallback_chunks("plain text only", {"filename": "x"})

        self.assertTrue(chunks)
        self.assertTrue(chunks[0]["metadata"]["parser_fallback"])

    def test_document_loader_preserves_code_blocks_across_hierarchy(self):
        loader = DocumentLoader()
        chunks = loader._split_page_to_three_levels(
            "```python\nprint(\"hello\")\n```",
            {"filename": "code.md", "page_number": 1},
            0,
        )

        self.assertTrue(chunks)
        self.assertTrue(all('print("hello")' in chunk["text"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
