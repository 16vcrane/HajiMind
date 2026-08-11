# Eval Datasets

Use `json`, `jsonl`, or `csv`.

Required fields:
- `question`
- `gold_chunk_ids`
- `expected_answer`

Optional:
- `dataset_type`: `gold` or `synthetic`
- any extra metadata you want to preserve

Examples:

```json
{
  "question": "What is RAG?",
  "gold_chunk_ids": ["doc1::p1::l3::0"],
  "expected_answer": "Retrieval augmented generation combines retrieval and generation.",
  "dataset_type": "gold"
}
```
