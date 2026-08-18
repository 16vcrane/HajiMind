# Changelog

## Unreleased

- Added multi-step Planner with structured `Plan` and `Task` models.
- Added dependency-aware `TaskExecutor` with sequential and parallel execution.
- Added task status, result, failure, retry, timeout, cancellation, and execution trace tracking.
- Added Planner streaming events for task lifecycle visibility.
- Added unified Agent Trace with Router, Planner, Tool, Task, RAG, latency, error, and final status observability.
- Added Agent Workflow Panel for frontend trace visualization.
- Added Agent reliability controls: bounded tool/task retries, exponential backoff,
  circuit breaking, maximum agent steps, loop detection, and recovery traces.
- Added Planner recovery for repeated tasks and failed web search fallback to RAG.
- Added structure-aware document parsing and semantic chunking with code, table, and
  multimodal-ready image metadata support.
- Added configurable BM25 parameters, Hybrid dense/sparse weights, sparse retrieval,
  and offline RAG retrieval/rerank evaluation with JSON/CSV reports.
- Added explainable multi-document conflict detection to RAG traces and knowledge-base
  tool output without automatically resolving conflicting source claims.
- Added structured working, conversation, long-term, and preference memory with
  privacy-aware extraction, user-isolated PostgreSQL CRUD, and Agent Trace integration.
