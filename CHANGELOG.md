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
