from typing import Optional

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None
_RAG_STEP_LOOP = None


def set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """Get the latest RAG context, preserving the existing agent integration."""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def reserve_knowledge_tool_call() -> bool:
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return False
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1
    return True


def set_rag_step_queue(queue):
    """Set the queue and capture the active event loop for worker-thread events."""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio

        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is None or _RAG_STEP_LOOP is None:
        return

    step = {"icon": icon, "label": label, "detail": detail}
    try:
        if not _RAG_STEP_LOOP.is_closed():
            _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
    except Exception:
        return
