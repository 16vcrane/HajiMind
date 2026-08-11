class ToolExecutionError(Exception):
    """A handled tool error that can be returned to the agent safely."""

    def __init__(self, message: str, *, retryable: bool = False, metadata: dict | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.metadata = metadata or {}


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool exceeds its configured execution timeout."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)
