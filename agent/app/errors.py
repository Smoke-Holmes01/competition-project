class AgentError(Exception):
    """Expected failure safe to show to a user (never raw provider response bodies)."""

    def __init__(self, message: str, *, retryable: bool = False, status: int = 400):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
