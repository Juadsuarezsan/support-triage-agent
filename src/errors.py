"""Exception hierarchy shared by every layer of the service."""

from __future__ import annotations


class TriageError(Exception):
    """Base class for every error raised by the triage domain."""


class LLMError(TriageError):
    """The LLM provider could not be reached or returned an API error."""


class LLMOutputError(LLMError):
    """The LLM answered, but the payload did not match the expected schema."""


class RetrievalError(TriageError):
    """The similar-ticket store failed to index or search."""


class PersistenceError(TriageError):
    """The audit store failed to read or write."""
