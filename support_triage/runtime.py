"""Session-owned credentials; never stored in graph state or process environment."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values


@dataclass(frozen=True)
class Credentials:
    openai: str = field(default="", repr=False)
    anthropic: str = field(default="", repr=False)

    @classmethod
    def from_environment(cls):
        values = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
        return cls(
            os.getenv("OPENAI_API_KEY", values.get("OPENAI_API_KEY", "")) or "",
            os.getenv("ANTHROPIC_API_KEY", values.get("ANTHROPIC_API_KEY", "")) or "",
        )


def safe_failure(exc):
    reference = uuid4().hex[:12]
    # Retain exception type and stack locations, never exception text/customer data.
    import traceback

    frames = traceback.extract_tb(exc.__traceback__)
    logging.getLogger(__name__).error(
        "Failure %s type=%s frames=%s",
        reference,
        type(exc).__name__,
        [(f.filename, f.lineno) for f in frames],
    )
    return f"Operation failed. Check provider configuration or contact the operator. Reference: {reference}"


def setting(name, default=None):
    values = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    return os.getenv(name, values.get(name, default)) or default
