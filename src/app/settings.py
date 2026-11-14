"""Shared product name for the API title and OpenAPI docs."""

import os

APP_NAME = "FlimFlex"


def recommendations_debug_enabled() -> bool:
    """Debug recommendation fields stay off unless this process is explicitly in dev mode."""
    return os.environ.get("RECS_DEBUG", "").strip().lower() in {"1", "true", "yes"}
