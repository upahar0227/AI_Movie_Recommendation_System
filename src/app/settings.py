"""Shared product name for the API title and OpenAPI docs."""

import os

APP_NAME = "FlimFlex"


def recommendations_debug_enabled() -> bool:
    """Return whether recommendation responses may include score breakdowns.

    Debug fields stay off unless this process is explicitly in dev mode.

    Returns:
        True when ``RECS_DEBUG`` is ``1``, ``true``, or ``yes``, ignoring case and surrounding spaces.

    Side effects:
        Reads the ``RECS_DEBUG`` environment variable.
    """
    return os.environ.get("RECS_DEBUG", "").strip().lower() in {"1", "true", "yes"}
