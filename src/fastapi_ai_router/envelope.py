"""Envelope: how AIRouter shapes responses to the client.

Default mode wraps the dispatched response in a metadata-rich envelope so
clients can see what the LLM decided (and why). ?raw=true bypasses the
envelope entirely and returns the dispatched response as-is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

_TRUTHY = {"true", "1", "yes", "on"}


def is_raw_requested(query_params: Mapping[str, str], *, raw_param: str) -> bool:
    """Check if the client asked for the raw (un-enveloped) response."""
    val = query_params.get(raw_param)
    return val is not None and val.lower() in _TRUTHY


def wrap_envelope(
    *,
    endpoint: str,
    args: dict[str, Any],
    reasoning: str | None,
    response: httpx.Response,
) -> dict[str, Any]:
    """Build the envelope body for a successful (or even 4xx/5xx) dispatch."""
    return {
        "endpoint": endpoint,
        "args": args,
        "result": _decode_result(response),
        "reasoning": reasoning,
        "result_status": response.status_code,
    }


def _decode_result(response: httpx.Response) -> Any:
    ctype = response.headers.get("content-type", "").lower()
    if "json" in ctype:
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text


__all__ = ["is_raw_requested", "wrap_envelope"]
