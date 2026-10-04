"""Minimal HTTP client with a declared User-Agent, retries and a per-host rate limit."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request

__all__ = ["RateLimiter", "fetch_bytes", "user_agent"]

DEFAULT_USER_AGENT = "varclock/0.1 (open-source research tool; set VARCLOCK_USER_AGENT)"


def user_agent() -> str:
    """Declared User-Agent; set ``VARCLOCK_USER_AGENT`` to ``"Name email"`` for BLS and similar."""
    return os.environ.get("VARCLOCK_USER_AGENT", DEFAULT_USER_AGENT)


class RateLimiter:
    """Enforces a minimum interval between consecutive calls."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def fetch_bytes(
    url: str,
    *,
    limiter: RateLimiter | None = None,
    retries: int = 3,
    timeout: float = 60.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    """GET ``url`` and return the body; retries transient failures with a growing pause."""
    req_headers = {"User-Agent": user_agent(), "Accept": "*/*", **(headers or {})}
    last_error: Exception | None = None
    for attempt in range(retries):
        if limiter is not None:
            limiter.wait()
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last_error}") from last_error
