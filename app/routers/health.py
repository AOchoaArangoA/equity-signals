"""app.routers.health — liveness / readiness endpoint.

No authentication required.  Used by Railway and load balancers to verify
that the service is running.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["Health"])

_VERSION = "0.1.0"


@router.get(
    "/health",
    summary="Health check",
    response_description="Service status and environment",
)
def health() -> dict[str, str]:
    """Return service liveness status.

    Returns:
        JSON with keys ``status``, ``environment``, and ``version``.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.environment,
        "version": _VERSION,
    }
