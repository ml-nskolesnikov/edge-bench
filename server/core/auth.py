"""
Shared-secret authentication dependency for API endpoints.

Usage:
    router = APIRouter(dependencies=[Depends(require_api_secret)])

Set EDGEBENCH_AGENT_SECRET to enable. Leave empty to disable (dev / trusted LAN).
"""

import hmac

from fastapi import Header, HTTPException

from server.core.config import settings


async def require_api_secret(x_agent_secret: str = Header(default='')) -> None:
    """Validate shared secret when EDGEBENCH_AGENT_SECRET is configured."""
    if not settings.AGENT_SECRET:
        return  # auth disabled
    if not hmac.compare_digest(
        x_agent_secret.encode('utf-8'), settings.AGENT_SECRET.encode('utf-8')
    ):
        raise HTTPException(403, 'Invalid or missing X-Agent-Secret header')
