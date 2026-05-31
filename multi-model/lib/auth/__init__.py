"""
Authentication package for multi-tenant API.

Provides dual-mode auth (JWT + API Key), Supabase client management,
and FastAPI dependency injection for route protection.
"""

from .middleware import RequestContext, resolve_identity, hash_api_key
from .deps import require_auth, require_admin, require_platform_admin
from .config import (
    get_supabase_client,
    get_supabase_admin_client,
    get_supabase_jwt_secret,
    create_temp_client,
)

__all__ = [
    "RequestContext",
    "resolve_identity",
    "hash_api_key",
    "require_auth",
    "require_admin",
    "require_platform_admin",
    "get_supabase_client",
    "get_supabase_admin_client",
    "get_supabase_jwt_secret",
    "create_temp_client",
]
