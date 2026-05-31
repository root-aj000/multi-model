"""
FastAPI dependency functions for extracting authenticated identity.

Usage:
    from lib.auth.deps import require_auth, require_admin, require_platform_admin

    @router.post("/predict")
    async def predict(ctx: RequestContext = Depends(require_auth)):
        # ctx.user_id, ctx.tenant_id, ctx.role are guaranteed
        ...

    @router.delete("/settings/team/{user_id}")
    async def remove_member(
        user_id: str,
        ctx: RequestContext = Depends(require_admin),
    ):
        # Only admin/owner can reach this handler
        ...
"""

from fastapi import Depends, HTTPException

from .middleware import RequestContext, resolve_identity


async def require_auth(ctx: RequestContext = Depends(resolve_identity)) -> RequestContext:
    """
    Require any authenticated user (JWT or API key).

    This is the base dependency — all other auth dependencies
    build on top of it.

    Args:
        ctx: Resolved identity context from resolve_identity().

    Returns:
        The authenticated RequestContext.

    Raises:
        HTTPException: 401 if no valid authentication is provided.
    """
    return ctx


async def require_admin(ctx: RequestContext = Depends(require_auth)) -> RequestContext:
    """
    Require admin or owner role within the tenant.

    Members are rejected with 403. Platform admins also pass
    this check since they have elevated privileges.

    Args:
        ctx: Authenticated RequestContext from require_auth.

    Returns:
        The authenticated RequestContext (guaranteed admin+).

    Raises:
        HTTPException: 403 if the user is a member (not admin/owner).
    """
    if not ctx.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Admin access required. Your role is '%s'." % ctx.role,
        )
    return ctx


async def require_platform_admin(
    ctx: RequestContext = Depends(require_auth),
) -> RequestContext:
    """
    Require platform_admin role (cross-tenant access).

    This is the highest privilege level — only platform admins
    can access admin endpoints that span all tenants.

    Args:
        ctx: Authenticated RequestContext from require_auth.

    Returns:
        The authenticated RequestContext (guaranteed platform_admin).

    Raises:
        HTTPException: 403 if the user is not a platform admin.
    """
    if not ctx.is_platform_admin:
        raise HTTPException(
            status_code=403,
            detail="Platform admin access required. Your role is '%s'." % ctx.role,
        )
    return ctx
