"""Tenants — the platform boundary, created out of band.

Gated on `admin_token` rather than on a key scope: every API key belongs to a
tenant, so a key that could create tenants would not be tenant-scoped at all.

Creating a tenant mints its first key and returns the plaintext once. Nothing
re-reads it, so a lost first key means a new one issued by an operator.
"""

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError

from prism import tenancy
from prism.api.deps import AdminTokenDep, KeyDep
from prism.tenancy import AlreadyExistsError, TenantRow

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/tenants", tags=["tenants"])

_MAX_NAME_CHARS = 200


class TenantCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=_MAX_NAME_CHARS)]

    @field_validator("name")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name is empty")
        return value.strip()


class Tenant(BaseModel):
    id: UUID
    name: str
    created_at: str

    @classmethod
    def of(cls, row: TenantRow) -> "Tenant":
        return cls(id=row.id, name=row.name, created_at=row.created_at.isoformat())


class TenantCreated(Tenant):
    api_key: str = Field(description="Shown once. Not recoverable.")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_tenant(request: TenantCreate, _: AdminTokenDep) -> TenantCreated:
    """Create a tenant and its first key, which carries every scope."""
    try:
        row, key = await tenancy.create_tenant(request.name)
    except AlreadyExistsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    # The prefix identifies the key in logs; the plaintext appears only here.
    log.info("tenant_created", tenant_id=str(row.id), key_prefix=key.prefix)
    return TenantCreated(**Tenant.of(row).model_dump(), api_key=key.plaintext)


@router.get("")
async def list_tenants(_: AdminTokenDep) -> list[Tenant]:
    """Every tenant. Admin only — a tenant must not enumerate the others."""
    try:
        return [Tenant.of(row) for row in await tenancy.list_tenants()]
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/me")
async def read_own_tenant(key: KeyDep) -> Tenant:
    """The caller's own tenant, from their key. Discloses nothing they lack."""
    try:
        row = await tenancy.read_tenant(key.tenant_id)
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "tenant does not exist")
    return Tenant.of(row)
