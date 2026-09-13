"""API keys, always under the caller's own tenant.

Issuing and revoking need the admin scope, listing needs read. No response
carries `key_hash`, and a plaintext key appears only in the response that issued
it.
"""

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import AwareDatetime, BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError

from prism import tenancy
from prism.api.deps import AdminDep, ReadDep
from prism.auth import Scope
from prism.tenancy import KeyRow, LastAdminKeyError

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/keys", tags=["keys"])

_MAX_NAME_CHARS = 200


class KeyCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=_MAX_NAME_CHARS)]
    scopes: Annotated[set[Scope], Field(min_length=1)]
    expires_at: AwareDatetime | None = None

    @field_validator("name")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name is empty")
        return value.strip()

    @field_validator("expires_at")
    @classmethod
    def _in_the_future(cls, value: datetime | None) -> datetime | None:
        if value is not None and value <= datetime.now(UTC):
            raise ValueError("expires_at is not in the future")
        return value


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


class Key(BaseModel):
    id: UUID
    name: str
    prefix: str
    scopes: list[Scope]
    created_at: str
    last_used_at: str | None
    revoked_at: str | None
    expires_at: str | None

    @classmethod
    def of(cls, row: KeyRow) -> "Key":
        return cls(
            id=row.id,
            name=row.name,
            prefix=row.key_prefix,
            scopes=list(row.scopes),
            created_at=row.created_at.isoformat(),
            last_used_at=_iso(row.last_used_at),
            revoked_at=_iso(row.revoked_at),
            expires_at=_iso(row.expires_at),
        )


class KeyCreated(Key):
    api_key: str = Field(description="Shown once. Not recoverable.")


@router.post("", status_code=status.HTTP_201_CREATED)
async def issue_key(request: KeyCreate, key: AdminDep) -> KeyCreated:
    """Issue a key under the caller's tenant with exactly the scopes requested."""
    try:
        row, issued = await tenancy.create_key(
            tenant_id=key.tenant_id,
            name=request.name,
            scopes=request.scopes,
            expires_at=request.expires_at,
        )
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    log.info(
        "key_issued",
        tenant_id=str(key.tenant_id),
        key_id=str(row.id),
        key_prefix=row.key_prefix,
        issued_by=str(key.id),
    )
    return KeyCreated(**Key.of(row).model_dump(), api_key=issued.plaintext)


@router.get("")
async def list_keys(key: ReadDep) -> list[Key]:
    """The caller's own keys, revoked and expired ones included."""
    try:
        return [Key.of(row) for row in await tenancy.list_keys(tenant_id=key.tenant_id)]
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(key_id: UUID, key: AdminDep) -> None:
    """Revoke a key. Idempotent; 404 for another tenant's key."""
    try:
        row = await tenancy.revoke_key(key_id, tenant_id=key.tenant_id)
    except LastAdminKeyError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "cannot revoke the tenant's last non-expiring admin key; issue another first",
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"key {key_id} does not exist")

    log.info(
        "key_revoked", tenant_id=str(key.tenant_id), key_id=str(row.id), revoked_by=str(key.id)
    )
