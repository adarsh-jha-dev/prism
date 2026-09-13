"""Collections, always under the caller's own tenant.

`tenant_id` comes from the bearer key and is never accepted from the request, so
no shape of call creates or reads a collection under another tenant.

The embedding model comes from configuration rather than the request: one model
project-wide is a settled decision, and a collection built for another returns
neighbours that mean nothing.
"""

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError

from prism import tenancy
from prism.api.deps import AdminDep, ProviderDep, ReadDep
from prism.tenancy import AlreadyExistsError, CollectionRow

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/collections", tags=["collections"])

_MAX_NAME_CHARS = 200


class CollectionCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=_MAX_NAME_CHARS)]

    @field_validator("name")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name is empty")
        return value.strip()


class Collection(BaseModel):
    id: UUID
    tenant_id: UUID
    name: str
    embedding_model: str
    embedding_dim: int
    abstention_threshold: float
    created_at: str

    @classmethod
    def of(cls, row: CollectionRow) -> "Collection":
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            name=row.name,
            embedding_model=row.embedding_model,
            embedding_dim=row.embedding_dim,
            abstention_threshold=row.abstention_threshold,
            created_at=row.created_at.isoformat(),
        )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_collection(
    request: CollectionCreate, key: AdminDep, provider: ProviderDep
) -> Collection:
    """Create a collection under the caller's tenant."""
    try:
        row = await tenancy.create_collection(
            tenant_id=key.tenant_id,
            name=request.name,
            embedding_model=provider.model,
            embedding_dim=provider.dim,
        )
    except AlreadyExistsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    log.info(
        "collection_created",
        tenant_id=str(key.tenant_id),
        collection_id=str(row.id),
        name=row.name,
    )
    return Collection.of(row)


@router.get("")
async def list_collections(key: ReadDep) -> list[Collection]:
    """The caller's own collections. Another tenant's are not in the result."""
    try:
        return [
            Collection.of(row) for row in await tenancy.list_collections(tenant_id=key.tenant_id)
        ]
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/{collection_id}")
async def read_collection(collection_id: UUID, key: ReadDep) -> Collection:
    """404 for another tenant's collection, the same as for one that is absent."""
    try:
        row = await tenancy.read_collection(collection_id, tenant_id=key.tenant_id)
    except SQLAlchemyError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"collection {collection_id} does not exist")
    return Collection.of(row)
