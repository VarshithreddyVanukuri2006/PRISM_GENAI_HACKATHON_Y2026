from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.api.service import CodeLensService
from app.models.api_schemas import (
    EvaluationRequest, EvaluationResponse, EvolutionRequest, EvolutionResponse, GraphResponse,
    HealthResponse, RepositoryIndexRequest, RepositoryIndexResponse,
    RepositoryListResponse, SearchRequest, SearchResponse,
    VersionIndexRequest, VersionIndexResponse, VersionListResponse,
)

router = APIRouter(prefix="/api")


def _service(request: Request) -> CodeLensService:
    return request.app.state.service


def _require_writable_index() -> None:
    if os.environ.get("CODELENS_READ_ONLY", "").strip().casefold() in {"1", "true", "yes"}:
        raise HTTPException(status_code=403, detail="Repository indexing is disabled in the hosted read-only demo.")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok", service="CodeLens Agentic Code Intelligence",
        indexed_repositories=len(_service(request).list_repositories()),
    )


@router.get("/repositories", response_model=RepositoryListResponse)
def repositories(request: Request) -> RepositoryListResponse:
    return RepositoryListResponse(repositories=_service(request).list_repositories())


@router.post("/repositories/index", response_model=RepositoryIndexResponse,
             status_code=status.HTTP_201_CREATED)
def index_repository(body: RepositoryIndexRequest, request: Request) -> RepositoryIndexResponse:
    _require_writable_index()
    return _service(request).index_repository(body)


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    return _service(request).search(body)


@router.get("/graph", response_model=GraphResponse)
def graph(request: Request, repository: str = Query(min_length=1, max_length=512),
          chunk_id: str = Query(min_length=1, max_length=512)) -> GraphResponse:
    return _service(request).graph_for_chunk(repository, chunk_id)


@router.post("/versions/index", response_model=VersionIndexResponse,
             status_code=status.HTTP_201_CREATED)
def index_version(body: VersionIndexRequest, request: Request) -> VersionIndexResponse:
    _require_writable_index()
    return _service(request).index_version(body)


@router.get("/versions", response_model=VersionListResponse)
def versions(request: Request, repository: str | None = Query(default=None, min_length=1, max_length=512)) -> VersionListResponse:
    return _service(request).list_versions(repository)


@router.post("/evolution", response_model=EvolutionResponse)
def evolution(body: EvolutionRequest, request: Request) -> EvolutionResponse:
    return _service(request).evolve(body)


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(body: EvaluationRequest, request: Request) -> EvaluationResponse:
    return _service(request).evaluate(body)
