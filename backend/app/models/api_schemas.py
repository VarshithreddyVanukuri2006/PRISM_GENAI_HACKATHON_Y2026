from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(APIModel):
    status: Literal["ok"]
    service: str
    indexed_repositories: int


class RepositoryInfo(APIModel):
    repository_id: str
    name: str
    repository_path: str | None
    number_of_files: int
    number_of_chunks: int
    available_methods: list[str]
    has_versions: bool


class RepositoryListResponse(APIModel):
    repositories: list[RepositoryInfo]


class RepositoryIndexRequest(APIModel):
    repository_path: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    include_semantic: bool = False
    build_graph: bool = True
    model_name: str | None = Field(default=None, min_length=1, max_length=512)


class RepositoryIndexResponse(APIModel):
    repository_id: str
    name: str
    repository_path: str
    chunks_path: str
    bm25_index_path: str
    semantic_index_path: str | None
    graph_path: str | None
    number_of_files: int
    number_of_chunks: int
    skipped_files: int
    warnings: list[str]


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=4000)
    repository: str | None = Field(default=None, min_length=1, max_length=512)
    version: str | None = Field(default=None, min_length=1, max_length=128)
    top_k: int = Field(default=10, ge=1, le=100)
    retrieval_method: Literal["auto", "bm25", "semantic", "hybrid", "reranked", "multi_pass"] = "auto"
    candidate_k: int = Field(default=50, ge=1, le=500)
    seed_k: int = Field(default=10, ge=1, le=100)
    context_candidate_k: int = Field(default=25, ge=1, le=250)
    max_graph_hops: int = Field(default=3, ge=1, le=8)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace text")
        return value


class SearchResult(APIModel):
    rank: int
    chunk_id: str
    repository_id: str
    version_id: str | None
    version_label: str | None
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    code: str
    score: float
    retrieval_method: str
    metadata: dict[str, object]
    features: dict[str, float] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)
    lexical_score: float | None = None
    semantic_score: float | None = None


class SearchResponse(APIModel):
    query: str
    repository: RepositoryInfo
    version: str | None
    retrieval_method: str
    retrieval_latency_ms: float
    total_candidates: int
    retrieval_details: dict[str, object] = Field(default_factory=dict)
    results: list[SearchResult]


class GraphNodeResponse(APIModel):
    node_id: str
    kind: str
    chunk_id: str | None = None
    repository_id: str | None = None
    symbol_name: str | None = None
    symbol_type: str | None = None
    file_path: str | None = None
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    code: str | None = None
    selected: bool = False


class GraphEdgeResponse(APIModel):
    source: str
    target: str
    relation: str
    direction: Literal["incoming", "outgoing"]


class GraphSummaryResponse(APIModel):
    node_count: int
    edge_count: int
    node_types: dict[str, int] = Field(default_factory=dict)
    relations: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class GraphResponse(APIModel):
    repository_id: str
    selected_chunk_id: str
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    summary: GraphSummaryResponse


class VersionIndexRequest(APIModel):
    repository_path: str = Field(min_length=1, max_length=4096)
    revision: str = Field(min_length=1, max_length=256)
    label: str | None = Field(default=None, min_length=1, max_length=128)
    include_semantic: bool = False
    model_name: str | None = Field(default=None, min_length=1, max_length=512)


class VersionInfo(APIModel):
    repository_id: str
    repository_path: str
    version_id: str
    label: str
    commit_hash: str
    parents: list[str]
    changed_paths: list[str]
    number_of_files: int
    number_of_chunks: int
    semantic_indexed: bool
    model_name: str | None


class VersionIndexResponse(APIModel):
    catalog_path: str
    version: VersionInfo


class VersionListResponse(APIModel):
    repository_id: str | None
    versions: list[VersionInfo]


class EvolutionRequest(APIModel):
    query: str = Field(min_length=1, max_length=4000)
    repository: str = Field(min_length=1, max_length=512)
    per_version_top_k: int = Field(default=10, ge=1, le=100)
    candidate_pool_size: int = Field(default=50, ge=1, le=500)
    min_similarity: float = Field(default=0.45, ge=0, le=1)
    retrieval_mode: Literal["auto", "lexical", "hybrid"] = "auto"
    max_tracks: int = Field(default=50, ge=1, le=500)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain non-whitespace text")
        return value


class EvaluationCase(APIModel):
    query: str = Field(min_length=1, max_length=4000)
    # chunk_id -> integer relevance grade (0 means not relevant).
    relevance: dict[str, int] = Field(min_length=1, max_length=5000)

    @field_validator("relevance")
    @classmethod
    def validate_relevance_grades(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not chunk_id.strip() or grade < 0 or grade > 3 for chunk_id, grade in value.items()):
            raise ValueError("relevance keys must be chunk IDs and grades must be between 0 and 3")
        return value


class EvaluationRequest(APIModel):
    repository: str = Field(min_length=1, max_length=512)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=10, ge=10, le=100)
    retrieval_method: Literal["auto", "bm25", "semantic", "hybrid", "reranked", "multi_pass"] = "auto"


class EvaluationCaseResult(APIModel):
    query: str
    ndcg_at_10: float
    reciprocal_rank: float
    retrieved_chunk_ids: list[str]
    retrieval_latency_ms: float


class EvaluationResponse(APIModel):
    repository: str
    retrieval_method: str
    ndcg_at_10: float
    mrr: float
    mean_latency_ms: float
    cases: list[EvaluationCaseResult]


class EvolutionItemResponse(APIModel):
    version_id: str
    version_label: str
    commit_hash: str
    chunk_id: str
    symbol_name: str
    file_path: str
    start_line: int
    end_line: int
    relevance_score: float
    change: str
    code: str


class EvolutionTransitionResponse(APIModel):
    from_version: str
    to_version: str
    from_chunk_id: str
    to_chunk_id: str
    from_symbol: str
    to_symbol: str
    from_path: str
    to_path: str
    similarity_score: float
    change: str


class EvolutionTrackResponse(APIModel):
    items: list[EvolutionItemResponse]
    transitions: list[EvolutionTransitionResponse]


class EvolutionResponse(APIModel):
    query: str
    repository_id: str
    retrieval_mode: str
    retrieval_latency_ms: float
    versions_searched: list[str]
    tracks: list[EvolutionTrackResponse]
