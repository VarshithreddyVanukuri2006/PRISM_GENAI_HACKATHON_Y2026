from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.service import CodeLensService

ROOT_DIR = Path(__file__).resolve().parents[2]


def _default_origins() -> list[str]:
    configured = os.environ.get("CODELENS_CORS_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    frontend_url = os.environ.get("CODELENS_FRONTEND_URL")
    if frontend_url:
        origins.append(frontend_url.rstrip("/"))
    return origins


def create_app(service: CodeLensService | None = None,
               cors_origins: list[str] | None = None) -> FastAPI:
    index_dir = Path(os.environ.get("CODELENS_INDEX_DIR", ROOT_DIR / "data" / "indexes"))
    version_dir = Path(os.environ.get("CODELENS_VERSION_DIR", ROOT_DIR / "data" / "versions"))
    app = FastAPI(
        title="CodeLens Agentic Code Intelligence API",
        description="Repository indexing, code retrieval, version search, evolutionary retrieval, and evaluation.",
        version="0.1.0",
    )
    app.state.service = service or CodeLensService(index_dir, version_dir)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins is not None else _default_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(router)

    @app.exception_handler(FileNotFoundError)
    async def missing_resource_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid_request_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def unavailable_feature_handler(_: Request, exc: RuntimeError) -> JSONResponse:
        logging.getLogger("codelens.api").warning("Optional retrieval feature unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


app = create_app()
