# CodeLens — Agentic Code Intelligence

**Understand the question. Understand the code. Retrieve what matters.**

CodeLens is being built as a local, CPU-friendly code retrieval platform. This checkout implements **Phases A through I**: local repository ingestion, Python AST-based symbol chunking, metadata extraction, BM25 lexical retrieval, CPU-based semantic retrieval with Sentence Transformers and FAISS, hybrid candidate fusion, configurable code-aware reranking, a repository relationship graph, bounded multi-pass graph retrieval, Git-version indexing/search/matching, and query-driven evolutionary retrieval across versions.

## Phase A workflow

1. Walk a local repository and skip generated/dependency folders, unsupported file types, and source files over 1 MB.
2. Read supported source files as UTF-8.
3. Parse Python with the standard-library `ast` module into class, function, method, and nested-function chunks.
4. Attach stable chunk/repository IDs, file/language/symbol/line metadata, docstrings, imports, and called names.
5. Write one JSON object per chunk to a JSONL file.

The ingestion filter recognizes Python, JavaScript, TypeScript, Java, and C/C++ extensions. Symbol-level parsing is implemented for Python in this phase; the parser adapter is the extension point for additional AST/tree-sitter grammars. Malformed Python files produce no chunks and do not stop indexing. Empty repositories are valid and produce an empty JSONL file.

## Run indexing

Requires Python 3.11+; Phase A uses only the Python standard library.

```powershell
python scripts/index_repository.py .\data\demo
```

By default, chunk metadata is written to `data/indexes/<repository-id>.jsonl`. Choose another destination with `--output`:

```powershell
python scripts/index_repository.py C:\path\to\repository --output .\chunks.jsonl
```

## Demo repository

`data/demo` contains small authentication, payment, image-storage, and password modules with functions and calls that exercise symbol and relationship metadata.

## Phase B: BM25 lexical search

Build the Phase A chunks, then search them with BM25:

```powershell
python scripts/index_repository.py .\data\demo --output .\data\indexes\demo.jsonl
python scripts/search_bm25.py "Where is the password hashed?" --chunks .\data\indexes\demo.jsonl --top-k 5 --save-index .\data\indexes\demo.bm25.json
```

Later searches can load the saved BM25 index directly:

```powershell
python scripts/search_bm25.py "decode JWT token" --index .\data\indexes\demo.bm25.json
```

The BM25 corpus includes code, file paths, symbol names and types, docstrings, imports, and extracted call names. Tokenization preserves full identifiers and adds underscore/camel-case subterms. Results include their raw BM25 score; no semantic or hybrid score is implied.

## Phase C: semantic search

Install the optional CPU semantic-retrieval dependencies (ingestion and BM25 remain standard-library only):

```powershell
python -m pip install -r backend/requirements-semantic.txt
```

Build semantic vectors from the Phase A chunk file, then query the persisted FAISS index:

```powershell
python scripts/build_semantic_index.py .\data\indexes\demo.jsonl --output .\data\indexes\demo.semantic.faiss
python scripts/search_semantic.py "Where is the password checked before storing credentials?" --index .\data\indexes\demo.semantic.faiss --top-k 5
```

The default model is `sentence-transformers/all-MiniLM-L6-v2`; choose another Sentence Transformers model or local model path with `--model`. Building writes FAISS vectors to the requested index file and chunk/model metadata to a companion `.json` file. Vectors are encoded in batches and normalized; FAISS inner-product search therefore returns cosine similarity scores. Reuse the saved index for subsequent queries so document embeddings are not recomputed. Model files are loaded lazily and cached by Sentence Transformers; first-time model use may need network access unless the model is already cached locally.

## Phase D: hybrid retrieval

After building the BM25 and semantic indexes, retrieve candidates from both, normalize each score list with per-query min-max scaling, deduplicate by chunk ID, and fuse the normalized scores:

```powershell
python scripts/search_hybrid.py "Where is the password hashed before storing credentials?" --bm25-index .\data\indexes\demo.bm25.json --semantic-index .\data\indexes\demo.semantic.faiss --candidate-k 50 --top-k 10 --bm25-weight 0.5 --semantic-weight 0.5
```

Weights are configurable and are not claimed to be optimal. The fused score is the weighted mean of the normalized component scores. Results retain raw BM25/cosine scores, normalized scores, and each retriever's rank. Results retrieved by only one channel receive zero for the missing channel. Equal scores within a channel normalize to 1 so a single-result list still contributes its signal.

## Phase E: code-aware reranking

Run the Phase D candidate pool through a configurable feature scorer:

```powershell
python scripts/search_reranked.py "Where is the JWT validated before protected routes?" --bm25-index .\data\indexes\demo.bm25.json --semantic-index .\data\indexes\demo.semantic.faiss --candidate-k 50 --top-k 10
```

The default score uses normalized semantic and BM25 scores plus exact symbol, filename, explicit language, lightweight function-role, metadata, and import/call features. Feature weights are configurable with `--weight-*`; set a weight to `0` to disable it for an ablation. The output includes the feature values and matching signals used for each result. Graph proximity and version relevance are not included in this phase.

## Phase F: code relationship graph

Install NetworkX for graph building and inspection:

```powershell
python -m pip install -r backend/requirements-graph.txt
```

Build a graph from the Phase A chunk metadata and inspect a chunk's direct relationships by ID:

```powershell
python scripts/build_code_graph.py .\data\indexes\demo.jsonl --output .\data\indexes\demo.graph.json
python scripts/inspect_code_graph.py --graph .\data\indexes\demo.graph.json --chunk-id <chunk-id>
```

The graph contains file and parsed-symbol nodes, with `CONTAINS`, resolvable `CALLS`, repository-local `IMPORTS`, resolvable class `INHERITS`, and AST-resolved `REFERENCES` relationships. Ambiguous call targets are left unlinked. The saved graph JSON includes chunk metadata and can be loaded without reparsing the repository. Graph expansion into retrieval results belongs to a later phase.

## Phase G: multi-pass retrieval

Multi-pass retrieval uses the saved lexical, semantic, and graph indexes to run the five retrieval stages:

1. Retrieve a broad hybrid candidate set.
2. Rerank that set to select graph-expansion seeds.
3. Traverse resolved graph edges for a bounded number of hops.
4. Search again using the original query plus related symbol, metadata, and code context.
5. Fuse all pass results, retain graph-only discoveries, and rerank the full set.

```powershell
python scripts/search_multipass.py "Where is the database connection created before executing queries?" --bm25-index .\data\indexes\demo.bm25.json --semantic-index .\data\indexes\demo.semantic.faiss --graph .\data\indexes\demo.graph.json --candidate-k 50 --seed-k 10 --context-candidate-k 25 --max-graph-hops 3 --top-k 10
```

Graph-only candidates receive a graph feature of `1 / hops`; this is a simple proximity heuristic, not a probability or measured relevance confidence. Pass counts, expansion paths, and additional candidates are printed so each stage can be inspected. Candidate limits, hop depth, and graph weight are configurable.

## Phase H: Git-version retrieval

Index Git tags, branches, or commit IDs independently. This uses `git archive` snapshots and does not switch the repository's working tree:

```powershell
python scripts/index_git_version.py C:\path\to\repository v1 --label v1 --output-dir .\data\versions
python scripts/index_git_version.py C:\path\to\repository v2 --label v2 --output-dir .\data\versions
```

Each commit gets its own JSONL chunks and BM25 index. Add `--semantic` to build a per-version FAISS index as well; use the same `--model` for every version you intend to hybrid-search:

```powershell
python scripts/index_git_version.py C:\path\to\repository v1 --label v1 --output-dir .\data\versions --semantic
```

The script prints the catalog location. Search one version with `--version`; omit it to search across the catalog:

```powershell
python scripts/search_versions.py "Where is JWT validated?" --catalog .\data\versions\<repository-id>\catalog.json --version v1 --mode auto
python scripts/search_versions.py "Where is JWT validated?" --catalog .\data\versions\<repository-id>\catalog.json --mode auto
```

`auto` uses hybrid search when every selected version has a semantic index built with the same model; otherwise it uses BM25 for all selected versions so their retrieval mode is consistent. `--mode hybrid` requires semantic indexes for every selected version.

Compare likely related code units across two versions, including renamed symbols:

```powershell
python scripts/compare_versions.py --catalog .\data\versions\<repository-id>\catalog.json --from-version v1 --to-version v2
```

Pair scores combine semantic, file path, symbol, AST-structure, and Git parent/rename signals with configurable weights in `VersionSimilarityWeights`. The symbol name is one signal only. A similarity score is a ranking signal, not proof that two functions are equivalent.

## Phase I: evolutionary retrieval

Search for a function across every indexed version and follow likely continuations along known Git parent links:

```powershell
python scripts/search_evolution.py "Where is authentication handled?" --catalog .\data\versions\<repository-id>\catalog.json --per-version-top-k 10
```

Each track shows the relevant code unit per version and likely unchanged, modified, or renamed/moved transitions. The baseline uses lexical query retrieval and path, symbol, AST-shape, and Git-parent signals, so it does not download a model. To include embedding similarity, set `--semantic-weight` above zero and provide the optional semantic dependencies/model. For hybrid query retrieval, use `--mode hybrid`; every indexed version must have a semantic index made with the same model. Match scores are ranking evidence, not proof of semantic equivalence. The command reports retrieved units and likely links; it does not infer deletion or addition solely from an unmatched result.

## Verify Phases A through I

```powershell
python -m unittest discover -s backend/tests -v
python scripts/index_repository.py .\data\demo --output .\data\indexes\demo.jsonl
python scripts/search_bm25.py "Where is the password hashed?" --chunks .\data\indexes\demo.jsonl
```

Semantic integration requires the optional dependencies and configured embedding model. Graph building/search requires NetworkX. Semantic tests use an injected deterministic embedder and FAISS-compatible test double; graph and multi-pass tests use controlled graph/retriever test doubles. Version indexing/search tests create a local temporary Git repository and exercise commit snapshots without changing a working tree; version matching uses a deterministic embedder in tests. These tests cover the implemented phases without downloading a model. They are functional tests, not benchmark results.

## Frontend preview

The React + TypeScript interface is in `frontend/`. It uses the checked-in demo JSONL index and labels its client-side term-coverage ranking and measured browser latency as preview values. The current screen remains in local preview mode and does not call the FastAPI endpoints yet.

```powershell
cd frontend
npm install
npm run dev
```

## Step 12: FastAPI backend

The API wraps the existing BM25, semantic, hybrid, reranking, graph multi-pass, Git-version, and evolutionary retrieval implementations. It loads saved indexes on demand and caches loaded indexes by file timestamp. The default index directories are `data/indexes/` and `data/versions/`; override them with `CODELENS_INDEX_DIR` and `CODELENS_VERSION_DIR`. CORS defaults to the local Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`; set `CODELENS_CORS_ORIGINS` to a comma-separated allowlist to change them.

Install the API/test dependencies and start on loopback from the repository root:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`. Semantic indexing/search remains optional; install `backend/requirements-semantic.txt` and build semantic indexes to enable it. An explicit semantic, hybrid, reranked, or multi-pass request returns a clear error when required indexes are missing. `auto` uses multi-pass when both semantic and graph indexes exist, reranking when only a semantic index exists, and BM25 otherwise.

### Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service and indexed repository count |
| `GET` | `/api/repositories` | Available repositories, sizes, and retrieval methods |
| `POST` | `/api/repositories/index` | Index a local repository into JSONL/BM25 and optionally graph/semantic artifacts |
| `POST` | `/api/search` | Search a repository or indexed Git version |
| `GET` | `/api/graph?repository=<id>&chunk_id=<id>` | Get actual incoming/outgoing graph relationships for an indexed code unit |
| `POST` | `/api/versions/index` | Index a Git ref, tag, or commit |
| `GET` | `/api/versions?repository=<id-or-name>` | List indexed versions |
| `POST` | `/api/evolution` | Find query-relevant code tracks across indexed Git versions |
| `POST` | `/api/evaluate` | Evaluate retrieval against caller-supplied relevance judgments |

Index a local repository:

```bash
curl -X POST http://127.0.0.1:8000/api/repositories/index \
  -H "Content-Type: application/json" \
  -d '{"repository_path":"/path/to/repository","name":"my-service","build_graph":true}'
```

Example response shape (file paths, IDs, counts, and warnings are returned from the actual indexing run):

```json
{
  "repository_id": "<repository-id>",
  "name": "my-service",
  "repository_path": "/path/to/repository",
  "chunks_path": "<index-dir>/<repository-id>.jsonl",
  "bm25_index_path": "<index-dir>/<repository-id>.bm25.json",
  "semantic_index_path": null,
  "graph_path": "<index-dir>/<repository-id>.graph.json",
  "number_of_files": "<measured-count>",
  "number_of_chunks": "<measured-count>",
  "skipped_files": "<measured-count>",
  "warnings": []
}
```

Search (repository may be omitted when exactly one is indexed; `version` may be a label/commit or `all`):

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Where is the JWT token validated before protected routes?","repository":"my-service","top_k":5,"retrieval_method":"auto"}'
```

Each result includes its actual chunk ID, snippet, file and symbol metadata, line bounds, score, method, signal/features, and optional raw lexical/semantic scores. The response also reports the end-to-end measured `retrieval_latency_ms` and candidate count. BM25 returns the index's raw BM25 score; hybrid returns its configured fused score; reranked and multi-pass return their configured reranker score. Scores from different methods should not be compared as if calibrated to one scale.

```json
{
  "query": "<query>",
  "repository": {"repository_id":"<id>","name":"my-service","repository_path":"<path>","number_of_files":"<measured-count>","number_of_chunks":"<measured-count>","available_methods":["bm25"],"has_versions":false},
  "version": null,
  "retrieval_method": "bm25",
  "retrieval_latency_ms": "<measured-latency-ms>",
  "total_candidates": "<actual-count>",
  "retrieval_details": {},
  "results": [{"rank":"<actual-rank>","chunk_id":"<actual-chunk-id>","repository_id":"<id>","version_id":null,"version_label":null,"file_path":"<repository-relative-path>","language":"Python","symbol_name":"<actual-symbol>","symbol_type":"function","start_line":"<actual-line>","end_line":"<actual-line>","code":"<actual indexed code>","score":"<actual method-specific score>","retrieval_method":"bm25","metadata":{"docstring":"<actual docstring>","imports":[],"calls":[],"class_name":null,"parent_symbol":null,"commit_hash":null},"features":{},"signals":["BM25 lexical score from indexed code and metadata"],"lexical_score":"<actual BM25 score>","semantic_score":null}]
}
```

Index and query Git versions:

```bash
curl -X POST http://127.0.0.1:8000/api/versions/index \
  -H "Content-Type: application/json" \
  -d '{"repository_path":"/path/to/repository","revision":"v1.2.0","label":"v1.2.0"}'

curl "http://127.0.0.1:8000/api/versions?repository=my-service"

curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Where is authentication handled?","repository":"my-service","version":"all","top_k":10}'
```

Evolution accepts a repository ID/name and query; it returns likely code-unit tracks across directly linked indexed Git parents. Matching scores rank candidate continuations and do not prove equivalence. Evaluation requires explicit `chunk_id` relevance grades (0–3) for every query; the API computes measured NDCG@10, MRR, and per-query latency from those judgments and the real search results. It does not download or invent benchmark judgments.

```json
{
  "repository":"my-service",
  "retrieval_method":"bm25",
  "top_k":10,
  "cases":[{"query":"Where is authentication handled?","relevance":{"<judged-relevant-chunk-id>":2}}]
}
```

Run the complete existing and API test suite with `python -m pytest backend/tests -q`. The API tests create temporary repositories, index their real source files and Git commits, exercise every route, and check five natural-language retrieval queries against the source they indexed. For a live loopback-server smoke run, use `python scripts/check_api_server.py`; it starts Uvicorn with a temporary index, checks all routes, validates five retrieved snippets against their source files, and stops the server afterward.

## React demo frontend

The Vite frontend in `frontend/` connects to the FastAPI service and displays its indexed repositories, actual ranked snippets and scores, returned explanation signals, measured latency, indexed Git versions, evolutionary tracks, and code-graph edges. Graph data is available from `GET /api/graph?repository=<id>&chunk_id=<chunk-id>`; it uses the persisted graph index or builds and caches AST-resolved relationships from the repository's persisted lexical snippets when an older index has no graph file. The UI does not create local search results or scores.

Start the backend from the repository root:

```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

In another terminal, configure and start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The frontend defaults to `http://127.0.0.1:8000`; override it with `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The development server's Vite origin is allowed by the backend's default CORS configuration. Set `CODELENS_CORS_ORIGINS` to a comma-separated origin list when hosting the frontend elsewhere. To enable graph and multi-pass retrieval, index with `build_graph: true`; for a graph-only/CPU-light demo, semantic indexing can remain disabled. Select an indexed Git ref from the version selector to search it, or choose “All indexed versions.” Use “Trace query” to call evolutionary retrieval when the selected repository has indexed versions. The repository path used by “Index repository” must be accessible to the backend process.

Build the frontend for production with `npm run build` from `frontend/`.

### Deploy the full demo on Vercel

This repository is a monorepo, so create two Vercel projects from the same Git repository:

1. **API project:** use the repository root as the Root Directory. Vercel detects the FastAPI entry point in `main.py` and installs the root `requirements.txt`. The entry point points to the existing backend and its checked-in demo index. Vercel sets this deployment to read-only; repository/version indexing endpoints return `403` there because serverless files are not a durable index store.
2. **Frontend project:** set the Root Directory to `frontend` so Vercel builds the Vite application from that package.
3. After both projects have a production domain, set `VITE_API_BASE_URL` on the frontend project to the API project's `https://…vercel.app` URL. Set `CODELENS_CORS_ORIGINS` on the API project to the frontend project's exact `https://…vercel.app` origin, then redeploy both projects.

The deployed API serves search, repository listing, graph relationships, and version/evolution data included in the deployment. Indexing is intended to run locally or on a backend with persistent storage; this public demo does not accept arbitrary filesystem indexing requests.
