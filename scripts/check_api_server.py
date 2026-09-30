from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def run_git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args], check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def checked(client: httpx.Client, method: str, path: str, **kwargs):
    response = client.request(method, path, **kwargs)
    if not response.is_success:
        raise RuntimeError(f"{method} {path} returned {response.status_code}: {response.text}")
    return response.json() if response.content else None


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="codelens-api-live-") as temp_name:
        temporary = Path(temp_name)
        repository = temporary / "repo"
        repository.mkdir()
        run_git(repository, "init", "-b", "main")
        run_git(repository, "config", "user.name", "CodeLens API Smoke")
        run_git(repository, "config", "user.email", "codelens-smoke@example.invalid")
        (repository / "auth.py").write_text(
            'def authorize_request(bearer_token):\n'
            '    """Validate bearer token before serving a protected endpoint."""\n'
            '    return decode_session_token(bearer_token)\n\n'
            'def decode_session_token(token):\n'
            '    """Decode a signed session token."""\n'
            '    return {"subject": token}\n\n'
            'def parse_authorization(headers):\n'
            '    """Extract a bearer credential from request headers."""\n'
            '    return headers.get("Authorization", "").removeprefix("Bearer ")\n',
            encoding="utf-8",
        )
        (repository / "accounts.py").write_text(
            'def hash_password(password, salt):\n'
            '    """Hash an account password with salt before storing credentials."""\n'
            '    return hashlib.sha256((salt + password).encode()).hexdigest()\n',
            encoding="utf-8",
        )
        (repository / "media.py").write_text(
            'def resize_profile_image(image, width, height):\n'
            '    """Resize an uploaded profile image to requested dimensions."""\n'
            '    return image.resize((width, height))\n',
            encoding="utf-8",
        )
        (repository / "payments.py").write_text(
            'def capture_payment(card, amount):\n'
            '    """Check the amount, then charge the payment card."""\n'
            '    if amount <= 0:\n'
            '        raise ValueError("invalid amount")\n'
            '    return charge_card(card, amount)\n',
            encoding="utf-8",
        )
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-m", "initial smoke-test source")
        v1 = run_git(repository, "rev-parse", "HEAD")

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        environment["CODELENS_INDEX_DIR"] = str(temporary / "indexes")
        environment["CODELENS_VERSION_DIR"] = str(temporary / "versions")
        log_path = temporary / "server.log"

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", str(ROOT / "backend"),
                 "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                cwd=ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT,
            )
            try:
                with httpx.Client(base_url=base_url, timeout=15) as client:
                    for _ in range(80):
                        if process.poll() is not None:
                            raise RuntimeError("Uvicorn exited during startup:\n" + log_path.read_text(encoding="utf-8"))
                        try:
                            health = client.get("/api/health")
                            if health.is_success:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.25)
                    else:
                        raise RuntimeError("Uvicorn did not become healthy in 20 seconds")

                    assert health.json()["status"] == "ok"
                    indexed = checked(client, "POST", "/api/repositories/index", json={
                        "repository_path": str(repository), "name": "live-smoke", "build_graph": True,
                    })
                    repo_id = indexed["repository_id"]
                    assert indexed["number_of_chunks"] == 6
                    assert checked(client, "GET", "/api/repositories")["repositories"]

                    queries = [
                        ("Where is a bearer token validated before protected endpoint access?", "authorize_request"),
                        ("How are account passwords hashed with salt before storage?", "hash_password"),
                        ("Where is an uploaded profile image resized?", "resize_profile_image"),
                        ("Which function checks amount and charges a payment card?", "capture_payment"),
                        ("How is a bearer credential extracted from request headers?", "parse_authorization"),
                    ]
                    print("Five live natural-language searches:")
                    first_query_relevance = {}
                    for query, expected in queries:
                        response = checked(client, "POST", "/api/search", json={
                            "query": query, "repository": repo_id, "top_k": 5,
                            "retrieval_method": "bm25",
                        })
                        match = next((item for item in response["results"]
                                      if item["symbol_name"] == expected), None)
                        if match is None:
                            raise AssertionError(f"{expected} was not retrieved for '{query}'")
                        source = (repository / match["file_path"]).read_text(encoding="utf-8")
                        if match["code"] not in source:
                            raise AssertionError(f"Returned snippet {match['chunk_id']} is not present in indexed source")
                        if expected == "authorize_request":
                            first_query_relevance[match["chunk_id"]] = 2
                        print(f"  {response['retrieval_latency_ms']:.3f} ms  {expected}  {match['file_path']}:{match['start_line']}")

                    cors = client.options("/api/search", headers={
                        "Origin": "http://localhost:5173",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    })
                    assert cors.status_code == 200
                    assert cors.headers.get("access-control-allow-origin") == "http://localhost:5173"

                    v1_result = checked(client, "POST", "/api/versions/index", json={
                        "repository_path": str(repository), "revision": v1, "label": "v1",
                    })
                    auth_path = repository / "auth.py"
                    auth_path.write_text(auth_path.read_text(encoding="utf-8").replace(
                        "authorize_request", "authenticate_request"), encoding="utf-8")
                    run_git(repository, "add", ".")
                    run_git(repository, "commit", "-m", "rename auth entry point")
                    v2 = run_git(repository, "rev-parse", "HEAD")
                    v2_result = checked(client, "POST", "/api/versions/index", json={
                        "repository_path": str(repository), "revision": v2, "label": "v2",
                    })
                    versions = checked(client, "GET", "/api/versions", params={"repository": repo_id})
                    assert {item["label"] for item in versions["versions"]} == {"v1", "v2"}

                    version_search = checked(client, "POST", "/api/search", json={
                        "query": "validate bearer token", "repository": repo_id,
                        "version": "all", "top_k": 10,
                    })
                    assert {item["version_label"] for item in version_search["results"]} == {"v1", "v2"}
                    evolution = checked(client, "POST", "/api/evolution", json={
                        "query": "validate bearer token before protected endpoint",
                        "repository": repo_id, "retrieval_mode": "lexical", "min_similarity": 0.4,
                    })
                    assert evolution["tracks"]
                    evaluation = checked(client, "POST", "/api/evaluate", json={
                        "repository": repo_id, "retrieval_method": "bm25",
                        "cases": [{"query": queries[0][0], "relevance": first_query_relevance}],
                    })
                    assert evaluation["mrr"] > 0
                    assert evaluation["ndcg_at_10"] > 0
                    print("All 8 API endpoints and CORS preflight passed against the live Uvicorn server.")
                    print(f"Version indexing: {v1_result['version']['label']}, {v2_result['version']['label']}")
                    print(f"Judgment-based evaluation: NDCG@10={evaluation['ndcg_at_10']:.4f}, MRR={evaluation['mrr']:.4f}")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
