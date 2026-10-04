import { useCallback, useEffect, useMemo, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import type { Core, ElementDefinition } from "cytoscape";

type Repo = { repository_id: string; name: string; repository_path: string | null; number_of_files: number; number_of_chunks: number; available_methods: string[]; has_versions: boolean };
type Result = { rank: number; chunk_id: string; repository_id: string; version_id: string | null; version_label: string | null; file_path: string; language: string; symbol_name: string; symbol_type: string; start_line: number; end_line: number; code: string; score: number; retrieval_method: string; metadata: Record<string, unknown>; features: Record<string, number>; signals: string[]; lexical_score: number | null; semantic_score: number | null };
type SearchResponse = { query: string; repository: Repo; version: string | null; retrieval_method: string; retrieval_latency_ms: number; total_candidates: number; retrieval_details: Record<string, unknown>; results: Result[] };
type Version = { version_id: string; label: string; commit_hash: string; parents: string[]; changed_paths: string[]; number_of_chunks: number; number_of_files: number; semantic_indexed: boolean };
type GraphNode = { node_id: string; kind: string; chunk_id?: string; symbol_name?: string; file_path?: string; language?: string; start_line?: number; end_line?: number; code?: string; selected?: boolean };
type GraphData = { nodes: GraphNode[]; edges: { source: string; target: string; relation: string; direction: string }[]; summary: { node_count: number; edge_count: number } };
type Evolution = { retrieval_mode: string; retrieval_latency_ms: number; versions_searched: string[]; tracks: { items: { version_label: string; commit_hash: string; chunk_id: string; symbol_name: string; file_path: string; start_line: number; end_line: number; relevance_score: number; change: string; code: string }[]; transitions: { from_version: string; to_version: string; from_symbol: string; to_symbol: string; from_path: string; to_path: string; similarity_score: number; change: string }[] }[] };

const API = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const demos = ["Where is the JWT token validated?", "Where is the password hashed?", "Where is the database connection created?", "Where are uploaded images resized?", "Where is payment failure handled?"];

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); message = body.detail || message; } catch { /* keep HTTP detail */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function Button({ children, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) { return <button className={`button ${props.className || ""}`} {...props}>{children}</button>; }

function HighlightedCode({ code }: { code: string }) {
  const keywordSet = new Set(["async", "await", "break", "case", "catch", "class", "const", "def", "else", "elif", "export", "extends", "false", "finally", "for", "from", "function", "if", "import", "in", "let", "new", "null", "pass", "raise", "return", "self", "static", "switch", "this", "throw", "true", "try", "var", "while", "with", "yield"]);
  const rows: ReactNode[] = code.split("\n").map((line, lineNo) => {
    const matcher = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`|\/\/.*|#.*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_$][\w$]*\b|[^\w\s])/g;
    const parts: ReactNode[] = []; let last = 0; let match: RegExpExecArray | null;
    while ((match = matcher.exec(line)) !== null) {
      if (match.index > last) parts.push(line.slice(last, match.index));
      const token = match[0]; let kind = "";
      if (/^(#|\/\/)/.test(token)) kind = "syntax-comment";
      else if (/^["'`]/.test(token)) kind = "syntax-string";
      else if (/^\d/.test(token)) kind = "syntax-number";
      else if (keywordSet.has(token)) kind = "syntax-keyword";
      else if (/^[A-Za-z_$]/.test(token) && line.slice(matcher.lastIndex).trimStart().startsWith("(")) kind = "syntax-call";
      parts.push(kind ? <span className={kind} key={`${match.index}-${token}`}>{token}</span> : token); last = matcher.lastIndex;
    }
    if (last < line.length) parts.push(line.slice(last));
    return <span key={lineNo}>{parts}{lineNo < code.split("\n").length - 1 ? "\n" : ""}</span>;
  });
  return <>{rows}</>;
}

function RelationshipGraph({ data, onSelect }: { data: GraphData | null; onSelect: (node: GraphNode) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<Core | null>(null);
  useEffect(() => {
    if (!container.current || !data?.nodes.length) { instance.current?.destroy(); instance.current = null; return; }
    const elements: ElementDefinition[] = [
      ...data.nodes.map((node) => ({ data: { id: node.node_id, label: node.symbol_name || node.file_path || node.kind, kind: node.kind, selected: !!node.selected, chunkId: node.chunk_id } })),
      ...data.edges.map((edge, index) => ({ data: { id: `edge-${index}`, source: edge.source, target: edge.target, label: edge.relation } })),
    ];
    let disposed = false;
    instance.current?.destroy();
    void import("cytoscape").then(({ default: cytoscape }) => {
      if (disposed || !container.current) return;
      instance.current = cytoscape({ container: container.current, elements, layout: { name: "breadthfirst", directed: true, padding: 35, spacingFactor: 1.2 }, userZoomingEnabled: true, userPanningEnabled: true,
      style: [
        { selector: "node", style: { "background-color": "#26362f", "border-color": "#496c5d", "border-width": "1px", color: "#e4ece8", label: "data(label)", "font-size": "10px", "font-family": "ui-monospace, monospace", "text-wrap": "wrap", "text-max-width": "120px", width: "label", height: "label", padding: "12px", shape: "round-rectangle" } },
        { selector: "node[kind = 'file']", style: { "background-color": "#242b35", "border-color": "#424c5a", color: "#c4ced8" } },
        { selector: "node[selected = true]", style: { "background-color": "#22513f", "border-color": "#79d7ad", "border-width": 2, color: "#f3fff8" } },
        { selector: "edge", style: { width: "1.3px", "line-color": "#52665c", "target-arrow-color": "#78bc9c", "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", color: "#8b9d93", "font-size": "8px", "text-background-color": "#11161b", "text-background-opacity": 1, "text-background-padding": "2px" } },
      ],
      });
      instance.current.on("tap", "node", (event) => { const item = data.nodes.find((node) => node.node_id === event.target.id()); if (item?.chunk_id) onSelect(item); });
    });
    return () => { disposed = true; instance.current?.destroy(); instance.current = null; };
  }, [data, onSelect]);
  return <div className="graph-wrap">{data?.nodes.length ? <div className="graph-canvas" ref={container} /> : <div className="empty compact">Select a search result to inspect its indexed relationships. Graph edges appear only when the repository was indexed with graph support.</div>}</div>;
}

function App() {
  const [health, setHealth] = useState(false);
  const [repos, setRepos] = useState<Repo[]>([]);
  const [repoId, setRepoId] = useState("");
  const [versions, setVersions] = useState<Version[]>([]);
  const [version, setVersion] = useState("");
  const [query, setQuery] = useState(demos[0]);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selected, setSelected] = useState<Result | null>(null);
  const [graphFocus, setGraphFocus] = useState<GraphNode | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [evolution, setEvolution] = useState<Evolution | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [apiError, setApiError] = useState("");
  const [retryKind, setRetryKind] = useState<"connection" | "search" | "evolution" | "index" | null>(null);
  const [repoPath, setRepoPath] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [evolutionBusy, setEvolutionBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const reload = useCallback(async () => {
    try {
      const [status, data] = await Promise.all([api<{ status: string }>("/api/health"), api<{ repositories: Repo[] }>("/api/repositories")]);
      setHealth(status.status === "ok"); setRepos(data.repositories); setApiError("");
      setRepoId((current) => current && data.repositories.some((r) => r.repository_id === current) ? current : data.repositories[0]?.repository_id || "");
    } catch (reason) { setHealth(false); setApiError(reason instanceof Error ? reason.message : "API unavailable"); setRetryKind("connection"); }
  }, []);
  useEffect(() => { void reload(); const timer = window.setInterval(() => void reload(), 15000); return () => clearInterval(timer); }, [reload]);

  const repository = repos.find((item) => item.repository_id === repoId);
  useEffect(() => {
    setVersions([]); setVersion(""); setEvolution(null);
    if (!repoId) return;
    api<{ versions: Version[] }>(`/api/versions?repository=${encodeURIComponent(repoId)}`).then((data) => setVersions(data.versions)).catch(() => setVersions([]));
  }, [repoId]);

  const search = useCallback(async (q = query) => {
    if (!q.trim()) { setError("Enter a question to search the indexed repository."); return; }
    setBusy(true); setError(""); setRetryKind(null); setResponse(null); setSelected(null); setGraphFocus(null); setGraph(null); setEvolution(null);
    try {
      const data = await api<SearchResponse>("/api/search", { method: "POST", body: JSON.stringify({ query: q, repository: repoId || undefined, version: version || undefined, top_k: 10, retrieval_method: "auto" }) });
      setResponse(data); setSelected(data.results[0] || null); setRetryKind(null);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Search failed."); setRetryKind("search"); }
    finally { setBusy(false); }
  }, [query, repoId, version]);
  useEffect(() => { if (repoId) void search(query); }, [repoId, version]); // Keep results scoped to the selected repository/version.

  useEffect(() => {
    if (!selected || !repoId) { setGraph(null); return; }
    let live = true;
    api<GraphData>(`/api/graph?repository=${encodeURIComponent(repoId)}&chunk_id=${encodeURIComponent(selected.chunk_id)}`)
      .then((data) => { if (live) setGraph(data); }).catch(() => { if (live) setGraph(null); });
    return () => { live = false; };
  }, [selected, repoId]);

  const runEvolution = useCallback(async () => {
    if (!repository?.has_versions) return;
    setEvolutionBusy(true); setError(""); setRetryKind(null);
    try { setEvolution(await api<Evolution>("/api/evolution", { method: "POST", body: JSON.stringify({ query, repository: repoId, retrieval_mode: "auto", per_version_top_k: 5 }) })); setRetryKind(null); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Evolution search failed."); setRetryKind("evolution"); }
    finally { setEvolutionBusy(false); }
  }, [query, repoId, repository]);

  const indexRepository = async () => {
    if (!repoPath.trim()) return;
    setIndexing(true); setError(""); setRetryKind(null);
    try { await api("/api/repositories/index", { method: "POST", body: JSON.stringify({ repository_path: repoPath.trim(), build_graph: true, include_semantic: false }) }); await reload(); setRepoPath(""); setRetryKind(null); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Indexing failed."); setRetryKind("index"); }
    finally { setIndexing(false); }
  };

  const retry = () => {
    const action = retryKind;
    setError(""); setApiError(""); setRetryKind(null);
    if (action === "search") void search();
    else if (action === "evolution") void runEvolution();
    else if (action === "index") void indexRepository();
    else void reload();
  };

  const analyzedQuery = response?.query || query;
  const queryTerms = useMemo(() => [...new Set(analyzedQuery.toLowerCase().match(/[a-z0-9_]+/g) || [])].filter((word) => word.length > 2), [analyzedQuery]);
  const viewedCode = graphFocus?.code ? graphFocus : selected;
  const viewedCodeLines = viewedCode?.code?.split("\n") || [];
  const copy = async () => { if (viewedCode?.code) { await navigator.clipboard?.writeText(viewedCode.code); setCopied(true); window.setTimeout(() => setCopied(false), 1200); } };
  const selectGraphNode = useCallback((node: GraphNode) => {
    if (!node.chunk_id) return;
    setGraphFocus(node);
  }, []);

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#top"><span className="brand-icon">⌘</span><span><b>CodeLens</b><small>AGENTIC CODE INTELLIGENCE</small></span></a>
      <div className="side-label">WORKSPACE</div>
      <select className="repo-select" aria-label="Repository" value={repoId} onChange={(event) => { setRepoId(event.target.value); setResponse(null); setSelected(null); setGraph(null); }}><option value="">Select repository…</option>{repos.map((repo) => <option key={repo.repository_id} value={repo.repository_id}>{repo.name}</option>)}</select>
      <nav className="side-nav"><a className="active" href="#search">⌕ <span>Code search</span></a><a href="#relationship-graph">⌬ <span>Relationships</span></a><a href="#evolution-view">◷ <span>Evolution</span></a></nav>
      <div className="side-bottom"><div className="connection"><i className={health ? "online" : "offline"} /><span>Backend {health ? "connected" : "disconnected"}</span></div><small className="api-url">{API}</small><div className="index-box"><span>INDEXED SOURCE</span><b>{repository?.number_of_chunks.toLocaleString() ?? "—"} snippets</b><small>{repository ? `${repository.number_of_files} files · ${repository.available_methods.join(" / ")}` : "No repository selected"}</small></div></div>
    </aside>
    <main className="main-area" id="top">
      <header className="topbar"><div><span className="muted">Intelligence</span><span className="slash">/</span><b>Code search</b></div><span className={`status-pill ${health ? "good" : "bad"}`}><i />{health ? "API ONLINE" : "API OFFLINE"}</span></header>
      <div className="content">
        <div className="intro"><div><div className="eyebrow">RETRIEVAL WORKSPACE</div><h1>Search your codebase</h1><p>Ask about behavior. Trace the implementation to its source.</p></div><label className="version-control"><span>VERSION SCOPE</span><select value={version} onChange={(event) => { setVersion(event.target.value); setResponse(null); setSelected(null); setGraph(null); }} disabled={!repoId}><option value="">Working tree index</option>{versions.length > 0 && <option value="all">All indexed versions</option>}{versions.map((item) => <option value={item.label} key={item.version_id}>{item.label} · {item.commit_hash.slice(0, 8)}</option>)}</select></label></div>
        <section className="search-panel" id="search"><label htmlFor="query">NATURAL LANGUAGE QUERY</label><div className="search-row"><textarea id="query" rows={2} value={query} placeholder="Describe the code behavior you want to find…" onChange={(event) => { setQuery(event.target.value); setEvolution(null); }} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") void search(); }} /><Button className="primary" onClick={() => void search()} disabled={busy || !health}>{busy ? "Searching…" : "⌕  Search code"}</Button></div><div className="demo-row"><span>EXAMPLES</span>{demos.map((item) => <button key={item} onClick={() => { setQuery(item); void search(item); }}>{item}</button>)}</div></section>
        {(apiError || error) && <div className="error-banner">{apiError ? `Cannot reach CodeLens API: ${apiError}` : error}<button onClick={retry}>Retry</button></div>}
        <div className="metric-grid">
          <Metric label="RETRIEVAL LATENCY" value={response ? `${response.retrieval_latency_ms.toFixed(1)} ms` : "—"} detail={response?.retrieval_method || "Waiting for search"} />
          <Metric label="INDEXED SNIPPETS" value={repository?.number_of_chunks.toLocaleString() || "—"} detail={repository?.name || "No repository"} />
          <Metric label="CANDIDATES" value={response ? String(response.total_candidates) : "—"} detail={response ? `${response.results.length} returned` : "Backend reported"} />
          <Metric label="INDEXING" value={repository ? "Ready" : "Unavailable"} detail={repository ? `${repository.number_of_files} source files` : "Index a repository below"} />
        </div>

        <div className="analysis-strip"><div><span className="section-tag">QUERY ANALYSIS</span><b>{response?.retrieval_method || "Awaiting retrieval"}</b></div><div className="analysis-detail"><span>Terms from query</span>{queryTerms.length ? queryTerms.map((term) => <code key={term}>{term}</code>) : <small>Type a question</small>}</div><div className="analysis-detail"><span>Strategy</span><b>{response?.retrieval_method || "—"}</b></div><div className="analysis-detail"><span>Candidate pool</span><b>{response ? response.total_candidates : "—"}</b></div></div>

        <div className="section-heading"><div><span className="section-tag">RETRIEVAL OUTPUT</span><h2>Ranked results</h2></div><span>{response ? `${response.results.length} results · ranked by backend` : "No search run"}</span></div>
        <div className="work-grid">
          <section className="result-column" aria-label="Ranked search results">
            {busy && <div className="empty"><div className="spinner" />Retrieving indexed code…</div>}
            {!busy && response?.results.length === 0 && <div className="empty"><b>No results returned</b><span>The backend did not find matching snippets for this query.</span></div>}
            {!response && !busy && !error && <div className="empty"><b>Ready to search</b><span>Choose a repository and submit a natural-language question.</span></div>}
            {response?.results.map((item) => <button className={`result-card ${selected?.chunk_id === item.chunk_id ? "chosen" : ""}`} key={`${item.version_id || "working"}-${item.chunk_id}`} onClick={() => { setSelected(item); setGraphFocus(null); }}>
              <div className="result-top"><span className="rank">{String(item.rank).padStart(2, "0")}</span><div className="result-title"><b>{item.symbol_name}</b><span>{item.symbol_type}{item.version_label ? ` · ${item.version_label}` : ""}</span></div><div className="score"><b>{item.score.toPrecision(4)}</b><small>score</small></div></div>
              <div className="filepath">{item.file_path}<span>:{item.start_line}–{item.end_line}</span><i>{item.language}</i></div>
              <pre className="code-compact"><code><HighlightedCode code={item.code.split("\n").slice(0, 4).join("\n") + (item.code.split("\n").length > 4 ? "\n…" : "")} /></code></pre>
              <div className="result-foot"><span>{item.retrieval_method}</span><span>{item.signals.length ? item.signals.length + " retrieval signals" : "No signal details"}</span></div>
            </button>)}
          </section>
          <aside className="inspector">
            <section className="panel"><div className="panel-head"><div><span className="section-tag">EXPLAINABILITY</span><h3>Why this result?</h3></div>{selected && <span className="rank-chip">#{selected.rank}</span>}</div>
              {selected ? <><div className="selected-symbol"><b>{selected.symbol_name}</b><span>{selected.file_path}:{selected.start_line}–{selected.end_line}</span></div><div className="score-summary"><span>Backend ranking score</span><b>{selected.score.toPrecision(5)}</b></div><ul className="signal-list">{selected.signals.map((signal, i) => <li key={`${i}-${signal}`}>{signal}</li>)}</ul>{!selected.signals.length && <p className="muted-copy">The selected API result does not include retrieval signal details.</p>}{Object.entries(selected.features).length > 0 && <div className="feature-list"><b>Backend features</b>{Object.entries(selected.features).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><code>{value.toPrecision(4)}</code></div>)}</div>}</> : <p className="muted-copy">Select a ranked result to inspect its backend explanations.</p>}
            </section>
            <section className="panel code-viewer"><div className="panel-head"><div><span className="section-tag">SOURCE {graphFocus ? "· GRAPH NODE" : ""}</span><h3>Code viewer</h3></div><Button onClick={() => void copy()} disabled={!viewedCode}>{copied ? "Copied" : "Copy"}</Button></div>{viewedCode ? <><div className="source-path">{viewedCode.file_path}<span> · {viewedCode.language || selected?.language || "code"} · lines {viewedCode.start_line}–{viewedCode.end_line}</span></div><div className="source-code">{viewedCodeLines.map((line, index) => <div className="source-line" key={index}><span>{(viewedCode.start_line || 1) + index}</span><code><HighlightedCode code={line || " "} /></code></div>)}</div></> : <p className="muted-copy">Select a result to view its indexed source code.</p>}</section>
          </aside>
        </div>

        <div className="lower-grid"><section className="panel" id="relationship-graph"><div className="panel-head"><div><span className="section-tag">STRUCTURAL CONTEXT</span><h3>Code relationships</h3></div><span className="count-tag">{graph ? `${graph.summary.node_count} nodes · ${graph.summary.edge_count} edges` : "—"}</span></div><p className="muted-copy">Resolved relationships from the repository's indexed code graph.</p><RelationshipGraph data={graph} onSelect={selectGraphNode} /></section>
          <section className="panel" id="evolution-view"><div className="panel-head"><div><span className="section-tag">VERSION INTELLIGENCE</span><h3>Evolutionary retrieval</h3></div><Button onClick={() => void runEvolution()} disabled={!repository?.has_versions || evolutionBusy}>{evolutionBusy ? "Tracing…" : "Trace query"}</Button></div><p className="muted-copy">Search indexed Git versions and show related implementations and transitions.</p>{repository?.has_versions ? <div className="evolution-content">{evolution ? <><div className="evo-meta">{evolution.versions_searched.length} versions · {evolution.retrieval_latency_ms.toFixed(1)} ms · {evolution.retrieval_mode}</div>{evolution.tracks.length ? evolution.tracks.map((track, i) => <div className="evo-track" key={i}>{track.items.map((item) => <article key={item.chunk_id}><div><b>{item.symbol_name}</b><span>{item.version_label} · {item.commit_hash.slice(0, 8)}</span></div><small>{item.file_path}:{item.start_line}–{item.end_line} · relevance {item.relevance_score.toPrecision(4)}</small><p>{item.change}</p></article>)}{track.transitions.map((transition, j) => <div className="transition" key={j}>{transition.from_version} → {transition.to_version}: {transition.from_symbol} → {transition.to_symbol}<small>{transition.change} · similarity {transition.similarity_score.toPrecision(3)}</small></div>)}</div>) : <div className="empty compact">The backend returned no evolutionary tracks for this query.</div>}</> : <div className="empty compact">No indexed Git versions for this repository. Index versions through the API to enable evolutionary retrieval.</div>}</div> : <div className="empty compact">Select a repository with indexed Git versions to run evolutionary retrieval.</div>}</section></div>
        <section className="panel index-panel"><div className="panel-head"><div><span className="section-tag">REPOSITORY MANAGEMENT</span><h3>Index a local repository</h3></div></div><p className="muted-copy">The API indexes a directory accessible to the backend process. Graph relationships are built from parsed source.</p><div className="index-row"><input value={repoPath} onChange={(event) => setRepoPath(event.target.value)} placeholder="Backend-visible repository path" aria-label="Repository path" /><Button className="primary" disabled={indexing || !repoPath.trim() || !health} onClick={() => void indexRepository()}>{indexing ? "Indexing…" : "Index repository"}</Button></div></section>
        <footer>CodeLens <span>·</span> Live retrieval powered by the connected FastAPI service</footer>
      </div>
    </main>
  </div>;
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) { return <div className="metric"><span>{label}</span><b>{value}</b><small>{detail}</small></div>; }
export default App;
