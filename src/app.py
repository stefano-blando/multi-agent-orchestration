"""
Demo Streamlit Hackapizza 2.0
- Modalita' Agent (LLM orchestrator)
- Modalita' Structured PoC (schema-first, offline-friendly)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.agent import check_constraints_batch, run_structured_orchestration, run_with_trace


DEFAULT_QUERY = "Quali piatti sono senza glutine e vegani?"
DEFAULT_CONSTRAINTS = "senza glutine, vegano"
DEFAULT_CANDIDATES = "Pizza Nebulosa\nInsalata Orbitale\nTofu Meteorico\nRisotto Lunare"

SCENARIOS = {
    "Free-form": {
        "query": DEFAULT_QUERY,
        "constraints": DEFAULT_CONSTRAINTS,
        "candidates": DEFAULT_CANDIDATES,
        "mode": "Structured PoC",
    },
    "Allergeni stretti": {
        "query": "Seleziona solo piatti senza glutine, senza lattosio e senza frutta a guscio.",
        "constraints": "senza glutine, senza lattosio, senza frutta a guscio",
        "candidates": "Pizza Nebulosa\nRisotto Lunare\nTofu Meteorico\nInsalata Orbitale",
        "mode": "Structured PoC",
    },
    "Stress multi-vincolo": {
        "query": "Dammi i piatti vegani, senza glutine e adatti a celiaci.",
        "constraints": "vegano, senza glutine",
        "candidates": "Pizza Nebulosa\nBurger Cosmico\nInsalata Orbitale\nRavioli Cometa",
        "mode": "Agent",
    },
}


def _parse_csv_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    out = []
    for part in raw.split(","):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out


def _parse_lines_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    out = []
    for line in raw.splitlines():
        item = line.strip()
        if item and item not in out:
            out.append(item)
    return out


def _set_runtime_env(
    embedding_mode: str,
    rerank_mode: str,
    min_confidence: float,
    validator_top_k: int,
    evidence_limit: int,
    trace_enabled: bool,
    trace_path: str,
    structured_fallback: bool,
    timeout_seconds: int,
    hybrid_cache_ttl: int,
    batch_cache_ttl: int,
):
    os.environ["EMBEDDING_MODE"] = embedding_mode
    os.environ["RERANK_MODE"] = rerank_mode
    os.environ["VALIDATOR_TOP_K"] = str(int(validator_top_k))
    os.environ["VALIDATOR_EVIDENCE_LIMIT"] = str(int(evidence_limit))
    os.environ["VALIDATOR_MIN_CONFIDENCE"] = str(min_confidence)
    os.environ["TRACE_ENABLED"] = "1" if trace_enabled else "0"
    os.environ["TRACE_LOG_PATH"] = trace_path
    os.environ["ENABLE_STRUCTURED_FALLBACK"] = "1" if structured_fallback else "0"
    os.environ["APP_TIMEOUT_SECONDS"] = str(int(timeout_seconds))
    os.environ["HYBRID_CACHE_TTL_SECONDS"] = str(int(hybrid_cache_ttl))
    os.environ["BATCH_CACHE_TTL_SECONDS"] = str(int(batch_cache_ttl))


def _op_timeout_seconds() -> int:
    return max(3, int(os.getenv("APP_TIMEOUT_SECONDS", "25")))


def _run_with_timeout(label: str, fn, *args, **kwargs):
    timeout = _op_timeout_seconds()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise RuntimeError(
                f"Timeout su `{label}` dopo {timeout}s. "
                "Riduci top_k/evidence_limit o passa a Structured PoC."
            ) from exc


def _load_recent_trace_events(path: str, limit: int = 120) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    events = []
    for row in lines[-limit:]:
        try:
            events.append(json.loads(row))
        except json.JSONDecodeError:
            continue
    return events


def _render_summary_metrics(payload: dict):
    answer = payload.get("answer", [])
    summary = payload.get("summary", {})
    safe_items = summary.get("safe_items", []) if isinstance(summary, dict) else []
    total_checks = summary.get("total_checks", 0) if isinstance(summary, dict) else 0
    cols = st.columns(3)
    cols[0].metric("Risposta", len(answer))
    cols[1].metric("Safe Items", len(safe_items))
    cols[2].metric("Constraint Checks", int(total_checks))


def _render_item_summary_table(summary: dict):
    rows = summary.get("item_summary", []) if isinstance(summary, dict) else []
    if not rows:
        st.info("Nessun riepilogo item disponibile.")
        return
    table = []
    for row in rows:
        table.append(
            {
                "item": row.get("item", ""),
                "is_safe": row.get("is_safe", False),
                "passed_constraints": ", ".join(row.get("passed_constraints", [])),
                "failed_count": len(row.get("failed_constraints", [])),
            }
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _summarize_explainability(results: list[dict], min_confidence: float) -> tuple[list[dict], list[dict]]:
    explain_rows = []
    risk_rows = []
    for row in results:
        evidence = row.get("evidence", [])
        first_evidence = evidence[0] if isinstance(evidence, list) and evidence else {}
        ref = first_evidence.get("evidence_id", "")
        snippet = first_evidence.get("text", "")
        explain = {
            "item": row.get("item", ""),
            "constraint": row.get("constraint", ""),
            "status": row.get("status", "NON CONFORME"),
            "confidence": round(float(row.get("confidence", 0.0)), 3),
            "reason": row.get("reason", ""),
            "top_evidence": ref,
            "signal": first_evidence.get("signal", "neutral"),
            "snippet": snippet[:120],
        }
        explain_rows.append(explain)
        if explain["status"] != "CONFORME" or explain["confidence"] < min_confidence:
            risk_rows.append(explain)
    return explain_rows, risk_rows


def _render_explainability(results: list[dict], min_confidence: float):
    if not results:
        st.info("Nessuna explainability disponibile (mancano risultati batch).")
        return
    explain_rows, risk_rows = _summarize_explainability(results, min_confidence=min_confidence)
    if risk_rows:
        st.markdown("**Alert vincoli (NON CONFORME o confidence bassa)**")
        st.dataframe(risk_rows, use_container_width=True, hide_index=True)
    else:
        st.success("Nessuna violazione critica nei check vincoli.")
    with st.expander("Tutti i check con evidenze", expanded=False):
        st.dataframe(explain_rows, use_container_width=True, hide_index=True)


def _ensure_state():
    defaults = {
        "query_input": DEFAULT_QUERY,
        "constraints_input": DEFAULT_CONSTRAINTS,
        "candidates_input": DEFAULT_CANDIDATES,
        "last_run_payload": {},
        "last_batch_payload": {},
        "run_history": [],
        "execution_mode_state": "Structured PoC",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _push_history(entry: dict):
    st.session_state.run_history.insert(0, entry)
    st.session_state.run_history = st.session_state.run_history[:12]


st.set_page_config(page_title="Hackapizza Agent Console", page_icon="HP", layout="wide")
_ensure_state()

st.markdown(
    """
<style>
:root {
  --bg-main: #060606;
  --bg-soft: #131313;
  --card: #ffffff;
  --ink: #f5f5f5;
  --ink-soft: #c8c8c8;
  --ink-dark: #131313;
  --line-dark: #2b2b2b;
  --line-light: #e4e4e4;
  --brand-red: #d73333;
  --brand-red-dark: #a91f1f;
}
.stApp {
  background: radial-gradient(circle at 15% -5%, #1f1f1f 0%, var(--bg-main) 55%);
  color: var(--ink);
}
.block-container {
  max-width: 1200px;
  padding-top: 1.5rem;
}
.hero-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  color: var(--ink);
}
.hero-sub {
  color: var(--ink-soft);
  margin-top: -0.4rem;
}
.pill {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border: 1px solid var(--brand-red-dark);
  border-radius: 999px;
  margin-right: 0.35rem;
  margin-top: 0.2rem;
  font-size: 0.78rem;
  color: #f3f3f3;
  background: #171717;
}
.panel {
  border: 1px solid var(--line-light);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  background: var(--card);
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
  color: var(--ink-dark);
}
.panel * {
  color: var(--ink-dark);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f0f0f 0%, #060606 100%);
  border-right: 1px solid var(--line-dark);
}
section[data-testid="stSidebar"] * {
  color: #f3f3f3;
}
div[data-baseweb="tab-list"] button {
  color: #f0f0f0;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] {
  color: var(--brand-red) !important;
}
button[kind="primary"] {
  background: var(--brand-red) !important;
  color: #ffffff !important;
  border: 1px solid var(--brand-red-dark) !important;
}
button[kind="primary"]:hover {
  background: var(--brand-red-dark) !important;
}
.stButton > button {
  border-radius: 10px;
}
div[data-testid="stMetric"] {
  border: 1px solid var(--line-light);
  border-radius: 10px;
  padding: 0.2rem 0.5rem;
}
div[data-testid="stMetric"] * {
  color: var(--ink-dark) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='hero-title'>Hackapizza Agent Console</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>Demo pronta per hackathon: orchestrazione multi-agent, validazione batch, trace runtime.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<span class='pill'>Offline-ready</span>"
    "<span class='pill'>Schema-first fallback</span>"
    "<span class='pill'>Validator evidence-based</span>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Runtime")
    preset = st.selectbox("Scenario preset", options=list(SCENARIOS.keys()), index=0)
    if st.button("Carica preset", use_container_width=True):
        choice = SCENARIOS[preset]
        st.session_state.query_input = choice["query"]
        st.session_state.constraints_input = choice["constraints"]
        st.session_state.candidates_input = choice["candidates"]
        st.session_state.execution_mode_state = choice["mode"]
        st.success(f"Preset '{preset}' caricato.")

    mode = st.radio(
        "Execution Mode",
        options=["Structured PoC", "Agent"],
        key="execution_mode_state",
        help="Structured PoC: deterministico/offline. Agent: orchestrator LLM con fallback opzionale.",
    )
    embedding_mode = st.selectbox("Embedding Mode", ["mock", "local", "openai"], index=0)
    rerank_mode = st.selectbox("Rerank Mode", ["heuristic", "cross_encoder", "none"], index=0)
    min_confidence = st.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    validator_top_k = st.number_input("Validator top_k", min_value=1, max_value=15, value=4, step=1)
    evidence_limit = st.number_input("Evidence limit", min_value=1, max_value=20, value=8, step=1)
    timeout_seconds = st.number_input("Operation timeout (s)", min_value=3, max_value=180, value=25, step=1)
    hybrid_cache_ttl = st.number_input("Hybrid cache TTL (s)", min_value=0, max_value=600, value=90, step=10)
    batch_cache_ttl = st.number_input("Batch cache TTL (s)", min_value=0, max_value=600, value=120, step=10)
    structured_fallback = st.toggle("Structured fallback", value=True)
    trace_enabled = st.toggle("Trace enabled", value=True)
    trace_path = st.text_input("Trace log path", value="data/traces/agent_trace.jsonl")
    if st.button("Pulisci output", use_container_width=True):
        st.session_state.last_run_payload = {}
        st.session_state.last_batch_payload = {}
        st.session_state.run_history = []

    _set_runtime_env(
        embedding_mode=embedding_mode,
        rerank_mode=rerank_mode,
        min_confidence=min_confidence,
        validator_top_k=int(validator_top_k),
        evidence_limit=int(evidence_limit),
        trace_enabled=trace_enabled,
        trace_path=trace_path,
        structured_fallback=structured_fallback,
        timeout_seconds=int(timeout_seconds),
        hybrid_cache_ttl=int(hybrid_cache_ttl),
        batch_cache_ttl=int(batch_cache_ttl),
    )

    st.divider()
    st.subheader("Preflight")
    key_present = bool(os.getenv("OPENAI_API_KEY"))
    st.caption(f"OPENAI_API_KEY: {'disponibile' if key_present else 'assente'}")
    if mode == "Agent" and not key_present:
        st.warning("Modalita' Agent senza API key: verra' usato mock + fallback.")
    st.caption(f"Embedding: {embedding_mode}")
    st.caption(f"Rerank: {rerank_mode}")
    st.caption(f"Timeout: {int(timeout_seconds)}s")
    st.caption(f"Cache TTL: hybrid={int(hybrid_cache_ttl)}s, batch={int(batch_cache_ttl)}s")

tab_run, tab_validator, tab_trace, tab_runbook = st.tabs(
    ["Orchestrazione", "Validator Lab", "Trace", "Runbook Demo"]
)

with tab_run:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Input")
        st.text_area("Query", key="query_input", height=100)
        st.text_input("Constraints (comma separated)", key="constraints_input")
        st.text_area("Candidates (one per line, opzionale)", key="candidates_input", height=160)
        run_btn = st.button("Esegui Orchestrazione", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Output")
        payload = st.session_state.last_run_payload
        if payload:
            mode_badge = payload.get("mode", "n/a")
            elapsed_ms = payload.get("elapsed_ms")
            elapsed_info = f" | {elapsed_ms} ms" if elapsed_ms is not None else ""
            st.caption(f"Ultimo run: {payload.get('ts', '-')} | mode={mode_badge}{elapsed_info}")
            answer = payload.get("answer", [])
            if answer:
                st.success(f"Risposta: {answer}")
            else:
                st.warning("Nessuna risposta sicura trovata.")
            if payload.get("fallback_used"):
                st.warning("Fallback structured attivato durante run Agent.")
            _render_summary_metrics(payload)
            summary = payload.get("summary", {})
            if summary:
                st.markdown("**Item summary**")
                _render_item_summary_table(summary)
            _render_explainability(payload.get("batch_results", []), min_confidence=min_confidence)
            with st.expander("Payload JSON", expanded=False):
                st.json(payload, expanded=False)
        else:
            st.info("Nessuna esecuzione effettuata.")
        st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        query = st.session_state.query_input.strip()
        constraints = _parse_csv_list(st.session_state.constraints_input)
        candidates = _parse_lines_list(st.session_state.candidates_input)
        if not query:
            st.error("Inserisci una query prima di eseguire.")
        else:
            with st.spinner("Esecuzione in corso..."):
                try:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    started = time.perf_counter()
                    if mode == "Agent":
                        trace = _run_with_timeout("run_with_trace", run_with_trace, query)
                        fallback_used = trace.get("mode") == "structured_poc"
                        run_payload = {
                            "ts": now,
                            "mode": trace.get("mode", "agent"),
                            "query": query,
                            "answer": trace.get("answer", []),
                            "tools_used": trace.get("tools_used", []),
                            "usage": trace.get("usage", ""),
                            "raw_text": trace.get("raw_text", ""),
                            "constraints": constraints,
                            "candidates": candidates,
                            "fallback_used": fallback_used,
                        }
                        if constraints and candidates:
                            batch_json = _run_with_timeout(
                                "check_constraints_batch",
                                check_constraints_batch,
                                items=candidates,
                                constraints=constraints,
                                min_confidence=min_confidence,
                            )
                            batch_obj = json.loads(batch_json)
                            run_payload["summary"] = batch_obj.get("summary", {})
                            run_payload["batch_results"] = batch_obj.get("results", [])
                        else:
                            run_payload["summary"] = {"safe_items": [], "item_summary": [], "total_checks": 0}
                            run_payload["batch_results"] = []
                    else:
                        result = _run_with_timeout(
                            "run_structured_orchestration",
                            run_structured_orchestration,
                            {
                                "query": query,
                                "constraints": constraints,
                                "candidates": candidates,
                                "min_confidence": min_confidence,
                            }
                        )
                        batch_obj = result.get("batch", {})
                        run_payload = {
                            "ts": now,
                            "mode": "structured_poc",
                            "query": query,
                            "answer": result.get("answer", []),
                            "constraints": result.get("constraints", []),
                            "candidates": result.get("candidates", []),
                            "summary": batch_obj.get("summary", {}),
                            "batch_results": batch_obj.get("results", []),
                            "fallback_used": False,
                        }
                    run_payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)

                    st.session_state.last_run_payload = run_payload
                    _push_history(
                        {
                            "ts": run_payload["ts"],
                            "mode": run_payload["mode"],
                            "query": query,
                            "answer_count": len(run_payload.get("answer", [])),
                            "elapsed_ms": run_payload.get("elapsed_ms", 0),
                        }
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Errore esecuzione: {exc}")

with tab_validator:
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Batch Validation")
        st.caption("Usa gli stessi input della tab Orchestrazione o modificali qui.")
        validator_constraints_raw = st.text_input(
            "Constraints (comma separated)",
            value=st.session_state.constraints_input,
            key="validator_constraints_input",
        )
        validator_candidates_raw = st.text_area(
            "Candidates (one per line)",
            value=st.session_state.candidates_input,
            key="validator_candidates_input",
            height=220,
        )
        batch_btn = st.button("Valida Candidati", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Batch Output")
        batch_payload = st.session_state.last_batch_payload
        if batch_payload:
            summary = batch_payload.get("summary", {})
            safe_items = summary.get("safe_items", []) if isinstance(summary, dict) else []
            elapsed_ms = batch_payload.get("elapsed_ms")
            if elapsed_ms is not None:
                st.caption(f"Tempo batch: {elapsed_ms} ms")
            st.success(f"Safe items: {safe_items}")
            _render_summary_metrics({"answer": safe_items, "summary": summary})
            _render_item_summary_table(summary)
            _render_explainability(batch_payload.get("results", []), min_confidence=min_confidence)
            with st.expander("Batch JSON", expanded=False):
                st.json(batch_payload, expanded=False)
        else:
            st.info("Nessuna validazione batch eseguita.")
        st.markdown("</div>", unsafe_allow_html=True)

    if batch_btn:
        constraints = _parse_csv_list(validator_constraints_raw)
        candidates = _parse_lines_list(validator_candidates_raw)
        if not constraints or not candidates:
            st.error("Inserisci almeno 1 vincolo e 1 candidato.")
        else:
            with st.spinner("Validazione batch in corso..."):
                try:
                    started = time.perf_counter()
                    batch_json = _run_with_timeout(
                        "check_constraints_batch",
                        check_constraints_batch,
                        items=candidates,
                        constraints=constraints,
                        min_confidence=min_confidence,
                    )
                    batch_obj = json.loads(batch_json)
                    batch_obj["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
                    st.session_state.last_batch_payload = batch_obj
                    st.rerun()
                except Exception as exc:
                    st.error(f"Errore validator batch: {exc}")

with tab_trace:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Trace Log")
    events = _load_recent_trace_events(trace_path, limit=200)
    if not events:
        st.info("Nessun evento trace trovato. Esegui prima una run.")
    else:
        flat = []
        for idx, event in enumerate(events):
            flat.append(
                {
                    "idx": idx,
                    "ts": event.get("ts", ""),
                    "event": event.get("event", ""),
                    "query": event.get("query", ""),
                    "request_id": event.get("request_id", ""),
                }
            )
        st.dataframe(flat[-40:], use_container_width=True, hide_index=True)
        selected = st.slider("Evento da ispezionare", min_value=0, max_value=len(events) - 1, value=len(events) - 1)
        st.json(events[selected], expanded=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Run History")
    history = st.session_state.run_history
    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.caption("Nessuna run nello storico locale.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_runbook:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Script Demo 3 minuti")
    st.markdown(
        "1. Seleziona preset `Allergeni stretti` e premi `Carica preset`.\n"
        "2. Esegui in `Structured PoC` per mostrare pipeline deterministica senza API key.\n"
        "3. Apri `Validator Lab` e fai vedere `safe_items` + `item_summary`.\n"
        "4. Passa a `Agent` e rilancia per mostrare orchestrazione + fallback.\n"
        "5. Apri `Trace` e mostra eventi JSONL (audit rapido delle decisioni)."
    )
    st.markdown("</div>", unsafe_allow_html=True)
