"""
Agentic AI system con datapizza-ai — Hackapizza 2.0.

Architettura: OrchestratorAgent + ValidatorAgent con ReAct loop e planning.
L'orchestratore decide autonomamente quali tool/agent richiamare,
in che ordine, e quante iterazioni fare prima di rispondere.

Caratteristiche agentiche:
  - planning_interval=1: pianifica esplicitamente prima di agire
  - terminate_on_text=False: DEVE chiamare submit_answer per rispondere
  - submit_answer(end=True): forza output strutturato (lista) per scoring Jaccard
  - max_steps=15: evita loop infiniti

Tool disponibili:
  - hybrid_search: BM25 + semantic search con RRF fusion
  - get_document: recupera tutti i chunk di un documento specifico
  - validator (agent-as-tool): validazione puntuale o batch dei vincoli
  - submit_answer: risposta finale strutturata (end=True)
"""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Literal, Optional
from dotenv import load_dotenv
from datapizza.agents import Agent
from datapizza.tools import tool
from pydantic import BaseModel, Field, ValidationError, ConfigDict

load_dotenv()

_BATCH_CACHE: dict[str, tuple[float, str]] = {}


def _validator_top_k() -> int:
    return max(1, int(os.getenv("VALIDATOR_TOP_K", "4")))


def _validator_evidence_limit() -> int:
    return max(1, int(os.getenv("VALIDATOR_EVIDENCE_LIMIT", "8")))


def _validator_min_confidence() -> float:
    return max(0.0, min(1.0, float(os.getenv("VALIDATOR_MIN_CONFIDENCE", "0.6"))))


def _trace_enabled() -> bool:
    return os.getenv("TRACE_ENABLED", "1").strip() not in {"0", "false", "False"}


def _trace_log_path() -> str:
    return os.getenv("TRACE_LOG_PATH", "data/traces/agent_trace.jsonl")


def _batch_cache_ttl_seconds() -> int:
    return max(0, int(os.getenv("BATCH_CACHE_TTL_SECONDS", "120")))


def _batch_cache_fingerprint() -> str:
    path = os.getenv("BM25_CORPUS_PATH", "data/index/bm25_corpus.jsonl")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return json.dumps(
        {
            "bm25_path": path,
            "bm25_mtime": round(float(mtime), 3),
            "qdrant_collection": os.getenv("QDRANT_COLLECTION", "hackapizza"),
            "embedding_mode": os.getenv("EMBEDDING_MODE", "local"),
            "rerank_mode": os.getenv("RERANK_MODE", "heuristic"),
            "validator_top_k": _validator_top_k(),
            "validator_evidence_limit": _validator_evidence_limit(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _append_trace_event(event: str, payload: dict):
    if not _trace_enabled():
        return
    path = _trace_log_path()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    source: str
    index: int
    score: float
    signal: Literal["positive", "negative", "neutral"]
    text: str


class ConstraintCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: str
    constraint: str
    status: Literal["CONFORME", "NON CONFORME"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    queries: list[str]
    evidence: list[EvidenceItem]
    evidence_refs: list[str]


class FailedConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraint: str
    status: Literal["CONFORME", "NON CONFORME"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ItemValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: str
    is_safe: bool
    failed_constraints: list[FailedConstraint]
    passed_constraints: list[str]


class ConstraintBatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_confidence: float = Field(ge=0.0, le=1.0)
    safe_items: list[str]
    item_summary: list[ItemValidationSummary]
    total_checks: int


class ConstraintBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[ConstraintCheckResult]
    summary: ConstraintBatchSummary


class StructuredOrchestrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    constraints: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class StructuredOrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    query: str
    answer: list[str]
    constraints: list[str]
    candidates: list[str]
    batch: ConstraintBatchResponse


class RunTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    query: str
    answer: list[str]
    tools_used: list
    usage: str
    raw_text: str
    mode: Literal["agent", "structured_poc"]


@dataclass
class ConstraintProfile:
    raw: str
    mode: str
    targets: list[str]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _build_constraint_profile(constraint: str) -> ConstraintProfile:
    rule = _normalize_text(constraint)
    if not rule:
        return ConstraintProfile(raw=constraint, mode="generic", targets=[])

    absence_prefixes = ["senza ", "no ", "privo di ", "free from "]
    presence_prefixes = ["con ", "contains ", "include ", "contiene "]

    for prefix in absence_prefixes:
        if rule.startswith(prefix):
            target = rule[len(prefix) :].strip(" .,:;")
            return ConstraintProfile(raw=constraint, mode="absence", targets=[target] if target else [])

    for prefix in presence_prefixes:
        if rule.startswith(prefix):
            target = rule[len(prefix) :].strip(" .,:;")
            return ConstraintProfile(raw=constraint, mode="presence", targets=[target] if target else [])

    if any(term in rule for term in ["vegano", "vegan"]):
        return ConstraintProfile(raw=constraint, mode="vegan", targets=["vegano", "vegan"])
    if any(term in rule for term in ["vegetariano", "vegetarian"]):
        return ConstraintProfile(raw=constraint, mode="vegetarian", targets=["vegetariano", "vegetarian"])

    return ConstraintProfile(raw=constraint, mode="generic", targets=[rule])


def _search_queries(item: str, profile: ConstraintProfile) -> list[str]:
    base = [f"{item} {profile.raw}"]
    if profile.mode in {"absence", "presence"} and profile.targets:
        t = profile.targets[0]
        base.extend(
            [
                f"{item} {t}",
                f"{item} contiene {t}",
                f"{item} senza {t}",
                f"{item} allergeni {t}",
            ]
        )
    elif profile.mode == "vegan":
        base.extend(
            [
                f"{item} vegano",
                f"{item} ingredienti",
                f"{item} carne uova latte formaggio",
            ]
        )
    elif profile.mode == "vegetarian":
        base.extend(
            [
                f"{item} vegetariano",
                f"{item} ingredienti",
                f"{item} carne pesce",
            ]
        )
    else:
        base.extend([f"{item} {profile.targets[0] if profile.targets else profile.raw}", f"{item} requisito"])

    deduped: list[str] = []
    seen: set[str] = set()
    for q in base:
        k = _normalize_text(q)
        if k and k not in seen:
            seen.add(k)
            deduped.append(q)
    return deduped


def _contains_phrase(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases if p)


def _focus_text_on_item(text: str, item_tokens: list[str]) -> str:
    """Estrae le porzioni del testo che menzionano l'item per ridurre cross-contamination."""
    if not item_tokens:
        return text
    segments = re.split(r"[.\n;]+", text)
    focused_idxs = []
    normalized = [_normalize_text(seg) for seg in segments]
    for idx, seg_norm in enumerate(normalized):
        if seg_norm and any(tok in seg_norm for tok in item_tokens):
            # Include segmento corrente + successivo per catturare continuazioni,
            # ma evita il precedente per ridurre contaminazione da altri item.
            focused_idxs.extend([idx, idx + 1])

    focused = []
    seen = set()
    for idx in focused_idxs:
        if idx < 0 or idx >= len(normalized):
            continue
        seg_norm = normalized[idx]
        if seg_norm and seg_norm not in seen:
            seen.add(seg_norm)
            focused.append(seg_norm)
    return " ".join(focused) if focused else text


def _evaluate_constraint_with_evidence(
    item: str, constraint: str, evidence_rows: list[dict]
) -> tuple[str, str, float, list[dict]]:
    """
    Valuta il vincolo usando evidenze retrieved.
    Ritorna: status, reason, confidence [0..1], evidenze annotate.
    """
    profile = _build_constraint_profile(constraint)
    if not evidence_rows:
        return (
            "NON CONFORME",
            "Nessuna evidenza recuperata dal corpus.",
            0.25,
            [],
        )

    negative_clues = [
        "non conforme",
        "vietato",
        "proibito",
        "non consentito",
        "evitare",
        "contains",
        "contiene",
        "tracce di",
    ]
    absence_clues = ["senza", "no", "privo di", "free from", "gluten free", "lactose free"]

    vegan_forbidden = ["carne", "pesce", "uova", "latte", "burro", "formaggio", "latticini", "gelatina"]
    vegetarian_forbidden = ["carne", "pesce", "gelatina"]

    positive_hits = 0
    negative_hits = 0
    annotated: list[dict] = []
    item_tokens = [tok for tok in _normalize_text(item).split() if len(tok) >= 3]

    for row in evidence_rows:
        text = _normalize_text(row.get("text", ""))
        signal = "neutral"
        mentions_item = True if not item_tokens else any(tok in text for tok in item_tokens)
        scoped_text = _focus_text_on_item(text, item_tokens)

        if not mentions_item:
            signal = "neutral"
        elif profile.mode == "absence" and profile.targets:
            target = _normalize_text(profile.targets[0])
            escaped_target = re.escape(target)
            positive_pattern = rf"(senza|no|privo di|free from)\s+[^.\n]{{0,40}}{escaped_target}"
            negative_pattern = rf"(contiene|con|include|contains|tracce di|presenza di)\s+[^.\n]{{0,40}}{escaped_target}"
            has_positive = re.search(positive_pattern, scoped_text) is not None
            has_negative = re.search(negative_pattern, scoped_text) is not None

            if has_positive and has_negative:
                signal = "negative"
            elif has_positive:
                signal = "positive"
            elif has_negative:
                signal = "negative"
            elif target in scoped_text and _contains_phrase(scoped_text, ["vietato", "non consentito", "evitare"]):
                signal = "negative"
        elif profile.mode == "presence" and profile.targets:
            target = _normalize_text(profile.targets[0])
            escaped_target = re.escape(target)
            positive_pattern = rf"(contiene|con|include|contains|presenza di)\s+[^.\n]{{0,40}}{escaped_target}"
            negative_pattern = rf"(senza|no|privo di|free from)\s+[^.\n]{{0,40}}{escaped_target}"
            has_positive = re.search(positive_pattern, scoped_text) is not None
            has_negative = re.search(negative_pattern, scoped_text) is not None

            if has_positive and has_negative:
                signal = "negative"
            elif has_positive:
                signal = "positive"
            elif has_negative:
                signal = "negative"
        elif profile.mode == "vegan":
            has_positive = _contains_phrase(scoped_text, ["vegano", "vegan", "100% vegetale", "plant-based"])
            has_negative = _contains_phrase(scoped_text, vegan_forbidden)
            if has_positive and has_negative:
                signal = "negative"
            elif has_positive:
                signal = "positive"
            elif has_negative:
                signal = "negative"
        elif profile.mode == "vegetarian":
            has_positive = _contains_phrase(scoped_text, ["vegetariano", "vegetarian"])
            has_negative = _contains_phrase(scoped_text, vegetarian_forbidden)
            if has_positive and has_negative:
                signal = "negative"
            elif has_positive:
                signal = "positive"
            elif has_negative:
                signal = "negative"
        else:
            target = _normalize_text(profile.targets[0]) if profile.targets else _normalize_text(constraint)
            if target and target in scoped_text and not _contains_phrase(scoped_text, ["non " + target]):
                signal = "positive"
            if _contains_phrase(scoped_text, negative_clues):
                signal = "negative"

        if signal == "positive":
            positive_hits += 1
        elif signal == "negative":
            negative_hits += 1

        annotated.append(
            {
                "evidence_id": f"{row.get('source', 'unknown')}#{int(row.get('index', -1))}",
                "source": row.get("source", "unknown"),
                "index": int(row.get("index", -1)),
                "score": round(float(row.get("score", 0.0)), 4),
                "signal": signal,
                "text": row.get("text", "")[:240],
            }
        )

    total_hits = positive_hits + negative_hits
    if negative_hits > 0:
        if positive_hits > 0:
            return (
                "NON CONFORME",
                "Evidenze contraddittorie: trovato sia supporto sia violazioni. Scelta prudenziale.",
                0.55,
                annotated[:5],
            )
        return (
            "NON CONFORME",
            "Trovata almeno una evidenza negativa esplicita.",
            min(0.95, 0.55 + 0.1 * negative_hits),
            annotated[:5],
        )
    if positive_hits > 0:
        return (
            "CONFORME",
            "Trovate evidenze positive senza contraddizioni esplicite.",
            min(0.95, 0.5 + 0.1 * positive_hits),
            annotated[:5],
        )
    if total_hits == 0:
        return (
            "NON CONFORME",
            "Nessuna evidenza utile trovata per validare il vincolo.",
            0.4,
            annotated[:5],
        )
    return (
        "NON CONFORME",
        "Evidenze insufficienti per confermare il vincolo in modo affidabile.",
        0.45,
        annotated[:5],
    )


def _check_constraint_data(item: str, constraint: str) -> dict:
    """Esegue la validazione completa di un singolo vincolo e ritorna un dict strutturato."""
    from src.retrieval import hybrid_search as _hybrid_search

    profile = _build_constraint_profile(constraint)
    queries = _search_queries(item, profile)

    merged: dict[tuple[str, int, str], dict] = {}
    for query in queries:
        for row in _hybrid_search(query, top_k=_validator_top_k()):
            key = (
                row.get("source", "unknown"),
                int(row.get("index", -1)),
                row.get("text", ""),
            )
            if key not in merged:
                merged[key] = row
            else:
                merged[key]["score"] = max(
                    float(merged[key].get("score", 0.0)),
                    float(row.get("score", 0.0)),
                )

    evidence_rows = sorted(
        merged.values(),
        key=lambda r: float(r.get("score", 0.0)),
        reverse=True,
    )[: _validator_evidence_limit()]

    status, reason, confidence, evidence = _evaluate_constraint_with_evidence(
        item=item,
        constraint=constraint,
        evidence_rows=evidence_rows,
    )
    # Hard evidence policy: un "CONFORME" senza evidenza positiva non e' accettato.
    has_positive = any(e.get("signal") == "positive" for e in evidence)
    if status == "CONFORME" and not has_positive:
        status = "NON CONFORME"
        confidence = min(confidence, 0.49)
        reason = "Nessuna evidenza positiva attribuibile all'item."

    result_obj = ConstraintCheckResult(
        item=item,
        constraint=constraint,
        status=status,
        confidence=round(confidence, 3),
        reason=reason,
        queries=queries,
        evidence=evidence,
        evidence_refs=[e["evidence_id"] for e in evidence],
    )
    result = result_obj.model_dump()
    _append_trace_event(
        "constraint_check",
        {
            "item": item,
            "constraint": constraint,
            "status": result["status"],
            "confidence": result["confidence"],
            "evidence_refs": result["evidence_refs"],
        },
    )
    return result


def _summarize_batch_results(results: list[dict], min_confidence: float = 0.6) -> dict:
    """Aggrega i risultati di validazione in una vista per item."""
    per_item: dict[str, dict] = {}
    for result in results:
        item = result.get("item", "")
        if not item:
            continue
        if item not in per_item:
            per_item[item] = {
                "item": item,
                "is_safe": True,
                "failed_constraints": [],
                "passed_constraints": [],
            }

        status = result.get("status", "NON CONFORME")
        confidence = float(result.get("confidence", 0.0))
        constraint = result.get("constraint", "")

        if status == "CONFORME" and confidence >= min_confidence:
            per_item[item]["passed_constraints"].append(constraint)
        else:
            per_item[item]["is_safe"] = False
            per_item[item]["failed_constraints"].append(
                {
                    "constraint": constraint,
                    "status": status,
                    "confidence": round(confidence, 3),
                    "reason": result.get("reason", ""),
                }
            )

    item_summary = [ItemValidationSummary(**row).model_dump() for row in per_item.values()]
    safe_items = [row["item"] for row in item_summary if row["is_safe"]]

    summary_obj = ConstraintBatchSummary(
        min_confidence=min_confidence,
        safe_items=safe_items,
        item_summary=item_summary,
        total_checks=len(results),
    )
    return summary_obj.model_dump()


def build_client():
    api_key = os.getenv("OPENAI_API_KEY")
    competition = os.getenv("COMPETITION_MODE", "0").strip() not in {"0", "false", "False"}
    if api_key:
        from datapizza.clients.openai import OpenAIClient
        return OpenAIClient(
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )
    if competition:
        raise RuntimeError(
            "COMPETITION_MODE attivo ma OPENAI_API_KEY mancante! "
            "Imposta la key nel .env prima di lanciare in modalità gara."
        )
    from datapizza.clients.mock_client import MockClient
    return MockClient()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def hybrid_search(query: str, top_k: int = 5) -> str:
    """
    Cerca nei documenti indicizzati usando ricerca ibrida (BM25 + semantica).
    Usa questo tool ogni volta che hai bisogno di trovare informazioni nei documenti.
    Puoi chiamarlo più volte con query diverse per esplorare aspetti differenti.

    Args:
        query: la query di ricerca in linguaggio naturale
        top_k: numero di risultati da ritornare (default 5, max 10)

    Returns:
        Lista di chunk rilevanti con fonte e score di rilevanza.
    """
    from src.retrieval import hybrid_search as _hybrid_search
    results = _hybrid_search(query, top_k=min(top_k, 10))
    if not results:
        return "Nessun documento trovato per questa query. Prova a riformularla."
    return "\n\n---\n\n".join(
        f"[Fonte: {r['source']} | Score: {r['score']:.3f}]\n{r['text']}"
        for r in results
    )


@tool
def get_document(source: str) -> str:
    """
    Recupera il contenuto completo di un documento specifico per nome file.
    Usa questo tool quando hai trovato un documento rilevante tramite hybrid_search
    e vuoi approfondire il suo contenuto completo.

    Args:
        source: nome del file documento (es. 'menu_galattico.pdf')

    Returns:
        Tutto il testo del documento.
    """
    from src.retrieval import get_document_chunks

    rows = get_document_chunks(source)
    if not rows:
        return f"Documento '{source}' non trovato nel corpus."
    return f"[Documento: {source}]\n\n" + "\n\n".join(row["text"] for row in rows)


@tool
def check_constraint(item: str, constraint: str) -> str:
    """
    Verifica se un item (piatto, ingrediente, prodotto) rispetta un vincolo specifico.
    Usa questo tool per validare ogni candidato rispetto ai requisiti della query
    prima di includerlo nella risposta finale.

    Args:
        item: nome del piatto o ingrediente da verificare
        constraint: descrizione del vincolo (es. 'senza glutine', 'vegano', 'normativa X')

    Returns:
        'CONFORME' o 'NON CONFORME' con motivazione.
    """
    result = ConstraintCheckResult(**_check_constraint_data(item=item, constraint=constraint))
    return result.model_dump_json()


@tool
def check_constraints_batch(items: list, constraints: list, min_confidence: float = 0.6) -> str:
    """
    Valida piu' item contro piu' vincoli e ritorna una sintesi pronta per l'orchestratore.

    Args:
        items: lista item candidati
        constraints: lista vincoli da validare su ogni item
        min_confidence: soglia minima per considerare affidabile un "CONFORME"

    Returns:
        JSON con risultati dettagliati + safe_items aggregati.
    """
    norm_items = []
    for item in items:
        item_text = str(item).strip()
        if item_text and item_text not in norm_items:
            norm_items.append(item_text)

    norm_constraints = []
    for constraint in constraints:
        rule = str(constraint).strip()
        if rule and rule not in norm_constraints:
            norm_constraints.append(rule)

    min_conf = max(0.0, min(1.0, min_confidence))
    cache_key = json.dumps(
        {
            "items": norm_items,
            "constraints": norm_constraints,
            "min_confidence": round(min_conf, 3),
            "fingerprint": _batch_cache_fingerprint(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    ttl = _batch_cache_ttl_seconds()
    now = time.time()
    if ttl > 0:
        cached = _BATCH_CACHE.get(cache_key)
        if cached and (now - cached[0]) <= ttl:
            _append_trace_event(
                "constraint_batch_cache_hit",
                {
                    "n_items": len(norm_items),
                    "n_constraints": len(norm_constraints),
                    "min_confidence": min_conf,
                },
            )
            return cached[1]

    if not norm_items or not norm_constraints:
        empty = ConstraintBatchResponse(
            results=[],
            summary=ConstraintBatchSummary(
                min_confidence=min_conf,
                safe_items=[],
                item_summary=[],
                total_checks=0,
            ),
        )
        return empty.model_dump_json()

    results = []
    for item in norm_items:
        for constraint in norm_constraints:
            results.append(ConstraintCheckResult(**_check_constraint_data(item=item, constraint=constraint)).model_dump())

    summary = _summarize_batch_results(results, min_confidence=min_conf)
    response = ConstraintBatchResponse(results=results, summary=summary)
    _append_trace_event(
        "constraint_batch",
        {
            "n_items": len(norm_items),
            "n_constraints": len(norm_constraints),
            "safe_items": response.summary.safe_items,
            "min_confidence": response.summary.min_confidence,
        },
    )
    payload = response.model_dump_json()
    if ttl > 0:
        _BATCH_CACHE[cache_key] = (now, payload)
        # Limite soft per evitare crescita infinita.
        if len(_BATCH_CACHE) > 256:
            oldest_keys = sorted(_BATCH_CACHE.keys(), key=lambda k: _BATCH_CACHE[k][0])[:64]
            for key in oldest_keys:
                _BATCH_CACHE.pop(key, None)
    return payload


@tool(end=True)
def submit_answer(items: list) -> str:
    """
    Sottomette la risposta finale come lista di item.
    CHIAMARE SOLO quando sei completamente sicuro della risposta.
    NON chiamare prima di aver verificato tutti i vincoli della query.
    La risposta verrà valutata con Jaccard Similarity — includi SOLO item certi.

    Args:
        items: lista di nomi (piatti, prodotti, etc.) che rispondono alla query

    Returns:
        JSON della risposta finale.
    """
    return json.dumps({"answer": items}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM_PROMPT = """Sei il ValidatorAgent.
Valuti la conformita' di item rispetto a vincoli usando solo evidenze dai tool.
Regole:
- Usa `check_constraint` per verifiche puntuali.
- Usa `check_constraints_batch` se devi validare piu' item e/o piu' vincoli.
- In caso di evidenze contraddittorie, preferisci NON CONFORME.
- Rispondi sempre in JSON sintetico, senza testo narrativo superfluo.
"""

ORCHESTRATOR_SYSTEM_PROMPT = """Sei l'OrchestratorAgent per query complesse.
Operi in un loop ReAct: Ragiona -> Agisci -> Osserva -> Ragiona.

REGOLE FONDAMENTALI:
1. Prima di agire, pianifica: "Cosa devo trovare per rispondere a questa query?"
2. Usa hybrid_search più volte con angolazioni diverse se necessario
3. Se trovi un documento rilevante, approfondiscilo con get_document
4. Per validare i vincoli usa SEMPRE il tool agentico `validator`
5. Chiedi al validator preferibilmente verifiche batch quando hai piu' candidati/vincoli
6. Includi nella risposta SOLO item di cui sei certo
7. Termina SEMPRE chiamando submit_answer con la lista finale, mai rispondere in prosa
8. Interpreta l'output JSON del validator: usa status, confidence ed evidence

STRATEGIA DI RICERCA:
- Inizia con ricerche generali, poi raffina
- Cerca anche le eccezioni e le esclusioni
- Se la query menziona vincoli multipli, verifica ognuno separatamente
- In caso di dubbio su un item, escludilo dalla risposta
- Se confidence < 0.6 o evidenze contraddittorie, considera l'item NON sicuro
"""


def build_validator_agent(client) -> Agent:
    is_mock = client.__class__.__name__ == "MockClient"
    return Agent(
        name="validator",
        client=client,
        tools=[check_constraint, check_constraints_batch],
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        planning_interval=0 if is_mock else 1,
        max_steps=8,
    )


def build_agent() -> Agent:
    client = build_client()
    is_mock = client.__class__.__name__ == "MockClient"
    orchestrator = Agent(
        name="orchestrator",
        client=client,
        tools=[hybrid_search, get_document, submit_answer],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        # MockClient non supporta bene planning strutturato: fallback a 0.
        planning_interval=0 if is_mock else 1,
        max_steps=15,             # evita loop infiniti
        terminate_on_text=False,  # DEVE usare submit_answer
    )
    orchestrator.can_call(build_validator_agent(client))
    return orchestrator


def _infer_constraints_from_query(query: str) -> list[str]:
    text = _normalize_text(query)
    inferred = []
    if "senza glutine" in text or "gluten free" in text or "celiach" in text:
        inferred.append("senza glutine")
    if "vegano" in text or "vegan" in text:
        inferred.append("vegano")
    if "vegetar" in text:
        inferred.append("vegetariano")
    if "senza lattosio" in text or "lactose" in text:
        inferred.append("senza lattosio")
    return inferred


def run_structured_orchestration(request: dict | StructuredOrchestrationRequest) -> dict:
    """
    Orchestrazione schema-first senza dipendere dalla policy del LLM.
    Utile come PoC/demo e fallback pre-hackathon.
    """
    if isinstance(request, StructuredOrchestrationRequest):
        req = request
    else:
        req = StructuredOrchestrationRequest(**request)

    candidates = [c.strip() for c in req.candidates if c and c.strip()]
    constraints = [c.strip() for c in req.constraints if c and c.strip()]

    if not constraints:
        constraints = _infer_constraints_from_query(req.query)

    if not candidates:
        from src.retrieval import suggest_candidates

        candidates = suggest_candidates(req.query, top_k_docs=10, max_candidates=25)

    if not constraints:
        # Nessun vincolo esplicito: restituiamo i candidati ranked direttamente.
        # Non ha senso validare contro vincoli inesistenti (uccide il recall).
        batch = ConstraintBatchResponse(
            results=[],
            summary=ConstraintBatchSummary(
                min_confidence=req.min_confidence,
                safe_items=candidates,
                item_summary=[
                    ItemValidationSummary(
                        item=c, is_safe=True, failed_constraints=[], passed_constraints=[]
                    ).model_dump()
                    for c in candidates
                ],
                total_checks=0,
            ),
        )
    else:
        batch_json = check_constraints_batch(
            items=candidates,
            constraints=constraints,
            min_confidence=req.min_confidence,
        )
        try:
            batch = ConstraintBatchResponse.model_validate_json(batch_json)
        except ValidationError:
            batch = ConstraintBatchResponse(
                results=[],
                summary=ConstraintBatchSummary(
                    min_confidence=req.min_confidence,
                    safe_items=[],
                    item_summary=[],
                    total_checks=0,
                ),
            )

    result = StructuredOrchestrationResult(
        request_id=str(uuid.uuid4()),
        query=req.query,
        answer=batch.summary.safe_items,
        constraints=constraints,
        candidates=candidates,
        batch=batch,
    )
    _append_trace_event(
        "structured_orchestration",
        {
            "request_id": result.request_id,
            "query": req.query,
            "n_candidates": len(candidates),
            "n_constraints": len(constraints),
            "answer": result.answer,
        },
    )
    return result.model_dump()


def _extract_answer_from_result(result) -> list[str]:
    """
    Fallback policy conservativa:
    se l'output non e' JSON valido con lista `answer`, ritorna [].
    """
    if not result or not getattr(result, "text", None):
        return []
    try:
        data = json.loads(result.text)
    except json.JSONDecodeError:
        return []

    raw_answer = data.get("answer", [])
    if not isinstance(raw_answer, list):
        return []

    normalized = []
    seen = set()
    for item in raw_answer:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(query: str) -> list[str]:
    """
    Esegue il pipeline agentico su una query.

    Returns:
        Lista di item che rispondono alla query (per scoring Jaccard).
    """
    agent = build_agent()
    result = agent.run(query)
    answer = _extract_answer_from_result(result)
    # Fallback opzionale per demo/offline: se l'agente non produce output strutturato.
    use_fallback = os.getenv("ENABLE_STRUCTURED_FALLBACK", "1").strip() not in {"0", "false", "False"}
    if not answer and use_fallback:
        structured = run_structured_orchestration(
            {
                "query": query,
                "constraints": [],
                "candidates": [],
                "min_confidence": _validator_min_confidence(),
            }
        )
        answer = structured.get("answer", [])
    return answer


def run_with_trace(query: str) -> dict:
    """
    Esegue la query e ritorna anche metadati utili per debug locale pre-hackathon.
    """
    agent = build_agent()
    result = agent.run(query)
    answer = _extract_answer_from_result(result)
    mode: Literal["agent", "structured_poc"] = "agent"

    use_fallback = os.getenv("ENABLE_STRUCTURED_FALLBACK", "1").strip() not in {"0", "false", "False"}
    if not answer and use_fallback:
        structured = run_structured_orchestration(
            {
                "query": query,
                "constraints": [],
                "candidates": [],
                "min_confidence": _validator_min_confidence(),
            }
        )
        answer = structured.get("answer", [])
        mode = "structured_poc"

    trace_obj = RunTrace(
        request_id=str(uuid.uuid4()),
        query=query,
        answer=answer,
        tools_used=getattr(result, "tools_used", []) if result else [],
        usage=str(getattr(result, "usage", "")) if result else "",
        raw_text=getattr(result, "text", "") if result else "",
        mode=mode,
    )
    trace = trace_obj.model_dump()
    _append_trace_event("run_trace", trace)
    return trace


if __name__ == "__main__":
    query = input("Query: ")
    answer = run(query)
    print(f"\nRisposta: {answer}")
    print(f"N. item: {len(answer)}")
