"""
Chainlit demo app per Hackapizza 2.0.

Obiettivo:
- offrire una UX chat-first per mostrare orchestration agentica
- riusare lo stesso backend di Streamlit (agent + structured PoC + validator batch)
"""

import json
import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Any

import chainlit as cl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import check_constraints_batch, run_structured_orchestration, run_with_trace


DEFAULT_SETTINGS = {
    "mode": "structured_poc",
    "constraints": ["senza glutine", "vegano"],
    "candidates": ["Pizza Nebulosa", "Insalata Orbitale", "Tofu Meteorico", "Risotto Lunare"],
    "min_confidence": 0.6,
    "embedding_mode": "mock",
    "rerank_mode": "heuristic",
    "trace_enabled": True,
    "structured_fallback": True,
    "validator_top_k": 4,
    "validator_evidence_limit": 8,
    "timeout_seconds": 25,
}

QUICK_QUERIES = {
    "demo_base": "Quali piatti sono senza glutine e vegani?",
    "demo_allergeni": "Seleziona solo piatti senza glutine, senza lattosio e senza frutta a guscio.",
    "demo_stress": "Dammi i piatti vegani, senza glutine e adatti a celiaci.",
}


def _normalize_constraints(raw: str) -> list[str]:
    out = []
    for part in raw.split(","):
        item = part.strip()
        if item and item not in out:
            out.append(item)
    return out


def _normalize_candidates(raw: str) -> list[str]:
    out = []
    for part in raw.replace(";", "\n").splitlines():
        item = part.strip()
        if item and item not in out:
            out.append(item)
    return out


def _set_runtime_env(settings: dict[str, Any]):
    os.environ["EMBEDDING_MODE"] = str(settings["embedding_mode"])
    os.environ["RERANK_MODE"] = str(settings["rerank_mode"])
    os.environ["VALIDATOR_MIN_CONFIDENCE"] = str(settings["min_confidence"])
    os.environ["VALIDATOR_TOP_K"] = str(settings["validator_top_k"])
    os.environ["VALIDATOR_EVIDENCE_LIMIT"] = str(settings["validator_evidence_limit"])
    os.environ["TRACE_ENABLED"] = "1" if settings["trace_enabled"] else "0"
    os.environ["ENABLE_STRUCTURED_FALLBACK"] = "1" if settings["structured_fallback"] else "0"
    os.environ["APP_TIMEOUT_SECONDS"] = str(settings["timeout_seconds"])
    os.environ.setdefault("HYBRID_CACHE_TTL_SECONDS", "90")
    os.environ.setdefault("BATCH_CACHE_TTL_SECONDS", "120")
    os.environ.setdefault("TRACE_LOG_PATH", "data/traces/agent_trace.jsonl")


async def _run_with_timeout(label: str, fn, *args, **kwargs):
    timeout = max(3, int(_get_settings().get("timeout_seconds", 25)))
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=timeout)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result, elapsed_ms
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Timeout su `{label}` dopo {timeout}s. "
            "Riduci top_k/evidence_limit o passa a /mode structured."
        ) from exc


def _quick_actions() -> list[cl.Action]:
    return [
        cl.Action(
            name="set_mode",
            payload={"mode": "structured_poc"},
            label="Mode Structured",
            tooltip="Imposta la modalita structured_poc",
        ),
        cl.Action(
            name="set_mode",
            payload={"mode": "agent"},
            label="Mode Agent",
            tooltip="Imposta la modalita agent",
        ),
        cl.Action(
            name="run_query",
            payload={"query": QUICK_QUERIES["demo_base"]},
            label="Run Demo Base",
            tooltip="Esegue query demo standard",
        ),
        cl.Action(
            name="run_query",
            payload={"query": QUICK_QUERIES["demo_stress"]},
            label="Run Demo Stress",
            tooltip="Esegue query multi-vincolo",
        ),
        cl.Action(
            name="run_query",
            payload={"query": QUICK_QUERIES["demo_allergeni"]},
            label="Run Demo Allergeni",
            tooltip="Esegue query allergeni stretti",
        ),
        cl.Action(
            name="show_settings",
            payload={},
            label="Show Settings",
            tooltip="Mostra configurazione corrente",
        ),
        cl.Action(
            name="show_help",
            payload={},
            label="Help",
            tooltip="Mostra comandi disponibili",
        ),
    ]


async def _send_quick_actions():
    await cl.Message(
        content="Azioni rapide disponibili:",
        actions=_quick_actions(),
    ).send()


def _get_settings() -> dict[str, Any]:
    settings = cl.user_session.get("settings")
    if not settings:
        settings = DEFAULT_SETTINGS.copy()
        cl.user_session.set("settings", settings)
    return settings


def _save_settings(settings: dict[str, Any]):
    cl.user_session.set("settings", settings)
    _set_runtime_env(settings)


def _compact_summary(payload: dict[str, Any]) -> str:
    mode = payload.get("mode", "n/a")
    answer = payload.get("answer", [])
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    safe_items = summary.get("safe_items", [])
    total_checks = summary.get("total_checks", 0)
    tools = payload.get("tools_used", [])
    elapsed_ms = payload.get("elapsed_ms")
    fallback_used = payload.get("fallback_used", False)

    lines = [
        f"Modalita: `{mode}`",
        f"Risposta ({len(answer)}): `{answer}`",
        f"Safe items: `{safe_items}`",
        f"Constraint checks: `{total_checks}`",
    ]
    if elapsed_ms is not None:
        lines.append(f"Tempo: `{elapsed_ms} ms`")
    if fallback_used:
        lines.append("Fallback: `structured_poc attivato`")
    if tools:
        lines.append(f"Tools usati: `{tools}`")
    return "\n".join(f"- {line}" for line in lines)


def _help_text() -> str:
    return (
        "Comandi disponibili:\n"
        "- `/mode structured` oppure `/mode agent`\n"
        "- `/constraints senza glutine, vegano`\n"
        "- `/candidates Pizza A; Pizza B; Insalata C`\n"
        "- `/confidence 0.6`\n"
        "- `/topk 4`\n"
        "- `/evidence_limit 8`\n"
        "- `/timeout 25`\n"
        "- `/embedding mock|local|openai`\n"
        "- `/rerank heuristic|cross_encoder|none`\n"
        "- `/trace on|off`\n"
        "- `/fallback on|off`\n"
        "- `/settings` (stato corrente)\n"
        "- `/reset` (ripristina default)\n\n"
        "Ogni messaggio non comando viene trattato come query da eseguire.\n"
        "In alternativa puoi usare i bottoni quick action in chat."
    )


def _extract_tool_entries(tools_raw: Any) -> list[dict[str, Any]]:
    if not isinstance(tools_raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for idx, tool in enumerate(tools_raw, start=1):
        if isinstance(tool, dict):
            name = str(tool.get("name") or tool.get("tool") or f"tool_{idx}")
            tool_input = tool.get("input") or tool.get("args") or {}
            tool_output = tool.get("output") or tool.get("result") or {}
            entries.append(
                {
                    "name": name,
                    "input": tool_input,
                    "output": tool_output,
                }
            )
        else:
            entries.append(
                {
                    "name": str(tool),
                    "input": {},
                    "output": {},
                }
            )
    return entries


async def _emit_tool_timeline(tools_raw: Any):
    entries = _extract_tool_entries(tools_raw)
    for idx, entry in enumerate(entries, start=1):
        async with cl.Step(
            name=f"Tool {idx}: {entry['name']}",
            type="tool",
            show_input="json",
            default_open=False,
        ) as step:
            step.input = entry.get("input", {})
            step.output = entry.get("output", {})


def _build_explainability_rows(results: list[dict], min_confidence: float) -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    risk_rows: list[dict] = []
    for row in results:
        evidence = row.get("evidence", []) if isinstance(row.get("evidence"), list) else []
        top = evidence[0] if evidence else {}
        line = {
            "item": row.get("item", ""),
            "constraint": row.get("constraint", ""),
            "status": row.get("status", "NON CONFORME"),
            "confidence": round(float(row.get("confidence", 0.0)), 3),
            "reason": row.get("reason", ""),
            "evidence": top.get("evidence_id", ""),
            "signal": top.get("signal", "neutral"),
        }
        all_rows.append(line)
        if line["status"] != "CONFORME" or line["confidence"] < min_confidence:
            risk_rows.append(line)
    return all_rows, risk_rows


async def _emit_explainability(results: list[dict], min_confidence: float):
    all_rows, risk_rows = _build_explainability_rows(results, min_confidence=min_confidence)
    if not all_rows:
        return
    async with cl.Step(
        name="Explainability",
        type="tool",
        show_input="json",
        default_open=False,
    ) as step:
        step.input = {"checks": len(all_rows), "min_confidence": min_confidence}
        step.output = {"risk_rows": risk_rows[:8], "all_rows_preview": all_rows[:12]}
    if risk_rows:
        await cl.Message(
            content="Vincoli a rischio:\n```json\n"
            + json.dumps(risk_rows[:8], ensure_ascii=False, indent=2)
            + "\n```"
        ).send()


async def _execute_query(query: str, trigger: str = "chat"):
    settings = _get_settings()
    _set_runtime_env(settings)
    await cl.Message(content=f"Esecuzione in corso (`{settings['mode']}`)...").send()

    try:
        if settings["mode"] == "agent":
            async with cl.Step(
                name="Agent Orchestration",
                type="run",
                show_input="json",
                default_open=True,
            ) as step:
                step.input = {
                    "query": query,
                    "trigger": trigger,
                    "mode": settings["mode"],
                    "embedding_mode": settings["embedding_mode"],
                    "rerank_mode": settings["rerank_mode"],
                    "min_confidence": settings["min_confidence"],
                }
                trace, elapsed_ms = await _run_with_timeout("run_with_trace", run_with_trace, query)
                step.output = {
                    "mode": trace.get("mode", "agent"),
                    "answer": trace.get("answer", []),
                    "usage": trace.get("usage", ""),
                    "tools_used_count": len(trace.get("tools_used", [])),
                    "elapsed_ms": elapsed_ms,
                }

            payload: dict[str, Any] = {
                "mode": trace.get("mode", "agent"),
                "query": query,
                "answer": trace.get("answer", []),
                "tools_used": trace.get("tools_used", []),
                "usage": trace.get("usage", ""),
                "raw_text": trace.get("raw_text", ""),
                "fallback_used": trace.get("mode") == "structured_poc",
                "elapsed_ms": elapsed_ms,
                "batch_results": [],
            }

            await _emit_tool_timeline(trace.get("tools_used", []))

            if settings["constraints"] and settings["candidates"]:
                async with cl.Step(
                    name="Validator Batch",
                    type="tool",
                    show_input="json",
                    default_open=False,
                ) as step:
                    step.input = {
                        "items": settings["candidates"],
                        "constraints": settings["constraints"],
                        "min_confidence": float(settings["min_confidence"]),
                    }
                    batch_json, batch_elapsed_ms = await _run_with_timeout(
                        "check_constraints_batch",
                        check_constraints_batch,
                        items=settings["candidates"],
                        constraints=settings["constraints"],
                        min_confidence=float(settings["min_confidence"]),
                    )
                    batch_obj = json.loads(batch_json)
                    payload["summary"] = batch_obj.get("summary", {})
                    payload["batch_results"] = batch_obj.get("results", [])
                    payload["batch_elapsed_ms"] = batch_elapsed_ms
                    step.output = {**payload["summary"], "elapsed_ms": batch_elapsed_ms}
                    await _emit_explainability(
                        payload["batch_results"],
                        min_confidence=float(settings["min_confidence"]),
                    )
        else:
            async with cl.Step(
                name="Structured Orchestration",
                type="run",
                show_input="json",
                default_open=True,
            ) as step:
                step.input = {
                    "query": query,
                    "constraints": settings["constraints"],
                    "candidates": settings["candidates"],
                    "min_confidence": float(settings["min_confidence"]),
                    "trigger": trigger,
                }
                result, elapsed_ms = await _run_with_timeout(
                    "run_structured_orchestration",
                    run_structured_orchestration,
                    {
                        "query": query,
                        "constraints": settings["constraints"],
                        "candidates": settings["candidates"],
                        "min_confidence": float(settings["min_confidence"]),
                    }
                )
                step.output = {
                    "answer": result.get("answer", []),
                    "constraints": result.get("constraints", []),
                    "candidates_count": len(result.get("candidates", [])),
                    "elapsed_ms": elapsed_ms,
                }

            batch = result.get("batch", {})
            payload = {
                "mode": "structured_poc",
                "query": query,
                "answer": result.get("answer", []),
                "constraints": result.get("constraints", []),
                "candidates": result.get("candidates", []),
                "summary": batch.get("summary", {}),
                "batch_results": batch.get("results", []),
                "elapsed_ms": elapsed_ms,
                "fallback_used": False,
            }

            async with cl.Step(
                name="Validator Summary",
                type="tool",
                show_input="json",
                default_open=False,
            ) as step:
                step.input = {
                    "safe_items": payload.get("summary", {}).get("safe_items", []),
                    "total_checks": payload.get("summary", {}).get("total_checks", 0),
                }
                step.output = payload.get("summary", {})
            await _emit_explainability(
                payload["batch_results"],
                min_confidence=float(settings["min_confidence"]),
            )
    except Exception as exc:
        await cl.Message(content=f"Errore durante l'esecuzione: `{exc}`").send()
        return

    if not payload.get("answer"):
        await cl.Message(content="Nessun item sicuro trovato con i vincoli correnti.").send()
    if payload.get("fallback_used"):
        await cl.Message(content="Fallback structured attivato durante run agent.").send()
    await cl.Message(content=_compact_summary(payload), actions=_quick_actions()).send()
    await cl.Message(content="```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```").send()


async def _handle_command(text: str):
    settings = _get_settings()
    command, _, arg = text.partition(" ")
    cmd = command.lower().strip()
    value = arg.strip()

    if cmd == "/help":
        await cl.Message(content=_help_text()).send()
        return

    if cmd == "/mode":
        if value.lower() not in {"structured", "agent"}:
            await cl.Message(content="Uso: `/mode structured` oppure `/mode agent`").send()
            return
        settings["mode"] = "structured_poc" if value.lower() == "structured" else "agent"
        _save_settings(settings)
        await cl.Message(content=f"Modalita impostata: `{settings['mode']}`").send()
        return

    if cmd == "/constraints":
        constraints = _normalize_constraints(value)
        settings["constraints"] = constraints
        _save_settings(settings)
        await cl.Message(content=f"Constraints aggiornati: `{constraints}`").send()
        return

    if cmd == "/candidates":
        candidates = _normalize_candidates(value)
        settings["candidates"] = candidates
        _save_settings(settings)
        await cl.Message(content=f"Candidates aggiornati ({len(candidates)}): `{candidates}`").send()
        return

    if cmd == "/confidence":
        try:
            threshold = float(value)
            if threshold < 0 or threshold > 1:
                raise ValueError
        except ValueError:
            await cl.Message(content="Uso: `/confidence <numero tra 0 e 1>`").send()
            return
        settings["min_confidence"] = round(threshold, 2)
        _save_settings(settings)
        await cl.Message(content=f"Min confidence: `{settings['min_confidence']}`").send()
        return

    if cmd == "/topk":
        try:
            top_k = int(value)
            if top_k < 1 or top_k > 20:
                raise ValueError
        except ValueError:
            await cl.Message(content="Uso: `/topk <intero 1..20>`").send()
            return
        settings["validator_top_k"] = top_k
        _save_settings(settings)
        await cl.Message(content=f"Validator top_k: `{top_k}`").send()
        return

    if cmd == "/evidence_limit":
        try:
            limit = int(value)
            if limit < 1 or limit > 40:
                raise ValueError
        except ValueError:
            await cl.Message(content="Uso: `/evidence_limit <intero 1..40>`").send()
            return
        settings["validator_evidence_limit"] = limit
        _save_settings(settings)
        await cl.Message(content=f"Validator evidence_limit: `{limit}`").send()
        return

    if cmd == "/timeout":
        try:
            timeout = int(value)
            if timeout < 3 or timeout > 180:
                raise ValueError
        except ValueError:
            await cl.Message(content="Uso: `/timeout <intero 3..180>`").send()
            return
        settings["timeout_seconds"] = timeout
        _save_settings(settings)
        await cl.Message(content=f"Timeout operativo: `{timeout}s`").send()
        return

    if cmd == "/embedding":
        if value not in {"mock", "local", "openai"}:
            await cl.Message(content="Uso: `/embedding mock|local|openai`").send()
            return
        settings["embedding_mode"] = value
        _save_settings(settings)
        await cl.Message(content=f"Embedding mode: `{value}`").send()
        return

    if cmd == "/rerank":
        if value not in {"heuristic", "cross_encoder", "none"}:
            await cl.Message(content="Uso: `/rerank heuristic|cross_encoder|none`").send()
            return
        settings["rerank_mode"] = value
        _save_settings(settings)
        await cl.Message(content=f"Rerank mode: `{value}`").send()
        return

    if cmd == "/trace":
        if value not in {"on", "off"}:
            await cl.Message(content="Uso: `/trace on|off`").send()
            return
        settings["trace_enabled"] = value == "on"
        _save_settings(settings)
        await cl.Message(content=f"Trace: `{value}`").send()
        return

    if cmd == "/fallback":
        if value not in {"on", "off"}:
            await cl.Message(content="Uso: `/fallback on|off`").send()
            return
        settings["structured_fallback"] = value == "on"
        _save_settings(settings)
        await cl.Message(content=f"Structured fallback: `{value}`").send()
        return

    if cmd == "/settings":
        await cl.Message(
            content="```json\n" + json.dumps(settings, ensure_ascii=False, indent=2) + "\n```"
        ).send()
        return

    if cmd == "/reset":
        settings = DEFAULT_SETTINGS.copy()
        _save_settings(settings)
        await cl.Message(content="Settings ripristinati ai default.").send()
        return

    await cl.Message(content="Comando non riconosciuto. Usa `/help`.").send()


@cl.on_chat_start
async def on_chat_start():
    settings = DEFAULT_SETTINGS.copy()
    _save_settings(settings)
    key_present = bool(os.getenv("OPENAI_API_KEY"))
    await cl.Message(
        content=(
            "Hackapizza Chainlit Console pronta.\n\n"
            f"- OPENAI_API_KEY: `{'presente' if key_present else 'assente'}`\n"
            "- Modalita default: `structured_poc`\n"
            "- Runtime default: `top_k=4`, `evidence_limit=8`, `timeout=25s`\n"
            "- Scrivi una query oppure usa `/help` per i comandi."
        )
    ).send()
    await _send_quick_actions()


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()
    if not text:
        await cl.Message(content="Messaggio vuoto. Inserisci una query o un comando `/help`.").send()
        return

    if text.startswith("/"):
        await _handle_command(text)
        return

    await _execute_query(text, trigger="chat")


@cl.action_callback("set_mode")
async def on_set_mode(action: cl.Action):
    mode = str(action.payload.get("mode", "")).strip()
    if mode not in {"structured_poc", "agent"}:
        await cl.Message(content="Payload mode non valido.").send()
        return
    settings = _get_settings()
    settings["mode"] = mode
    _save_settings(settings)
    await cl.Message(content=f"Modalita impostata: `{mode}`").send()


@cl.action_callback("run_query")
async def on_run_query(action: cl.Action):
    query = str(action.payload.get("query", "")).strip()
    if not query:
        await cl.Message(content="Query mancante nel payload action.").send()
        return
    await _execute_query(query, trigger="quick_action")


@cl.action_callback("show_settings")
async def on_show_settings(action: cl.Action):
    settings = _get_settings()
    await cl.Message(
        content="```json\n" + json.dumps(settings, ensure_ascii=False, indent=2) + "\n```"
    ).send()


@cl.action_callback("show_help")
async def on_show_help(action: cl.Action):
    await cl.Message(content=_help_text()).send()
