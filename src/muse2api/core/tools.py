"""Prompt-level tool calling for the muse web chat.

muse.ai has no function-calling API, and its Hatch agent runs file/terminal asks in
its own VM when it sees them. A strict system block makes it answer with
``CALL {"name": ..., "arguments": {...}}`` lines instead; those are parsed back into
OpenAI ``tool_calls``. Tool history is replayed as CALL / RESULT text.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

_CALL = re.compile(r"CALL\s*(?=\{)")
NUDGE = ("Your last reply had no CALL line, so the program will treat it as the FINAL answer "
         "and stop. If the task still needs actions, reply now with only the CALL lines. If the "
         "task is really complete, reply with exactly: DONE")


def needs_nudge(messages: list[dict[str, Any]], text: str) -> bool:
    """Short text right after a tool result is usually narration ("now I will read...")."""
    return bool(messages) and messages[-1].get("role") == "tool" and len(text.strip()) < 400
_DECODER = json.JSONDecoder()


def _signature(fn: dict[str, Any]) -> str:
    params = fn.get("parameters") or {}
    required = set(params.get("required") or [])
    args = []
    for name, schema in (params.get("properties") or {}).items():
        typ = schema.get("type", "any") if isinstance(schema, dict) else "any"
        if isinstance(typ, list):
            typ = "|".join(map(str, typ))
        args.append(f"{name}{'' if name in required else '?'}: {typ}")
    desc = " ".join((fn.get("description") or "").split())[:300]
    return f"- {fn.get('name')}({', '.join(args)}): {desc}"


def tool_prompt(tools: list[dict[str, Any]]) -> str:
    fns = "\n".join(_signature(t.get("function") or t) for t in tools)
    return (
        "You are the text-only reasoning core of an external agent program. You cannot act "
        "yourself: do NOT use your own tools, sandbox, computer, code execution, browsing or "
        "files, and never claim you ran something. The program performs actions for you.\n"
        "To request an action, reply with CALL lines only, one per line:\n"
        'CALL {"name": "<function>", "arguments": {...}}\n'
        "Several independent actions may go in one reply as several CALL lines. The program "
        "replies with RESULT messages; then CALL again or give the final answer.\n"
        "A reply without any CALL line is treated as your FINAL answer and the task ends. So "
        "never announce or describe the next step in words (no \"now I will read...\"): if "
        "more actions are needed, send the CALL lines right away. Only write plain text when "
        "the whole task is done or no action is needed.\n"
        f"Functions:\n{fns}"
    )


def _call_line(call: dict[str, Any]) -> str:
    fn = call.get("function") or {}
    args = fn.get("arguments") or "{}"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return "CALL " + json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False)


def apply_tools(messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Rewrite tool history as plain text turns and prepend the tool block."""
    names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            lines = [c for c in [msg.get("content")] if isinstance(c, str) and c.strip()]
            for call in msg["tool_calls"]:
                names[call.get("id", "")] = (call.get("function") or {}).get("name", "")
                lines.append(_call_line(call))
            out.append({"role": "assistant", "content": "\n".join(lines)})
        elif role == "tool":
            name = msg.get("name") or names.get(msg.get("tool_call_id", ""), "")
            content = msg.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            # role user so a follow-up on an open thread sends only the result
            out.append({"role": "user", "content": f"RESULT {name}\n{content}".strip()})
        else:
            out.append(msg)
    if tools:
        # Last, so a long agent system prompt and big tool results don't bury it.
        # ponytail: this breaks the follow-up prefix match, so every step resends the full
        # prompt in a fresh thread; fine while muse.ai accepts the size.
        out.append({"role": "system", "content": tool_prompt(tools)})
    return out


def parse_reply(text: str, tools: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Split a reply into leftover text and OpenAI ``tool_calls``."""
    known = {(t.get("function") or t).get("name") for t in tools}
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for m in _CALL.finditer(text):
        try:
            obj, end = _DECODER.raw_decode(text, m.end())
        except ValueError:
            continue
        if not isinstance(obj, dict) or obj.get("name") not in known:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        spans.append((m.start(), end))
        fn = {"name": obj["name"], "arguments": json.dumps(args, ensure_ascii=False)}
        if any(c["function"] == fn for c in calls):  # muse sometimes repeats a line
            continue
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": fn})
    if not calls:
        return text, []
    rest = text
    for start, end in reversed(spans):
        rest = rest[:start] + rest[end:]
    rest = rest.strip().strip("`").strip()
    return (rest or None), calls
