from __future__ import annotations

import json

from muse2api.core.prompt import followup_text, message_turns
from muse2api.core.tools import apply_tools, parse_reply

TOOLS = [{"type": "function", "function": {
    "name": "terminal", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


def test_parse_reply_call_and_text():
    content, calls = parse_reply('CALL {"name": "terminal", "arguments": {"command": "df -h"}}', TOOLS)
    assert content is None
    assert calls[0]["function"]["name"] == "terminal"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "df -h"}
    # unknown name, broken json and plain text stay text
    for text in ('CALL {"name": "rm", "arguments": {}}', "CALL {oops", "Halo!"):
        assert parse_reply(text, TOOLS) == (text, [])
    # fenced, multi-line JSON, two calls
    text = '```\nCALL {"name": "terminal",\n "arguments": {"command": "ls"}}\n```\nCALL {"name": "terminal", "arguments": "{\\"command\\": \\"pwd\\"}"}'
    content, calls = parse_reply(text, TOOLS)
    assert content is None and [json.loads(c["function"]["arguments"])["command"] for c in calls] == ["ls", "pwd"]


def test_apply_tools_history_and_followup():
    first = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "disk?"}]
    second = first + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
         "function": {"name": "terminal", "arguments": '{"command": "df -h"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "68% used"}]
    a, b = apply_tools(first, TOOLS), apply_tools(second, TOOLS)
    assert a[-1]["role"] == "system" and "terminal(command: string)" in a[-1]["content"]
    assert b[2] == {"role": "assistant", "content": 'CALL {"name": "terminal", "arguments": {"command": "df -h"}}'}
    assert b[3] == {"role": "user", "content": "RESULT terminal\n68% used"}
    # without tools the tool result still continues an open muse thread
    assert followup_text(message_turns(apply_tools(first, None)),
                         message_turns(apply_tools(second, None))) == "RESULT terminal\n68% used"
    assert apply_tools(first, None) == first


def test_chat_tool_calls(client, auth, monkeypatch):
    from muse2api.drivers.mock import MockDriver

    async def reply(self, account, req):
        yield 'CALL {"name": "terminal", "arguments": {"command": "df -h"}}'

    monkeypatch.setattr(MockDriver, "chat_stream", reply)
    msgs = [{"role": "user", "content": 'CALL {"name": "terminal", "arguments": {"command": "df -h"}}'}]
    r = client.post("/v1/chat/completions", headers=auth, json={"messages": msgs, "tools": TOOLS})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "terminal"
    with client.stream("POST", "/v1/chat/completions", headers=auth,
                       json={"messages": msgs, "tools": TOOLS, "stream": True}) as r:
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    first = json.loads(lines[0])["choices"][0]["delta"]
    assert first["tool_calls"][0]["index"] == 0
    assert json.loads(lines[1])["choices"][0]["finish_reason"] == "tool_calls"
    # tool_choice none: plain text
    r = client.post("/v1/chat/completions", headers=auth,
                    json={"messages": msgs, "tools": TOOLS, "tool_choice": "none"})
    assert r.json()["choices"][0]["finish_reason"] == "stop"
