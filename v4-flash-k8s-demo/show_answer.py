#!/usr/bin/env python3
"""Print the model's answer from a saved /v1/chat/completions response.

    python3 show_answer.py /tmp/just-ask.json

A separate file rather than `python3 -c '...'` inside the justfile: multi-line inline Python
has broken the justfile parser four times, because continuation lines land at column 0 and
`just` tokenises them as recipe syntax. The comment saying so was already in the repo; this
is it being obeyed.

Reports what actually came back instead of raising. `just ask` used to die with a raw
JSONDecodeError traceback whenever it was pointed at a Service that nothing was serving —
which, during strategy C, was its own default.
"""
import json
import sys

if len(sys.argv) < 2:
    sys.exit("usage: show_answer.py <response.json>")

try:
    raw = open(sys.argv[1]).read()
except OSError as exc:
    sys.exit(f"could not read the response: {exc}")

if not raw.strip():
    sys.exit("empty response body — that endpoint accepted the connection but said nothing")

try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    sys.exit(f"reply was not JSON (first 200 chars): {raw[:200]!r}")

if "choices" not in payload:
    sys.exit(f"no choices in reply: {json.dumps(payload)[:300]}")

message = payload["choices"][0].get("message", {})
text = (message.get("content") or message.get("reasoning_content") or "").strip()
print(text or "(the model returned an empty message)")
print("--", payload.get("usage"))
