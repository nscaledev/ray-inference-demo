#!/usr/bin/env python3
"""Build a /v1/chat/completions request body.

    python3 make_request.py <model> <prompt> [max_tokens]

Exists for the same reason as show_answer.py: inline multi-line Python in the justfile has
broken its parser four times. Also guarantees the prompt is JSON-escaped, so a prompt
containing a quote or a colon cannot corrupt the request or the recipe.
"""
import json
import sys


# Duplicated in bench.py, longctx.py and make_request.py on purpose: the in-cluster Jobs mount
# a ConfigMap containing exactly ONE file, so a shared module would not be present at runtime.
# Six lines beats adding a second --from-file to two Job templates.
def describe_call(url, body, prompt_key="messages", limit=100):
    """One auditable line per API call: what was sent, where, with the prompt trimmed.

    Printed so the audience can see the demo is a plain OpenAI-compatible POST and nothing is
    hidden in the harness — the whole talk asks people to rerun this, so the request has to be
    visible. Newlines are collapsed and the prompt is truncated to `limit` chars with its true
    length shown, because a 1M-token prompt is ~5.8 MB and must never be echoed in full.
    """
    msgs = body.get(prompt_key) or []
    prompt = (msgs[0].get("content", "") if msgs else "")
    flat = " ".join(prompt.split())
    shown = flat[:limit] + ("…" if len(flat) > limit else "")
    opts = " ".join(f"{k}={body[k]}" for k in ("max_tokens", "temperature", "stream") if k in body)
    return (f"→ POST {url}\n"
            f"    model={body.get('model')} {opts}\n"
            f'    prompt[{len(prompt):,} chars]: "{shown}"')


if len(sys.argv) < 3:
    sys.exit("usage: make_request.py <model> <prompt> [max_tokens]")
model, prompt = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200
body = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0.0,          # deterministic, so a garbled answer means a broken engine
}
# The request goes to stderr so the caller sees it while stdout stays pure JSON for curl -d.
url = sys.argv[4] if len(sys.argv) > 4 else "/v1/chat/completions"
print(describe_call(url, body), file=sys.stderr)
print(json.dumps(body))
