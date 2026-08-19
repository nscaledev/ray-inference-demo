#!/usr/bin/env python3
"""Build a /v1/chat/completions request body.

    python3 make_request.py <model> <prompt> [max_tokens]

Exists for the same reason as show_answer.py: inline multi-line Python in the justfile has
broken its parser four times. Also guarantees the prompt is JSON-escaped, so a prompt
containing a quote or a colon cannot corrupt the request or the recipe.
"""
import json
import sys


# Copied verbatim in bench.py and longctx.py, which is deliberate: each runs in-cluster from a
# ConfigMap holding exactly ONE file, so a shared module would not exist at runtime. This third
# copy in make_request.py is NOT covered by that argument -- it runs on the laptop -- and the
# three had already drifted apart (this one grew a prompt_key parameter the others lack), which
# is how triplication normally fails. Kept identical to the other two on purpose: a shared
# module importable by only one of three callers buys nothing.
def describe_call(url, body, limit=100):
    """One auditable line per API call: what was sent, where, with the prompt trimmed.

    Printed so the audience can see the demo is a plain OpenAI-compatible POST with nothing
    hidden in the harness — the talk asks people to rerun this, so the request has to be
    visible. Newlines are collapsed and the prompt is truncated to `limit` chars with its true
    length shown, because a 1M-token prompt is ~5.8 MB and must never be echoed in full.
    """
    msgs = body.get("messages") or []
    prompt = (msgs[0].get("content", "") if msgs else "")
    flat = " ".join(prompt.split())
    shown = flat[:limit] + ("\u2026" if len(flat) > limit else "")
    opts = " ".join(f"{k}={body[k]}" for k in ("max_tokens", "temperature", "stream") if k in body)
    return (f"\u2192 POST {url}\n"
            f"    model={body.get('model')} {opts}\n"
            f'    prompt[{len(prompt):,} chars]: "{shown}"')


if len(sys.argv) < 3:
    sys.exit("usage: make_request.py <model> <prompt> [max_tokens] [url] [temperature]")
model, prompt = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200
# Temperature is the CALLER's choice, because the two uses want opposite things:
#   diagnostics (is this engine sane? is it generating yet?) want 0.0 -- deterministic, so a
#   garbled answer means a broken engine rather than an unlucky sample;
#   the live "it talks" beat wants the model card's 1.0 (README: temperature 1.0, top_p 0.95
#   agentic / 1.0 otherwise). Greedy decoding on a model specified for 1.0 is the setting most
#   likely to produce a visibly repetitive answer in front of an audience.
temperature = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
body = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": temperature,
}
# The request goes to stderr so the caller sees it while stdout stays pure JSON for curl -d.
url = sys.argv[4] if len(sys.argv) > 4 else "/v1/chat/completions"
print(describe_call(url, body), file=sys.stderr)
print(json.dumps(body))
