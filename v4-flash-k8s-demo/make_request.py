#!/usr/bin/env python3
"""Build a /v1/chat/completions request body.

    python3 make_request.py <model> <prompt> [max_tokens]

Exists for the same reason as show_answer.py: inline multi-line Python in the justfile has
broken its parser four times. Also guarantees the prompt is JSON-escaped, so a prompt
containing a quote or a colon cannot corrupt the request or the recipe.
"""
import json
import sys

if len(sys.argv) < 3:
    sys.exit("usage: make_request.py <model> <prompt> [max_tokens]")
model, prompt = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0.0,          # deterministic, so a garbled answer means a broken engine
}))
