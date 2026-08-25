#!/usr/bin/env python3
"""Print the model's answer, then its latency and token counts, in one consistent line.

    python3 show_answer.py /tmp/just-ask.json

A separate file rather than `python3 -c '...'` inside the justfile: multi-line inline Python
puts continuation lines at column 0, where `just` tokenises them as recipe syntax.

Reports what actually came back instead of raising, so pointing this at a Service that nothing
is serving prints a sentence rather than a JSONDecodeError traceback.

WHY THIS SAYS "latency" AND NOT "TTFT". The request is not streamed, so the server buffers
until generation finishes and time-to-first-byte IS time-to-last-token. Labelling that TTFT
would be a number whose name lies about what it measures -- the same class of error the output
sanity gate exists to catch. bench.py streams and therefore reports a real TTFT and TPOT; these
ask recipes report end-to-end latency, which is the honest thing they can measure.

The timing comes from curl's own %{time_total}, measured INSIDE the pod, so it excludes
kubectl-exec setup. The justfile appends it as a trailing __TIMING__ line.
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

# Split off curl's trailing timing line so the rest still parses as pure JSON.
latency = None
body_lines = []
for line in raw.splitlines():
    if line.startswith("__TIMING__"):
        try:
            latency = float(line.split()[1])
        except (IndexError, ValueError):
            pass
        continue
    body_lines.append(line)
raw = "\n".join(body_lines)

try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    sys.exit(f"reply was not JSON (first 200 chars): {raw[:200]!r}")

if "choices" not in payload:
    sys.exit(f"no choices in reply: {json.dumps(payload)[:300]}")

message = payload["choices"][0].get("message", {})
text = (message.get("content") or message.get("reasoning_content") or "").strip()
print(text or "(the model returned an empty message)")

# One line, same vocabulary the rest of the repo uses: seconds, tokens, tok/s.
usage = payload.get("usage") or {}
prompt_tok = usage.get("prompt_tokens")
completion_tok = usage.get("completion_tokens")
bits = []
if latency is not None:
    bits.append(f"latency {latency:.2f}s")
if latency and completion_tok:
    bits.append(f"{completion_tok / latency:.1f} tok/s")
if prompt_tok is not None:
    bits.append(f"{prompt_tok} prompt + {completion_tok or 0} completion tokens")
print("-- " + " · ".join(bits) if bits else "-- (no timings available)")
