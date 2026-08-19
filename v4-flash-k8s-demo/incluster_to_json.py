#!/usr/bin/env python3
"""Turn the in-cluster bench pod's log into a metrics/*.json file.

    kubectl -n <ns> logs job/bench-in-cluster \
      | python3 incluster_to_json.py NAME SVC CONC [RUNS] [KV_TOKENS] [LABEL] [CLAIM]

Extra args are optional so older 3-arg callers keep working. They exist because the Job's
pod is gone by the time we parse its log, so anything we do not carry through here is lost:
the label and claim sed'd into the Job template, the real number of single-stream runs
(0 means the single-stream pass was skipped), and the KV cache size from the engine.

`kubectl cp` can't be used for this: the bench pod has already exited by the time the Job
reports complete, so its filesystem is gone. The log is the durable artefact.
"""
import json
import re
import sys

name, svc, conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
# "0" is what the scrape echoes when it FAILS, and "0".isdigit() is true — recording that
# as a measured zero would put a confident "0" in the comparison table.
kv_tokens = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else None
kv_tokens = kv_tokens or None
label = sys.argv[6] if len(sys.argv) > 6 else f"measured in-cluster via svc/{svc}"
claim = sys.argv[7] if len(sys.argv) > 7 else "no port-forward in the path"
log = sys.stdin.read()


def grab(pattern, cast=float):
    m = re.search(pattern, log)
    return cast(m.group(1)) if m else None


# The sanity line is the difference between "fast" and "fast AND right". If bench.py could
# not get a correct answer to 2+2, every timing in this file describes garbage.
m = re.search(r"sanity: .* -> (.*) \[(PASS|FAIL)\]", log)
sane = (m.group(2) == "PASS") if m else None
sane_answer = m.group(1) if m else None

out = {
    "name": name,
    "output_sane": sane,
    "sanity_answer": sane_answer,
    "label": label,
    "claim": claim,
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    # bench.py now prints max_tokens on its loaded-pass line; the old pattern matched a
    # line it never emitted, so this always silently fell back to 256 — wrong as soon as

    "max_tokens": grab(r"max_tokens=(\d+)", int) or 256,
    "kv_cache_tokens": kv_tokens,
    "single_stream": {
        "runs": runs,
        "ttft_s_p50": grab(r"TTFT p50 ([0-9.]+)s"),
        "tpot_ms_p50": grab(r"TPOT p50 ([0-9.]+)ms"),
        "output_tok_s": grab(r"· ([0-9.]+) tok/s"),
    },
    "under_load": {
        "concurrency": conc,
        "endpoints": 1,
        "aggregate_output_tok_s": grab(r"aggregate ([0-9.]+) tok/s"),
        "ttft_s_p95": grab(r"TTFT p95 ([0-9.]+)s"),
        "failed": grab(r"failed (\d+)", int),
        "errors": [],
    },
}
with open(f"metrics/{name}.json", "w") as fh:
    fh.write(json.dumps(out, indent=2) + "\n")
print(f"wrote metrics/{name}.json")
