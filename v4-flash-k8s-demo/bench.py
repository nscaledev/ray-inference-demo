#!/usr/bin/env python3
"""Measure one serving strategy and write metrics/<name>.json.

Stdlib only, so it runs from the laptop against a port-forward with no venv:

    python3 bench.py --name A-tep8 --port 8000 --label "TEP8 · latency shape"

Two passes, because that is the whole argument of the talk:
  single  — one request at a time: TTFT and TPOT, the latency shape's home turf
  loaded  — N concurrent requests: aggregate output tok/s, the throughput shape's

TTFT/TPOT come from the SSE stream itself (first chunk vs. subsequent chunks), so
they measure what a user feels, not what the server reports about itself.
"""
import argparse, json, re, statistics, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

PROMPT = (
    "Explain, in about 200 words, why expert routing in a sparse mixture-of-experts "
    "model is far more sensitive to interconnect bandwidth than pipeline parallelism is."
)


# Copied verbatim into longctx.py and make_request.py. This file and longctx.py run in-cluster
# from a ConfigMap holding exactly ONE file, so a shared module would not exist at runtime --
# that is the real constraint. make_request.py runs on the laptop and copies only to stay
# identical; keep the three in sync or delete two of them, but do not let them drift again.
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


def stream_once(url, model, max_tokens, prompt=PROMPT):
    """One streaming completion. Returns (ttft_s, tpot_s, output_tokens, total_s)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": True,
        # Ask for the usage block. Counting SSE chunks is NOT counting tokens: vLLM emits
        # one chunk per engine step, and with speculative decoding a step can carry
        # several accepted tokens. Chunk-counting therefore flatters non-speculative
        # configs and penalises speculative ones — it made DSpark look ~1.75x worse than
        # it is. usage.completion_tokens is the ground truth.
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    start = time.perf_counter()
    ttft = None
    chunks = 0
    reported_tokens = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                reported_tokens = usage["completion_tokens"]   # final usage-only chunk
            choices = obj.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if not delta.get("content"):
                continue                      # role-only chunk, or the usage chunk
            if ttft is None:
                ttft = time.perf_counter() - start
            chunks += 1
    total = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("no content chunks came back")
    # Prefer the server's token count; fall back to chunk count only if usage is absent
    # (then TPOT is per-chunk and speculative configs read worse than they are).
    tokens = reported_tokens or chunks
    tpot = (total - ttft) / max(1, tokens - 1)
    return ttft, tpot, tokens, total


# A deterministic question with one right answer. This exists because "0 failed requests" is
# not the same as "correct output": strategy C's prefill pool returned HTTP 200 with fluent
# nonsense ("What is 2+2?" -> " \u2014 \nB.2.2.2.2.2...") for hours, and this harness happily
# reported 1,703.9 tok/s for it, because it counted tokens without ever reading one. Any
# throughput number from an engine that fails this check is measuring the speed of garbage.
SANITY_Q = "What is 2+2? Reply with only the number."
SANITY_A = "4"


def sanity(url, model):
    """-> (passed, answer). Never raises; a failure here must not lose the whole run."""
    try:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": SANITY_Q}],
            "max_tokens": 16,
            "temperature": 0.0,
            "stream": False,
        }).encode()
        req = urllib.request.Request(url, data=body,
                                    headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            msg = json.loads(resp.read())["choices"][0]["message"]
        answer = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    except Exception as exc:                       # noqa: BLE001 - report, never crash
        return False, f"<error: {exc!r}>"
    return sane(answer), answer


def sane(answer):
    """Strict: the reply must BE the answer, not merely contain it.

    A substring test is the wrong instrument here. Fluent garbage is exactly what this gate
    exists to catch, and a reply like "o $ object:；QQ2vivoGz3.## 4660 (almosterCinvasive
    speciese" contains a 4 -- so `SANITY_A in answer` passes it. Checking that a number appears
    somewhere in the output is the same error as counting tokens without reading them, one
    level down.

    Normalise away punctuation, spacing and case, then require the whole reply to be the
    answer. A short tolerance is kept ("4.", "= 4") because the prompt says "reply with only
    the number" and correct engines here answer exactly "4", but a stray period should not
    fail a good engine. Sixty characters of word salad cannot survive it.
    """
    norm = re.sub(r"[^a-z0-9]", "", (answer or "").lower())
    if norm in ("4", "four"):
        return True
    return len(norm) <= 6 and "4" in norm


def pass_single(url, model, max_tokens, runs):
    ttfts, tpots, rates = [], [], []
    for _ in range(runs):
        ttft, tpot, ntok, total = stream_once(url, model, max_tokens)
        ttfts.append(ttft)
        tpots.append(tpot)
        rates.append(ntok / total)
    return {
        "runs": runs,
        "ttft_s_p50": round(statistics.median(ttfts), 3),
        "ttft_s_min": round(min(ttfts), 3),
        "tpot_ms_p50": round(statistics.median(tpots) * 1000, 2),
        "output_tok_s": round(statistics.median(rates), 1),
    }


def warmup(urls, model, concurrency, max_tokens=64):
    """Fire requests of the MEASURED shape at every endpoint and discard the results.

    Readiness is not warmth. /health returns OK as soon as the API server is up, but the first
    request of a given shape can trigger lazy kernel autotune inside the engine — tens of
    seconds. The single-stream pass only ever touches urls[0], so behind a multi-replica Service
    every OTHER replica must be warmed explicitly or it meets the measured load cold: requests
    queue behind compilation, TTFT p95 blows out, and the aggregate halves.

    The warm-up shape has to MATCH the measured shape in BOTH dimensions. Concurrency keys
    CUDA-graph capture and MoE autotune, and decode length keys it too — so warming at 64 tokens
    before measuring at 256 leaves the 256-token path cold and the result comes out bimodal even
    with warm-up nominally in place. Measured per replica, the effect is a one-time cost that
    follows whichever pod has not recently served the measured shape, not a permanently sick pod:
        first pass on a pod    ~400 tok/s, TTFT p95 18-40 s
        every pass after that  ~2050-2490 tok/s, six consecutive passes, 1.22x spread
    and it recurs after the pod sits idle, which is why this runs before every measurement
    rather than once at deploy time.
    """
    # Same concurrency (batch size keys CUDA-graph and MoE autotune) AND the same token
    # count as the measured pass. Warmup costs one extra pass; a wrong number costs a slide.
    n = max(concurrency, 4 * len(urls))       # enough spread to land on every replica
    tokens = max_tokens
    threads = []
    for i in range(n):
        t = threading.Thread(target=lambda i=i: _quiet(urls[i % len(urls)], model, tokens))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def _quiet(url, model, tokens):
    try:
        stream_once(url, model, tokens, prompt="warmup")
    except Exception:
        pass


def pass_loaded(urls, model, max_tokens, concurrency):
    """urls is a list — request i goes to urls[i % len(urls)], so a multi-replica
    Service is actually exercised instead of whichever pod port-forward picked."""
    results, errors, lock = [], [], threading.Lock()

    def worker(i):
        try:
            ttft, tpot, ntok, total = stream_once(urls[i % len(urls)], model, max_tokens,
                                                 prompt=f"{PROMPT} (variant {i})")
            with lock:
                results.append((ttft, ntok, total))
        except Exception as exc:                       # a failed stream is a data point
            with lock:
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    if not results:
        return {"concurrency": concurrency, "failed": concurrency, "errors": errors[:3]}
    ttfts = sorted(r[0] for r in results)
    tokens = sum(r[1] for r in results)
    return {
        "concurrency": concurrency,
        "endpoints": len(urls),
        "completed": len(results),
        "failed": len(errors),
        "wall_s": round(wall, 2),
        "aggregate_output_tok_s": round(tokens / wall, 1),
        "ttft_s_p50": round(statistics.median(ttfts), 3),
        "ttft_s_p95": round(ttfts[max(0, int(len(ttfts) * 0.95) - 1)], 3),
        "errors": errors[:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="metrics file stem, e.g. A-tep8")
    ap.add_argument("--label", default="", help="human label for the comparison table")
    ap.add_argument("--claim", default="", help="what this strategy is supposed to show")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ports", default="",
                    help="comma-separated ports to round-robin across. Needed for "
                         "strategy D: `kubectl port-forward svc/...` pins to ONE pod, "
                         "so a single port measures one replica, not the Service.")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3,
                    help="single-stream runs; 0 skips that pass entirely")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--kv-tokens", type=int, default=0,
                    help="GPU KV cache size the engine logged, for the comparison table")
    ap.add_argument("--out", default="metrics")
    args = ap.parse_args()

    ports = [int(x) for x in args.ports.split(",") if x.strip()] or [args.port]
    urls = [f"http://{args.host}:{p}/v1/chat/completions" for p in ports]
    url = urls[0]
    # Warm FIRST. wait-ready only polls /v1/models, so without this the single-stream pass
    # is the first completion the engine ever serves — and per warmup()'s docstring that
    # request can trigger tens of seconds of lazy autotune. Warming after it meant the
    # latency numbers (the deck's "A is the latency shape") were gathered cold while the
    # throughput numbers were gathered warm.
    # Correctness BEFORE speed. A garbage engine must not produce a clean-looking column.
    print(describe_call(url, {"model": args.model,
                              "messages": [{"role": "user", "content": SANITY_Q}],
                              "max_tokens": 16, "temperature": 0.0}), flush=True)
    ok, answer = sanity(url, args.model)
    print(f"sanity: {SANITY_Q!r} -> {answer!r} [{'PASS' if ok else 'FAIL'}]", flush=True)
    if not ok:
        print("  !! OUTPUT IS NOT SANE. Timings below measure the speed of wrong answers.",
              flush=True)

    print(f"warming every endpoint before measuring ({len(urls)} endpoint(s))", flush=True)
    warmup(urls, args.model, args.concurrency, args.max_tokens)

    if args.runs > 0:
        print(f"single-stream pass ({args.runs} runs) against {url}", flush=True)
        single = pass_single(url, args.model, args.max_tokens, args.runs)
        print(f"  TTFT p50 {single['ttft_s_p50']}s · TPOT p50 {single['tpot_ms_p50']}ms · "
              f"{single['output_tok_s']} tok/s", flush=True)
    else:
        print("skipping the single-stream pass (--runs 0)", flush=True)
        single = {"runs": 0}

    print(describe_call(urls[0], {"model": args.model,
                                  "messages": [{"role": "user", "content": PROMPT}],
                                  "max_tokens": args.max_tokens, "temperature": 1.0,
                                  "stream": True}), flush=True)
    print(f"loaded pass (concurrency {args.concurrency} across {len(urls)} "
          f"endpoint(s): {ports}) max_tokens={args.max_tokens}", flush=True)
    loaded = pass_loaded(urls, args.model, args.max_tokens, args.concurrency)
    print(f"  aggregate {loaded.get('aggregate_output_tok_s')} tok/s · "
          f"TTFT p95 {loaded.get('ttft_s_p95')}s · failed {loaded.get('failed')}", flush=True)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    dest = out / f"{args.name}.json"
    dest.write_text(json.dumps({
        "name": args.name,
        "label": args.label,
        "claim": args.claim,
        "model": args.model,
        # Recorded, not just printed. incluster_to_json.py scrapes the printed line for the
        # in-cluster route, but this file is also written directly when bench.py runs from a
        # laptop (see the docstring) -- and a MISSING output_sane reads in compare.py as
        # "predates the check", a mild note, where a recorded False triggers the outright
        # refusal. So the laptop route was the one that downgraded a garbage engine.
        "output_sane": ok,
        "sanity_answer": answer,
        "max_tokens": args.max_tokens,
        "kv_cache_tokens": args.kv_tokens or None,
        "single_stream": single,
        "under_load": loaded,
    }, indent=2) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach the endpoint ({exc}). Is the port-forward up and a strategy running?")
