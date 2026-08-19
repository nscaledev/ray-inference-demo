#!/usr/bin/env python3
"""Needle-in-a-haystack for the long-context claim: does the engine READ the window?

    python3 longctx.py --port 8000 --tokens 128000 --depth 0.5

Accepting a long prompt is not the same as using it. This builds a haystack of roughly
--tokens tokens, hides a fact at --depth through it (0.0 = start, 1.0 = end), asks for that
fact back, and reports:

  prompt_tokens  what the server actually counted (the only trustworthy length)
  ttft_s         time to first token — long-context cost lands almost entirely in prefill
  found          whether the answer contains the needle

Why depth matters: a model that only attends to the head and tail of its window still
passes a needle test at depth 0.0 or 1.0. Sweeping the middle is what tests CSA/HCA, the
two-stage compression the deck's long-context slide is about.

Stdlib only, so it runs against a port-forward with no venv.
"""
import argparse
import json

import time
import urllib.error
import urllib.request

# MEASURED on this tokenizer with this filler: 29,098 chars -> 5,027 tokens = 5.79 chars/tok.
# The generic "~4 chars/token" rule undershot by 45%, so a "250K" request was really 155K.
CHARS_PER_TOKEN = 5.79

FILLER = (
    "The maintenance log for turbine bay {i} records nominal vibration, inlet pressure "
    "within tolerance, and no operator intervention during the shift. Coolant flow was "
    "steady and the bearing temperature trend stayed flat across the reporting window. "
)
NEEDLE = ("Buried maintenance note: the calibration passphrase for turbine bay 7 is "
          "{secret}. Remember it exactly.")


def build(tokens, depth, secret, tag):
    """Return the prompt. Length is approximate by design — the server reports the truth.

    `tag` goes in the FIRST line to defeat prefix caching. Without it, successive runs share
    the filler prefix and the engine serves most of the window from cache: a sweep measured
    TTFT 7.9s / 39.8s / 42.7s / 6.0s at 5K / 20K / 79K / 155K prompt tokens — the LONGEST
    prompt looked cheapest, because it was mostly a cache hit. Prefix caching is a real and
    desirable feature, but it is not what "prefill cost at 1M" means.
    """
    target_chars = int(tokens * CHARS_PER_TOKEN)
    needle = NEEDLE.format(secret=secret)
    body, i, n = [], 0, 0
    while n < target_chars:
        chunk = FILLER.format(i=i % 12)
        body.append(chunk)
        n += len(chunk)
        i += 1
    cut = max(0, min(len(body), int(len(body) * depth)))
    body.insert(cut, needle)
    return (
        f"Archive session {tag}. "
        "You are given a long maintenance archive. Read it and answer only from it.\n\n"
        + "".join(body)
        + "\n\nQuestion: what is the calibration passphrase for turbine bay 7? "
          "Answer with the passphrase only."
    )


def ask(url, model, prompt, max_tokens=48, timeout=1800):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"content-type": "application/json"})
    start = time.perf_counter()
    ttft, text, prompt_tokens = None, [], None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
            if obj.get("usage"):
                prompt_tokens = obj["usage"].get("prompt_tokens") or prompt_tokens
            choices = obj.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if not piece:
                continue
            if ttft is None:
                ttft = time.perf_counter() - start
            text.append(piece)
    return ttft, time.perf_counter() - start, "".join(text), prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--tokens", type=int, default=128000)
    ap.add_argument("--depth", type=float, default=0.5, help="0.0 start .. 1.0 end")
    ap.add_argument("--secret", default="ORCHID-4417-VERDANT")
    ap.add_argument("--tag", default="", help="unique marker to defeat prefix caching "
                                             "(default: a fresh timestamp)")
    args = ap.parse_args()
    tag = args.tag or f"{time.time():.6f}"

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    prompt = build(args.tokens, args.depth, args.secret, tag)
    print(f"  target ~{args.tokens:,} tok · {len(prompt):,} chars · needle at depth "
          f"{args.depth:.2f}", flush=True)
    try:
        ttft, total, answer, ptok = ask(url, args.model, prompt)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf8", "ignore")[:300]
        print(f"  REJECTED  HTTP {exc.code}: {detail}")
        return 2
    except urllib.error.URLError as exc:
        print(f"  UNREACHABLE  {exc.reason} — endpoint down, or the tunnel dropped under a "
              f"{len(prompt)/1e6:.1f} MB body. Prefer the in-cluster Job.")
        return 4
    # ttft is None when the stream carried no content at all. That happened for real: a 1M
    # prompt is a ~5.8 MB body, `kubectl port-forward` died mid-request, and this line raised
    # TypeError on None instead of saying so — then every later depth failed with "connection
    # refused". A measurement harness must report its own failure, not crash on it.
    if ttft is None:
        print(f"  NO CONTENT returned after {total:.1f}s — the stream closed without a token. "
              f"Check the engine is up and that this is not going through a tunnel "
              f"(a 1M prompt is ~{len(prompt)/1e6:.1f} MB; run it in-cluster).")
        return 3
    found = args.secret.lower() in answer.lower()
    got = f"{ptok:,}" if ptok else "unreported"
    ratio = f" · {len(prompt)/ptok:.2f} chars/tok" if ptok else ""
    print(f"  prompt_tokens {got} · TTFT {ttft:.1f}s · total {total:.1f}s · "
          f"needle {'FOUND' if found else 'MISSED'}{ratio}")
    if not found:
        print(f"    answered: {answer.strip()[:160]!r}")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
