#!/usr/bin/env python3
"""Print whatever strategies have been measured, side by side, each next to its point.

    python3 compare.py [metrics_dir]        # or: just compare

Designed for a partly-run tour — on demo day you may only get through A and B. So:
  - strategies with no metrics file are NOT shown as columns; they are listed at the
    bottom with the exact command that would produce them
  - extra runs (variants like B-without-DSpark) become their own
    columns automatically, after the canonical four
  - a missing field inside a file prints "—" rather than raising
  - the concurrency and endpoint count are printed as rows, because comparing two
    strategies measured at different offered load is the easiest way to lie to yourself
"""
import json
import statistics
import sys
from pathlib import Path

# canonical order, with the claim each one is supposed to demonstrate
STRATEGIES = [
    ("A-tep8", "A · TEP8", "just run-a",
     "lowest single-stream latency", "gives up aggregate throughput"),
    ("B-dp8-ep", "B · DP8xEP", "just run-b",
     "the published reference shape", "measured slower here, both axes"),
    ("C-pd", "C · P/D 4+4", "just run-c",
     "pools scale independently; biggest cache budget", "a router sits in the path"),
    ("D-replicas", "D · replicas", "just run-d",
     "capacity scales per machine, close to linearly, at equal load per replica",
     "buys no per-request speed, and a request already streaming from a killed pod "
     "cannot be rescued by the infrastructure — the client must retry"),
]

FIELDS = [
    ("TTFT p50 (single)",   lambda d: d["single_stream"]["ttft_s_p50"], "s"),
    ("TPOT p50 (single)",   lambda d: d["single_stream"]["tpot_ms_p50"], "ms"),
    ("tok/s (single)",      lambda d: d["single_stream"]["output_tok_s"], ""),
    ("tok/s (aggregate)",   lambda d: d["under_load"]["aggregate_output_tok_s"], ""),
    ("TTFT p95 under load", lambda d: d["under_load"]["ttft_s_p95"], "s"),
    ("concurrency",         lambda d: d["under_load"]["concurrency"], ""),
    ("endpoints",           lambda d: d["under_load"].get("endpoints", 1), ""),
    ("KV cache (tokens)",   lambda d: f'{d["kv_cache_tokens"]:,}' if d.get("kv_cache_tokens") else None, ""),
    ("failed requests",     lambda d: d["under_load"]["failed"], ""),
    # Last row on purpose: it is the one that invalidates every row above it.
    ("output sane (2+2)",   lambda d: {True: "yes", False: "NO — GARBAGE", None: "—"}[d.get("output_sane")], ""),
]


def load(dirname):
    """Return {stem: data} using WARM MEDIANS where repeat snapshots exist.

    metrics/<stem>.json holds only the LAST run, and the last run is not the strategy: the
    first passes after a deploy are still warming (D read 2278 against a warm 4836). Reading
    the single file made this table disagree with `just claims` and with slide 8, both of which
    use warm medians — A showed 2410.6 here and 2740.5 there. One source of truth instead.
    """
    root = Path(dirname)
    out = {}
    for f in sorted(root.glob("*.json")):
        if f.stem.startswith("_"):        # _archive-* runs are kept but never charted
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping {f.name}: {exc}", file=sys.stderr)
            continue
        snaps = [q for q in sorted((root / "_repeat").glob(f"{f.stem}__*.json"))
                 if not q.name.startswith("_")]
        runs = []
        for q in snaps:
            try:
                runs.append(json.loads(q.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
        warm = runs[1:] if len(runs) >= 3 else runs
        if warm:
            for sec, fld in (("under_load", "aggregate_output_tok_s"),
                             ("under_load", "ttft_s_p95"),
                             ("single_stream", "ttft_s_p50"),
                             ("single_stream", "tpot_ms_p50"),
                             ("single_stream", "output_tok_s")):
                vals = [r[sec][fld] for r in warm
                        if isinstance(r.get(sec, {}).get(fld), (int, float))]
                if vals and sec in data:
                    data[sec][fld] = round(statistics.median(vals), 3)
            data["_warm_n"] = len(warm)
        out[f.stem] = data
    return out


def cell(data, getter, unit):
    """A present-but-null field must render as "—", not "None".

    try/except only catches getters that RAISE; `f"{None}s"` happily produces "Nones".
    A run invoked with --runs 0 produces null single-stream fields by design.
    """
    try:
        value = getter(data)
    except (KeyError, TypeError, ValueError):
        return "—"
    if value is None or value == "None":
        return "—"
    return f"{value}{unit}"


def main():
    dirname = sys.argv[1] if len(sys.argv) > 1 else "metrics"
    data = load(dirname)
    if not data:
        print(f"No metrics in {dirname}/ yet. Run a strategy first, e.g.:\n")
        for _, label, cmd, *_ in STRATEGIES:
            print(f"  {cmd:<14} {label}")
        return

    known = [s[0] for s in STRATEGIES]
    measured = [(stem, label) for stem, label, *_ in STRATEGIES if stem in data]
    extras = [(stem, stem) for stem in data if stem not in known]
    cols = measured + extras
    missing = [(label, cmd) for stem, label, cmd, *_ in STRATEGIES if stem not in data]

    label_w = 22
    col_w = max(14, min(22, 100 // max(1, len(cols))))
    print()
    print("(warm medians where repeats exist — same source as `just claims`)\n")
    print("MEASURED".ljust(label_w) + "".join(c[1][:col_w - 1].ljust(col_w) for c in cols))
    print("-" * (label_w + col_w * len(cols)))
    # A column that failed the output-sanity check has its TIMINGS BLANKED, not merely
    # footnoted. The number on screen is what gets read out loud, so a caveat printed below
    # the table loses to a throughput figure printed inside it. Only the sanity row and the
    # shape rows stay visible, which is exactly enough to see why it is blanked.
    KEEP_WHEN_INSANE = {"concurrency", "endpoints", "KV cache (tokens)", "output sane (2+2)"}
    for name, getter, unit in FIELDS:
        row = name.ljust(label_w)
        for stem, _ in cols:
            insane = data[stem].get("output_sane") is False
            value = ("✗" if insane and name not in KEEP_WHEN_INSANE
                     else cell(data[stem], getter, unit))
            row += value.ljust(col_w)
        print(row)

    print("\nTHE POINT")
    print("-" * 78)
    for stem, label, cmd, plus, minus in STRATEGIES:
        state = "measured" if stem in data else f"not measured — {cmd}"
        print(f"{label}  [{state}]")
        print(f"  + {plus}")
        print(f"  - {minus}")
    for stem, _ in extras:
        d = data[stem]
        print(f"{stem}  [measured]")
        if d.get("claim"):
            print(f"  · {d['claim']}")

    if missing:
        print("\nNOT YET MEASURED — run these to fill the table:")
        for label, cmd in missing:
            print(f"  {cmd:<14} {label}")

    # the two ways this table lies if you don't look at the rows above
    concs = {cell(data[s], lambda d: d["under_load"]["concurrency"], "") for s, _ in cols}
    if len(concs) > 1:
        print(f"\n⚠ columns were measured at different concurrency ({', '.join(sorted(concs))}). "
              "Aggregate tok/s is only comparable at equal offered load per engine — for a "
              "multi-replica strategy that means 32 per replica, not 32 in total.")
    bad = [s for s, _ in cols if data[s].get("output_sane") is False]
    if bad:
        print(f"\n⛔ {', '.join(bad)} FAILED the output-sanity check — timings above are shown "
              f"as ✗ rather than as numbers. The engine answered 2+2 "
              "wrongly, so its throughput is the throughput of garbage. Do not quote it. This "
              "happened for real — a P/D prefill pool with kv_role=kv_both returned 200s full "
              "of nonsense while this table showed it at 1,703.9 tok/s.")
    unknown = [s for s, _ in cols if data[s].get("output_sane") is None]
    if unknown:
        print(f"\n⚠ {', '.join(unknown)} predate the sanity check — re-measure before quoting.")
    print("⚠ compare within one image only: the same manifest on two vLLM builds gave a "
          "45x different KV budget and ~20x different throughput.")


if __name__ == "__main__":
    main()
