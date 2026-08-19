#!/usr/bin/env python3
"""Derive every claim the deck and abstract make, from the measurements, with provenance.

    python3 claims.py          # or: just claims

Exists because the ratios were hand-computed several times and drifted: "37% more tok/s" and
"4.7x sooner" both came from an engine that answered "What is 2+2?" with Java package
declarations, and "close to linear" scale-out came from dividing by that same wrong number.
Nothing here is typed in. Every figure is read from metrics/, and every claim prints the two
numbers it divides plus how many runs each rests on, so a wrong claim is visible rather than
merely wrong.

Two rules encoded here:

  WARM MEDIAN. The first one or two passes after a deploy are not measurements of the strategy;
  they are measurements of an engine still warming. Measured on this cluster, four identical
  benches in a row:
      A  2312.8  2260.3  2688.7  2792.3
      B  1361.3   915.6  1697.6  1653.0
      D  4291.4  2278.2  4614.8  5056.6      <- a 2.1x error if you quote the second pass
  So the earliest snapshot is dropped. bench.py already fires a shape-matched warmup and the
  pod's startupProbe already made it generate, and it is still not enough.

  SANITY FIRST. A strategy whose output failed the 2+2 check contributes nothing. Throughput
  cannot report incorrectness, so a number without a read answer behind it is not evidence.
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

M = Path(sys.argv[1] if len(sys.argv) > 1 else "metrics")

FIELDS = {
    "agg":    ("under_load", "aggregate_output_tok_s"),
    "p95":    ("under_load", "ttft_s_p95"),
    "ttft":   ("single_stream", "ttft_s_p50"),
    "tpot":   ("single_stream", "tpot_ms_p50"),
    "single": ("single_stream", "output_tok_s"),
}


def load(stem):
    """-> dict of warm medians, plus n and whether every run passed the sanity gate."""
    snaps = [f for f in sorted(glob.glob(str(M / "_repeat" / f"{stem}__*.json")))
             if "_archive" not in f]
    runs = [json.loads(Path(f).read_text()) for f in snaps]
    if not runs:
        single = M / f"{stem}.json"
        if not single.is_file():
            return None
        runs = [json.loads(single.read_text())]
    warm = runs[1:] if len(runs) >= 3 else runs      # drop the still-warming first snapshot
    out = {"n": len(runs), "n_warm": len(warm),
           "sane": all(r.get("output_sane") for r in runs),
           "unknown": any(r.get("output_sane") is None for r in runs)}
    for key, (sec, fld) in FIELDS.items():
        vals = [r[sec][fld] for r in warm
                if isinstance(r.get(sec, {}).get(fld), (int, float))]
        out[key] = st.median(vals) if vals else None
    return out


def main():
    WANTED = ("A-tep8", "B-dp8-ep", "B-dp8-ep-nospec", "D-replicas")
    loaded = {s: load(s) for s in WANTED}
    missing = [k for k, v in loaded.items() if v is None]
    # Narrow to the measured ones so the rest of this function cannot index a None. The early
    # return below already guarantees it, but a reader (and a type checker) should not have to
    # prove that by tracing control flow.
    S = {k: v for k, v in loaded.items() if v is not None}
    print("\nWARM MEDIANS  (earliest snapshot dropped; every run gate-passed unless flagged)\n")
    print(f"  {'strategy':<18}{'n':>3}{'warm':>5}  {'agg':>8}{'single':>8}{'tpot':>8}"
          f"{'ttft50':>8}{'p95':>7}  sane")
    for k in WANTED:
        v = loaded[k]
        if v is None:
            print(f"  {k:<18}  — not measured —"); continue
        flag = "yes" if v["sane"] else ("UNKNOWN" if v["unknown"] else "NO — GARBAGE")
        print(f"  {k:<18}{v['n']:>3}{v['n_warm']:>5}  {v['agg']:>8.1f}{v['single']:>8.1f}"
              f"{v['tpot']:>8.2f}{v['ttft']:>8.3f}{v['p95']:>7.3f}  {flag}")
    if missing:
        print(f"\n  cannot derive claims: {', '.join(missing)} unmeasured")
        return 1

    A, B, N, D = (S[k] for k in ("A-tep8", "B-dp8-ep", "B-dp8-ep-nospec", "D-replicas"))
    bad = [k for k, v in S.items() if not v["sane"]]
    if bad:
        print(f"\n  ⛔ {', '.join(bad)} failed the output-sanity gate — claims below are void")
        return 1

    print("\nCLAIMS  (each shows the two numbers it divides)\n")
    rows = [
        ("A beats the reference shape on throughput",
         f"{A['agg']:.0f} / {B['agg']:.0f}", A['agg'] / B['agg'],
         f"{(A['agg']/B['agg']-1)*100:.0f}% more tok/s"),
        ("A's first token arrives sooner (single stream)",
         f"{B['ttft']:.3f}s / {A['ttft']:.3f}s", B['ttft'] / A['ttft'], "x sooner"),
        ("A's later tokens arrive faster (TPOT)",
         f"{B['tpot']:.2f}ms / {A['tpot']:.2f}ms", B['tpot'] / A['tpot'], "x faster"),
        ("A's single-stream throughput",
         f"{A['single']:.1f} / {B['single']:.1f}", A['single'] / B['single'], "x"),
        ("DSpark helps rather than costs",
         f"{B['agg']:.0f} / {N['agg']:.0f}", B['agg'] / N['agg'], "x aggregate"),
        ("Scale-out across two machines",
         f"{D['agg']:.0f} / {A['agg']:.0f}", D['agg'] / A['agg'], "x"),
        ("Per-replica efficiency inside the fleet",
         f"({D['agg']:.0f}/2) / {A['agg']:.0f}", (D['agg'] / 2) / A['agg'], "of a solo machine"),
    ]
    for label, prov, ratio, unit in rows:
        print(f"  {label:<48} {ratio:>5.2f}  {unit:<22} [{prov}]")

    print("\n  TTFT p95 under load is NOT a claim: "
          f"A {A['p95']:.3f}s vs B {B['p95']:.3f}s — inside the noise, do not assert a winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
