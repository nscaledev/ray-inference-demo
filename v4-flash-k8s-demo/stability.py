#!/usr/bin/env python3
"""Summarise repeat measurements: median, spread, and whether a number is trustworthy.

    python3 stability.py [metrics/_repeat]      # or: just stability

Every bench writes metrics/<stem>.json, so repeats are snapshotted as
metrics/_repeat/<stem>__r<round>-<i>.json. A single sample of any of these numbers is not
evidence: this session produced 424 and 4652 tok/s from the same pods minutes apart, and
300 vs 911 from the same manifest under two harnesses. Spread is the point.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# (label, json path, noise floor). Below the floor a ratio is meaningless: TTFT p95 of
# 0.4s vs 0.7s is 1.93x and reads as UNSTABLE, but 0.3s of absolute movement on a
# sub-second number is scheduling jitter, not an unreliable measurement. Only flag a
# metric when it is BOTH proportionally wide and absolutely far apart.
FIELDS = [
    ("aggregate tok/s", ("under_load", "aggregate_output_tok_s"), 150.0),
    ("TTFT p95 (load)", ("under_load", "ttft_s_p95"), 1.0),
    ("tok/s single",    ("single_stream", "output_tok_s"), 15.0),
    ("TPOT ms single",  ("single_stream", "tpot_ms_p50"), 4.0),
]


def dig(d, path):
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "metrics/_repeat")
    if not root.is_dir():
        sys.exit(f"no repeats yet in {root} — run `just repeat-tour` first")
    groups = defaultdict(list)
    for f in sorted(root.glob("*.json")):
        # `_`-prefixed files are archived runs kept as evidence, not results — the same rule
        # compare.py uses. Without this, snapshots archived because they measured a HARNESS
        # BUG get charted as if they were strategies, and their spread is reported as if the
        # strategy were unstable.
        if f.stem.startswith("_"):
            continue
        stem = f.stem.split("__")[0]
        try:
            groups[stem].append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f"  ! unreadable: {f.name}", file=sys.stderr)

    print(f"\nREPEATABILITY  ({sum(len(v) for v in groups.values())} runs "
          f"across {len(groups)} strategies)\n")
    hdr = (f"{'strategy':<22}{'metric':<18}{'n':>3}  {'median':>10}{'min':>10}{'max':>10}"
           f"  spread  {'warm med':>9}")
    print(hdr)
    print("-" * len(hdr))
    # WARM-UP IS REAL AND IT LASTS MORE THAN ONE PASS. Measured on this cluster, four identical
    # benches against the same pod, in order:
    #   A: 2312.8  2260.3  2688.7  2792.3   tok/s     p95 0.913 0.916 0.510 0.373
    #   B: 1361.3   915.6  1697.6  1653.0             p95 1.191 4.242 0.407 0.406
    # The first two are not measurements of the strategy, they are measurements of an engine
    # still warming — even though bench.py already fires a shape-matched warmup and the pod's
    # startupProbe already made it generate. So a median over ALL passes is an artefact of how
    # many passes you happened to run. Reporting both keeps that visible instead of picking one:
    # "all" is what the raw snapshots say, "warm" drops the earliest snapshot (which follows the
    # deploy-time bench, so it is pass 2 of the engine's life) and is what should be quoted.
    for stem in sorted(groups):
        runs = groups[stem]
        warm = runs[1:] if len(runs) >= 3 else runs
        for label, path, floor in FIELDS:
            vals = [v for v in (dig(r, path) for r in runs) if isinstance(v, (int, float))]
            if not vals:
                continue
            wvals = [v for v in (dig(r, path) for r in warm) if isinstance(v, (int, float))]
            med, lo, hi = statistics.median(vals), min(vals), max(vals)
            wmed = statistics.median(wvals) if wvals else med
            spread = (hi / lo) if lo else float("inf")
            # Judge on the WARM values. The raw spread is dominated by the first pass after a
            # deploy — D read 2278 against a warm 4836, a 2.22x raw spread with a known cause
            # and a clean warm median. Flagging that as UNSTABLE points at the wrong thing.
            wspread = (max(wvals) / min(wvals)) if wvals and min(wvals) else spread
            wide = wspread >= 1.5 and len(wvals) > 1
            wrange = (max(wvals) - min(wvals)) if wvals else (hi - lo)
            flag = "  <-- UNSTABLE" if wide and wrange >= floor else \
                   "  (jitter)" if wide else ""
            print(f"{stem[:21]:<22}{label:<18}{len(vals):>3}  {med:>10.1f}{lo:>10.1f}{hi:>10.1f}"
                  f"  {spread:>5.2f}x  {wmed:>9.1f}{flag}")
        print()
    print("UNSTABLE is judged on the WARM values, not the raw ones: the raw spread is")
    print("dominated by the first post-deploy pass, whose cause is known and excluded.")
    print("UNSTABLE = warm spread >= 1.5x AND the warm range is past the metric's noise")
    print("floor: the number is not yet a measurement, so find the cause before quoting it.")
    print("(jitter) = proportionally wide but absolutely tiny — a ratio on a near-zero value.")
    print("Archived _*.json runs are excluded; they are evidence, not results.")
    print("warm med = median excluding the earliest snapshot, which is still warming up.")
    print("QUOTE THE WARM MEDIAN. The first two passes of an engine's life are not the strategy.")


if __name__ == "__main__":
    main()
