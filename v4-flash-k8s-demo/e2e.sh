#!/usr/bin/env bash
# Full end-to-end rerun from a clean state, using only `just` recipes.
#
#     bash e2e.sh            # ~90 min, logs everything it does
#
# Clean state means: no engine serving anything, but the RayCluster from 03 and the model PVC
# from 01/02 stay up — re-downloading 167 GB and rebuilding the Ray head proves nothing and
# costs half an hour.
#
# Every strategy is gated on output sanity by bench.py, so a column that answers "What is 2+2?"
# incorrectly cannot contribute a number. That gate is why this rerun exists: it caught
# strategy A — the deck's winner — emitting word salad at 2,295 tok/s, and TP4 doing the same
# at 1,704 tok/s, both with zero failed requests, after eight rounds of review had passed over
# them looking only at the measurement plumbing.
#
# Order is dictated by the 24 GPUs on three machines, with the Ray head permanently holding 8:
#     A  (Deployment, 8)          8 + 8 = 16   ok
#     B  (RayJob on the head)         8        ok
#     B-nospec (RayJob)               8        ok
#     D  (2 replicas, 16)        16 + 8 = 24   exactly full
#     1M (Deployment, 8)          8 + 8 = 16   ok, but only once D is down
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-$(cd .. && pwd)/ray-demo-kubeconfig.yaml}"
cd "$(dirname "$0")"

say() { echo; echo "════════ $* — $(date -u +%H:%M:%SZ)"; }

# A failed deploy-time bench is NOT cosmetic. bench.py's warmup plus the startupProbe do not
# finish warming these engines -- strategy A is still climbing at its third loaded pass
# (measured: 2336 2509 2813 2815 on one node, 2410 2356 2501 2731 on another). So if `run-X`'s
# bench dies, the repeat passes that follow are the engine's 1st-3rd rather than its 2nd-4th,
# the warm median lands mid-ramp, and the strategy reads ~14% slow for no physical reason.
# That happened here and read as post-stress degradation until two pinned reruns refuted it.
# Never let it pass quietly again.
run_leg() {   # run_leg "<label>" <just-recipe>
  local label="$1"; shift
  if ! "$@" 2>&1 | keep; then
    echo "⛔⛔ $label: THE DEPLOY-TIME BENCH FAILED. The repeat passes below start one pass"
    echo "⛔⛔ earlier in this engine's life than every other strategy's, so its warm median"
    echo "⛔⛔ is NOT comparable. Re-run this leg before quoting or regenerating anything."
  fi
}
keep() { grep -E "ready:|sanity:|aggregate |TTFT p50|wrote metrics|snapshot|needle|prompt_tokens|REFUSING|FAIL|PASS|replicas:|head idle" || true; }

say "CLEAN STATE"
just teardown 2>&1 | tail -2
kubectl -n v4-flash-demo delete rayjob --all --ignore-not-found >/dev/null 2>&1
just _wait-head-idle
echo "--- nothing should be serving, but the cluster and PVC must remain ---"
just state 2>/dev/null | sed -n '/RayCluster/,$p'

say "A · TEP8 (10b, Deployment with its own Ray)"
run_leg "A · TEP8" just run-a
just ask-a "What is 2+2? Reply with only the number." 0.0 2>&1 | tail -2
just repeat-bench bench-a A-tep8 3 "e2e" 2>&1 | keep

say "B · DP8xEP + MegaMoE + DSpark"
run_leg "B · DP8xEP" just run-b
just ask-b "What is the capital of France? One word." 0.0 2>&1 | tail -2
just repeat-bench bench-b B-dp8-ep 3 "e2e" 2>&1 | keep

say "B - DSpark"
run_leg "B - DSpark" just run-b-nospec
just repeat-bench bench-b-nospec B-dp8-ep-nospec 3 "e2e" 2>&1 | keep

say "D · two replicas"
run_leg "D · two replicas" just run-d
just probe-d 2>&1 | tail -4
just repeat-bench bench-d D-replicas 3 "e2e" 2>&1 | keep

say "1M window"
just teardown 2>&1 | tail -1
kubectl apply -f 50-longctx-1m.yaml >/dev/null 2>&1
kubectl -n v4-flash-demo rollout status deploy/v4-flash-longctx --timeout=40m 2>&1 | tail -1
just ask-1m "In one sentence: why does a longer context cost cache rather than compute?" 2>&1 | tail -2
just longctx 2>&1 | keep
just longctx-sweep 2>&1 | keep

say "EVERY LIVE ENDPOINT, KNOWN-ANSWER CHECK"
just verify-output 2>&1

say "FINAL TABLE"
just compare 2>&1

say "REPEATABILITY"
just stability 2>&1

say "DONE"
