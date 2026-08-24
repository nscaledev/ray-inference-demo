#!/usr/bin/env bash
#
# demo-e2e.sh — everything in demo.md, prep and demo, as one unattended run.
# Built for recording: it prints what you are about to see, then the command, then the output.
#
#     bash demo-e2e.sh                # prep + demo (~15 min; ~25 if the nodes are cold)
#     bash demo-e2e.sh --demo-only    # skip prep, engines must already be warm (~5 min)
#     bash demo-e2e.sh --prep-only    # bring the cluster up and stop
#     PAUSE=0 bash demo-e2e.sh        # no pauses between beats (default 4s, for narration)
#     PAUSE=read bash demo-e2e.sh     # wait for ENTER between beats
#
# ⚠ PREP IS DESTRUCTIVE. `just teardown` deletes every strategy Deployment and RayJob in the
#   namespace. It keeps the RayCluster and the model PVC, so no weights are re-downloaded, but
#   anything you had serving is gone. --demo-only touches nothing.
#
# WHAT IT LEAVES RUNNING: strategy A, strategy B and the 1M engine — 24 of 24 GPUs across three
# machines. Strategy C is deliberately not started; the demo shows its recorded artifact.
set -uo pipefail
cd "$(dirname "$0")"
export KUBECONFIG="${KUBECONFIG:-$(cd .. && pwd)/ray-demo-kubeconfig.yaml}"

# We ARE the orchestrator, so bypass teardown's concurrency guard for our own calls. That
# guard exists to stop a SEPARATE shell from tearing down mid-run; without this it also blocks
# prep step 2 below, which is this script's own teardown -- a guard that deadlocks the thing it
# protects. Any teardown typed in another terminal still refuses, which is the point.
export FORCE=1

PAUSE="${PAUSE:-4}"
DO_PREP=1; DO_DEMO=1
case "${1:-}" in
  --demo-only) DO_PREP=0 ;;
  --prep-only) DO_DEMO=0 ;;
  "") ;;
  *) echo "unknown option: $1"; exit 2 ;;
esac

RULE="════════════════════════════════════════════════════════════════════════════"

beat() {                        # beat "<title>" "<what you'll see>"
  echo; echo "$RULE"; echo "  $1"; echo "$RULE"
  [ -n "${2:-}" ] && { echo "  WHAT YOU'LL SEE: $2"; echo; }
  case "$PAUSE" in
    read) printf "  [ENTER to run] "; read -r _ </dev/tty || true ;;
    0) ;;
    *) sleep "$PAUSE" ;;
  esac
}

run() { echo "  \$ $*"; echo; "$@"; local rc=$?; echo; [ $rc -ne 0 ] && echo "  (exit $rc)"; return 0; }

t0=$(date +%s)
elapsed() { local s=$(( $(date +%s) - t0 )); printf "%dm%02ds" $((s/60)) $((s%60)); }

# =========================================================================================
if [ "$DO_PREP" = 1 ]; then
echo "$RULE"
echo "  PART 1 — PREP.  Destructive: tears down every strategy, keeps the RayCluster + PVC."
echo "  Expect 10 min if the weights are page-cached, up to 25 from genuinely cold nodes."
echo "$RULE"

beat "PREP 1/7 · point kubectl at this cluster" \
     "the cluster's context name, and the kubeconfig path being exported."
run just enable-kube

beat "PREP 2/7 · clean slate" \
     "deletions scrolling past, then a wait until the head holds no strategy GPUs, then 'RayCluster ready'."
run just teardown
run just _ensure-cluster

beat "PREP 3/7 · bring up A, B and the 1M engine in one apply" \
     "six or seven 'created' lines. They warm in parallel on three different machines."
run kubectl apply -f 10b-strategy-tep8-deployment.yaml \
                  -f 20-strategy-dp8-ep.yaml \
                  -f 50-longctx-1m.yaml

beat "PREP 4/7 · wait for all three" \
     "nothing for several minutes, then three lines: A ready, B ready, 1M ready. B usually lands first because its check is weaker — it only polls /v1/models, while A and 1M wait for a real generation."
{ kubectl -n v4-flash-demo rollout status deploy/v4-flash-tep8 --timeout=40m >/dev/null 2>&1 \
    && echo "  ✓ A ready   $(elapsed)" || echo "  ⛔ A FAILED"; } &
{ just wait-ready 8000 v4-flash-openai 2400 >/dev/null 2>&1 \
    && echo "  ✓ B ready   $(elapsed)" || echo "  ⛔ B FAILED"; } &
{ kubectl -n v4-flash-demo rollout status deploy/v4-flash-longctx --timeout=40m >/dev/null 2>&1 \
    && echo "  ✓ 1M ready  $(elapsed)" || echo "  ⛔ 1M FAILED"; } &
wait

beat "PREP 5/7 · pre-flight — the check that must say 3/3" \
     "a table of three endpoints, each with its strategy letter and parallelism, each answering '4'. The last line must read 3/3. If a row is missing, an engine is serving unchecked."
run just verify-output

beat "PREP 6/7 · prove the million-token needle before showtime" \
     "a target line, then about 75 seconds of silence, then one line with the real token count, time to first token, and whether the needle was found."
run just longctx

beat "PREP 7/7 · confirm the closing beat's inputs" \
     "the comparison table. Strategy C's column must be ✗ rather than numbers — the demo's beat 4 depends on that."
run just compare

echo "$RULE"; echo "  PREP DONE in $(elapsed)"; echo "$RULE"
fi

# =========================================================================================
if [ "$DO_DEMO" = 1 ]; then
echo; echo "$RULE"
echo "  PART 2 — THE DEMO.  Six beats, about 16 minutes when narrated."
echo "$RULE"

beat "BEAT 1 · 0:00 · Cold open — what this cluster is" \
     "six sections: three machines with 8 GPUs each plus three control-plane nodes, then who holds those GPUs (three engines, ending '24 GPU(s) held'), then the RayCluster, the workloads, the PVC, and the endpoints answering /v1/models. POINT AT the 24 and the three machine names."
run just state

beat "BEAT 2 · 1:00 · It talks" \
     "a header naming the engine, a line saying this is a SHORT prompt (not a million-token one), the POST, the prompt, the answer, a token count. Comes back in about two seconds."
run just ask

beat "BEAT 3a · 2:00 · Strategy A answers" \
     "the same shape of output, labelled A — attention sliced 8 ways, whole experts per GPU. A correct answer about expert routing."
run just ask-a

beat "BEAT 3b · 3:00 · Strategy B answers" \
     "the same again, labelled B — the shape DeepSeek published. Also correct. Note it hit a DIFFERENT Service: these are two separate engines on two separate machines."
run just ask-b

beat "BEAT 3c · 4:00 · and the measured difference" \
     "a warm-median table, then seven claims. Each claim ends with the two numbers it divided, in brackets, so the arithmetic is checkable."
run just claims

beat "BEAT 4a · 6:00 · What a throughput benchmark cannot see" \
     "the comparison table again. Four columns carry numbers; the column headed 'C · P/D 4+4' carries ✗ in every row, and a line underneath explains that C failed the output check so its timings are withheld."
run just compare

beat "BEAT 4b · 7:30 · the artifact behind that refusal" \
     "a short block: C's throughput of 1922.8 tok/s, zero failed requests, output_sane False, the question it was asked, and the answer it gave — a fragment of Java. SAY OUT LOUD that C is not running; this is a file."
run just receipt

beat "BEAT 5 · 9:00 · The 1M window, served and read" \
     "the target line, then ~75 silent seconds — fill them with the cache arithmetic from slide 6 — then one line: 996,068 tokens, time to first token, needle FOUND."
run just longctx

beat "BEAT 6 · 13:00 · The punchline" \
     "the same claims list as beat 3c. POINT AT the bracketed source numbers, and at the closing line naming the one comparison that is deliberately NOT claimed because it sits inside the noise."
run just claims

echo "$RULE"; echo "  DEMO DONE.  Total run $(elapsed)"; echo "$RULE"
fi

echo
echo "  Still serving: A, B and the 1M engine — 24 of 24 GPUs."
echo "  Tear down with: just teardown     (keeps the RayCluster and the weights)"
