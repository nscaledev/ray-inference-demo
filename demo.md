# Demo runbook — Serving a Frontier Open Model with Ray and vLLM

Two sections: **setup** you run 60 minutes before, and the **demo** you run on stage.
Every command below was executed against the live cluster; measured timings are noted.

**Run everything from `v4-flash-k8s-demo/`.** Not the repo root — `just` does not search
upward for the justfile, so every command here fails one directory up. Every demo command
is a `just` recipe precisely so there is one rule to remember and no bare file paths to
get wrong on stage.

**The through-line, so you can improvise back to it if a question derails you:** a throughput
number is not a result until somebody reads an answer. Slide 8 exists to be doubted; the demo
is where the doubt gets settled.

**Three engines, three machines, 24 of 24 GPUs, nothing started on stage:**

| service | strategy | parallelism | where it runs |
|---|---|---|---|
| `v4-flash-tep8` | **A** — the latency shape | TP8 + EP | a Deployment with its own local Ray |
| `v4-flash-openai` | **B** — the published reference | DP8 × EP + DSpark | a RayJob inside the Ray head's 8 GPUs |
| `v4-flash-longctx` | **1M window** | TP8 + EP, `--max-model-len 1048576` | a Deployment |

**Strategy C is deliberately not running.** It is the broken shape — TP4 prefill answers in
fluent nonsense — and everything the demo needs from it is already on disk in
`metrics/C-pd.json`. Running it live would cost 8 GPUs and a 20-minute warm-up to reproduce
garbage you can already show. Dropping it is what frees a whole machine for **B**, which turns
the deck's headline claim from an assertion into a live comparison.

---

## 1 · Pre-demo setup — start 60 minutes out (measured 9m 15s; budget 30)

A cold engine start is **not the weights** — it is ~1,200 DeepGEMM JIT compiles, a FlashInfer
autotune and CUDA graph capture. That is longer than the whole slot, which is why nothing starts
on stage. Strategy **D** is never started; its numbers come from `metrics/*.json`, which is why
`just compare` works for shapes that aren't serving.

**Measured, this cluster, all three warming in parallel:**

| engine | ready after | readiness signal |
|---|---|---|
| B | **5m 16s** | `/v1/models` answers — *weak, see below* |
| A | **8m 30s** | `startupProbe`, which requires a real generation |
| 1M | **8m 53s** | same |
| **total from `just teardown`** | **9m 15s** | |

Two caveats, both load-bearing:

**The three "ready" lines do not mean the same thing.** B is a RayJob with no Deployment to roll
out, so it is polled with `wait-ready`, which only checks that `/v1/models` responds — an engine
can answer that while still compiling kernels. A and the 1M engine wait on a `startupProbe` that
passes only after a token has actually been generated. B arriving first may just mean its bar is
lower. Step 5 is what actually settles it, for all three.

**The weights were already in the nodes' page cache** from earlier runs. On genuinely cold nodes
expect longer — the 11–25 minute range is what this stack does from truly cold. Budget 30
minutes and be pleasantly surprised.

```bash
# 1. Point BARE kubectl at this cluster.
#    The eval is required: a just recipe runs in a child process and cannot set the parent
#    shell's env, so `just enable-kube` alone would look like it worked and change nothing.
#    It also probes /readyz and prints the context to stderr — a wrong-cluster mistake
#    surfaces here rather than on stage.
eval "$(just enable-kube)"

# 2. One clean slate for all three engines.
#    teardown removes every strategy deployment and rayjob but KEEPS the RayCluster and the
#    model PVC — re-downloading 167 GB proves nothing. _ensure-cluster re-applies the
#    RayCluster only if it has gone missing. B needs that cluster: it runs as a RayJob.
just teardown && just _ensure-cluster

# 3. All three engines in ONE declarative apply.
#    They warm IN PARALLEL because they occupy different machines: B runs inside the Ray
#    head's 8 GPUs, A and the 1M engine take a node each. That is 24 of 24 GPUs.
#    Re-running is safe — whatever is already serving is left alone.
#    Do NOT use `just run-a` or `just run-b` here: each begins with its own teardown, so the
#    second would delete the first and they would warm sequentially (~50 min, not ~25).
kubectl apply -f 10b-strategy-tep8-deployment.yaml \
              -f 20-strategy-dp8-ep.yaml \
              -f 50-longctx-1m.yaml

# 4. Wait for all three AT ONCE.
#    This doesn't make it faster — they were already warming together from step 3. It makes a
#    failure visible immediately instead of hiding behind a 20-minute wait on another engine.
#    Each branch labels its own verdict; `wait` blocks until all three finish, so step 5
#    cannot run against a half-warm cluster.
#    B is a RayJob, so there is no Deployment to roll out — poll its endpoint instead.
{ kubectl -n v4-flash-demo rollout status deploy/v4-flash-tep8 --timeout=40m >/dev/null \
    && echo "✓ A ready" || echo "⛔ A FAILED"; } &
{ just wait-ready 8000 v4-flash-openai 2400 >/dev/null 2>&1 \
    && echo "✓ B ready" || echo "⛔ B FAILED"; } &
{ kubectl -n v4-flash-demo rollout status deploy/v4-flash-longctx --timeout=40m >/dev/null \
    && echo "✓ 1M engine ready" || echo "⛔ 1M engine FAILED"; } &
wait

# 5. PRE-FLIGHT CHECK. Asks every live endpoint "What is 2+2?" at temperature 0 and reads the
#    reply. With C gone, ALL THREE MUST PASS — expect:
#
#      endpoint               strategy parallelism         part           verdict
#      v4-flash-tep8:8000     A        TP8 + EP            whole engine   ✓ correct
#      v4-flash-openai:8000   B        DP8 x EP + DSpark   whole engine   ✓ correct
#      v4-flash-longctx:8000  1M       TP8 + EP            whole engine   ✓ correct
#
#      pre-flight: 3/3 endpoints behaved as the runbook expects.
#
#    The exit code is the mismatch count, so 0 means every engine behaved as this runbook
#    says. Anything else: read the table before you walk on.
#
#    COUNT THE ROWS. It must say 3/3, and A must be one of them. On the first from-scratch
#    run of this setup it said "2/2 ... behaved as expected" and exited 0 with A serving and
#    never asked a question -- `_endpoints` had a hardcoded service list that `v4-flash-tep8`
#    was missing from. Fixed, but the lesson generalises: a gate reports on what it looks at,
#    and a narrowed set looks exactly like a clean one.
just verify-output

# 6. Prove the needle BEFORE you're on stage (measured 77 s, TTFT 63.9 s).
#    A passphrase sits at the MIDPOINT of a 996K-token prompt — not the head, not the tail,
#    because head-and-tail attention passes an edge needle test for free.
#    If it fails on the day, quote the number from slide 6 instead of running it live.
just longctx

# 7. Sanity-check the closing beat's inputs.
#    compare reads metrics/*.json, so it also shows D (not serving) and C (never started, and
#    rendered as ✗ because it failed the gate). CONFIRM THAT ✗ COLUMN IS THERE — beat 4
#    depends on it. Leave the output in a spare terminal tab as network insurance.
just compare
```

**Not scriptable, do it now:** open the Ray dashboard (`just dashboard`), open the Nscale
console page **and log in** — it's behind SSO — and have the recorded clips of each beat ready
as a fallback.

---

## 2 · The demo — ~16 minutes, six beats

Each beat says what will appear on screen before the command, so you know what you are pointing
at. Run them one at a time and talk in between.

**Cut order if squeezed:** beat 2, then beat 3's second command, then beat 5's depth sweep.
Beat 4 is already down to two commands. Don't let it grow back on stage.

### Beat 1 · 0:00–1:00 · Cold open

**What you'll see:** six sections. A node list — three machines with 8 GPUs each, plus three
control-plane nodes with none. Then who holds those GPUs: three lines, one per engine, ending
in "24 GPU(s) held". Then the RayCluster, the RayJobs and Deployments, the PVC with the weights,
and last a list of the endpoints that answer `/v1/models`.

**Point at:** the 24, and the three separate machine names.

```bash
just state
```

> "Three machines, twenty-four B200s, one PVC holding the weights — and all twenty-four GPUs are
> busy right now: strategy A on one machine, the published reference shape B on the Ray head, and
> the million-token engine on the third. All three started before the talk, and I'll tell you
> exactly why in a minute."

Then about twenty seconds on the Nscale console, so they can see it is a real cluster.

### Beat 2 · 1:00–2:00 · It talks

**What you'll see:** a header naming the engine, the POST it is about to make, the model and
temperature, the prompt, then the answer, then a token count. The prompt is 19 tokens and the
answer comes back in about two seconds.

**Watch out:** the command is called `ask`, and it asks the million-token engine — but with a
short prompt. Say so, or half the room will think they just watched the million-token test. That
is beat 5.

```bash
just ask
```

> "One live answer before any benchmarks, so you know there's a real model here."

Temperature is 1.0, the model card's setting, so the wording changes every run. Read out whatever
it says.

### Beat 3 · 2:00–6:00 · A versus B, live

**What you'll see:** the same shape of output twice. First A, labelled TP8 + EP, answering a
question about expert routing. Then B, labelled DP8 × EP + DSpark, answering about speculative
decoding. Both answers are correct. Each block shows which Service it hit, so it is visible that
these are two different engines.

**Point at:** the two different service names, and the fact that both are right.

```bash
just ask-a
just ask-b
```

> "Same weights, same image, same machine count. A is tensor-plus-expert parallel across eight
> GPUs — attention sliced, whole experts per GPU. B is what DeepSeek published as the reference:
> data parallel by eight with expert parallelism, and their speculator switched on. Watch the
> first token on each."

Then the measured numbers:

**What you'll see:** a table of four strategies with their warm medians, then a list of claims.
Each claim ends with the two numbers it divided, in brackets.

```bash
just claims
```

> "Roughly fifty percent more tokens per second than the reference shape, and a first token about
> twice as fast. Every ratio there prints the two numbers it divided, so you can check my
> arithmetic instead of trusting it."

**Cut the second command first** if you are behind. The two asks carry this beat, and `claims`
comes back in beat 6.

### Beat 4 · 6:00–9:00 · What a throughput benchmark cannot see

Slide 3 planted this: TP4 ran 13% faster than the reference shape, and every answer was garbage.
Two commands, then move on. The lesson is the point, not the configuration — nobody should
deploy TP4 on this build, so don't spend the room's attention on how prefill/decode works.

**What you'll see, first command:** the comparison table. Four columns have numbers. The column
headed "C · P/D 4+4" has ✗ in every row instead. Below the table, a line explaining that C
failed the output check so its timings are not shown.

**What you'll see, second command:** a short block with C's throughput, its failed-request count
of zero, `output_sane False`, the question it was asked, and the answer it actually gave — which
is a fragment of Java.

**Point at:** ✗ in a column that has a throughput number sitting in a file.

```bash
just compare
just receipt
```

> "Every shape side by side. Four columns have numbers. One doesn't — strategy C shows ✗ all the
> way down. That isn't a missing measurement; the harness measured it fine and then refused to
> show it."
>
> …then `just receipt`…
>
> "Nineteen hundred and twenty-two tokens a second. Zero failed requests. HTTP two hundred every
> time. And the answer to 'what is two plus two' was a Java method. That ran for twenty-two
> benchmark runs before anybody read an answer — because a throughput number cannot report
> incorrectness. Only reading an answer can. The fix isn't clever, it's structural: ask a question
> with a known answer before you time anything, and refuse to print timings for wrong answers.
> That's why the three engines running right now are quotable and that one isn't."

**Say that C is not running.** You are showing a file, not a live pod, and someone will check
`just state`.

**Don't take questions on TP4's internals.** The answer is "it's broken on this build, so it's
not a shape you'd choose." The gate is the interesting part, not the bug. If someone really
wants it, the findings are in `30-strategy-pd-single-node.yaml`.

### Beat 5 · 9:00–13:00 · The 1M window

**What you'll see:** first a line saying the target size — about a million tokens, 5.8 million
characters, needle at depth 0.50. Then the POST. Then roughly 75 seconds of nothing. Then one
line: the real token count, time to first token, and whether the needle was found.

**Watch out:** the wait is long and silent. Fill it, don't stare at it.

```bash
just longctx
```

> "A passphrase is buried at the midpoint of a prompt just under a million tokens — not the head,
> not the tail, the middle, because head-and-tail attention passes a needle test at the edges for
> free."

While it runs, give the cache arithmetic from slide 6: four times the window costs 1.4× in
concurrency, and the per-token cache cost falls 2.9×.

> "Sixty-four seconds to first token at a million tokens of context, served from one machine's
> HBM. No offloading."

Then one sentence on D, which is not demoed:

> "Scale-out isn't demoed because 'add machines, get throughput' is self-evident — the non-obvious
> part was which shape you replicate. We replicate the winner: 1.85× on two machines."

### Beat 6 · 13:00–16:00 · The punchline

**What you'll see:** the same output as beat 3's second command — the warm-median table, then
seven claims, each with its two source numbers in brackets, and a closing line saying which
comparison is *not* being claimed because it sits inside the noise.

**Point at:** the bracketed numbers, and the line about what is deliberately not claimed.

```bash
just claims
```

> "The published reference shape lost to the simpler one on my hardware. Every ratio on that list
> prints the two numbers behind it, and every one of those numbers came from an engine that
> answered a question correctly first. The flags are one kubectl apply away. Measure your own."

---

**If a question needs a number you didn't show,** it's on slide 8 or in `just claims`.

**Never use `just _ask`.** It picks the first live endpoint instead of a named strategy. Every
public `ask*` recipe names its strategy and refuses unless that strategy is serving.
