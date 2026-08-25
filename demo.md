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

| service | strategy | parallelism | how it is served |
|---|---|---|---|
| `v4-flash-tep8` | **A** — the latency shape | TP8 + EP | RayService (`v4-flash-a`) |
| `v4-flash-openai` | **B** — the published reference | DP8 × EP + DSpark | RayService (`v4-flash-b`) |
| `v4-flash-longctx` | **1M window** | TP8 + EP, `max_model_len: 1048576` | RayService (`v4-flash-1m`) |

All three are RayServices: one CRD per strategy, each owning its own RayCluster with the head
holding no GPUs and one 8-GPU worker group. The service names above are stable aliases, so the
tooling and the recorded measurements refer to one address per strategy.

**Strategy C is deliberately not running.** It is the broken shape — TP4 prefill answers in
fluent nonsense — and everything the demo needs from it is already on disk in
`metrics/C-pd.json`. Running it live would cost 8 GPUs and a 20-minute warm-up to reproduce
garbage you can already show. Dropping it is what frees a whole machine for **B**, which turns
the deck's headline claim from an assertion into a live comparison.

---

## 1 · Pre-demo setup — start 60 minutes out (measured 22m 06s; budget 40)

A cold engine start is **not the weights** — it is ~1,200 DeepGEMM JIT compiles, a FlashInfer
autotune and CUDA graph capture. That is longer than the whole slot, which is why nothing starts
on stage. Strategy **D** is never started; its numbers come from `metrics/*.json`, which is why
`just compare` works for shapes that aren't serving.

**Measured, this cluster, all three warming in parallel:**

| engine | ready after | readiness signal |
|---|---|---|
| B | **13m 46s** | RayService `Ready` condition |
| 1M | **16m 35s** | same |
| A | **22m 06s** | same |
| **total from `just teardown`** | **22m 06s** | they warm in parallel, so the total is the slowest |

Two caveats, both load-bearing:

**"Ready" now means the same thing for all three.** Each engine is a RayService, so each
publishes a `Ready` condition set from Serve's own health check of the application — one signal,
the same signal, waited on the same way. Even so, ready is not warm: kernels for a request shape
compile on that shape's first request. Step 5 is what actually settles it.

**22 minutes is with warm caches, and applying all three at once is the slow part.** All three
read the 167 GB checkpoint from the same RWX PVC simultaneously, so they contend for it: the same
engine loaded its weights in 2m 30s when it had the PVC to itself. On genuinely cold nodes expect
longer still — a cold DeepGEMM warm-up alone is ~1,250 kernels, and one fully cold start measured
27m 44s. Budget 40 minutes. If you are short of time, apply them one at a time rather than
together; the total is similar but the first engine is usable much sooner.

```bash
# 1. Point BARE kubectl at this cluster.
#    The eval is required: a just recipe runs in a child process and cannot set the parent
#    shell's env, so `just enable-kube` alone would look like it worked and change nothing.
#    It also probes /readyz and prints the context to stderr, so a wrong-cluster surfaces here
#    rather than on stage.
eval "$(just enable-kube)"

# 2. One clean slate for all three engines.
#    teardown deletes every RayService — and with it the RayCluster each one owns — but KEEPS
#    the model PVC, because re-downloading 167 GB proves nothing.
just teardown

# 3. All three engines in ONE declarative apply.
#    They warm IN PARALLEL: each RayService schedules its 8-GPU worker group on a different
#    machine, so this is 24 of 24 GPUs.
#    Re-running is safe — whatever is already serving is left alone.
#    Do NOT use `just run-a` or `just run-b` here: each begins with its own teardown, so the
#    second would delete the first and they would warm sequentially.
kubectl apply -f 11-strategy-tep8-rayservice.yaml \
              -f 21-strategy-dp8-ep-rayservice.yaml \
              -f 51-longctx-1m-rayservice.yaml

# 4. Wait for all three AT ONCE.
#    This doesn't make it faster — they were already warming together from step 3. It makes a
#    failure visible immediately instead of hiding behind a 20-minute wait on another engine.
#    One signal for all three now: the RayService Ready condition, which Serve sets from its own
#    health check of the application rather than from whether a process is alive.
{ kubectl -n v4-flash-demo wait --for=condition=Ready rayservice/v4-flash-a --timeout=40m >/dev/null \
    && echo "✓ A ready" || echo "⛔ A FAILED"; } &
{ kubectl -n v4-flash-demo wait --for=condition=Ready rayservice/v4-flash-b --timeout=40m >/dev/null \
    && echo "✓ B ready" || echo "⛔ B FAILED"; } &
{ kubectl -n v4-flash-demo wait --for=condition=Ready rayservice/v4-flash-1m --timeout=40m >/dev/null \
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

# 6. Prove the needle BEFORE you're on stage (measured TTFT 63.7 s at 996,068 prompt tokens).
#    A passphrase sits at the MIDPOINT of a 996K-token prompt — not the head, not the tail,
#    because head-and-tail attention passes an edge needle test for free.
#    If it fails on the day, quote the number from slide 6 instead of running it live.
just longctx

# 7. Warm every endpoint at the shape the demo will ask for.
#    READY IS NOT WARM. The kernels for a request shape compile on that shape's FIRST request,
#    so without this the demo's first asks pay the compile on stage — and an unwarmed engine
#    reports a fraction of its real throughput while looking perfectly healthy.
just warm

# 8. Sanity-check the closing beat's inputs.
#    compare reads metrics/*.json, so it also shows D (not serving) and C (never started, and
#    rendered as ✗ because it failed the gate). CONFIRM THAT ✗ COLUMN IS THERE — beat 4
#    depends on it. Leave the output in a spare terminal tab as network insurance.
just compare
```

**Not scriptable, do it now:** open the Ray dashboard for the strategy you'll talk about first
(`just dashboard a` — there is one dashboard per strategy now), open the Nscale
console page **and log in** — it's behind SSO — and have the recorded clips of each beat ready
as a fallback.

---

## 2 · The demo — ~16 minutes, six beats

**The whole demo, in one block.** Run them one at a time and talk in between — the commands total
under three minutes, so the rest of the slot is you. The annotated beats follow underneath.

```bash
# 1. The cluster, honestly: three machines, who holds all 24 GPUs, which endpoints answer.
#    Point at the 24 and the three separate machine names.
just state

# 1b. The Ray dashboard. NAME THE STRATEGY — there are three RayClusters up, one per
#     strategy, so there are three dashboards. `just dashboard` alone means A.
#     BLOCKS until Ctrl-C, so give it its own tab; do not paste it mid-sequence.
#     A second one needs a second port: `just dashboard b 8266`.
just dashboard a

# 2. It talks. One live answer, ~2 s. Says on screen that this is a SHORT prompt,
#    not the million-token one — otherwise the room assumes they just saw that test.
just ask

# 3. Strategy A: TP8 + expert parallel, attention sliced 8 ways. Answers correctly.
just ask-a

# 4. Strategy B: the shape DeepSeek published, DP8 × EP + DSpark. Also correct.
#    Note it hits a DIFFERENT Service — two engines, two machines, same weights.
just ask-b

# 5. The measured difference: warm medians, then seven claims, each printing the two
#    numbers it divided so the arithmetic is checkable on screen.
just claims

# 6. What a throughput benchmark cannot see. Four columns carry numbers; C's carries ✗
#    in every row because it failed the output check, so its timings are withheld.
just compare

# 7. The receipt behind that refusal: C ran at 1922.8 tok/s, zero failed requests,
#    and answered "2+2" with a fragment of Java. Say out loud that C is not running.
just receipt

# 8. The 1M window, served and read: a passphrase at the midpoint of 996,068 tokens.
#    ~65 silent seconds — fill them with the cache arithmetic from slide 6.
just longctx

# 9. Close on the same list. "The published reference shape lost to the simpler one."
just claims
```

Each beat below says what will appear on screen before the command, so you know what you are
pointing at.

**Cut order if squeezed:** beat 2, then beat 3's second command, then beat 5's depth sweep.
Beat 4 is already down to two commands. Don't let it grow back on stage.

### Beat 1 · 0:00–1:00 · Cold open

**What you'll see:** six sections. A node list — three machines with 8 GPUs each, plus three
control-plane nodes with none. Then who holds those GPUs: three lines, one per engine, ending
in "24 GPU(s) held". Then the RayServices and the clusters they own, the PVC with the weights,
and last a list of the endpoints that answer `/v1/models`.

**Point at:** the 24, and the three separate machine names.

```bash
just state
```

> "Three machines, twenty-four B200s, one PVC holding the weights — and all twenty-four GPUs are
> busy right now: strategy A on one machine, the published reference shape B on the second, and
> the million-token engine on the third. Each one is a RayService with its own cluster. All three
> started before the talk, and I'll tell you exactly why in a minute."

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
not a shape you'd choose." The gate is the interesting part, not the bug. The measurement that
says so is on disk in `metrics/C-pd.json`, which `just receipt` prints.

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

While it runs, give the cache arithmetic from slide 6: four times the window costs 1.3× in
concurrency, and the per-token cache cost falls 3.1×.

> "Sixty-four seconds to first token at a million tokens of context, served from one machine's
> HBM. No offloading."

Then one sentence on D, which is not demoed:

> "Scale-out isn't demoed because 'add machines, get throughput' is self-evident — the non-obvious
> part is which shape you replicate. We replicate the winner: 1.75× on two machines, and each
> replica still does 87% of what it did alone."

**If asked why it's 1.75× and not 2×:** that 0.87 per-replica efficiency is the honest answer, and
it held at exactly 0.87 before the RayService migration too — so it is a property of the shape,
not of the orchestration.

**The thing worth telling, if there's time:** D's first measurement on this architecture was
2751 / 491 / 2641 tok/s with TTFT p95 up to 30.5 s — a 5.6× spread that looked like a broken
fleet. Nothing was wrong with the GPUs. `build_openai_app` creates **one** ingress actor no matter
how many engine replicas sit behind it, so both replicas were relaying every streamed token
through a single Python process, and at 64 concurrent that saturated before the hardware did. One
line — `ingress_deployment_config: {num_replicas: 2}` — moved it to 3384–4381 with p95 ~1 s.
Ask what the narrowest thing in the request path is before believing an aggregate.

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
