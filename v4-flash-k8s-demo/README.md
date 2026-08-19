# DeepSeek-V4-Flash on Ray + vLLM — Ray Summit demo (Kubernetes, B200)

Serving `deepseek-ai/DeepSeek-V4-Flash-0731` (284B MoE / 13B active, FP4+FP8,
167 GB measured, DSpark speculator built in) on Nscale B200 nodes. The model fits
one 8× B200 machine with room to spare, so **every bandwidth-hungry strategy
runs inside one machine's NVLink domain** — and extra machines are used for
what they're best at: whole replicas.

The weights are downloaded **once** to a shared RWX PVC; each strategy is a
`kubectl apply` / `delete` — engines restart in minutes, nothing re-downloads.

## Strategy map

| # | File | Shape | Where | Story |
|---|------|-------|-------|-------|
| A | `10-strategy-tep8.yaml` | TEP8 (RayJob) | 1 machine (the Ray head — see `03`) | the latency shape — and, on this hardware, also the throughput winner |
| B | `20-strategy-dp8-ep.yaml` | DP8×EP + MegaMoE + DSpark (RayJob) | 1 machine | the model card's reference config, widened from DP4/4×GB300 |
| B′ | `20b-strategy-dp8-ep-nospec.yaml` | same, `--speculative-config` removed | 1 machine | isolates what DSpark costs — the only honest way to claim a number for it |
| C | `30-strategy-pd-single-node.yaml` | TP4 prefill → DP4×EP decode over NIXL, two RayJobs + `vllm-router` | 1 machine, 4+4 GPUs placed by Ray | P/D end-to-end on one machine, and the largest cache budget |
| D | `40-strategy-replicas.yaml` (+ optional `40b`) | whole replicas of **A** behind a Service or `vllm-router` | 2 machines | scale-out. Replicate the winner, not the reference |
| 1M | `50-longctx-1m.yaml` | strategy A's shape with `--max-model-len 1048576` | 1 machine | the native window, *served* — the cache-is-the-product claim, demonstrated rather than stated |

**Numbers live in `metrics/`, not in this README** — run `just compare` for the table and
`just stability` for the spread across repeats. Hardcoding them here went stale twice in one
day: once when the harness moved from counting SSE chunks to counting tokens (which alone
moved B from 300 to 911 tok/s, because chunk-counting penalises the one config using
speculative decoding), and once when warming moved ahead of the latency pass. Quote a number
only if `just stability` shows a spread below ~1.5× for it.

## This cluster (measured on the Nscale `ray-demo` cluster, not assumed)

| | |
|---|---|
| Compute | 3x `g.192.b200.8` **bare metal** — 8x B200 (183 GB each), 192 CPU, 2 TB RAM = **24 GPUs** |
| Control plane | 3x 4-CPU/16 GB VMs, tainted `NoSchedule`, ~2.5 GB RAM free each |
| Block storage | only `cinder`, and it **cannot attach to the bare-metal nodes** (attach times out; fine on the control-plane VMs) |
| Shared storage | none out of the box → we add an NFS provisioner (below) for the RWX class |
| Fabric | `mlx5_bond_0` Ethernet 200 Gb/s ACTIVE; GPU-side IB links DOWN. Irrelevant here: every strategy but D stays inside one machine's NVLink domain |
| Stage setup | `just stage-up` runs **C (8 GPUs) + D (16 GPUs)** together and warms them — the demo never starts an engine, because cold starts are 11–25 min |
| Model | `DeepSeek-V4-Flash-0731` measures **167 GB** across 74 files, DSpark inside |

## Prereqs (one-time, cluster-level)

Everything cluster-level lives in one idempotent script — namespace, `ib_umad`
loader, gpu-operator (driver 580.126.20 = CUDA 13), KubeRay, and the NFS server
that provides the `shared-nfs` RWX class. Pinned to the versions verified here:

```bash
./prereq-install.sh      # applies prereq-node-prep.yaml, then 3 helm charts, then verifies
```

The script applies `prereq-node-prep.yaml` first because the node image doesn't load `ib_umad`,
and without it the driver container's fabric manager can't reach the NVLink subnet
manager: fabric state sticks at `In Progress`, every CUDA call returns "system not
yet initialized", and `nvidia.com/gpu` never appears. If you install in the wrong
order, `kubectl delete pod -n gpu-operator -l app=nvidia-driver-daemonset` afterwards
— the driver container only sees `/dev/infiniband/umad*` if they existed at its start.

The script ends by printing the three things that must look right: 8 GPUs per
B200 node, `Fabric State: Completed / Success`, and the `shared-nfs` class.

## One-time setup (before the talk)

```bash
kubectl apply -f 01-model-pvc.yaml               # RWX on shared-nfs
kubectl apply -f 02-model-download-job.yaml      # 167 GB + the ray overlay, once
kubectl -n v4-flash-demo logs -f job/v4-flash-download
kubectl apply -f 03-raycluster.yaml              # Ray head owns the machine's 8 B200s
```

**Warm the page cache before you present.** Measured from a B200 node: cold NFS
read 549 MB/s, cached read 6.4 GB/s. So the first engine start pulls 167 GB in
~5 minutes, and every later start on the same node reads from that node's page
cache (2 TB of RAM — the whole checkpoint fits) in well under a minute. Read the
weights once per machine before the talk:

```bash
w=$(kubectl -n v4-flash-demo get pod -l ray.io/node-type=head -o name | head -1)
kubectl -n v4-flash-demo exec $w -- sh -c 'cat /models/DeepSeek-V4-Flash-0731/*.safetensors > /dev/null'
# repeat on the replica machines before strategy D — or just run each strategy once
```

## Measuring each strategy (`just`)

```bash
just                 # list recipes
just state           # nodes, who holds the GPUs, RayCluster, PVC, and which model is served

just run-a           # teardown -> apply 10- -> wait -> measure -> leave it serving
just run-b           # same for 20-  (tears down A first)
just run-c           # same for 30-
just run-d           # tears down the RayCluster too — D wants both machines
just run-all         # the whole tour in talk order, ending in `compare`

just compare         # all strategies side by side, each next to the point it makes
just probe-d         # measure each D replica ALONE — a fleet number hides a cold pod
just longctx         # needle-in-a-haystack at the full 1M window
just longctx-sweep   # TTFT vs context length + needle depth sweep
just logs            # follow the running RayJob's driver log
just teardown        # free the GPUs, keep the RayCluster + weights
just teardown-cluster # also drop the RayCluster
```

**24 GPUs, so some strategies coexist and some don't.** A, B and C each want 8 and D wants
16: `C + D = 24` (what `just stage-up` builds for the demo) and `A + B + C = 24` both fit;
all four (40) do not. The `run-*` recipes still tear down first, because measuring one at a
time is what makes `just compare`'s columns comparable — they block until the GPUs are
actually released, so the tour can't wedge itself half-deployed. `bench-*` measures whatever
is already serving.

`bench.py` is stdlib-only (no venv): it streams completions and takes TTFT and TPOT
from the SSE chunks themselves, in two passes — single-stream (the latency claim)
and concurrency 32 (the throughput claim). Results are JSON under `metrics/`, so
`just compare` still works after the pods are gone. Every strategy manifest repeats
its own `kubectl apply` + `just bench-*` at the bottom of the file.

Three candidate images all claim V4-Flash support and none are interchangeable. The pin is
**`vllm/vllm-openai:v0.27.1`** — the only *released* tag with both `dspark` and
`deep_gemm_mega_moe`, so slide 7's command runs verbatim:

| | `deepseekv4-cu130` | `dsv4-megamoe-…4ba0a72` | **`v0.27.1` (pinned)** |
|---|---|---|---|
| `--moe-backend deep_gemm_mega_moe` | ✗ (only `deep_gemm`) | ✓ | ✓ |
| `--speculative-config method=dspark` | ✗ | ✗ | **✓** |
| torch.compile on this checkpoint | **crashes** (Inductor assertion) | ok | ok |
| strategy A: does it serve at all? | only with `--enforce-eager` | yes | **yes** |

`deepseekv4-cu130` needs `--enforce-eager` to start at all, and eager mode costs ~20×
(5.8 vs 105 tok/s single-stream). The megamoe dev build is fast but has no `dspark`.
**Never compare numbers across these images** — the same manifest gave a 45× different KV
budget between builds, which is how a "45× TP-vs-DP finding" turned out to be an artefact.

## The 1M window

Strategies A–D all serve `--max-model-len 262144`. That is a deliberate setting, not a
limit of the model: 256K is the demo's point on the context ↔ throughput dial. `50` exists
so the 1M claim can be shown.

```
kubectl -n v4-flash-demo scale deploy/v4-flash-replicas --replicas=1   # free a machine
kubectl apply -f 50-longctx-1m.yaml
just longctx                 # needle-in-a-haystack at ~1M tokens
just longctx-sweep           # TTFT vs context length, then the needle at five depths
```

**Measured** — in-cluster, prefix caching defeated, needle at mid-depth:

| prompt tokens | TTFT | step |
|---|---|---|
| 127,592 | 7.9 s | |
| 249,098 | 12.0 s | ×1.95 tokens → ×1.52 TTFT |
| 498,074 | 19.8 s | ×2.00 tokens → ×1.65 TTFT |
| **996,068** | **57.9 s** | ×2.00 tokens → ×2.92 TTFT ← the knee |

Overall **7.8× the context for 7.3× the TTFT — roughly linear, not sub-linear.** Cost grows
slower than context up to ~500K, then the last doubling is where it turns. Don't blend this
with a curve from the 262144 engine; different `max_model_len` means a different KV layout,
and mixing them is the same error `just compare` warns about for images.

Needle **FOUND at all five depths** (0.05 / 0.25 / 0.50 / 0.75 / 0.95) at 996,068 tokens.

**Why a needle test and not just a long prompt.** Accepting a 1M-token prompt proves
nothing; answering from a fact buried in the middle of it does. `longctx.py` hides a
passphrase at a chosen depth and checks it comes back. The depth sweep matters because a
model that attends only to the head and tail of its window still passes a needle test at
depth 0.0 or 1.0 — the middle depths are what exercise CSA's token selection.

**Two ways this measurement lies, both hit while building it:**

1. **Prefix caching makes long prompts look cheap.** Successive runs sharing a filler prefix
   let the engine serve most of the window from cache. A sweep produced TTFT
   7.9 / 39.8 / 42.7 / **6.0 s** at 5K / 20K / 79K / 155K prompt tokens — the longest prompt
   looked cheapest. `longctx.py` now puts a unique tag in the first line of every prompt to
   defeat it. Prefix caching is a real feature and worth demoing on its own; it is just not
   what "prefill cost at 1M" means.
2. **"~4 chars per token" is wrong by 45% here.** This tokenizer and filler measure 5.81
   chars/token, so a request built for "250K" was really 155K. The script reports measured
   chars/token every run, and only `usage.prompt_tokens` is trusted for length.

**Cache arithmetic — measured, and it contradicts the intuitive version.** The naive story
is "A reports ~5.5M KV tokens, so a 1M request eats a fifth and ~5 fit". Measured, that is
wrong. Both engines are the same TEP8 shape on the same 8× B200 with the same 34 engine args;
only `--max-model-len` differs:

| window | KV cache tokens | bytes/token | max concurrency |
|---|---|---|---|
| 262,144 | 5,503,901 | 26,973 | 21.0× |
| 1,048,576 | **15,626,973** | **9,431** | 14.9× |

Same ~138 GiB of KV memory, 2.84× more tokens in it. Per-token cost *falls* 2.86× as the
configured window grows, so a **4× window costs 1.41× in concurrency, not 4×**. The likely
mechanism — inferred from the measurement, not read from the source — is the two-tier
attention: with a longer window a larger share of each sequence sits in HCA's heavily
compressed tier, so average bytes/token drops. It is the concrete version of the model card's
"~10% of V3.2's cache at 1M".

Reproduce with `kubectl -n v4-flash-demo logs deploy/v4-flash-longctx | grep -E "KV cache size|Maximum concurrency"`.

## Demo flow (~18 min)

### 1. Strategy A — TEP8, the latency shape

```bash
kubectl apply -f 10-strategy-tep8.yaml
kubectl -n v4-flash-demo port-forward svc/v4-flash-openai 8000:8000 8265:8265 &
# Ray dashboard on :8265 — placement group across all 8 GPUs of one machine
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "messages": [{"role":"user","content":"One sentence: why keep expert routing on NVLink?"}],
  "max_tokens": 128
}' | jq -r '.choices[0].message.content'
```

### 2. Strategy B — DP8×EP + MegaMoE + DSpark, the published reference shape

```bash
kubectl -n v4-flash-demo delete rayjob serve-tep8
kubectl apply -f 20-strategy-dp8-ep.yaml
```

Talking points: same weights, no re-download; MegaMoE keeps expert routing on
NVLink; DSpark visibly speeds single-stream decode. Compare tok/s vs strategy A
at concurrency 1 and 32.

### 3. Strategy C — P/D disaggregation on one machine

```bash
kubectl -n v4-flash-demo delete rayjob serve-dp8-ep
kubectl apply -f 30-strategy-pd-single-node.yaml
kubectl -n v4-flash-demo port-forward svc/pd-router 8080:8000 &
```

Talking points: TP4 prefill (GPUs 0–3) → DP4×EP decode (GPUs 4–7); NIXL moves
paged KV + the indexer cache between pools per request; different pool shapes,
one connector. `watch nvidia-smi` on the node makes the two pools visible.

### 4. Strategy D — replicas across machines + failure

`40-` replicates **strategy A's** shape, not B's: replication multiplies whatever you put in
it (measured close to linear per machine), so replicating the slower config produced less on
two machines than one machine of A did on its own. Scale out the winner. Current numbers:
`just compare`.

Three layers decide whether a dying replica loses requests:
1. **Graceful drain** (in `40-`): `terminationGracePeriodSeconds: 300` + a preStop sleep.
   Fixes planned disruptions — on SIGTERM the pod leaves the Service's endpoints while
   in-flight generations finish. Without it (30 s default, 91 s generations) a
   `kubectl delete pod` killed 31 of 64 in-flight requests.
2. **A retrying router** (`40b-replicas-router.yaml`, optional): `vllm-router` retries 5×
   with backoff and circuit-breaks a bad worker. This covers failures *before* the first
   token. A bare Service cannot — it has no concept of a failed request.
3. **Client retry**: the only cure once a 200 and some tokens are already on the wire.
   No proxy can resume someone else's half-sent stream.

```bash
kubectl delete -f 30-strategy-pd-single-node.yaml      # both RayJobs + the router
kubectl -n v4-flash-demo delete raycluster v4-flash   # frees the head's 8 GPUs
kubectl apply -f 40-strategy-replicas.yaml
kubectl -n v4-flash-demo port-forward svc/v4-flash-replicas 8090:8000 &
# failure demo (kills a replica that is actually being loaded, and measures it):
kubectl -n v4-flash-demo delete pod -l app=v4-flash-replica --field-selector=... # pick one
```

Talking points: throughput multiplies with zero model traffic between machines;
the Service reroutes around the killed replica while k8s restarts it.

## Cleanup

```bash
kubectl delete ns v4-flash-demo   # removes the PVC (and the download) — keep it if re-running
```

## Notes / gotchas found on this cluster

- **The vLLM image ships no ray** — no `ray` binary, no `ray` module in
  `vllm/vllm-openai:deepseekv4-cu130` (nor in the Kimi image). KubeRay's
  containers crashloop with `ray: command not found`. `02` stages
  `ray[default]==2.49.0` onto the PVC at `/models/ray-site`, pruning anything the
  image already provides so vllm's pinned deps aren't shadowed; `03` mounts it via
  `PATH`/`PYTHONPATH`. Verified: `import ray, vllm` both work off the overlay.
  **RayJob submitter pods need the same treatment** — `submissionMode: K8sJobMode`
  starts a submitter from this image, so either give its `submitterPodTemplate` the
  mount + env or switch to `submissionMode: HTTPMode`.
- **Image pinned to `vllm/vllm-openai:v0.27.1`** — the only released tag that has both
  `--moe-backend deep_gemm_mega_moe` and `--speculative-config method=dspark`, so every
  manifest here runs as written. Verified inside the image: `DSparkModelTypes =
  Literal["dspark"]`, `deep_gemm_mega_moe` in the `MoEBackend` literal,
  `use_fp4_indexer_cache` in `config/attention.py`, `deepseek_v4` tokenizer/tool/reasoning
  parsers, and `draft_sample_method: greedy`. The two dev tags each lack one of those —
  see the table above. **`dspark` needs vLLM ≥ 0.27.1**; `DSparkDraftModel` itself lives in
  the separate `speculators` package (0.7.0+), but you don't need it installed: vLLM reads
  the draft out of the target checkpoint, whose `config.json` carries `dspark_block_size`,
  `dspark_noise_token_id`, `dspark_target_layer_ids` and `dspark_markov_rank`.
- **torch.compile crashes on `deepseekv4-cu130`** (the older candidate):
  `InductorError: AssertionError` in `decompose_triton_kernel_wrapper_functional`
  during post-grad passes, on all 8 workers, right after the weights load. Only
  `--enforce-eager` gets past it — which costs ~20× (see the table above). **The pinned
  `v0.27.1` does not hit it**: every strategy in this repo ran on it with compilation
  enabled and full CUDA-graph capture (51 decode graphs, ~3.2 GiB graph pool), so this is
  verified by execution, not assumed. Don't "upgrade" back to the older `deepseekv4-cu130`
  tag to get MegaMoE — v0.27.1 has it.
- **A RayJob entrypoint needs a GPU on the pod it runs on.** `vllm serve` builds
  its `DeviceConfig` while argparse defaults are being constructed, so on a
  GPU-less pod it dies with `Failed to infer device type` before reading any flag
  — not even `--help` works. That's why `03` puts the 8 GPUs on the **head** and
  has no worker group. Ray is still the executor (`--distributed-executor-backend
  ray`), so the dashboard still shows the placement group and its 8 actors.
- **KubeRay's default health probes shell out to `wget`, which this image lacks.**
  Without the `curl`-based probes in `03` the head stays NotReady forever: no
  Service endpoints, and RayJob submission never starts.
- Cinder PVCs hang forever attaching to the bare-metal B200 nodes; the RWX class
  is an NFS server on a control-plane VM. Those VMs have ~2.5 GB RAM free, so keep
  the server's requests small.
- vLLM prefetches the checkpoint into page cache on its own (it logs
  `Filesystem type for checkpoints: NFS4. Checkpoint size: 155.43 GiB. Available
  RAM: 1934 GiB`), which is what makes the second and later engine starts fast.
- fp8 KV and `--block-size 256` travel together — every published V4-Flash
  command pins the pair.
- `--attention-config '{"use_fp4_indexer_cache": true}'` shrinks the
  sparse-attention indexer cache — a second cache to remember in memory budgets.
- DSpark draft weights are inside the main checkpoint: `--speculative-config`
  only, no draft-model path.
- Reasoning effort levels are low / high / max; DeepSeek allows up to 384K
  output tokens at high/max — cap `max_tokens` per workload. Sampling:
  temperature 1.0; top_p 0.95 for agentic use, 1.0 otherwise.
- The published spec-decode kernel caveat is for sm_120-class GPUs; B200 is a
  different Blackwell variant — still, dry-run DSpark on your exact image.
- The P/D proxy is the minimal two-phase example pattern — swap in your
  production router for real traffic.
