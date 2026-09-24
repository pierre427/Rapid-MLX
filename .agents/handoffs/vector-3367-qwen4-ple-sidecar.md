# Vector handoff: Qwen4 PLE file-backed sidecar (#3367)

- Owner: Vector
- Source PR: #3367
- Integration branch: `integration/pr-3367-ple-current`
- Mac mini worktree: `/Users/raullenmini/orca/worktrees/pr3367-ple`
- Base: `0c049254ed9229a97f436548f1d28478e35524e6`

## Scope

Port Pierre Lamy's opt-in Qwen4 q4/group32 PLE reader from the retired
`vllm_mlx` package path to current `rapid_mlx`, preserve attribution, and
establish whether its memory reduction justifies the synchronous host lookup
on real inference. This work must not trigger or prepare a release.

## Verified on Mac mini

- The 32 focused CPU contracts pass after the package-path port. Ruff check,
  Ruff format check, and `git diff --check` pass for the changed Python files.
- The Apple-Silicon CI lane now runs the focused sidecar contracts; the source
  PR left them under `scripts/`, outside normal pytest collection and without a
  workflow invocation.
- The existing Qwen4 suite reaches 114/115 tests in the local verification
  environment. The sole failure is an unchanged MTP test calling
  `mx.full_like`, which this Mac mini's installed MLX does not provide.
- A same-geometry synthetic lookup probe (128 shards, 160-wide q4/group32,
  16 rows per decode token) was bit-exact. Its file was only 12.5 MiB and hot
  in the page cache, so its timing is implementation smoke evidence only and
  must not be used as the production speed gate.

## Exact production artifact and capacity blocker

The published source checkpoint is
`rapid-mlx/Qwen3.8-Flash-Next-4bit` at immutable revision
`dcf657e4acda2aae72da99cde65b6c491cd96998`. Its config declares one PLE
layer, 128 shards, 320,001,536 total rows, 160 dimensions, and q4/group32 for
PLE. The packed table is therefore exactly 32,000,153,600 bytes. The public
index SHA-256 is
`1ffe41e4484e7dc137900b12b14ae29cf92d1737e89fbe72c4fa2dd1b42ee7f1` and
reports 104,681,488,408 bytes of model tensors.

This Mac mini has 32 GiB unified memory and about 34 GiB free disk. The
resident baseline requires roughly the full 104.7 GB tensor set; the offloaded
variant still leaves roughly 72.7 GB before runtime state. Neither arm can run
on this host, and downloading the source plus writing the 32.0 GB sidecar also
exceeds free disk. A same-host off/on A/B here would be fabricated evidence.

The immutable public repository has no `ple_rows.bin` or sidecar manifest. The
only located inventory receipt points to Pierre's local
`/Users/pierrelamy/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP/ple_rows.bin` and
records 32,000,153,600 bytes. The PR contains a reader but no artifact builder
or published sidecar, so users currently have no reproducible path to activate
it.

## Remaining merge gate

Run separate clean processes on a host that can load the 104.7 GB resident
baseline (practically 128 GB or larger), using one immutable checkpoint and a
source-bound sidecar. Record peak resident/Metal memory, TTFT, decode tok/s,
greedy output parity, the 32,000,153,600 removed bytes, exact hashes, and the
full invocation. Also supply a reproducible builder or publish the immutable
sidecar and manifest. Keep the PR out of the queue until both the artifact path
and isolated user-benefit gate exist.
