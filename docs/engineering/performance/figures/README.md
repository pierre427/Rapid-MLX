# Continuous self-MTP paper figure artifacts

These artifacts are candidates for the working paper. They do not modify the
paper source or rendered PDF.

## Figures

- `dense27b-throughput-scaling.svg` / `.png`: aggregate and per-lane scaling,
  with the single-process sweep, isolated fresh-process points, and the real
  controller-selected `N=40` run kept as distinct protocols.
- `dense27b-admission-calibration.svg` / `.png`: measured pre-fix admission and
  the source/test-derived post-fix curve after applying the measured `N=16`
  compute knee, plus the separately measured consequence of admitting `N=40`.
- `continuous-self-mtp-workflow.svg` / `.png`: conceptual comparison of normal
  autoregressive decode, single-request self-MTP, and continuous self-MTP
  batching.
- `flashnext-throughput-admission.svg` / `.png`: Flash-Next's memory-bound
  ladder and its real controller admission curve.
- `two-ceiling-model-comparison.svg` / `.png`: dense-27B and Flash-Next as the
  compute-bound and memory-bound cases of the same `min(memory, compute)` rule.
- `rapid-qwen35-random-head-smoke.csv`: recovery manifest for the later Rapid
  real-target/random-head diagnostic. Every row is session-trace-attested,
  zero-acceptance, and explicitly not a trained-head performance result.

The SVG files are the paper-quality vector masters. The PNG files are
1800-pixel-wide review/rendering copies. Regenerate them with:

```bash
python3 docs/engineering/performance/figures/generate_dense27b_figures.py
```

## Evidence boundary

The in-process `N=1,2,4,8,16` sweep is raw-log-attested by the lab-root file
`/Users/pierrelamy/Desktop/mlx-uag/results/window-dense27b-lane-ladder/static_dynamic_27b.log` (SHA-256
`7a891022361e98253839b2a34df8f298d64d1318275903841bd3f181358e5f58`).
The pre-fix controller curve and `N=40` execution are raw-log-attested by the
lab-root file `/Users/pierrelamy/Desktop/mlx-uag/results/window-dense27b-lane-ladder/dynamic_only_27b.log` (SHA-256
`921797adb07ea66e86cd4754618e04446270b7de349320e94c7fbe359570443a`).

The fresh-process `N=16,20,24,32` values were printed to the Claude terminal
without per-run log redirection. They are preserved in Claude session
`1c3a4880-ce64-415d-9bb0-b06f7a29f73c`, the campaign `SUMMARY.md`, and the two
user-supplied screenshots. They are therefore labelled
`transcript/screenshot-attested`, not raw-log-attested or artifact-complete.
The screenshot SHA-256 values are:

- throughput: `dfa65163c484efacae9d765dead15e426e097f36024235888511ad423b1705bb`
- admission: `2e9631ed8bfa9e7c51bde17e468854edd3f451fd0a57442ecebb0cef32c53bce`

The figures deliberately do not combine the accumulation-affected
`peak_27b.log` values with the clean fresh-process series.

The post-fix curve is derived by capping each measured pre-fix memory decision
at `N=16`, as implemented by
`SelfMTPLaneAdmissionController.SATURATION_LANE_CAP` in source commit
`1a0a2474`; it is not a second measured controller sweep. The same commit makes
the MoE-calibrated 1.76 GiB/lane transient configurable for dense deployments;
it does not replace the default with the dense estimate.

Flash-Next is raw-log-attested throughout:

- static ladder (lab-root path):
  `/Users/pierrelamy/Desktop/mlx-uag/results/window-flashnext-lane-ladder/flash_next_ladder.log`, SHA-256
  `1dffc73039c4f383023939ee3408a369d299a470532567009b12c1522441cbcf`
- controller decisions and `N=10` run (lab-root path):
  `/Users/pierrelamy/Desktop/mlx-uag/results/window-flashnext-lane-ladder/flash_next_dynamic.log`, SHA-256
  `6448fd3a9203c742009f5e65e3be7541da1bf58fd24a32cdae53096c406db767`

The Flash operating point requires the PLE-NVMe sidecar. The raw harness sets
`MLX_QWEN4_PLE_NVME` explicitly; without it, PLE remains resident and the
controller sees almost no memory above its hard reserve. The wiki companion
record is root commit `51045285`.

The lab-root logs above are not packaged in this Rapid worktree. The CSVs keep
their evidence grades adjacent to plotted values; publication should either
copy an immutable evidence bundle into the final docs PR or link a stable
artifact location.

The Rapid random-head recovery CSV is similarly not a raw benchmark artifact.
It exists to prevent the 4.74x diagnostic from being detached from its
test-only head and 0% acceptance boundary.
