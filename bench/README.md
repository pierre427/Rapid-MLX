# bench/

Dev-only micro-benchmarks (not packaged with `pip install rapid-mlx`; for
end-to-end serving benchmarks use `rapid-mlx bench`).

- `bench_radix_vs_hash.py` — multi-tenant prefix-cache index bench (#303):
  N tenants sharing a system prompt, measuring index lookup/insert cost.
- `bench_spec_decode_mtp.py` — MTP speculative-decode bench (#302): decode
  tok/s of `--spec-decode mtp` vs `none` on a Qwen3.5/3.6 MTP checkpoint,
  interleaved runs to avoid thermal drift.
- `continuous_self_mtp_campaign.py` — guarded, service-level Qwen3.8 27B and
  Flash-Next context/quality A/B campaign. Its default `validate`, `plan`, and
  `launch-command` operations are CPU-only; live clients require a two-part
  execution interlock. See
  `docs/engineering/performance/continuous-self-mtp-benchmark-protocol.md`.
