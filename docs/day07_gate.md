# Day 7 Probe Gate Runner

This project includes a deterministic Day 7 probe runner that evaluates object-presence linear probes under identical conditions for pretrained and random encoders.

## Run

```bash
python -m scripts.day7_probe_runner --ckpt %MVM_CKPT_PATH%
```

PowerShell equivalent:

```bash
python -m scripts.day7_probe_runner --ckpt $env:MVM_CKPT_PATH
```

Sanity-only fast rerun (scene overlap, token overlap, alignment logging, tiny overfit, shuffled-label collapse):

```bash
python -m scripts.day7_probe_runner --ckpt %MVM_CKPT_PATH% --only_sanity
```

If `--ckpt` is omitted (or `MVM_CKPT_PATH` is unset), the script still runs the random baseline and marks pretrained runs as missing checkpoint.

## Outputs

- `artifacts/day7_probe_report.json`
- `artifacts/day7_probe_report.md`

Both reports include:

- Config snapshot (paths, seeds, pooling, optimizer, epochs, sample caps)
- Per-run metrics for pretrained vs random
- Sanity-check outcomes
- PASS/NOT PASS decision with reasoning

## What the runner enforces

- Scene-disjoint deterministic split with zero train/val scene overlap
- Identical probe conditions for pretrained and random:
  - Same split
  - Same probe head seed/init
  - Same optimizer settings
  - Same epoch budget
  - Same feature extraction pipeline
- Metrics:
  - mAP computed from sigmoid probabilities
  - Macro-F1 threshold sweep at thresholds `0.1..0.9` with best threshold reported
- Finite ablations:
  - Pooling: `mean_visible`, `mean_all`, `mean_excluding_cls`
  - Optimizer: `adamw`, `sgd` (momentum `0.9`)
- Sanity checks (fail-fast):
  - Scene overlap check
  - Shuffled-label collapse check
  - Tiny-subset overfit check (200 samples, train mAP threshold)
  - Label/feature alignment logging for 10 random examples

## PASS criterion

Default gate config is:

- pooling: `mean_excluding_cls`
- optimizer: `adamw`
- seeds: `[1, 2, 3]`

PASS requires:

- `mean(val mAP_pretrained - val mAP_random) >= 0.01` across seeds `[1,2,3]`
- The runner enforces seeds `[1,2,3]` for this gate.

The report also includes standard deviation across seeds and top likely causes if NOT PASS.
