# Design Doc (v1) — Masked Video Modeling on nuScenes CAM_FRONT

## Problem & Goal
We want a self-supervised video representation from driving scenes that transfers to a simple downstream proxy task.
Success is not SOTA; success is a credible pipeline + scaling/system analysis.

## Data
Dataset: nuScenes v1.0, sensor CAM_FRONT only.
We sample short clips of length T=[6–8] with stride s=[1 default].
We will enforce deterministic sampling (seeded) and continuity checks.

## Self-Supervised Objective
We use masked video modeling: mask a high fraction of spatiotemporal patches and reconstruct only the masked content.
Loss is computed only on masked regions to force prediction from context.

## Model
- Patch embedding: CNN produces patch tokens per frame.
- Temporal encoder: transformer over time (and possibly space tokens, depending on implementation).
- Reconstruction head: lightweight decoder that predicts masked patches/pixels.
We prioritize stability and reproducibility over complexity.

## Observability & Debugging
We will log:
- reconstruction loss curve
- training stability signals (NaNs/inf, gradient norms if needed)
- embedding collapse detector (e.g., embedding norm / variance over batch)
We will produce sanity visualizations:
- original vs masked vs reconstructed frames (small batch)

## Evaluation Gate (must pass before scaling)
Freeze encoder. Train a linear probe on a proxy label task (TBD Day 7).
Compare probe accuracy (or appropriate metric) for pretrained vs random encoder.
If pretrained does not beat random → fix training/data before any scaling.

## Scaling Experiments (later days)
Primary: dataset scaling (×2, ×4) on RCAC; track probe performance vs data, throughput, memory.
Systems: DDP run to measure step time and comm overhead.
Secondary: temporal scaling (T=4,8,12) to show tradeoffs.

## Risks / Failure Modes
- Data pipeline bugs: non-contiguous frames, wrong ordering, nondeterminism.
- Collapse: embeddings become constant; probe ~= random.
- Reconstruction “cheating” if loss/mask implemented incorrectly.
- Overfitting small subset: good recon loss but poor probe transfer.
