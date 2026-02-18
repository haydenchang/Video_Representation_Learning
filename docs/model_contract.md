# Day 5 — Model Contract (Masked Tubelet Modeling)

## Fixed Inputs
- Video tensor: x ∈ R^{B×T×C×H×W}
- T=8, C=3, H=360, W=640
- Keyframes only (no sweeps), ordered by timestamp

## Tubelet Patchification
Tubelet size:
- temporal tubelet t=2
- spatial patch p=40

Derived:
- T' = T / t = 4
- H' = H / p = 9
- W' = W / p = 16
- N = T' * H' * W' = 576      (tokens per clip)
- D = C * t * p * p = 9600    (values per token; pixel-space tubelet)

Tokenization:
- tubeletify(x) -> tokens_gt ∈ R^{B×N×D}
- untubeletify(tokens, T,H,W) -> x_recon ∈ R^{B×T×C×H×W}
- Round-trip invariant: untubeletify(tubeletify(x)) == x (exact reshape/permutation)

## Masking
- mask ∈ {False,True}^{B×N}, where True = masked
- mask_ratio = TBD (will be set and locked in Day 5/6)
- ids_visible derived from mask for each sample:
  - ids_visible ∈ Z^{B×Nv}, values in [0, N-1]
  - Nv = N - num_masked

## Model Forward API (v1)
Inputs:
- tokens_visible ∈ R^{B×Nv×D}
- ids_visible    ∈ Z^{B×Nv}

Output:
- pred_tokens ∈ R^{B×N×D}

Notes:
- pred_tokens predicts pixel-space tubelet vectors for ALL N positions
- positional information is provided via ids_visible (exact encoding method is model-internal)

## Training Loss (pixel-space)
- target: tokens_gt ∈ R^{B×N×D}
- loss: masked_mse_loss(pred_tokens, tokens_gt, mask)
  - computes MSE only over masked positions
  - does NOT include visible positions in the loss

## Required Sanity Checks
1) Shapes:
   - tokens_gt: [B, 576, 9600]
   - mask:      [B, 576]
2) Loss correctness unit test:
   - perfect pred on masked => loss=0
   - pred=1, target=0 on masked => loss=1
3) Keyframes-only invariant:
   - no filenames under sweeps/
