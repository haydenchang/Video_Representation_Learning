import torch
from src.model.patchify import tubeletify, untubeletify

B, T, C, H, W = 2, 8, 3, 360, 640
t, p = 2, 40

x = torch.rand(B, T, C, H, W, dtype=torch.float32)
tok = tubeletify(x, t=t, p=p)

print("tokens shape:", tuple(tok.shape))

x2 = untubeletify(tok, t=t, p=p, T=T, H=H, W=W)
max_err = (x - x2).abs().max().item()

print("roundtrip max_err:", max_err)
assert max_err == 0.0, "Round-trip should be exact for pure reshape/permutation"
print("OK: round-trip exact")
