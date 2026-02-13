import torch
from src.model.masking import make_random_mask, masked_mse_loss

B, N, D = 2, 576, 9600
mask_ratio = 0.9

gen = torch.Generator().manual_seed(123)

target = torch.zeros(B, N, D)
pred = torch.zeros(B, N, D)

mask = make_random_mask(B=B, N=N, mask_ratio=mask_ratio, device=torch.device("cpu"), generator=gen)

# Case 1: perfect prediction => loss 0
loss0 = masked_mse_loss(pred, target, mask)
print("loss0:", float(loss0))
assert float(loss0) == 0.0

# Case 2: pred=1 on masked positions => loss should be 1
pred2 = pred.clone()
pred2[mask] = 1.0
loss1 = masked_mse_loss(pred2, target, mask)
print("loss1:", float(loss1))
assert abs(float(loss1) - 1.0) < 1e-6

print("OK")
