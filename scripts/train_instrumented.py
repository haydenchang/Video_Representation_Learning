# scripts/train_day6_instrumented.py
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from src.data.clip_dataset import ClipDataset
from src.model.patchify import tubeletify
from src.model.masking import make_random_mask, masked_mse_loss
from src.model.mvm import MaskedVideoModel

from src.train.guardrails import assert_finite_loss, assert_finite_grads, assert_finite_params
from src.train.stats import global_grad_norm, global_param_norm, pred_stats, latent_feat_std
from src.train.hooks import ActivationRecorder
from src.train.tripwires import check_tripwires


# ---------- Determinism ----------
def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


seed_all(1)

# ---------- Config ----------
DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

t, p = 2, 40
T, H, W = 8, 360, 640
N = (T // t) * (H // p) * (W // p)  # 576
D = 3 * t * p * p                  # 9600
mask_ratio = 0.90

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------- Data ----------
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
dataset = ClipDataset(nusc=nusc, dataroot=DATAROOT, index_path=INDEX_PATH)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)

# ---------- Model ----------
model = MaskedVideoModel(N=N, D=D, d_model=256, n_layers=2, n_heads=8).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# ---------- Hooks ----------
rec = ActivationRecorder()
rec.hook_output(model.in_proj, "in_proj_out")
rec.hook_output(model.encoder, "encoder_out", store_tensor=True)  # needed for latent_feat_std
rec.hook_output(model.out_proj, "out_proj_out")

# ---------- Seeded mask RNG ----------
mask_gen = torch.Generator(device="cpu").manual_seed(1)

# ---------- Loop ----------
model.train()
it = iter(loader)
max_steps = 1000

for step in range(max_steps):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)

    video = batch["video"].to(device)       # [B,T,C,H,W]
    tokens_gt = tubeletify(video, t=t, p=p) # [B,N,D]
    B = tokens_gt.shape[0]

    mask_cpu = make_random_mask(B=B, N=N, mask_ratio=mask_ratio, device=torch.device("cpu"), generator=mask_gen)
    mask = mask_cpu.to(device)

    ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    ids_visible = ids[~mask].view(B, -1)
    tokens_visible = tokens_gt[~mask].view(B, -1, D)

    rec.clear()
    pred = model(tokens_visible, ids_visible)
    loss = masked_mse_loss(pred, tokens_gt, mask)

    assert_finite_loss(loss)

    opt.zero_grad(set_to_none=True)
    loss.backward()

    assert_finite_grads(model)
    gnorm = global_grad_norm(model)

    opt.step()
    assert_finite_params(model)

    # Stats
    ps = pred_stats(pred)
    pnorm = global_param_norm(model)

    # Collapse metric on encoder latents (visible tokens)
    x_enc = rec.tensors.get("encoder_out")  # [B, Nv, d_model]
    lstd = latent_feat_std(x_enc) if x_enc is not None else float("nan")

    # Activation stats
    acts = rec.stats
    ip = acts.get("in_proj_out")
    eo = acts.get("encoder_out")
    op = acts.get("out_proj_out")

    def fmt(a):
        return "NA" if a is None else f"m={a.mean:.3f} s={a.std:.3f} a={a.absmax:.3f}"

    # Tripwires (warn/stop)
    act_abs = {}
    if ip is not None: act_abs["in_proj"] = ip.absmax
    if eo is not None: act_abs["encoder"] = eo.absmax
    if op is not None: act_abs["out_proj"] = op.absmax

    if step in [49, 199, 499, 999]:
        os.makedirs("artifacts/checkpoints", exist_ok=True)
        ckpt_path = f"artifacts/checkpoints/mvm_day6_seed1_step{step + 1}.pt"
        torch.save(
            {"model_state": model.state_dict(), "N": N, "D": D, "t": t, "p": p, "d_model": model.d_model},
            ckpt_path,
        )
        print("saved checkpoint:", ckpt_path)

    if step % 10 == 0 or step == max_steps - 1:
        check_tripwires(
            step=step,
            day=6,
            gnorm=gnorm,
            pred_absmax=ps["pred_absmax"],
            act_absmax_values=act_abs,
        )

        print(
            f"Step {step} (Day 6): loss={loss.item():.6f} "
            f"gnorm={gnorm:.4f} pnorm={pnorm:.2f} latent_feat_std={lstd:.6f} "
            f"pred_std={ps['pred_std']:.6f} pred_mean={ps['pred_mean']:.6f} pred_absmax={ps['pred_absmax']:.3f} | "
            f"in_proj[{fmt(ip)}] enc[{fmt(eo)}] out_proj[{fmt(op)}]"
        )


print("DONE")
