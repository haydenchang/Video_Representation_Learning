import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from nuscenes.nuscenes import NuScenes

from src.model.mvm import MaskedVideoModel
from src.probes.ego_motion_probe_dataset import EgoMotionProbeDataset
from src.probes.ego_motion_labels import EGO_LABELS
from src.probes.splits import scene_disjoint_split

def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def macro_f1_3class(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # y_true, y_pred: [N] int64 in {0,1,2}
    f1s = []
    eps = 1e-8
    for k in [0, 1, 2]:
        tp = ((y_pred == k) & (y_true == k)).sum().item()
        fp = ((y_pred == k) & (y_true != k)).sum().item()
        fn = ((y_pred != k) & (y_true == k)).sum().item()
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        f1s.append(f1)
    return float(np.mean(f1s))

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
CKPT_PATH = os.environ.get("MVM_CKPT_PATH", r"artifacts\checkpoints\mvm_day6_seed1_step1000.pt")  # must exist

device = torch.device("cpu")
seed_all(123)

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
print("Using CKPT_PATH:", CKPT_PATH)
N = ckpt["N"]; D = ckpt["D"]
t = ckpt["t"]; p = ckpt["p"]
d_model = ckpt["d_model"]

def make_encoder(pretrained: bool) -> MaskedVideoModel:
    enc = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=2, n_heads=8).to(device)
    if pretrained:
        missing, unexpected = enc.load_state_dict(ckpt["model_state"], strict=True)
        print("load_state_dict missing:", missing)
        print("load_state_dict unexpected:", unexpected)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc

def debug_encoder_diff():
    enc_rand = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=2, n_heads=8).to(device)
    enc_pre  = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=2, n_heads=8).to(device)

    # IMPORTANT: ensure load happens and report result
    missing, unexpected = enc_pre.load_state_dict(ckpt["model_state"], strict=True)
    print("load_state_dict missing:", missing)
    print("load_state_dict unexpected:", unexpected)

    # Compare a few key params + global diff
    with torch.no_grad():
        total_l2 = 0.0
        total_l2_pre = 0.0
        total_l2_rand = 0.0
        max_abs = 0.0
        max_name = None

        for (n1, p_rand), (n2, p_pre) in zip(enc_rand.named_parameters(), enc_pre.named_parameters()):
            assert n1 == n2
            d = (p_pre - p_rand).float()
            total_l2 += float(d.pow(2).sum().item())
            total_l2_pre += float(p_pre.float().pow(2).sum().item())
            total_l2_rand += float(p_rand.float().pow(2).sum().item())

            m = float(d.abs().max().item())
            if m > max_abs:
                max_abs = m
                max_name = n1

        print("rand param L2:", total_l2_rand ** 0.5)
        print("pre  param L2:", total_l2_pre ** 0.5)
        print("pre-vs-rand L2 diff:", total_l2 ** 0.5)
        print("max abs diff:", max_abs, "at", max_name)

debug_encoder_diff()


def run_probe(pretrained: bool, split_seed: int = 123) -> dict:
    encoder = make_encoder(pretrained=pretrained)

    ds = EgoMotionProbeDataset(
        nusc=nusc,
        dataroot=DATAROOT,
        index_path=INDEX_PATH,
        encoder_model=encoder,
        t=t,
        p=p,
        device=device,
    )

    train_idx, val_idx = scene_disjoint_split(ds.items, val_scene_frac=0.2, seed=split_seed)
    train_scenes = {ds.items[i].scene_idx for i in train_idx}
    val_scenes = {ds.items[i].scene_idx for i in val_idx}
    print(
        "TRAIN clips:", len(train_idx),
        "VAL clips:", len(val_idx),
        "TRAIN scenes:", len(train_scenes),
        "VAL scenes:", len(val_scenes),
        "scene overlap:", len(train_scenes & val_scenes),
    )

    probe_seed = 10_000 + split_seed
    torch.manual_seed(probe_seed)
    train_gen = torch.Generator(device="cpu").manual_seed(probe_seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=16, shuffle=True, generator=train_gen)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=16, shuffle=False)

    torch.manual_seed(probe_seed)
    clf = nn.Linear(d_model, 3).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # train
    clf.train()
    for epoch in range(25):
        for z, y_onehot in train_loader:
            z = z.to(device)
            y = y_onehot.argmax(dim=1).long().to(device)  # [B]
            logits = clf(z)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # eval
    clf.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for z, y_onehot in val_loader:
            y = y_onehot.argmax(dim=1).long()
            logits = clf(z.to(device)).cpu()
            pred = logits.argmax(dim=1).long()
            all_y.append(y)
            all_pred.append(pred)

    y_true = torch.cat(all_y, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    acc = float((y_true == y_pred).float().mean().item())
    f1 = macro_f1_3class(y_true, y_pred)
    return {"acc": acc, "macro_f1": f1, "val_size": int(y_true.numel())}

r_pre = run_probe(pretrained=True, split_seed=123)
r_rand = run_probe(pretrained=False, split_seed=123)

print("VAL size:", r_pre["val_size"])
print("pretrained acc/macro_f1:", r_pre["acc"], r_pre["macro_f1"])
print("random     acc/macro_f1:", r_rand["acc"], r_rand["macro_f1"])
