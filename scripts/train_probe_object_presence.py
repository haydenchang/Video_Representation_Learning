import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from nuscenes.nuscenes import NuScenes

from src.model.mvm import MaskedVideoModel
from src.probes.probe_dataset import ObjectPresenceProbeDataset
from src.probes.object_presence_labels import LABELS
from src.probes.metrics import mean_average_precision
from src.probes.splits import scene_disjoint_split


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def macro_f1(y_true: torch.Tensor, y_prob: torch.Tensor, thr: float = 0.5) -> float:
    """
    y_true: [N,K] {0,1}
    y_prob: [N,K] in [0,1]
    """
    y_pred = (y_prob >= thr).float()
    eps = 1e-8
    f1s = []
    for k in range(y_true.shape[1]):
        tp = (y_pred[:, k] * y_true[:, k]).sum()
        fp = (y_pred[:, k] * (1 - y_true[:, k])).sum()
        fn = ((1 - y_pred[:, k]) * y_true[:, k]).sum()
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        f1s.append(f1.item())
    return float(np.mean(f1s))

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
CKPT_PATH = os.environ.get("MVM_CKPT_PATH", r"artifacts\checkpoints\mvm_day6_seed1_step1000.pt")

device = torch.device("cpu")

seed_all(123)

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

# Load checkpoint metadata
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
    else:
        # DO NOT load anything.
        pass

    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def run_probe(pretrained: bool, split_seed: int) -> dict:
    encoder = make_encoder(pretrained=pretrained)

    ds = ObjectPresenceProbeDataset(
        nusc=nusc,
        dataroot=DATAROOT,
        index_path=INDEX_PATH,
        encoder_model=encoder,
        t=t,
        p=p,
        mask_ratio_visible=0.0,
        device=device,
    )

    train_idx, val_idx = scene_disjoint_split(ds.items, val_scene_frac=0.2, seed=split_seed)
    train_scenes = {ds.items[i].scene_idx for i in train_idx}
    val_scenes = {ds.items[i].scene_idx for i in val_idx}
    print(
        "TRAIN clips:", len(train_idx),
        "VAL clips:", len(val_idx),
        "TOTAL clips:", len(ds),
        "TRAIN scenes:", len(train_scenes),
        "VAL scenes:", len(val_scenes),
        "scene overlap:", len(train_scenes & val_scenes),
    )

    probe_seed = 10_000 + split_seed
    torch.manual_seed(probe_seed)

    train_gen = torch.Generator(device="cpu").manual_seed(probe_seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=8, shuffle=True, generator=train_gen)
    # Build pos_weight from TRAIN only (handles imbalance)
    Ys = []
    for _, y in train_loader:
        Ys.append(y)
    Y = torch.cat(Ys, dim=0)  # [N,K]

    pos = Y.sum(dim=0)
    neg = Y.shape[0] - pos
    pos_weight = (neg / (pos + 1e-6)).to(device)

    print("train prevalence:", (pos / Y.shape[0]).tolist())
    print("pos_weight:", pos_weight.cpu().tolist())

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("loss_fn pos_weight:", loss_fn.pos_weight.cpu().tolist())

    val_loader = DataLoader(Subset(ds, val_idx), batch_size=8, shuffle=False)

    K = len(LABELS)
    torch.manual_seed(probe_seed)  # ensure identical head init
    clf = nn.Linear(2 * d_model, K).to(device)
    print("probe_seed:", probe_seed, "clf_w0_sum:", float(clf.weight[0].sum().item()))
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.01)

    # train
    clf.train()
    for epoch in range(50):
        losses = []
        for z, y in train_loader:
            z = z.to(device)
            y = y.to(device)

            logits = clf(z)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        if epoch in [0, 1, 2, 5, 10, 20, 49]:
            print(("pretrained" if pretrained else "random"), "epoch", epoch, "train_loss", float(np.mean(losses)))

    def prob_stats(loader):
        probs = []
        ys = []
        with torch.no_grad():
            for z, y in loader:
                prob = torch.sigmoid(clf(z.to(device))).cpu()
                probs.append(prob)
                ys.append(y.cpu())
        P = torch.cat(probs, dim=0)  # [N,K]
        Y = torch.cat(ys, dim=0)
        return P.mean(0).tolist(), P.std(0, unbiased=False).tolist(), Y.mean(0).tolist()

    train_mean, train_std, train_prev = prob_stats(train_loader)
    print(("pretrained" if pretrained else "random"), "TRAIN prob std :", train_std)
    print(("pretrained" if pretrained else "random"), "TRAIN prob mean:", train_mean)
    print(("pretrained" if pretrained else "random"), "TRAIN prevalence:", train_prev)

    # eval
    clf.eval()
    all_y = []
    all_p = []
    with torch.no_grad():
        for z, y in val_loader:
            logits = clf(z.to(device))
            prob = torch.sigmoid(logits).cpu()
            all_p.append(prob)
            all_y.append(y.cpu())

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    f1 = macro_f1(y_true, y_prob, thr=0.5)
    map_score = mean_average_precision(y_true, y_prob)

    return {"macro_f1": f1, "mAP": map_score, "val_size": int(y_true.shape[0])}


seeds = [2, 3]

pre_maps = []
rand_maps = []

for s in seeds:
    r_pre = run_probe(pretrained=True, split_seed=s)
    r_rand = run_probe(pretrained=False, split_seed=s)

    pre_maps.append(r_pre["mAP"])
    rand_maps.append(r_rand["mAP"])

    print(f"seed {s}: pretrained mAP={r_pre['mAP']:.6f} | random mAP={r_rand['mAP']:.6f}")

import numpy as np
print("==== Summary (mAP) ====")
print(f"pretrained mean±std: {float(np.mean(pre_maps)):.6f} ± {float(np.std(pre_maps)):.6f}")
print(f"random     mean±std: {float(np.mean(rand_maps)):.6f} ± {float(np.std(rand_maps)):.6f}")
print(f"delta (pre-rand)    : {float(np.mean(pre_maps) - np.mean(rand_maps)):.6f}")

