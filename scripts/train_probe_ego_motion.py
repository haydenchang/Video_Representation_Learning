from __future__ import annotations

import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from nuscenes.nuscenes import NuScenes

# Allow direct script execution: python scripts/train_probe_ego_motion.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.clip_index import load_clip_index
from src.model.mvm import MaskedVideoModel
from src.probes.ego_motion_labels import EGO_LABELS
from src.probes.ego_motion_probe_dataset import EgoMotionProbeDataset
from src.probes.splits import scene_disjoint_split


def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tensor_md5(x: torch.Tensor) -> str:
    buf = x.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.md5(buf).hexdigest()


def linear_head_md5(head: nn.Linear) -> str:
    packed = torch.cat(
        [
            head.weight.detach().cpu().float().reshape(-1),
            head.bias.detach().cpu().float().reshape(-1),
        ],
        dim=0,
    )
    return tensor_md5(packed)


def class_counts(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    if y.dtype != torch.long:
        raise ValueError("Expected class indices tensor with dtype=torch.long")
    return torch.bincount(y, minlength=n_classes).to(torch.long)


def counts_to_dict(counts: torch.Tensor) -> dict[str, int]:
    return {EGO_LABELS[i]: int(counts[i].item()) for i in range(len(EGO_LABELS))}


def compute_class_weights(y_train: torch.Tensor, n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    counts = class_counts(y_train, n_classes=n_classes)
    if torch.any(counts <= 0):
        raise RuntimeError(
            f"Train split is missing at least one class (counts={counts.tolist()}); "
            "cannot compute stable class-weighted CE for all ego-motion buckets."
        )
    total = float(counts.sum().item())
    raw = torch.tensor(
        [total / (n_classes * float(c.item())) for c in counts],
        dtype=torch.float32,
    )
    weights = raw / raw.mean()
    return counts, weights


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> torch.Tensor:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    cm = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    denom = (2.0 * tp + fp + fn).clamp_min(1e-8)
    f1 = (2.0 * tp) / denom
    return float(f1.mean().item())


def balanced_accuracy_from_confusion(cm: torch.Tensor) -> float:
    tp = cm.diag().float()
    fn = cm.sum(dim=1).float() - tp
    recall = tp / (tp + fn).clamp_min(1e-8)
    return float(recall.mean().item())


def evaluate_multiclass(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> dict[str, Any]:
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, n_classes=n_classes)
    acc = float((y_true == y_pred).float().mean().item())
    macro_f1 = macro_f1_from_confusion(cm)
    bal_acc = balanced_accuracy_from_confusion(cm)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "confusion_matrix": cm,
    }


def standardize_features(
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x_train - mu) / sigma, (x_eval - mu) / sigma, mu, sigma


def format_confusion(cm: torch.Tensor) -> str:
    lines = ["rows=true [slow,medium,fast], cols=pred [slow,medium,fast]"]
    for r in range(cm.shape[0]):
        lines.append("[" + ", ".join(str(int(v)) for v in cm[r].tolist()) + "]")
    return "\n".join(lines)


def print_metrics(tag: str, metrics: dict[str, Any]) -> None:
    print(
        f"{tag}: acc={metrics['acc']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"bal_acc={metrics['balanced_acc']:.4f}"
    )
    print(f"{tag} confusion:\n{format_confusion(metrics['confusion_matrix'])}")


def make_encoder(
    *,
    pretrained: bool,
    ckpt: dict[str, Any],
    N: int,
    D: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    device: torch.device,
) -> MaskedVideoModel:
    enc = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    if pretrained:
        missing, unexpected = enc.load_state_dict(ckpt["model_state"], strict=True)
        print("load_state_dict missing:", missing)
        print("load_state_dict unexpected:", unexpected)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def debug_encoder_diff(
    *,
    ckpt: dict[str, Any],
    N: int,
    D: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    device: torch.device,
) -> None:
    enc_rand = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    enc_pre = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)

    missing, unexpected = enc_pre.load_state_dict(ckpt["model_state"], strict=True)
    print("load_state_dict missing:", missing)
    print("load_state_dict unexpected:", unexpected)

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

    print("rand param L2:", total_l2_rand**0.5)
    print("pre  param L2:", total_l2_pre**0.5)
    print("pre-vs-rand L2 diff:", total_l2**0.5)
    print("max abs diff:", max_abs, "at", max_name)


def assert_no_token_overlap(
    *,
    nusc: NuScenes,
    items: list[Any],
    train_idx: list[int],
    val_idx: list[int],
) -> None:
    sample_token_by_idx: dict[int, str] = {}
    for i in set(train_idx) | set(val_idx):
        sd = nusc.get("sample_data", items[i].start_sd_token)
        sample_token_by_idx[i] = sd["sample_token"]

    train_tokens = [sample_token_by_idx[i] for i in train_idx]
    val_tokens = [sample_token_by_idx[i] for i in val_idx]
    overlap = sorted(set(train_tokens) & set(val_tokens))
    if overlap:
        raise RuntimeError(
            f"Token overlap detected between train/val splits: {len(overlap)} overlaps. "
            f"Examples={overlap[:10]}"
        )
    print(
        "token overlap check:",
        "train_unique=", len(set(train_tokens)),
        "val_unique=", len(set(val_tokens)),
        "overlap=", len(overlap),
    )


def extract_features_and_labels(
    *,
    ds: EgoMotionProbeDataset,
    indices: list[int],
    tag: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_rows: list[torch.Tensor] = []
    y_rows: list[int] = []
    for j, idx in enumerate(indices):
        z, y_onehot = ds[idx]
        z_rows.append(z.float().cpu())
        y_rows.append(int(y_onehot.argmax().item()))
        if (j + 1) % 100 == 0 or (j + 1) == len(indices):
            print(f"[features:{tag}] {j + 1}/{len(indices)}")
    x = torch.stack(z_rows, dim=0).contiguous()
    y = torch.tensor(y_rows, dtype=torch.long)
    return x, y


def train_linear_head(
    *,
    x_train: torch.Tensor,
    y_train_true: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval_true: torch.Tensor,
    class_weights: torch.Tensor,
    probe_seed: int,
    n_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    shuffle_train_labels: bool,
) -> dict[str, Any]:
    if x_train.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("Expected x_train and x_eval to be 2D tensors [N,d].")
    if y_train_true.ndim != 1 or y_eval_true.ndim != 1:
        raise ValueError("Expected y_train_true and y_eval_true to be 1D class-index tensors [N].")
    if x_train.shape[0] != y_train_true.shape[0]:
        raise ValueError("x_train and y_train_true size mismatch.")
    if x_eval.shape[0] != y_eval_true.shape[0]:
        raise ValueError("x_eval and y_eval_true size mismatch.")

    y_train_used = y_train_true.clone()
    if shuffle_train_labels:
        shuf_gen = torch.Generator(device="cpu").manual_seed(probe_seed + 777_001)
        perm = torch.randperm(y_train_used.shape[0], generator=shuf_gen)
        y_train_used = y_train_used[perm]

    torch.manual_seed(probe_seed)
    head = nn.Linear(x_train.shape[1], n_classes).to(device)
    init_checksum = linear_head_md5(head)

    x_train_dev = x_train.to(device)
    y_train_dev = y_train_used.to(device)
    x_eval_dev = x_eval.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    order_gen = torch.Generator(device="cpu").manual_seed(probe_seed)

    head.train()
    for _ in range(epochs):
        perm = torch.randperm(x_train.shape[0], generator=order_gen)
        for st in range(0, x_train.shape[0], batch_size):
            ridx = perm[st : st + batch_size]
            logits = head(x_train_dev[ridx])
            loss = loss_fn(logits, y_train_dev[ridx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        pred_eval = head(x_eval_dev).argmax(dim=1).cpu()
    metrics = evaluate_multiclass(y_true=y_eval_true.cpu(), y_pred=pred_eval, n_classes=n_classes)
    return {
        "metrics": metrics,
        "head_init_checksum": init_checksum,
        "y_train_used_checksum": tensor_md5(y_train_used.to(torch.uint8)),
    }


def majority_baseline_metrics(
    *,
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
) -> dict[str, Any]:
    train_counts = class_counts(y_train, n_classes=n_classes)
    majority_class = int(torch.argmax(train_counts).item())
    y_pred = torch.full_like(y_val, fill_value=majority_class)
    metrics = evaluate_multiclass(y_true=y_val, y_pred=y_pred, n_classes=n_classes)
    return {
        "majority_class_idx": majority_class,
        "majority_class_name": EGO_LABELS[majority_class],
        "metrics": metrics,
    }


def run_probe(
    *,
    tag: str,
    pretrained: bool,
    nusc: NuScenes,
    dataroot: str,
    index_path: str,
    ckpt: dict[str, Any],
    N: int,
    D: int,
    t: int,
    p: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    device: torch.device,
    train_idx: list[int],
    val_idx: list[int],
    probe_seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    tiny_overfit_samples: int,
    tiny_overfit_epochs: int,
    tiny_overfit_acc_threshold: float,
    baseline_min_delta_macro_f1: float,
    shuffled_min_drop_macro_f1: float,
) -> dict[str, Any]:
    encoder = make_encoder(
        pretrained=pretrained,
        ckpt=ckpt,
        N=N,
        D=D,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        device=device,
    )
    ds = EgoMotionProbeDataset(
        nusc=nusc,
        dataroot=dataroot,
        index_path=index_path,
        encoder_model=encoder,
        t=t,
        p=p,
        device=device,
    )

    x_train, y_train = extract_features_and_labels(ds=ds, indices=train_idx, tag=f"{tag}:train")
    x_val, y_val = extract_features_and_labels(ds=ds, indices=val_idx, tag=f"{tag}:val")

    x_train_norm, x_val_norm, feat_mu, feat_sigma = standardize_features(x_train=x_train, x_eval=x_val)
    print(
        f"[{tag}] feature std stats after train-standardization: "
        f"mean_sigma={float(feat_sigma.mean().item()):.6f} "
        f"min_sigma={float(feat_sigma.min().item()):.6f} "
        f"max_sigma={float(feat_sigma.max().item()):.6f}"
    )

    fixed_batch = x_train_norm[: min(batch_size, x_train_norm.shape[0])].contiguous()
    feature_checksum = tensor_md5(fixed_batch.float())
    print(f"[{tag}] feature checksum (fixed batch): {feature_checksum}")

    train_counts, class_weights = compute_class_weights(y_train=y_train, n_classes=len(EGO_LABELS))
    print(f"[{tag}] train label counts:", counts_to_dict(train_counts))
    print(f"[{tag}] class weights (mean=1):", [round(float(v), 6) for v in class_weights.tolist()])

    baseline = majority_baseline_metrics(y_train=y_train, y_val=y_val, n_classes=len(EGO_LABELS))
    print(
        f"[{tag}] baseline (always '{baseline['majority_class_name']}'): "
        f"acc={baseline['metrics']['acc']:.4f} "
        f"macro_f1={baseline['metrics']['macro_f1']:.4f} "
        f"bal_acc={baseline['metrics']['balanced_acc']:.4f}"
    )

    ref = train_linear_head(
        x_train=x_train_norm,
        y_train_true=y_train,
        x_eval=x_val_norm,
        y_eval_true=y_val,
        class_weights=class_weights,
        probe_seed=probe_seed,
        n_classes=len(EGO_LABELS),
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        shuffle_train_labels=False,
    )
    print(f"[{tag}] head init checksum: {ref['head_init_checksum']}")
    print_metrics(f"[{tag}] trained", ref["metrics"])
    delta_macro_vs_baseline = ref["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"]
    print(f"[{tag}] delta_vs_baseline_macro_f1={delta_macro_vs_baseline:.4f}")
    if delta_macro_vs_baseline < baseline_min_delta_macro_f1:
        print(
            f"[WARN:{tag}] macro_f1 gain over baseline is only {delta_macro_vs_baseline:.4f} "
            f"(target >= {baseline_min_delta_macro_f1:.4f})."
        )

    tiny_n = min(tiny_overfit_samples, x_train.shape[0])
    tiny_gen = torch.Generator(device="cpu").manual_seed(probe_seed + 404)
    tiny_perm = torch.randperm(x_train.shape[0], generator=tiny_gen)[:tiny_n]
    x_tiny = x_train_norm[tiny_perm]
    y_tiny = y_train[tiny_perm]
    tiny_counts, tiny_weights = compute_class_weights(y_train=y_tiny, n_classes=len(EGO_LABELS))
    tiny = train_linear_head(
        x_train=x_tiny,
        y_train_true=y_tiny,
        x_eval=x_tiny,
        y_eval_true=y_tiny,
        class_weights=tiny_weights,
        probe_seed=probe_seed + 1,
        n_classes=len(EGO_LABELS),
        device=device,
        epochs=tiny_overfit_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        shuffle_train_labels=False,
    )
    print(f"[{tag}] tiny-overfit counts:", counts_to_dict(tiny_counts))
    print_metrics(f"[{tag}] tiny-overfit(train)", tiny["metrics"])
    if tiny["metrics"]["acc"] < tiny_overfit_acc_threshold:
        print(
            f"[WARN:{tag}] tiny-subset overfit acc={tiny['metrics']['acc']:.4f} "
            f"< {tiny_overfit_acc_threshold:.4f}; likely feature/label alignment issue."
        )

    shuffled = train_linear_head(
        x_train=x_train_norm,
        y_train_true=y_train,
        x_eval=x_val_norm,
        y_eval_true=y_val,
        class_weights=class_weights,
        probe_seed=probe_seed,
        n_classes=len(EGO_LABELS),
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        shuffle_train_labels=True,
    )
    drop_macro = ref["metrics"]["macro_f1"] - shuffled["metrics"]["macro_f1"]
    print_metrics(f"[{tag}] shuffled-label", shuffled["metrics"])
    print(f"[{tag}] shuffled-label collapse drop_macro_f1={drop_macro:.4f}")
    if drop_macro < shuffled_min_drop_macro_f1:
        raise RuntimeError(
            "Leakage or label not used: shuffled-label macro-F1 did not drop enough "
            f"({drop_macro:.4f} < {shuffled_min_drop_macro_f1:.4f})."
        )

    return {
        "tag": tag,
        "train_counts": train_counts,
        "class_weights": class_weights,
        "feature_checksum": feature_checksum,
        "baseline": baseline,
        "ref": ref,
        "tiny_overfit": tiny,
        "shuffled": shuffled,
        "val_size": int(y_val.numel()),
    }


def main() -> None:
    DATAROOT = r"C:\DS\TPV\nuScenes"
    INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
    CKPT_PATH = os.environ.get("MVM_CKPT_PATH", r"artifacts\checkpoints\mvm_day8_improved_seed1_step6000.pt")

    SPLIT_SEED = 123
    BASE_SEED = 123
    VAL_SCENE_FRAC = 0.2
    EPOCHS = 25
    BATCH_SIZE = 16
    LR = 1e-3
    WEIGHT_DECAY = 0.01
    TINY_OVERFIT_SAMPLES = 200
    TINY_OVERFIT_EPOCHS = 80
    TINY_OVERFIT_ACC_THRESHOLD = 0.80
    BASELINE_MIN_DELTA_MACRO_F1 = 0.02
    SHUFFLED_MIN_DROP_MACRO_F1 = 0.05
    N_LAYERS = 2
    N_HEADS = 8

    device = torch.device("cpu")
    seed_all(BASE_SEED)

    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    print("Using CKPT_PATH:", CKPT_PATH)
    N = int(ckpt["N"])
    D = int(ckpt["D"])
    t = int(ckpt["t"])
    p = int(ckpt["p"])
    d_model = int(ckpt["d_model"])

    debug_encoder_diff(
        ckpt=ckpt,
        N=N,
        D=D,
        d_model=d_model,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        device=device,
    )

    items = load_clip_index(INDEX_PATH)
    train_idx, val_idx = scene_disjoint_split(items, val_scene_frac=VAL_SCENE_FRAC, seed=SPLIT_SEED)
    train_scenes = {items[i].scene_idx for i in train_idx}
    val_scenes = {items[i].scene_idx for i in val_idx}
    scene_overlap = len(train_scenes & val_scenes)
    print(
        "TRAIN clips:", len(train_idx),
        "VAL clips:", len(val_idx),
        "TRAIN scenes:", len(train_scenes),
        "VAL scenes:", len(val_scenes),
        "scene overlap:", scene_overlap,
    )
    if scene_overlap != 0:
        raise RuntimeError(f"Scene overlap detected: {scene_overlap}")
    assert_no_token_overlap(nusc=nusc, items=items, train_idx=train_idx, val_idx=val_idx)

    probe_seed = 10_000 + SPLIT_SEED
    pre = run_probe(
        tag="pretrained",
        pretrained=True,
        nusc=nusc,
        dataroot=DATAROOT,
        index_path=INDEX_PATH,
        ckpt=ckpt,
        N=N,
        D=D,
        t=t,
        p=p,
        d_model=d_model,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        device=device,
        train_idx=train_idx,
        val_idx=val_idx,
        probe_seed=probe_seed,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        tiny_overfit_samples=TINY_OVERFIT_SAMPLES,
        tiny_overfit_epochs=TINY_OVERFIT_EPOCHS,
        tiny_overfit_acc_threshold=TINY_OVERFIT_ACC_THRESHOLD,
        baseline_min_delta_macro_f1=BASELINE_MIN_DELTA_MACRO_F1,
        shuffled_min_drop_macro_f1=SHUFFLED_MIN_DROP_MACRO_F1,
    )
    rnd = run_probe(
        tag="random",
        pretrained=False,
        nusc=nusc,
        dataroot=DATAROOT,
        index_path=INDEX_PATH,
        ckpt=ckpt,
        N=N,
        D=D,
        t=t,
        p=p,
        d_model=d_model,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        device=device,
        train_idx=train_idx,
        val_idx=val_idx,
        probe_seed=probe_seed,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        tiny_overfit_samples=TINY_OVERFIT_SAMPLES,
        tiny_overfit_epochs=TINY_OVERFIT_EPOCHS,
        tiny_overfit_acc_threshold=TINY_OVERFIT_ACC_THRESHOLD,
        baseline_min_delta_macro_f1=BASELINE_MIN_DELTA_MACRO_F1,
        shuffled_min_drop_macro_f1=SHUFFLED_MIN_DROP_MACRO_F1,
    )

    if pre["ref"]["head_init_checksum"] != rnd["ref"]["head_init_checksum"]:
        raise RuntimeError(
            "Head init checksum mismatch between pretrained and random runs; fairness condition violated."
        )
    if pre["feature_checksum"] == rnd["feature_checksum"]:
        print(
            "[WARN] Pretrained/random fixed-batch feature checksums are identical; "
            "expected them to differ."
        )

    print("\n=== Ego-motion Day7 Probe Summary (primary metric: macro_f1) ===")
    print("VAL size:", pre["val_size"])
    print(
        "baseline:",
        f"acc={pre['baseline']['metrics']['acc']:.4f}",
        f"macro_f1={pre['baseline']['metrics']['macro_f1']:.4f}",
        f"bal_acc={pre['baseline']['metrics']['balanced_acc']:.4f}",
    )
    print(
        "pretrained:",
        f"acc={pre['ref']['metrics']['acc']:.4f}",
        f"macro_f1={pre['ref']['metrics']['macro_f1']:.4f}",
        f"bal_acc={pre['ref']['metrics']['balanced_acc']:.4f}",
    )
    print(
        "random:",
        f"acc={rnd['ref']['metrics']['acc']:.4f}",
        f"macro_f1={rnd['ref']['metrics']['macro_f1']:.4f}",
        f"bal_acc={rnd['ref']['metrics']['balanced_acc']:.4f}",
    )
    print("pretrained confusion:\n" + format_confusion(pre["ref"]["metrics"]["confusion_matrix"]))
    print("random confusion:\n" + format_confusion(rnd["ref"]["metrics"]["confusion_matrix"]))
    print(
        "shuffled-label collapse:",
        f"pre_drop_macro_f1={pre['ref']['metrics']['macro_f1'] - pre['shuffled']['metrics']['macro_f1']:.4f}",
        f"rand_drop_macro_f1={rnd['ref']['metrics']['macro_f1'] - rnd['shuffled']['metrics']['macro_f1']:.4f}",
    )


if __name__ == "__main__":
    main()
