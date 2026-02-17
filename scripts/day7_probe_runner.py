from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from nuscenes.nuscenes import NuScenes

from src.config import DATA_CFG
from src.data.clip_dataset import ClipDataset
from src.data.clip_index import ClipSpec, load_clip_index
from src.model.mvm import MaskedVideoModel
from src.model.patchify import tubeletify
from src.probes.metrics import mean_average_precision
from src.probes.object_presence_labels import LABELS
from src.probes.object_presence_targets import clip_object_presence_multihot
from src.probes.splits import scene_disjoint_split


POOLING_MODES = ("mean_visible", "mean_all", "mean_excluding_cls")
OPTIMIZERS = ("adamw", "sgd")


@dataclass(frozen=True)
class ModelConfig:
    N: int
    D: int
    t: int
    p: int
    d_model: int
    n_layers: int
    n_heads: int


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


def parse_int_csv(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one integer seed.")
    return [int(p) for p in parts]


def parse_choice_csv(text: str, allowed: tuple[str, ...], name: str) -> list[str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Expected at least one {name}.")
    for p in parts:
        if p not in allowed:
            raise ValueError(f"Unsupported {name}={p!r}. Allowed: {allowed}")
    return parts


def cap_indices(indices: list[int], max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)
    out = list(indices)
    rng = random.Random(seed)
    rng.shuffle(out)
    out = out[:max_samples]
    out.sort()
    return out


def tensor_md5_uint8(x: torch.Tensor) -> str:
    x_u8 = x.detach().cpu().to(torch.uint8).contiguous()
    return hashlib.md5(x_u8.numpy().tobytes()).hexdigest()


def first_rows_positive_indices(y: torch.Tensor, n_rows: int = 5) -> list[list[int]]:
    n = min(n_rows, y.shape[0])
    out: list[list[int]] = []
    for i in range(n):
        pos = torch.where(y[i] > 0.5)[0]
        out.append([int(v) for v in pos.tolist()])
    return out


def make_shuffled_labels_with_evidence(
    y_train: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Build shuffled labels via deterministic permutation of flattened binary entries.
    This keeps the total number of positives identical while strongly breaking feature-label alignment.
    """
    if y_train.ndim != 2:
        raise ValueError(f"Expected y_train [N,K], got shape={tuple(y_train.shape)}")

    src_u8 = y_train.detach().cpu().to(torch.uint8)
    flat = src_u8.reshape(-1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(flat.shape[0], generator=gen)
    shuffled_u8 = flat[perm].reshape_as(src_u8)
    y_shuffled = shuffled_u8.float()

    before_rows = first_rows_positive_indices(src_u8.float(), n_rows=5)
    after_rows = first_rows_positive_indices(y_shuffled, n_rows=5)
    diff_rows = sum(1 for b, a in zip(before_rows, after_rows) if b != a)

    checksum_before = tensor_md5_uint8(src_u8)
    checksum_after = tensor_md5_uint8(shuffled_u8)

    if checksum_before == checksum_after:
        raise RuntimeError(
            "Shuffle not applied / labels regenerated after shuffle / using wrong tensor: "
            "train and shuffled train label checksums are identical."
        )

    min_required_diff = min(3, len(before_rows))
    if diff_rows < min_required_diff:
        raise RuntimeError(
            "Shuffle not applied / labels regenerated after shuffle / using wrong tensor: "
            f"only {diff_rows}/{len(before_rows)} first rows changed, expected >= {min_required_diff}."
        )

    if int(src_u8.sum().item()) != int(shuffled_u8.sum().item()):
        raise RuntimeError("Shuffled labels changed total positive count; expected pure permutation behavior.")

    evidence = {
        "method": "flattened-bit-permutation",
        "seed": int(seed),
        "train_checksum_md5_uint8": checksum_before,
        "shuffled_checksum_md5_uint8": checksum_after,
        "first5_positive_indices_before": before_rows,
        "first5_positive_indices_after": after_rows,
        "first5_row_diff_count": int(diff_rows),
        "total_positive_before": int(src_u8.sum().item()),
        "total_positive_after": int(shuffled_u8.sum().item()),
    }
    return y_shuffled, evidence


def macro_f1(y_true: torch.Tensor, y_prob: torch.Tensor, thr: float) -> float:
    y_pred = (y_prob >= thr).float()
    eps = 1e-8
    tp = (y_pred * y_true).sum(dim=0)
    fp = (y_pred * (1.0 - y_true)).sum(dim=0)
    fn = ((1.0 - y_pred) * y_true).sum(dim=0)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(f1.mean().item())


def sweep_macro_f1(y_true: torch.Tensor, y_prob: torch.Tensor) -> tuple[dict[str, float], float, float]:
    thresholds = [round(v, 1) for v in np.arange(0.1, 1.0, 0.1)]
    f1_by_thr: dict[str, float] = {}
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        f1 = macro_f1(y_true, y_prob, thr=thr)
        f1_by_thr[f"{thr:.1f}"] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return f1_by_thr, float(best_thr), float(best_f1)


def make_probe_seed(split_seed: int, pooling: str, optimizer_name: str) -> int:
    pidx = POOLING_MODES.index(pooling)
    oidx = OPTIMIZERS.index(optimizer_name)
    return 100_000 + split_seed * 1_000 + pidx * 100 + oidx * 10


def build_encoder(
    pretrained: bool,
    model_cfg: ModelConfig,
    ckpt_payload: dict[str, Any] | None,
    device: torch.device,
) -> MaskedVideoModel:
    enc = MaskedVideoModel(
        N=model_cfg.N,
        D=model_cfg.D,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
    ).to(device)
    if pretrained:
        if ckpt_payload is None:
            raise RuntimeError("Requested pretrained encoder but checkpoint payload is missing.")
        missing, unexpected = enc.load_state_dict(ckpt_payload["model_state"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Unexpected state_dict mismatch. missing={missing}, unexpected={unexpected}")
    enc.eval()
    for param in enc.parameters():
        param.requires_grad = False
    return enc


def pool_latents(x_lat: torch.Tensor, n_visible: int, pooling: str) -> torch.Tensor:
    has_cls = x_lat.shape[1] == n_visible + 1
    if pooling == "mean_visible":
        # Mean only over visible encoder tokens.
        x_tok = x_lat[:, 1:, :] if has_cls else x_lat
        return x_tok.mean(dim=1)
    if pooling == "mean_all":
        # Mean over all encoder outputs, including CLS if present.
        return x_lat.mean(dim=1)
    if pooling == "mean_excluding_cls":
        x_tok = x_lat[:, 1:, :] if has_cls else x_lat
        return x_tok.mean(dim=1)
    raise ValueError(f"Unsupported pooling={pooling!r}")


def extract_features_for_encoder(
    encoder: MaskedVideoModel,
    clip_ds: ClipDataset,
    items: list[ClipSpec],
    index_union: list[int],
    sample_id_by_clip_idx: dict[int, str],
    model_cfg: ModelConfig,
    pooling_modes: list[str],
    device: torch.device,
    progress_every: int,
) -> dict[str, dict[str, torch.Tensor]]:
    feats: dict[str, dict[str, torch.Tensor]] = {pooling: {} for pooling in pooling_modes}

    with torch.no_grad():
        for row_idx, clip_idx in enumerate(index_union):
            sample = clip_ds[clip_idx]
            spec = items[clip_idx]
            sample_id = sample_id_by_clip_idx[clip_idx]

            if sample["start_token"] != spec.start_sd_token:
                raise RuntimeError(
                    "Label/feature misalignment detected: dataset start token does not match clip index token "
                    f"(idx={clip_idx}, ds={sample['start_token']}, index={spec.start_sd_token})."
                )

            video = sample["video"].unsqueeze(0).to(device)  # [1,T,C,H,W]
            tokens = tubeletify(video, t=model_cfg.t, p=model_cfg.p)  # [1,N,D]
            ids_visible = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
            x_lat = encoder.encode_visible(tokens, ids_visible)  # [1, Nv(+1), d_model]

            for pooling in pooling_modes:
                z = pool_latents(x_lat, n_visible=tokens.shape[1], pooling=pooling)  # [1,d_model]
                feats[pooling][sample_id] = z.squeeze(0).cpu().float()

            if progress_every > 0 and ((row_idx + 1) % progress_every == 0 or (row_idx + 1) == len(index_union)):
                print(f"[feature-extract] processed {row_idx + 1}/{len(index_union)} clips")

    return feats


def labels_and_metadata(
    nusc: NuScenes,
    items: list[ClipSpec],
    index_union: list[int],
) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
    labels = torch.zeros((len(index_union), len(LABELS)), dtype=torch.float32)
    meta_by_clip_idx: dict[int, dict[str, Any]] = {}

    for row_idx, clip_idx in enumerate(index_union):
        spec = items[clip_idx]
        y_np = clip_object_presence_multihot(
            nusc=nusc,
            start_sd_token=spec.start_sd_token,
            T=spec.T,
            stride=spec.stride,
            keyframes_only=spec.keyframes_only,
        )
        labels[row_idx] = torch.from_numpy(y_np).float()

        sd = nusc.get("sample_data", spec.start_sd_token)
        meta_by_clip_idx[clip_idx] = {
            "clip_index": int(clip_idx),
            "scene_idx": int(spec.scene_idx),
            "start_sd_token": spec.start_sd_token,
            "sample_token": sd["sample_token"],
            "timestamp": int(sd["timestamp"]),
        }

    return labels, meta_by_clip_idx


def build_label_map_by_sample_id(
    clip_indices: list[int],
    idx_to_row: dict[int, int],
    y_all: torch.Tensor,
    sample_id_by_clip_idx: dict[int, str],
    shuffled_labels: bool,
) -> dict[str, torch.Tensor]:
    label_cache: dict[str, torch.Tensor] = {}
    if shuffled_labels and len(label_cache) != 0:
        raise RuntimeError("Internal error: shuffled_labels=True but label cache is not empty at start.")

    for clip_idx in clip_indices:
        sid = sample_id_by_clip_idx[clip_idx]
        if sid in label_cache:
            raise RuntimeError(f"Duplicate sample identifier in split: {sid}")
        label_cache[sid] = y_all[idx_to_row[clip_idx]].clone().cpu()
    return label_cache


def gather_split_data(
    feature_map_by_id: dict[str, torch.Tensor],
    label_map_by_id: dict[str, torch.Tensor],
    sample_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    for sid in sample_ids:
        if sid not in feature_map_by_id:
            raise RuntimeError(f"Missing feature for sample id: {sid}")
        if sid not in label_map_by_id:
            raise RuntimeError(f"Missing label for sample id: {sid}")
        x_rows.append(feature_map_by_id[sid])
        y_rows.append(label_map_by_id[sid])
    return torch.stack(x_rows, dim=0), torch.stack(y_rows, dim=0)


def assert_zero_token_overlap(
    train_ids: list[str],
    val_ids: list[str],
    id_label: str,
    seed: int,
) -> dict[str, Any]:
    train_set = set(train_ids)
    val_set = set(val_ids)
    overlap = sorted(train_set & val_set)
    result = {
        "seed": int(seed),
        "id_type": id_label,
        "train_unique": int(len(train_set)),
        "val_unique": int(len(val_set)),
        "overlap_count": int(len(overlap)),
        "overlap_examples": overlap[:10],
    }
    if overlap:
        raise RuntimeError(
            f"Token-level overlap detected for seed={seed}: {len(overlap)} overlapping ids. "
            f"Examples={overlap[:10]}"
        )
    return result


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_sample_ids: list[str],
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    optimizer_name: str,
    epochs: int,
    batch_size: int,
    probe_seed: int,
    adamw_lr: float,
    adamw_weight_decay: float,
    sgd_lr: float,
    sgd_weight_decay: float,
    device: torch.device,
    run_name: str,
    debug_first_batch_label_match: bool = False,
    train_label_lookup_by_id: dict[str, torch.Tensor] | None = None,
    y_val_reference: torch.Tensor | None = None,
    y_train_reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    if x_train.ndim != 2 or y_train.ndim != 2:
        raise ValueError("Expected x_train=[N,d] and y_train=[N,K].")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train size mismatch.")
    if x_eval.shape[0] != y_eval.shape[0]:
        raise ValueError("x_eval and y_eval size mismatch.")
    if len(train_sample_ids) != y_train.shape[0]:
        raise ValueError("train_sample_ids and y_train size mismatch.")

    torch.manual_seed(probe_seed)
    clf = nn.Linear(x_train.shape[1], y_train.shape[1]).to(device)

    y_train_dev = y_train.to(device)
    pos = y_train_dev.sum(dim=0)
    neg = y_train_dev.shape[0] - pos
    pos_weight = neg / (pos + 1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(clf.parameters(), lr=adamw_lr, weight_decay=adamw_weight_decay)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(clf.parameters(), lr=sgd_lr, momentum=0.9, weight_decay=sgd_weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}")

    x_train_dev = x_train.to(device)
    x_eval_dev = x_eval.to(device)

    order_gen = torch.Generator(device="cpu").manual_seed(probe_seed)
    n_train = x_train.shape[0]
    last_loss = float("nan")
    debug_batch_evidence: dict[str, Any] | None = None

    clf.train()
    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=order_gen)
        for start in range(0, n_train, batch_size):
            ridx = perm[start : start + batch_size]
            xb = x_train_dev[ridx]
            yb = y_train_dev[ridx]

            # Verify dataloader/index mapping on first batch when requested.
            if debug_first_batch_label_match and debug_batch_evidence is None:
                if train_label_lookup_by_id is None:
                    raise RuntimeError("Debug batch label check requested but train_label_lookup_by_id is missing.")
                batch_ids = [train_sample_ids[int(i)] for i in ridx.tolist()]
                from_lookup = torch.stack([train_label_lookup_by_id[sid] for sid in batch_ids], dim=0)
                checksum_batch_used = tensor_md5_uint8(yb.cpu())
                checksum_batch_lookup = tensor_md5_uint8(from_lookup)
                print(
                    f"[debug:{run_name}] first_batch_checksum_used={checksum_batch_used} "
                    f"first_batch_checksum_lookup={checksum_batch_lookup}"
                )
                if checksum_batch_used != checksum_batch_lookup:
                    raise RuntimeError(
                        "Dataset/DataLoader mapping bug: first batch labels used in loss do not match "
                        "stored labels for the same sample ids."
                    )
                debug_batch_evidence = {
                    "batch_size": int(len(batch_ids)),
                    "first_batch_sample_ids_preview": batch_ids[:8],
                    "batch_checksum_used": checksum_batch_used,
                    "batch_checksum_lookup": checksum_batch_lookup,
                }

            logits = clf(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    clf.eval()
    with torch.no_grad():
        eval_prob = torch.sigmoid(clf(x_eval_dev)).cpu()
    if eval_prob.shape[0] != y_eval.shape[0]:
        raise RuntimeError(
            f"Evaluation mismatch in {run_name}: pred_count={eval_prob.shape[0]} target_count={y_eval.shape[0]}"
        )

    checksum_eval_target = tensor_md5_uint8(y_eval)
    eval_checksums: dict[str, Any] = {
        "run_name": run_name,
        "eval_target_checksum_md5_uint8": checksum_eval_target,
        "eval_target_count": int(y_eval.shape[0]),
        "pred_count": int(eval_prob.shape[0]),
    }
    if y_val_reference is not None:
        checksum_y_val = tensor_md5_uint8(y_val_reference)
        eval_checksums["y_val_checksum_md5_uint8"] = checksum_y_val
        print(
            f"[debug:{run_name}] checksum_eval_target={checksum_eval_target} checksum_y_val={checksum_y_val}"
        )
        if checksum_eval_target != checksum_y_val:
            raise RuntimeError(
                "Evaluation target mismatch: eval target checksum does not equal Y_val checksum. "
                "Likely using wrong eval labels."
            )
    if y_train_reference is not None:
        checksum_y_train = tensor_md5_uint8(y_train_reference)
        eval_checksums["y_train_checksum_md5_uint8"] = checksum_y_train
        print(
            f"[debug:{run_name}] checksum_eval_target={checksum_eval_target} checksum_y_train={checksum_y_train}"
        )
        if y_train_reference.shape == y_eval.shape and checksum_eval_target == checksum_y_train:
            raise RuntimeError(
                "Evaluation target checksum equals Y_train checksum with equal shapes; expected Y_val targets."
            )

    eval_map = mean_average_precision(y_eval.cpu(), eval_prob)
    eval_f1_by_thr, best_thr, best_f1 = sweep_macro_f1(y_eval.cpu(), eval_prob)

    train_true_map: float | None = None
    if y_train_reference is not None and y_train_reference.shape[0] == x_train.shape[0]:
        with torch.no_grad():
            train_prob = torch.sigmoid(clf(x_train_dev)).cpu()
        train_true_map = float(mean_average_precision(y_train_reference.cpu(), train_prob))

    return {
        "mAP": float(eval_map),
        "macro_f1_by_threshold": eval_f1_by_thr,
        "best_threshold": float(best_thr),
        "best_macro_f1": float(best_f1),
        "last_train_loss": float(last_loss),
        "n_train": int(n_train),
        "n_eval": int(y_eval.shape[0]),
        "pos_weight": [float(x) for x in pos_weight.detach().cpu().tolist()],
        "eval_checksums": eval_checksums,
        "debug_first_batch": debug_batch_evidence,
        "train_true_mAP": train_true_map,
    }


def summarize_runs(
    runs: list[dict[str, Any]],
    pooling_modes: list[str],
    optimizers: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pooling in pooling_modes:
        for optimizer_name in optimizers:
            rows = [r for r in runs if r["pooling"] == pooling and r["optimizer"] == optimizer_name]
            pre_maps = [r["pretrained"]["mAP"] for r in rows if r.get("pretrained") is not None]
            rand_maps = [r["random"]["mAP"] for r in rows if r.get("random") is not None]
            deltas = [r["delta_mAP"] for r in rows if r.get("delta_mAP") is not None]
            out.append(
                {
                    "pooling": pooling,
                    "optimizer": optimizer_name,
                    "num_seeds": len(rows),
                    "pretrained_mean": float(np.mean(pre_maps)) if pre_maps else None,
                    "pretrained_std": float(np.std(pre_maps)) if pre_maps else None,
                    "random_mean": float(np.mean(rand_maps)) if rand_maps else None,
                    "random_std": float(np.std(rand_maps)) if rand_maps else None,
                    "delta_mean": float(np.mean(deltas)) if deltas else None,
                    "delta_std": float(np.std(deltas)) if deltas else None,
                }
            )
    return out


def infer_not_pass_causes(
    sanity_checks: dict[str, Any],
    ablation_summary: list[dict[str, Any]],
    gate_stats: dict[str, Any] | None,
    pass_delta: float,
    has_pretrained: bool,
    missing_ckpt_reason: str | None,
) -> list[str]:
    causes: list[str] = []

    if missing_ckpt_reason is not None:
        causes.append(missing_ckpt_reason)

    tiny = sanity_checks.get("tiny_subset_overfit")
    if tiny is not None and not tiny.get("passed", False):
        causes.append(
            f"Tiny-subset overfit failed (train mAP={tiny.get('train_mAP', 'NA')}, "
            f"threshold={tiny.get('threshold', 'NA')})."
        )

    shuffled = sanity_checks.get("shuffled_label")
    if shuffled is not None and not shuffled.get("passed", False):
        causes.append(
            "Shuffled-label sanity did not collapse enough; this strongly suggests leakage or feature/label mismatch."
        )

    if gate_stats is not None and has_pretrained:
        if gate_stats["delta_mean"] < pass_delta:
            causes.append(
                f"Gate delta is below threshold ({gate_stats['delta_mean']:.4f} < {pass_delta:.4f})."
            )
        if gate_stats["delta_std"] > 0.02:
            causes.append("Seed-to-seed variance is high, so probe conclusions are unstable.")

    valid_ablation = [r for r in ablation_summary if r.get("delta_mean") is not None]
    if valid_ablation:
        best = max(valid_ablation, key=lambda x: x["delta_mean"])
        if best["delta_mean"] < pass_delta:
            causes.append(
                "No pooling/optimizer ablation achieved the required pretrained advantage; checkpoint likely lacks probe signal."
            )
        elif gate_stats is not None and best["delta_mean"] > gate_stats["delta_mean"] + 0.01:
            causes.append(
                f"Probe is sensitive to head optimization/pooling (best {best['pooling']} + {best['optimizer']} "
                f"delta={best['delta_mean']:.4f})."
            )

    uniq: list[str] = []
    for c in causes:
        if c not in uniq:
            uniq.append(c)
    return uniq[:3]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def fmt_float(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{ndigits}f}"


def write_markdown_report(report: dict[str, Any], md_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Day 7 Probe Report")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at_utc']}`")
    lines.append(f"- Decision: **{report['decision']['status']}**")
    lines.append("")

    if report.get("error") is not None:
        lines.append("## Aborted")
        lines.append("")
        lines.append(f"- Error: `{report['error']}`")
        lines.append("")

    lines.append("## Config Snapshot")
    lines.append("")
    cfg = report["config"]
    lines.append(f"- dataroot: `{cfg['dataroot']}`")
    lines.append(f"- index_path: `{cfg['index_path']}`")
    lines.append(f"- ckpt_path: `{cfg['ckpt_path']}`")
    lines.append(f"- pretrained_available: `{cfg['pretrained_available']}`")
    lines.append(f"- pretrained_status: `{cfg['pretrained_status']}`")
    if cfg.get("pretrained_missing_reason") is not None:
        lines.append(f"- pretrained_missing_reason: `{cfg['pretrained_missing_reason']}`")
    lines.append(f"- seeds: `{cfg['seeds']}`")
    lines.append(f"- pooling_modes: `{cfg['pooling_modes']}`")
    lines.append(f"- optimizers: `{cfg['optimizers']}`")
    lines.append(f"- epochs: `{cfg['epochs']}`")
    lines.append(f"- sanity_epochs: `{cfg['sanity_epochs']}`")
    lines.append(f"- only_sanity: `{cfg['only_sanity']}`")
    lines.append(f"- max_train_samples: `{cfg['max_train_samples']}`")
    lines.append(f"- max_val_samples: `{cfg['max_val_samples']}`")
    lines.append("")

    lines.append("## Sanity Checks")
    lines.append("")
    sanity = report["sanity_checks"]
    lines.append(f"- all_passed: `{sanity.get('all_passed', False)}`")
    scene = sanity.get("scene_overlap")
    if scene is not None:
        lines.append(f"- scene_overlap_passed: `{scene.get('passed', False)}`")
    token_overlap = sanity.get("token_overlap")
    if token_overlap is not None:
        lines.append(f"- token_overlap_passed: `{token_overlap.get('passed', False)}`")
        per_seed = token_overlap.get("per_seed", [])
        if per_seed:
            overlap_desc = ", ".join(
                [
                    f"seed{row.get('seed')}:train={row.get('train_unique')} val={row.get('val_unique')} overlap={row.get('overlap_count')}"
                    for row in per_seed
                ]
            )
            lines.append(f"- token_overlap_counts: `{overlap_desc}`")
    tiny = sanity.get("tiny_subset_overfit")
    if tiny is not None:
        lines.append(
            f"- tiny_subset_overfit: `passed={tiny.get('passed', False)}` "
            f"`train_mAP={fmt_float(tiny.get('train_mAP'))}` "
            f"`threshold={fmt_float(tiny.get('threshold'))}`"
        )
    shuffled = sanity.get("shuffled_label")
    if shuffled is not None:
        collapse_metric = shuffled.get("collapse_metric", "mAP")
        lines.append(
            f"- shuffled_label: `passed={shuffled.get('passed', False)}` "
            f"`ref_mAP={fmt_float(shuffled.get('reference_mAP'))}` "
            f"`shuffled_mAP={fmt_float(shuffled.get('shuffled_mAP'))}` "
            f"`drop={fmt_float(shuffled.get('drop'))}`"
        )
        lines.append(f"- shuffled_collapse_metric: `{collapse_metric}`")
        if "reference_val_mAP" in shuffled and "shuffled_val_mAP" in shuffled:
            lines.append(
                f"- shuffled_val_mAP_check: `ref_val={fmt_float(shuffled.get('reference_val_mAP'))}` "
                f"`shuffled_val={fmt_float(shuffled.get('shuffled_val_mAP'))}`"
            )
        ev = shuffled.get("shuffle_evidence", {})
        if ev:
            lines.append(
                f"- shuffled_checksums: `train={ev.get('train_checksum_md5_uint8')}` "
                f"`shuffled={ev.get('shuffled_checksum_md5_uint8')}` "
                f"`first5_row_diff_count={ev.get('first5_row_diff_count')}`"
            )
        dbg = shuffled.get("debug_one_epoch", {})
        if dbg:
            lines.append(
                f"- first_batch_label_use: `used={dbg.get('batch_checksum_used')}` "
                f"`lookup={dbg.get('batch_checksum_lookup')}`"
            )
        eval_cs = shuffled.get("eval_checksums", {})
        if eval_cs:
            lines.append(
                f"- eval_target_checksums: `eval={eval_cs.get('eval_target_checksum_md5_uint8')}` "
                f"`val={eval_cs.get('y_val_checksum_md5_uint8')}` "
                f"`train={eval_cs.get('y_train_checksum_md5_uint8')}`"
            )
    lines.append("")

    gate = report.get("gate_stats")
    if gate is not None:
        lines.append("## PASS Gate")
        lines.append("")
        lines.append(f"- pooling: `{gate['pooling']}`")
        lines.append(f"- optimizer: `{gate['optimizer']}`")
        lines.append(f"- pretrained mean±std mAP: `{fmt_float(gate['pretrained_mean'])} ± {fmt_float(gate['pretrained_std'])}`")
        lines.append(f"- random mean±std mAP: `{fmt_float(gate['random_mean'])} ± {fmt_float(gate['random_std'])}`")
        lines.append(f"- delta mean±std: `{fmt_float(gate['delta_mean'])} ± {fmt_float(gate['delta_std'])}`")
        lines.append(f"- required delta: `{fmt_float(gate['required_delta'])}`")
        lines.append("")

    lines.append("## Ablation Summary")
    lines.append("")
    lines.append("| pooling | optimizer | pre_mean | rand_mean | delta_mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report.get("ablation_summary", []):
        lines.append(
            f"| {row['pooling']} | {row['optimizer']} | "
            f"{fmt_float(row['pretrained_mean'])} | {fmt_float(row['random_mean'])} | {fmt_float(row['delta_mean'])} |"
        )
    lines.append("")

    lines.append("## Run Metrics")
    lines.append("")
    lines.append("| pooling | optimizer | seed | pre_mAP | rand_mAP | delta_mAP | pre_bestF1 | rand_bestF1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.get("runs", []):
        pre = row.get("pretrained")
        rand = row.get("random")
        lines.append(
            f"| {row['pooling']} | {row['optimizer']} | {row['seed']} | "
            f"{fmt_float(None if pre is None else pre.get('mAP'))} | "
            f"{fmt_float(None if rand is None else rand.get('mAP'))} | "
            f"{fmt_float(row.get('delta_mAP'))} | "
            f"{fmt_float(None if pre is None else pre.get('best_macro_f1'))} | "
            f"{fmt_float(None if rand is None else rand.get('best_macro_f1'))} |"
        )
    lines.append("")

    lines.append("## Alignment Samples")
    lines.append("")
    lines.append("| clip_index | start_sd_token | sample_token | timestamp | num_positive_labels |")
    lines.append("|---:|---|---|---:|---:|")
    for row in sanity.get("alignment_examples", []):
        lines.append(
            f"| {row['clip_index']} | {row['start_sd_token']} | {row['sample_token']} | "
            f"{row['timestamp']} | {row['num_positive_labels']} |"
        )
    lines.append("")

    lines.append("## Decision Reasoning")
    lines.append("")
    for reason in report["decision"].get("reasons", []):
        lines.append(f"- {reason}")
    lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    env_ckpt = os.environ.get("MVM_CKPT_PATH", "").strip()
    parser = argparse.ArgumentParser(
        description="Finite and deterministic Day 7 linear-probe runner (pretrained vs random)."
    )
    parser.add_argument("--dataroot", type=str, default=r"C:\DS\TPV\nuScenes")
    parser.add_argument("--index-path", type=str, default=r"artifacts\clip_index_T8_s1_keyframes.json")
    parser.add_argument("--ckpt", type=str, default=env_ckpt)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--val-scene-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=10_000)
    parser.add_argument("--max-val-samples", type=int, default=3_000)

    parser.add_argument("--pooling-modes", type=str, default="mean_visible,mean_all,mean_excluding_cls")
    parser.add_argument("--optimizers", type=str, default="adamw,sgd")
    parser.add_argument("--gate-pooling", type=str, default="mean_excluding_cls", choices=POOLING_MODES)
    parser.add_argument("--gate-optimizer", type=str, default="adamw", choices=OPTIMIZERS)
    parser.add_argument("--pass-delta", type=float, default=0.01)

    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.01)
    parser.add_argument("--sgd-lr", type=float, default=5e-2)
    parser.add_argument("--sgd-weight-decay", type=float, default=0.0)

    parser.add_argument("--tiny-overfit-samples", type=int, default=200)
    parser.add_argument("--tiny-overfit-epochs", type=int, default=40)
    parser.add_argument("--tiny-overfit-threshold", type=float, default=0.60)
    parser.add_argument("--sanity-epochs", type=int, default=80)
    parser.add_argument("--shuffled-max-map", type=float, default=0.55)
    parser.add_argument("--shuffled-min-drop", type=float, default=0.05)

    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--progress-every", type=int, default=200)

    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--p", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)

    parser.add_argument("--report-json", type=str, default=r"artifacts\day7_probe_report.json")
    parser.add_argument("--report-md", type=str, default=r"artifacts\day7_probe_report.md")
    parser.add_argument("--only-sanity", "--only_sanity", dest="only_sanity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(args.base_seed)

    seeds = parse_int_csv(args.seeds)
    required_gate_seeds = [1, 2, 3]
    if seeds != required_gate_seeds:
        raise ValueError(
            f"Day 7 gate requires seeds exactly {required_gate_seeds}; got {seeds}. "
            "This avoids moving the PASS criterion."
        )
    pooling_modes = parse_choice_csv(args.pooling_modes, POOLING_MODES, "pooling mode")
    optimizers = parse_choice_csv(args.optimizers, OPTIMIZERS, "optimizer")
    if args.gate_pooling not in pooling_modes:
        raise ValueError("--gate-pooling must be included in --pooling-modes")
    if args.gate_optimizer not in optimizers:
        raise ValueError("--gate-optimizer must be included in --optimizers")

    report_json = Path(args.report_json)
    report_md = Path(args.report_md)

    ckpt_path = args.ckpt.strip()
    ckpt_payload: dict[str, Any] | None = None
    missing_ckpt_reason: str | None = None
    if ckpt_path:
        if Path(ckpt_path).exists():
            ckpt_payload = torch.load(ckpt_path, map_location="cpu")
        else:
            missing_ckpt_reason = (
                f"Pretrained checkpoint missing at {ckpt_path!r}; "
                "random baseline was run, pretrained runs marked missing."
            )
            ckpt_path = ""
    else:
        missing_ckpt_reason = (
            "MVM_CKPT_PATH/--ckpt not provided; random baseline was run, pretrained runs marked missing."
        )

    if ckpt_payload is not None:
        model_cfg = ModelConfig(
            N=int(ckpt_payload["N"]),
            D=int(ckpt_payload["D"]),
            t=int(ckpt_payload["t"]),
            p=int(ckpt_payload["p"]),
            d_model=int(ckpt_payload["d_model"]),
            n_layers=int(args.n_layers),
            n_heads=int(args.n_heads),
        )
    else:
        n_tokens = (DATA_CFG.T // args.t) * (DATA_CFG.out_h // args.p) * (DATA_CFG.out_w // args.p)
        d_token = 3 * args.t * args.p * args.p
        model_cfg = ModelConfig(
            N=int(n_tokens),
            D=int(d_token),
            t=int(args.t),
            p=int(args.p),
            d_model=int(args.d_model),
            n_layers=int(args.n_layers),
            n_heads=int(args.n_heads),
        )

    device = torch.device(args.device)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataroot": args.dataroot,
            "index_path": args.index_path,
            "ckpt_path": ckpt_path if ckpt_path else None,
            "pretrained_available": ckpt_payload is not None,
            "pretrained_status": "available" if ckpt_payload is not None else "missing_checkpoint",
            "pretrained_missing_reason": missing_ckpt_reason,
            "seeds": seeds,
            "pooling_modes": pooling_modes,
            "optimizers": optimizers,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_train_samples": int(args.max_train_samples),
            "max_val_samples": int(args.max_val_samples),
            "val_scene_frac": float(args.val_scene_frac),
            "gate_pooling": args.gate_pooling,
            "gate_optimizer": args.gate_optimizer,
            "pass_delta": float(args.pass_delta),
            "tiny_overfit_samples": int(args.tiny_overfit_samples),
            "tiny_overfit_epochs": int(args.tiny_overfit_epochs),
            "tiny_overfit_threshold": float(args.tiny_overfit_threshold),
            "sanity_epochs": int(args.sanity_epochs),
            "shuffled_max_map": float(args.shuffled_max_map),
            "shuffled_min_drop": float(args.shuffled_min_drop),
            "only_sanity": bool(args.only_sanity),
            "adamw_lr": float(args.adamw_lr),
            "adamw_weight_decay": float(args.adamw_weight_decay),
            "sgd_lr": float(args.sgd_lr),
            "sgd_weight_decay": float(args.sgd_weight_decay),
            "model": {
                "N": model_cfg.N,
                "D": model_cfg.D,
                "t": model_cfg.t,
                "p": model_cfg.p,
                "d_model": model_cfg.d_model,
                "n_layers": model_cfg.n_layers,
                "n_heads": model_cfg.n_heads,
            },
        },
        "sanity_checks": {},
        "runs": [],
        "ablation_summary": [],
        "gate_stats": None,
        "decision": {"status": "NOT PASS", "reasons": []},
        "error": None,
    }

    try:
        print("[setup] Loading nuScenes and clip index...")
        nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
        items = load_clip_index(args.index_path)
        clip_ds = ClipDataset(nusc=nusc, dataroot=args.dataroot, index_path=args.index_path)
        if len(items) != len(clip_ds):
            raise RuntimeError(f"Clip index mismatch: len(items)={len(items)} len(clip_ds)={len(clip_ds)}")

        split_by_seed: dict[int, dict[str, Any]] = {}
        all_needed_indices: set[int] = set()
        scene_rows: list[dict[str, Any]] = []
        for seed in seeds:
            train_idx, val_idx = scene_disjoint_split(items, val_scene_frac=args.val_scene_frac, seed=seed)
            train_scenes = {items[i].scene_idx for i in train_idx}
            val_scenes = {items[i].scene_idx for i in val_idx}
            overlap = len(train_scenes & val_scenes)
            scene_rows.append(
                {
                    "seed": int(seed),
                    "train_scene_count": int(len(train_scenes)),
                    "val_scene_count": int(len(val_scenes)),
                    "scene_overlap": int(overlap),
                }
            )
            if overlap != 0:
                raise RuntimeError(
                    f"Scene overlap detected for seed={seed}: overlap={overlap}. "
                    "Split policy must be scene-disjoint."
                )

            train_idx = cap_indices(train_idx, args.max_train_samples, seed=seed * 31 + 1)
            val_idx = cap_indices(val_idx, args.max_val_samples, seed=seed * 31 + 2)

            split_by_seed[seed] = {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_count": len(train_idx),
                "val_count": len(val_idx),
            }
            all_needed_indices.update(train_idx)
            all_needed_indices.update(val_idx)

        report["sanity_checks"]["scene_overlap"] = {"passed": True, "per_seed": scene_rows}

        index_union = sorted(all_needed_indices)
        idx_to_row = {clip_idx: row_idx for row_idx, clip_idx in enumerate(index_union)}
        print(f"[setup] Total unique clips needed across all seeds: {len(index_union)}")

        print("[labels] Building labels once for all required clips...")
        y_all, meta_by_clip_idx = labels_and_metadata(nusc=nusc, items=items, index_union=index_union)

        print("[alignment] Logging 10 random alignment examples...")
        rng = random.Random(args.base_seed + 7_777)
        alignment_count = min(10, len(index_union))
        alignment_indices = rng.sample(index_union, k=alignment_count)
        alignment_examples: list[dict[str, Any]] = []
        for clip_idx in alignment_indices:
            row = idx_to_row[clip_idx]
            record = dict(meta_by_clip_idx[clip_idx])
            record["num_positive_labels"] = int(y_all[row].sum().item())
            alignment_examples.append(record)
            print(
                f"  clip_idx={record['clip_index']} "
                f"start_sd_token={record['start_sd_token']} "
                f"sample_token={record['sample_token']} "
                f"timestamp={record['timestamp']} "
                f"num_pos={record['num_positive_labels']}"
            )
        report["sanity_checks"]["alignment_examples"] = alignment_examples

        sample_id_by_clip_idx = {i: meta_by_clip_idx[i]["sample_token"] for i in index_union}
        if len(set(sample_id_by_clip_idx.values())) != len(sample_id_by_clip_idx):
            raise RuntimeError("Sample-token identifiers are not unique; cannot build stable-id cache.")

        token_overlap_rows: list[dict[str, Any]] = []
        for seed in seeds:
            train_ids = [sample_id_by_clip_idx[i] for i in split_by_seed[seed]["train_idx"]]
            val_ids = [sample_id_by_clip_idx[i] for i in split_by_seed[seed]["val_idx"]]
            tok_row = assert_zero_token_overlap(
                train_ids=train_ids,
                val_ids=val_ids,
                id_label="sample_token",
                seed=seed,
            )
            token_overlap_rows.append(tok_row)
            print(
                f"[overlap] seed={seed} train_unique={tok_row['train_unique']} "
                f"val_unique={tok_row['val_unique']} overlap={tok_row['overlap_count']}"
            )
        report["sanity_checks"]["token_overlap"] = {"passed": True, "per_seed": token_overlap_rows}

        feature_cache: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

        print("[features] Extracting random-encoder features (shared across all runs)...")
        rand_encoder = build_encoder(
            pretrained=False,
            model_cfg=model_cfg,
            ckpt_payload=ckpt_payload,
            device=device,
        )
        feature_cache["random"] = extract_features_for_encoder(
            encoder=rand_encoder,
            clip_ds=clip_ds,
            items=items,
            index_union=index_union,
            sample_id_by_clip_idx=sample_id_by_clip_idx,
            model_cfg=model_cfg,
            pooling_modes=pooling_modes,
            device=device,
            progress_every=args.progress_every,
        )

        if ckpt_payload is not None:
            print("[features] Extracting pretrained-encoder features (shared across all runs)...")
            pre_encoder = build_encoder(
                pretrained=True,
                model_cfg=model_cfg,
                ckpt_payload=ckpt_payload,
                device=device,
            )
            feature_cache["pretrained"] = extract_features_for_encoder(
                encoder=pre_encoder,
                clip_ds=clip_ds,
                items=items,
                index_union=index_union,
                sample_id_by_clip_idx=sample_id_by_clip_idx,
                model_cfg=model_cfg,
                pooling_modes=pooling_modes,
                device=device,
                progress_every=args.progress_every,
            )

        # -------------------------
        # Sanity checks (fail-fast)
        # -------------------------
        sanity_variant = "pretrained" if ckpt_payload is not None else "random"
        sanity_seed = seeds[0]
        sanity_train_idx = split_by_seed[sanity_seed]["train_idx"]
        sanity_val_idx = split_by_seed[sanity_seed]["val_idx"]
        sanity_train_sample_ids = [sample_id_by_clip_idx[i] for i in sanity_train_idx]
        sanity_val_sample_ids = [sample_id_by_clip_idx[i] for i in sanity_val_idx]
        y_train_sanity_lookup = build_label_map_by_sample_id(
            clip_indices=sanity_train_idx,
            idx_to_row=idx_to_row,
            y_all=y_all,
            sample_id_by_clip_idx=sample_id_by_clip_idx,
            shuffled_labels=False,
        )
        y_val_sanity_lookup = build_label_map_by_sample_id(
            clip_indices=sanity_val_idx,
            idx_to_row=idx_to_row,
            y_all=y_all,
            sample_id_by_clip_idx=sample_id_by_clip_idx,
            shuffled_labels=False,
        )

        x_train_sanity, y_train_sanity = gather_split_data(
            feature_map_by_id=feature_cache[sanity_variant][args.gate_pooling],
            label_map_by_id=y_train_sanity_lookup,
            sample_ids=sanity_train_sample_ids,
        )
        x_val_sanity, y_val_sanity = gather_split_data(
            feature_map_by_id=feature_cache[sanity_variant][args.gate_pooling],
            label_map_by_id=y_val_sanity_lookup,
            sample_ids=sanity_val_sample_ids,
        )

        print("[sanity] tiny-subset overfit check...")
        tiny_n = min(args.tiny_overfit_samples, len(sanity_train_idx))
        tiny_sample_ids = sanity_train_sample_ids[:tiny_n]
        tiny_label_lookup = {sid: y_train_sanity_lookup[sid] for sid in tiny_sample_ids}
        x_tiny, y_tiny = gather_split_data(
            feature_map_by_id=feature_cache[sanity_variant][args.gate_pooling],
            label_map_by_id=tiny_label_lookup,
            sample_ids=tiny_sample_ids,
        )
        tiny_probe_seed = make_probe_seed(sanity_seed, args.gate_pooling, args.gate_optimizer) + 1
        tiny_metrics = train_linear_probe(
            x_train=x_tiny,
            y_train=y_tiny,
            train_sample_ids=tiny_sample_ids,
            x_eval=x_tiny,
            y_eval=y_tiny,
            optimizer_name=args.gate_optimizer,
            epochs=args.tiny_overfit_epochs,
            batch_size=min(args.batch_size, max(1, tiny_n)),
            probe_seed=tiny_probe_seed,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
            sgd_lr=args.sgd_lr,
            sgd_weight_decay=args.sgd_weight_decay,
            device=device,
            run_name="sanity_tiny_overfit",
        )
        tiny_pass = tiny_metrics["mAP"] >= args.tiny_overfit_threshold
        report["sanity_checks"]["tiny_subset_overfit"] = {
            "passed": bool(tiny_pass),
            "variant": sanity_variant,
            "seed": int(sanity_seed),
            "sample_count": int(tiny_n),
            "train_mAP": float(tiny_metrics["mAP"]),
            "threshold": float(args.tiny_overfit_threshold),
        }
        if not tiny_pass:
            raise RuntimeError(
                f"Tiny-subset overfit failed: train mAP={tiny_metrics['mAP']:.4f} "
                f"< {args.tiny_overfit_threshold:.4f}. Likely feature/label misalignment."
            )

        print("[sanity] shuffled-label collapse check...")
        reference_seed = make_probe_seed(sanity_seed, args.gate_pooling, args.gate_optimizer) + 2
        y_train_shuffled, shuffle_evidence = make_shuffled_labels_with_evidence(
            y_train=y_train_sanity,
            seed=args.base_seed + 9_101,
        )
        print(
            f"[sanity] shuffle checksums train={shuffle_evidence['train_checksum_md5_uint8']} "
            f"shuffled={shuffle_evidence['shuffled_checksum_md5_uint8']} "
            f"first5_row_diff_count={shuffle_evidence['first5_row_diff_count']}"
        )
        print("[sanity] first5 positives before shuffle:", shuffle_evidence["first5_positive_indices_before"])
        print("[sanity] first5 positives after  shuffle:", shuffle_evidence["first5_positive_indices_after"])

        ref_metrics = train_linear_probe(
            x_train=x_train_sanity,
            y_train=y_train_sanity,
            train_sample_ids=sanity_train_sample_ids,
            x_eval=x_val_sanity,
            y_eval=y_val_sanity,
            optimizer_name=args.gate_optimizer,
            epochs=args.sanity_epochs,
            batch_size=args.batch_size,
            probe_seed=reference_seed,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
            sgd_lr=args.sgd_lr,
            sgd_weight_decay=args.sgd_weight_decay,
            device=device,
            run_name="sanity_reference",
            y_val_reference=y_val_sanity,
            y_train_reference=y_train_sanity,
        )
        shuffled_label_lookup: dict[str, torch.Tensor] = {}
        if len(shuffled_label_lookup) != 0:
            raise RuntimeError(
                "Internal error: shuffled_labels=True but label cache is not empty at start."
            )
        for sid, yrow in zip(sanity_train_sample_ids, y_train_shuffled):
            shuffled_label_lookup[sid] = yrow.clone().cpu()

        shuffled_seed = reference_seed
        shuffled_debug_metrics = train_linear_probe(
            x_train=x_train_sanity,
            y_train=y_train_shuffled,
            train_sample_ids=sanity_train_sample_ids,
            x_eval=x_val_sanity,
            y_eval=y_val_sanity,
            optimizer_name=args.gate_optimizer,
            epochs=1,
            batch_size=args.batch_size,
            probe_seed=shuffled_seed,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
            sgd_lr=args.sgd_lr,
            sgd_weight_decay=args.sgd_weight_decay,
            device=device,
            run_name="sanity_shuffled_debug_epoch1",
            debug_first_batch_label_match=True,
            train_label_lookup_by_id=shuffled_label_lookup,
            y_val_reference=y_val_sanity,
            y_train_reference=y_train_sanity,
        )

        shuffled_metrics = train_linear_probe(
            x_train=x_train_sanity,
            y_train=y_train_shuffled,
            train_sample_ids=sanity_train_sample_ids,
            x_eval=x_val_sanity,
            y_eval=y_val_sanity,
            optimizer_name=args.gate_optimizer,
            epochs=args.sanity_epochs,
            batch_size=args.batch_size,
            probe_seed=shuffled_seed,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
            sgd_lr=args.sgd_lr,
            sgd_weight_decay=args.sgd_weight_decay,
            device=device,
            run_name="sanity_shuffled",
            y_val_reference=y_val_sanity,
            y_train_reference=y_train_sanity,
        )
        if ref_metrics.get("train_true_mAP") is None or shuffled_metrics.get("train_true_mAP") is None:
            raise RuntimeError("Missing train_true_mAP for shuffled-label sanity collapse check.")
        drop = float(ref_metrics["train_true_mAP"] - shuffled_metrics["train_true_mAP"])
        shuffled_pass = drop >= args.shuffled_min_drop
        report["sanity_checks"]["shuffled_label"] = {
            "passed": bool(shuffled_pass),
            "variant": sanity_variant,
            "seed": int(sanity_seed),
            "collapse_metric": "train_true_mAP",
            "reference_mAP": float(ref_metrics["train_true_mAP"]),
            "shuffled_mAP": float(shuffled_metrics["train_true_mAP"]),
            "drop": float(drop),
            "min_required_drop": float(args.shuffled_min_drop),
            "reference_val_mAP": float(ref_metrics["mAP"]),
            "shuffled_val_mAP": float(shuffled_metrics["mAP"]),
            "shuffle_evidence": shuffle_evidence,
            "debug_first_batch": shuffled_metrics.get("debug_first_batch"),
            "debug_one_epoch": shuffled_debug_metrics.get("debug_first_batch"),
            "eval_checksums": shuffled_metrics.get("eval_checksums"),
            "reference_eval_checksums": ref_metrics.get("eval_checksums"),
        }
        if not shuffled_pass:
            raise RuntimeError(
                "Shuffled-label sanity failed: shuffled-label mAP did not collapse enough. "
                "Likely leakage/bug in split, labels, or feature-target alignment."
            )

        report["sanity_checks"]["all_passed"] = True

        if args.only_sanity:
            report["runs"] = []
            report["ablation_summary"] = []
            report["gate_stats"] = None
            report["decision"] = {
                "status": "PASS",
                "reasons": [
                    "Sanity-only run requested; all Day 7 sanity checks passed.",
                    "PASS gate (pretrained vs random delta) was not evaluated in --only_sanity mode.",
                ],
            }
            raise StopIteration("only_sanity_completed")

        # -------------------------
        # Finite ablation suite
        # -------------------------
        print("[run] Running finite ablation grid...")
        print("pooling              | optimizer | seed | pre_mAP | rand_mAP | delta")
        print("---------------------+-----------+------+---------+----------+--------")

        runs: list[dict[str, Any]] = []
        for pooling in pooling_modes:
            for optimizer_name in optimizers:
                for seed in seeds:
                    split = split_by_seed[seed]
                    train_idx = split["train_idx"]
                    val_idx = split["val_idx"]
                    train_sample_ids = [sample_id_by_clip_idx[i] for i in train_idx]
                    val_sample_ids = [sample_id_by_clip_idx[i] for i in val_idx]
                    probe_seed = make_probe_seed(seed, pooling, optimizer_name)

                    y_train_lookup = build_label_map_by_sample_id(
                        clip_indices=train_idx,
                        idx_to_row=idx_to_row,
                        y_all=y_all,
                        sample_id_by_clip_idx=sample_id_by_clip_idx,
                        shuffled_labels=False,
                    )
                    y_val_lookup = build_label_map_by_sample_id(
                        clip_indices=val_idx,
                        idx_to_row=idx_to_row,
                        y_all=y_all,
                        sample_id_by_clip_idx=sample_id_by_clip_idx,
                        shuffled_labels=False,
                    )

                    x_train_rand, y_train_rand = gather_split_data(
                        feature_map_by_id=feature_cache["random"][pooling],
                        label_map_by_id=y_train_lookup,
                        sample_ids=train_sample_ids,
                    )
                    x_val_rand, y_val_rand = gather_split_data(
                        feature_map_by_id=feature_cache["random"][pooling],
                        label_map_by_id=y_val_lookup,
                        sample_ids=val_sample_ids,
                    )
                    rand_metrics = train_linear_probe(
                        x_train=x_train_rand,
                        y_train=y_train_rand,
                        train_sample_ids=train_sample_ids,
                        x_eval=x_val_rand,
                        y_eval=y_val_rand,
                        optimizer_name=optimizer_name,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        probe_seed=probe_seed,
                        adamw_lr=args.adamw_lr,
                        adamw_weight_decay=args.adamw_weight_decay,
                        sgd_lr=args.sgd_lr,
                        sgd_weight_decay=args.sgd_weight_decay,
                        device=device,
                        run_name=f"ablation_random_{pooling}_{optimizer_name}_seed{seed}",
                        y_val_reference=y_val_rand,
                        y_train_reference=y_train_rand,
                    )

                    pre_metrics: dict[str, Any] | None = None
                    if ckpt_payload is not None:
                        x_train_pre, y_train_pre = gather_split_data(
                            feature_map_by_id=feature_cache["pretrained"][pooling],
                            label_map_by_id=y_train_lookup,
                            sample_ids=train_sample_ids,
                        )
                        x_val_pre, y_val_pre = gather_split_data(
                            feature_map_by_id=feature_cache["pretrained"][pooling],
                            label_map_by_id=y_val_lookup,
                            sample_ids=val_sample_ids,
                        )
                        pre_metrics = train_linear_probe(
                            x_train=x_train_pre,
                            y_train=y_train_pre,
                            train_sample_ids=train_sample_ids,
                            x_eval=x_val_pre,
                            y_eval=y_val_pre,
                            optimizer_name=optimizer_name,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            probe_seed=probe_seed,
                            adamw_lr=args.adamw_lr,
                            adamw_weight_decay=args.adamw_weight_decay,
                            sgd_lr=args.sgd_lr,
                            sgd_weight_decay=args.sgd_weight_decay,
                            device=device,
                            run_name=f"ablation_pretrained_{pooling}_{optimizer_name}_seed{seed}",
                            y_val_reference=y_val_pre,
                            y_train_reference=y_train_pre,
                        )

                    delta = None if pre_metrics is None else float(pre_metrics["mAP"] - rand_metrics["mAP"])
                    row = {
                        "pooling": pooling,
                        "optimizer": optimizer_name,
                        "seed": int(seed),
                        "probe_seed": int(probe_seed),
                        "split": {
                            "train_count": int(split["train_count"]),
                            "val_count": int(split["val_count"]),
                        },
                        "pretrained": pre_metrics,
                        "random": rand_metrics,
                        "delta_mAP": delta,
                    }
                    runs.append(row)

                    pre_map_text = "NA" if pre_metrics is None else f"{pre_metrics['mAP']:.4f}"
                    delta_text = "NA" if delta is None else f"{delta:.4f}"
                    print(
                        f"{pooling:<21}| {optimizer_name:<10}| {seed:>4} | "
                        f"{pre_map_text:>7} | {rand_metrics['mAP']:.4f}   | {delta_text:>6}"
                    )

        report["runs"] = runs
        report["ablation_summary"] = summarize_runs(runs, pooling_modes=pooling_modes, optimizers=optimizers)

        gate_rows = [
            r
            for r in runs
            if r["pooling"] == args.gate_pooling
            and r["optimizer"] == args.gate_optimizer
            and r["pretrained"] is not None
        ]
        gate_stats: dict[str, Any] | None = None
        if gate_rows:
            pre_maps = [r["pretrained"]["mAP"] for r in gate_rows]
            rand_maps = [r["random"]["mAP"] for r in gate_rows]
            deltas = [r["delta_mAP"] for r in gate_rows]
            gate_stats = {
                "pooling": args.gate_pooling,
                "optimizer": args.gate_optimizer,
                "seeds": [int(r["seed"]) for r in gate_rows],
                "pretrained_mean": float(np.mean(pre_maps)),
                "pretrained_std": float(np.std(pre_maps)),
                "random_mean": float(np.mean(rand_maps)),
                "random_std": float(np.std(rand_maps)),
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas)),
                "required_delta": float(args.pass_delta),
                "pass": bool(float(np.mean(deltas)) >= args.pass_delta),
            }
        report["gate_stats"] = gate_stats

        if ckpt_payload is None:
            decision_status = "NOT PASS"
            decision_reasons = [
                missing_ckpt_reason
                or "Pretrained checkpoint missing; cannot evaluate pretrained vs random gate.",
            ]
        elif gate_stats is None:
            decision_status = "NOT PASS"
            decision_reasons = ["No gate rows available for pretrained runs."]
        else:
            decision_status = "PASS" if gate_stats["pass"] else "NOT PASS"
            if decision_status == "PASS":
                decision_reasons = [
                    "Pretrained exceeds random by >= 0.01 mAP on gate config across seeds [1,2,3].",
                    (
                        f"Gate delta mean±std = {gate_stats['delta_mean']:.4f} ± {gate_stats['delta_std']:.4f}, "
                        f"pretrained mAP mean±std = {gate_stats['pretrained_mean']:.4f} ± {gate_stats['pretrained_std']:.4f}, "
                        f"random mAP mean±std = {gate_stats['random_mean']:.4f} ± {gate_stats['random_std']:.4f}."
                    ),
                ]
            else:
                decision_reasons = infer_not_pass_causes(
                    sanity_checks=report["sanity_checks"],
                    ablation_summary=report["ablation_summary"],
                    gate_stats=gate_stats,
                    pass_delta=args.pass_delta,
                    has_pretrained=True,
                    missing_ckpt_reason=missing_ckpt_reason,
                )
                if not decision_reasons:
                    decision_reasons = ["Gate criterion not met and no dominant single cause identified."]

        report["decision"] = {"status": decision_status, "reasons": decision_reasons}

    except StopIteration as stop_exc:
        if str(stop_exc) != "only_sanity_completed":
            raise
    except Exception as exc:
        report["error"] = str(exc)
        if "all_passed" not in report["sanity_checks"]:
            report["sanity_checks"]["all_passed"] = False

        fallback_reasons = infer_not_pass_causes(
            sanity_checks=report["sanity_checks"],
            ablation_summary=report.get("ablation_summary", []),
            gate_stats=report.get("gate_stats"),
            pass_delta=float(args.pass_delta),
            has_pretrained=bool(ckpt_payload is not None),
            missing_ckpt_reason=missing_ckpt_reason,
        )
        if not fallback_reasons:
            fallback_reasons = [f"Run aborted with error: {exc}"]
        report["decision"] = {"status": "NOT PASS", "reasons": fallback_reasons}

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    write_markdown_report(report, report_md)

    print("")
    print(f"[report] JSON: {report_json}")
    print(f"[report] MD  : {report_md}")
    print(f"[decision] {report['decision']['status']}")
    for reason in report["decision"]["reasons"]:
        print(f"  - {reason}")

    if report.get("error") is not None:
        print(f"[error] {report['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
