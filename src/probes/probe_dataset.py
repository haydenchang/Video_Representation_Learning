from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes

from src.data.clip_index import ClipSpec, load_clip_index
from src.data.clip_dataset import ClipDataset
from src.model.patchify import tubeletify
from src.model.masking import make_random_mask
from src.probes.object_presence_targets import clip_object_presence_multihot


class ObjectPresenceProbeDataset(Dataset):
    """
    Returns:
      z: torch.FloatTensor [d_model]  (representation)
      y: torch.FloatTensor [K]        (multi-hot labels)
    """
    def __init__(
        self,
        nusc: NuScenes,
        dataroot: str,
        index_path: str,
        encoder_model,
        t: int,
        p: int,
        mask_ratio_visible: float = 0.0,
        device: torch.device | None = None,
    ):
        self.nusc = nusc
        self.items: list[ClipSpec] = load_clip_index(index_path)
        self.clip_ds = ClipDataset(nusc=nusc, dataroot=dataroot, index_path=index_path)

        self.encoder = encoder_model
        self.t = t
        self.p = p
        self.mask_ratio_visible = mask_ratio_visible  # for probing we typically use all tokens visible -> 0.0
        self.device = device if device is not None else torch.device("cpu")

        # Deterministic mask generator if ever used
        self.mask_gen = torch.Generator(device="cpu").manual_seed(777)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        spec = self.items[idx]
        item = self.clip_ds[idx]
        video = item["video"].unsqueeze(0).to(self.device)  # [1,T,C,H,W]

        # Tokenize
        tokens = tubeletify(video, t=self.t, p=self.p)  # [1,N,D]
        _, N, D = tokens.shape

        # For probing, default is NO masking of visible tokens
        if self.mask_ratio_visible > 0.0:
            mask_cpu = make_random_mask(B=1, N=N, mask_ratio=self.mask_ratio_visible, device=torch.device("cpu"), generator=self.mask_gen)
            mask = mask_cpu.to(self.device)
            ids = torch.arange(N, device=self.device).unsqueeze(0)
            ids_visible = ids[~mask].view(1, -1)
            tokens_visible = tokens[~mask].view(1, -1, D)
        else:
            ids_visible = torch.arange(N, device=self.device).unsqueeze(0)   # [1,N]
            tokens_visible = tokens                                           # [1,N,D]

        with torch.no_grad():
            x_lat = self.encoder.encode_visible(tokens_visible, ids_visible)  # [1, 1+Nv, d_model] if CLS exists
            # Drop CLS if present (assume CLS at index 0)
            if x_lat.shape[1] == tokens_visible.shape[1] + 1:
                x_tok = x_lat[:, 1:, :]  # [1, Nv, d_model]
            else:
                x_tok = x_lat  # [1, Nv, d_model]

            mu = x_tok.mean(dim=1)  # [1, d_model]
            sd = x_tok.std(dim=1, unbiased=False)  # [1, d_model]
            z = torch.cat([mu, sd], dim=1).squeeze(0).cpu()  # [2*d_model]

        y = clip_object_presence_multihot(
            self.nusc,
            start_sd_token=spec.start_sd_token,
            T=spec.T,
            stride=spec.stride,
            keyframes_only=spec.keyframes_only,
        )
        y = torch.from_numpy(y)  # [K]

        return z, y
