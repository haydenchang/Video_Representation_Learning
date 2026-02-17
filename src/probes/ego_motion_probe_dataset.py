from __future__ import annotations

import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes

from src.data.clip_index import ClipSpec, load_clip_index
from src.data.clip_dataset import ClipDataset
from src.model.patchify import tubeletify
from src.model.masking import make_random_mask
from src.probes.ego_motion_targets import clip_ego_motion_bucket_onehot
from src.probes.ego_motion_labels import EGO_LABELS


class EgoMotionProbeDataset(Dataset):
    """
    Returns:
      z: torch.FloatTensor [d_model]
      y: torch.FloatTensor [3]  (slow/medium/fast one-hot)
    """
    def __init__(
        self,
        nusc: NuScenes,
        dataroot: str,
        index_path: str,
        encoder_model,
        t: int,
        p: int,
        device: torch.device,
    ):
        self.nusc = nusc
        self.items: list[ClipSpec] = load_clip_index(index_path)
        self.clip_ds = ClipDataset(nusc=nusc, dataroot=dataroot, index_path=index_path)
        self.encoder = encoder_model
        self.t = t
        self.p = p
        self.device = device

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        spec = self.items[idx]
        item = self.clip_ds[idx]

        video = item["video"].unsqueeze(0).to(self.device)  # [1,T,C,H,W]
        tokens = tubeletify(video, t=self.t, p=self.p)

        B, N, D = tokens.shape
        ids_visible = torch.arange(N, device=self.device).unsqueeze(0)
        tokens_visible = tokens

        with torch.no_grad():
            x_lat = self.encoder.encode_visible(tokens_visible, ids_visible)
            # Drop CLS if present (encode_visible may prepend it)
            if x_lat.shape[1] == tokens_visible.shape[1] + 1:
                x_tok = x_lat[:, 1:, :]
            else:
                x_tok = x_lat
            z = x_tok.mean(dim=1).squeeze(0).cpu()

        y_np = clip_ego_motion_bucket_onehot(
            self.nusc,
            start_sd_token=spec.start_sd_token,
            T=spec.T,
            stride=spec.stride,
            keyframes_only=spec.keyframes_only,
        )
        y = torch.from_numpy(y_np)

        return z, y
