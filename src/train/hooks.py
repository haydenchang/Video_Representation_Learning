from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ActStat:
    mean: float
    std: float
    absmax: float


class ActivationRecorder:
    def __init__(self):
        self.stats: dict[str, ActStat] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @torch.no_grad()
    def _record(self, name: str, x: torch.Tensor, store_tensor: bool) -> None:
        x_det = x.detach()
        x_f = x_det.float()
        self.stats[name] = ActStat(
            mean=float(x_f.mean().item()),
            std=float(x_f.std().item()),
            absmax=float(x_f.abs().max().item()),
        )
        if store_tensor:
            self.tensors[name] = x_det

    def hook_output(self, module: torch.nn.Module, name: str, store_tensor: bool = False) -> None:
        def _hook(_mod, _inp, out):
            out0 = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(out0):
                self._record(name, out0, store_tensor=store_tensor)
        self._handles.append(module.register_forward_hook(_hook))

    def clear(self) -> None:
        self.stats.clear()
        self.tensors.clear()
