from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TripwireConfig:
    # hard stops
    max_grad_norm: float = 50.0
    max_pred_absmax: float = 50.0
    max_act_absmax: float = 50.0

    # warnings (soft)
    warn_grad_norm: float = 5.0
    warn_pred_absmax: float = 5.0
    warn_act_absmax: float = 10.0


def check_tripwires(
    step: int,
    day: int,
    gnorm: float,
    pred_absmax: float,
    act_absmax_values: dict[str, float],
    cfg: TripwireConfig = TripwireConfig(),
) -> None:
    # soft warnings
    if gnorm > cfg.warn_grad_norm:
        print(f"[WARN] Step {step} (Day {day}): grad norm high: {gnorm:.3f}")
    if pred_absmax > cfg.warn_pred_absmax:
        print(f"[WARN] Step {step} (Day {day}): pred absmax high: {pred_absmax:.3f}")
    for name, v in act_absmax_values.items():
        if v > cfg.warn_act_absmax:
            print(f"[WARN] Step {step} (Day {day}): activation absmax high: {name}={v:.3f}")

    # hard stops
    if gnorm > cfg.max_grad_norm:
        raise RuntimeError(f"Tripwire stop: grad norm {gnorm:.3f} > {cfg.max_grad_norm}")
    if pred_absmax > cfg.max_pred_absmax:
        raise RuntimeError(f"Tripwire stop: pred absmax {pred_absmax:.3f} > {cfg.max_pred_absmax}")
    for name, v in act_absmax_values.items():
        if v > cfg.max_act_absmax:
            raise RuntimeError(f"Tripwire stop: activation {name} absmax {v:.3f} > {cfg.max_act_absmax}")
