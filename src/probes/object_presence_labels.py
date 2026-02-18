from __future__ import annotations

from typing import Optional

LABELS = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]
LABEL_TO_IDX = {k: i for i, k in enumerate(LABELS)}


def map_nuscenes_category_to_label(cat: str) -> Optional[str]:
    """
    Maps nuScenes fine-grained category_name -> coarse label.
    Returns None if category is not in our label set.
    """
    # Vehicles
    if cat == "vehicle.car":
        return "car"
    if cat == "vehicle.truck":
        return "truck"
    if cat == "vehicle.bus.bendy" or cat == "vehicle.bus.rigid":
        return "bus"
    if cat == "vehicle.motorcycle":
        return "motorcycle"
    if cat == "vehicle.bicycle":
        return "bicycle"

    # Humans (all pedestrians collapse to pedestrian)
    if cat.startswith("human.pedestrian."):
        return "pedestrian"

    return None
