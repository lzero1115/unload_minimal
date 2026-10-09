from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MotionPlanUnavailableError(RuntimeError):
    """The selected grasp and unload targets have no complete executable plan."""


class TaskPhase(str, Enum):
    IDLE = "idle"
    MOVE_TO_APPROACH = "move_to_approach"
    APPROACH_TO_CONTACT = "approach_to_contact"
    ATTACH = "attach"
    UNLOAD_WITH_PHYSICS = "unload_with_physics"
    RETURN_TO_REST_WITH_PHYSICS = "return_to_rest_with_physics"
    RELEASE_REMOVE = "release_remove"
    POST_SETTLE = "post_settle"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class MotionPlan:
    box_index: int
    frame_index: int
    roll_index: int
    joint_names: tuple[str, ...]
    forward_q_path: np.ndarray
    contact_q_path: np.ndarray
    unload_q_path: np.ndarray
    return_q_path: np.ndarray
    contact_position_w: np.ndarray
    contact_quat_wxyz_w: np.ndarray
    unload_direction_w: np.ndarray
    unload_distance_m: float
    unload_target_position_w: np.ndarray
    unload_target_quat_wxyz_w: np.ndarray
    attempted_targets: int
    status: str
