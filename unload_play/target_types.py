from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from configs import task_configs as tc


class DecisionPhase(str, Enum):
    AWAIT_IK = "await_manual_ik"
    COMPUTING_IK = "computing_ik"
    AWAIT_BOX = "await_box"
    AWAIT_FRAME = "await_frame"
    AWAIT_PLAN = "await_plan"
    READY_TO_EXECUTE = "ready_to_execute"
    EXECUTING = "executing"
    SETTLING = "settling"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class PileAdjustment:
    translation_x_m: float = 0.0
    translation_y_m: float = 0.0
    yaw_deg: float = 0.0


def adjusted_pile_poses(*, positions_w, quats_wxyz, active_mask, adjustment):
    """Rigid XY/yaw adjustment about the original pile center, as in unload_replay."""
    positions = np.asarray(positions_w, dtype=np.float64).copy()
    quaternions = np.asarray(quats_wxyz, dtype=np.float64).copy()
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    if positions.shape != (len(active), 3) or quaternions.shape != (len(active), 4):
        raise ValueError("Pile poses do not match the active mask")
    if not np.any(active):
        raise ValueError("Cannot adjust an empty pile")
    center = np.mean(positions[active], axis=0)
    center[2] = 0.0
    yaw = Rotation.from_euler("z", adjustment.yaw_deg, degrees=True)
    translation = np.array([adjustment.translation_x_m, adjustment.translation_y_m, 0.0])
    positions[active] = center + yaw.apply(positions[active] - center) + translation
    xyzw = quaternions[active][:, [1, 2, 3, 0]]
    quaternions[active] = (yaw * Rotation.from_quat(xyzw)).as_quat()[:, [3, 0, 1, 2]]
    return positions, quaternions


@dataclass(frozen=True)
class FrameAdjustment:
    tangent_u_m: float = 0.0
    tangent_v_m: float = 0.0
    normal_m: float = 0.0
    rotation_u_deg: float = 0.0
    rotation_v_deg: float = 0.0
    rotation_normal_deg: float = 0.0


@dataclass(frozen=True)
class UnloadAdjustment:
    tilt_u_deg: float = 0.0
    tilt_v_deg: float = 0.0
    distance_m: float = float(tc.GRASP_APPROACH_CLEARANCE)


@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    box_index: int
    face_axis: int
    face_sign: int
    roll_index: int
    position_w: np.ndarray
    quat_wxyz_w: np.ndarray
    normal_w: np.ndarray
    tangent_u_w: np.ndarray
    tangent_v_w: np.ndarray
    openness_score: int
    face_area: float
    initially_ik_reachable: bool = True

    def adjusted_position(self, adjustment: FrameAdjustment) -> np.ndarray:
        return np.asarray(self.position_w, dtype=np.float64) + (
            float(adjustment.tangent_u_m)
            * np.asarray(self.tangent_u_w, dtype=np.float64)
            + float(adjustment.tangent_v_m)
            * np.asarray(self.tangent_v_w, dtype=np.float64)
            + float(adjustment.normal_m)
            * np.asarray(self.normal_w, dtype=np.float64)
        )

    def adjusted_rotation(self, adjustment: FrameAdjustment) -> np.ndarray:
        quat_wxyz = np.asarray(self.quat_wxyz_w, dtype=np.float64).reshape(4)
        quat_wxyz /= max(
            float(np.linalg.norm(quat_wxyz)), np.finfo(np.float64).eps
        )
        base = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        # The EE local +Z axis points into the surface, opposite the displayed
        # outward face normal. Negate the normal-axis angle accordingly.
        local_delta = Rotation.from_euler(
            "xyz",
            (
                float(adjustment.rotation_u_deg),
                float(adjustment.rotation_v_deg),
                -float(adjustment.rotation_normal_deg),
            ),
            degrees=True,
        ).as_matrix()
        return base @ local_delta

    def adjusted_quat_wxyz(self, adjustment: FrameAdjustment) -> np.ndarray:
        quat_xyzw = Rotation.from_matrix(self.adjusted_rotation(adjustment)).as_quat()
        return quat_xyzw[[3, 0, 1, 2]]

    def adjusted_tangent_u(self, adjustment: FrameAdjustment) -> np.ndarray:
        return self.adjusted_rotation(adjustment)[:, 0].copy()

    def adjusted_tangent_v(self, adjustment: FrameAdjustment) -> np.ndarray:
        return self.adjusted_rotation(adjustment)[:, 1].copy()

    def adjusted_normal(self, adjustment: FrameAdjustment) -> np.ndarray:
        return -self.adjusted_rotation(adjustment)[:, 2].copy()


@dataclass(frozen=True)
class CandidateSnapshot:
    query: dict
    proxy_mask: np.ndarray
    ik_mask: np.ndarray
    frames_by_box: dict[int, tuple[FrameCandidate, ...]]
    ik_total_s: float


@dataclass(frozen=True)
class SelectedTarget:
    candidate: FrameCandidate
    adjustment: FrameAdjustment

    @property
    def box_index(self) -> int:
        return int(self.candidate.box_index)

    @property
    def position_w(self) -> np.ndarray:
        return self.candidate.adjusted_position(self.adjustment)

    @property
    def quat_wxyz_w(self) -> np.ndarray:
        return self.candidate.adjusted_quat_wxyz(self.adjustment)

    @property
    def normal_w(self) -> np.ndarray:
        return self.candidate.adjusted_normal(self.adjustment)

    @property
    def tangent_u_w(self) -> np.ndarray:
        return self.candidate.adjusted_tangent_u(self.adjustment)

    @property
    def tangent_v_w(self) -> np.ndarray:
        return self.candidate.adjusted_tangent_v(self.adjustment)


def unload_direction_w(
    target: SelectedTarget,
    adjustment: UnloadAdjustment,
) -> np.ndarray:
    direction = normalized(target.normal_w, name="adjusted target normal")
    rotate_u = Rotation.from_rotvec(
        np.deg2rad(float(adjustment.tilt_u_deg))
        * normalized(target.tangent_u_w, name="adjusted tangent u")
    )
    direction = rotate_u.apply(direction)
    rotate_v = Rotation.from_rotvec(
        np.deg2rad(float(adjustment.tilt_v_deg))
        * normalized(target.tangent_v_w, name="adjusted tangent v")
    )
    return normalized(rotate_v.apply(direction), name="unload direction")


def unload_target_position_w(
    target: SelectedTarget,
    adjustment: UnloadAdjustment,
) -> np.ndarray:
    return np.asarray(target.position_w, dtype=np.float64) + unload_direction_w(
        target,
        adjustment,
    ) * float(adjustment.distance_m)


def normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} has zero length.")
    return value / norm
