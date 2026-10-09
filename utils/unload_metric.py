from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_UTILS_ROOT = Path(__file__).resolve().parent
_TASK_ROOT = _UTILS_ROOT.parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
for _p in (_UTILS_ROOT, _CONFIG_ROOT, _TASK_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import task_configs as tc

DEFAULT_INTER_STEADY_METRIC_ENERGY_SCALE = float(tc.METRIC_ALPHA_ENERGY)
DEFAULT_INTER_STEADY_METRIC_POSE_SCALE = float(tc.METRIC_BETA_POSE)
DEFAULT_INTER_STEADY_METRIC_TRANS_SCALE = float(tc.METRIC_LAMBDA_TRANS)
DEFAULT_INTER_STEADY_METRIC_ROT_SCALE = float(tc.METRIC_LAMBDA_ROT)
DEFAULT_INTER_STEADY_METRIC_STEP_DEADBAND = float(getattr(tc, "METRIC_STEP_DEADBAND", 0.0))
DEFAULT_INTER_STEADY_METRIC_LENGTH_SCALE = float(getattr(tc, "METRIC_LENGTH_SCALE", 1.0))
_EPS = 1e-15


def _quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(n, _EPS)


def quat_wxyz_geodesic_angle(*, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Minimum rotation angle in radians per row for WXYZ quaternions."""

    a = _quat_wxyz_normalize(q0)
    b = _quat_wxyz_normalize(q1)
    d = np.abs(np.sum(a * b, axis=1))
    d = np.minimum(d, 1.0)
    return 2.0 * np.arccos(d)


def collect_box_poses_mujoco(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_boxes: int,
    box_name_prefix: str = "box_",
) -> tuple[np.ndarray, np.ndarray]:
    """Collect current box positions and WXYZ quats by body name.

    Unlike the old unload_mujoco metric helper, this does not assume boxes occupy
    body ids ``1..N``. These scenes also contain robot bodies and optional
    per-box mocap anchors, so name-based lookup is the only safe convention.
    """

    n = int(n_boxes)
    if n < 0:
        raise ValueError(f"n_boxes must be non-negative, got {n_boxes!r}.")
    pos = np.zeros((n, 3), dtype=np.float64)
    quat = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        body_name = f"{box_name_prefix}{i}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise RuntimeError(f"MuJoCo model is missing body {body_name!r}.")
        pos[i] = np.asarray(data.xpos[bid], dtype=np.float64)
        quat[i] = np.asarray(data.xquat[bid], dtype=np.float64)
    return pos, quat


def body_diag_from_half_extents(*, half_extents: np.ndarray) -> np.ndarray:
    half = np.asarray(half_extents, dtype=np.float64).reshape(-1, 3)
    return 2.0 * np.linalg.norm(half, axis=1)


def box_volumes_from_half_extents(*, half_extents: np.ndarray) -> np.ndarray:
    half = np.asarray(half_extents, dtype=np.float64).reshape(-1, 3)
    return 8.0 * half[:, 0] * half[:, 1] * half[:, 2]


def _active_mask_bool(active_mask: np.ndarray) -> np.ndarray:
    return np.asarray(active_mask, dtype=bool).reshape(-1)


def potential_energy_proxy_per_box(
    *,
    pos: np.ndarray,
    half_extents: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Per-box potential-energy proxy ``volume * center_z``; inactive boxes are zero."""

    pos = np.asarray(pos, dtype=np.float64)
    active = _active_mask_bool(active_mask)
    vols = box_volumes_from_half_extents(half_extents=half_extents)
    n = int(active.shape[0])
    if pos.shape != (n, 3):
        raise ValueError(f"pos shape expects ({n}, 3); got {pos.shape}.")
    if vols.shape != (n,):
        raise ValueError(f"volume shape {vols.shape} vs n_boxes={n}.")

    out = np.zeros((n,), dtype=np.float64)
    out[active] = vols[active] * pos[active, 2]
    return out


def potential_energy_proxy_sum(
    *,
    pos: np.ndarray,
    half_extents: np.ndarray,
    active_mask: np.ndarray,
) -> float:
    return float(
        np.sum(
            potential_energy_proxy_per_box(
                pos=pos,
                half_extents=half_extents,
                active_mask=active_mask,
            )
        )
    )


def relative_pose_metric_components_per_box(
    *,
    pos_prev: np.ndarray,
    quat_prev: np.ndarray,
    pos_curr: np.ndarray,
    quat_curr: np.ndarray,
    body_diag: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-box normalized translation and rotation disturbance components."""

    pos_prev = np.asarray(pos_prev, dtype=np.float64)
    pos_curr = np.asarray(pos_curr, dtype=np.float64)
    quat_prev = np.asarray(quat_prev, dtype=np.float64)
    quat_curr = np.asarray(quat_curr, dtype=np.float64)
    diag = np.asarray(body_diag, dtype=np.float64).reshape(-1)
    active = _active_mask_bool(active_mask)

    n = int(active.shape[0])
    if pos_prev.shape != (n, 3) or pos_curr.shape != (n, 3):
        raise ValueError(f"pos shapes expect ({n}, 3); got {pos_prev.shape}, {pos_curr.shape}.")
    if quat_prev.shape != (n, 4) or quat_curr.shape != (n, 4):
        raise ValueError(
            f"quat shapes expect ({n}, 4); got {quat_prev.shape}, {quat_curr.shape}."
        )
    if diag.shape != (n,):
        raise ValueError(f"body_diag shape {diag.shape} vs n_boxes={n}.")

    trans = np.zeros((n,), dtype=np.float64)
    rot = np.zeros((n,), dtype=np.float64)
    if not np.any(active):
        return trans, rot

    dp = pos_curr[active] - pos_prev[active]
    trans[active] = np.linalg.norm(dp, axis=1) / np.maximum(diag[active], _EPS)
    rot[active] = quat_wxyz_geodesic_angle(
        q0=quat_prev[active],
        q1=quat_curr[active],
    ) / np.pi
    return trans, rot


def relative_pose_metric_per_box(
    *,
    pos_prev: np.ndarray,
    quat_prev: np.ndarray,
    pos_curr: np.ndarray,
    quat_curr: np.ndarray,
    body_diag: np.ndarray,
    active_mask: np.ndarray,
    lambda_trans: float = 1.0,
    lambda_rot: float = DEFAULT_INTER_STEADY_METRIC_ROT_SCALE,
) -> np.ndarray:
    """Per-box dimensionless pose disturbance; inactive boxes are zero."""

    trans, rot = relative_pose_metric_components_per_box(
        pos_prev=pos_prev,
        quat_prev=quat_prev,
        pos_curr=pos_curr,
        quat_curr=quat_curr,
        body_diag=body_diag,
        active_mask=active_mask,
    )
    return float(lambda_trans) * trans + float(lambda_rot) * rot


def relative_pose_metric_weighted(
    *,
    pos_prev: np.ndarray,
    quat_prev: np.ndarray,
    pos_curr: np.ndarray,
    quat_curr: np.ndarray,
    half_extents: np.ndarray,
    body_diag: np.ndarray,
    active_mask: np.ndarray,
    initial_total_volume: float,
    lambda_trans: float = 1.0,
    lambda_rot: float = DEFAULT_INTER_STEADY_METRIC_ROT_SCALE,
) -> dict[str, Any]:
    """Volume-weighted pose disturbance normalized by the initial active volume."""

    active = _active_mask_bool(active_mask)
    vols = box_volumes_from_half_extents(half_extents=half_extents)
    n = int(active.shape[0])
    if vols.shape != (n,):
        raise ValueError(f"volume shape {vols.shape} vs n_boxes={n}.")

    pose_per_box = relative_pose_metric_per_box(
        pos_prev=pos_prev,
        quat_prev=quat_prev,
        pos_curr=pos_curr,
        quat_curr=quat_curr,
        body_diag=body_diag,
        active_mask=active_mask,
        lambda_trans=lambda_trans,
        lambda_rot=lambda_rot,
    )
    trans_per_box, rot_per_box = relative_pose_metric_components_per_box(
        pos_prev=pos_prev,
        quat_prev=quat_prev,
        pos_curr=pos_curr,
        quat_curr=quat_curr,
        body_diag=body_diag,
        active_mask=active_mask,
    )

    if not np.any(active):
        zeros = np.zeros((n,), dtype=np.float64)
        return {
            "pose_term": 0.0,
            "pose_term_trans_only": 0.0,
            "pose_term_rot_only": 0.0,
            "active_volume_current": 0.0,
            "pose_term_per_box": zeros,
            "pose_term_trans_only_per_box": zeros,
            "pose_term_rot_only_per_box": zeros,
        }

    denom = max(float(initial_total_volume), _EPS)
    vol_frac = np.zeros((n,), dtype=np.float64)
    vol_frac[active] = vols[active] / denom
    pose_term_per_box = vol_frac * pose_per_box
    pose_term_trans_only_per_box = vol_frac * (float(lambda_trans) * trans_per_box)
    pose_term_rot_only_per_box = vol_frac * (float(lambda_rot) * rot_per_box)
    return {
        "pose_term": float(np.sum(pose_term_per_box)),
        "pose_term_trans_only": float(np.sum(pose_term_trans_only_per_box)),
        "pose_term_rot_only": float(np.sum(pose_term_rot_only_per_box)),
        "active_volume_current": float(np.sum(vols[active])),
        "pose_term_per_box": pose_term_per_box,
        "pose_term_trans_only_per_box": pose_term_trans_only_per_box,
        "pose_term_rot_only_per_box": pose_term_rot_only_per_box,
    }


def inter_steady_metric_new(
    *,
    pos_prev: np.ndarray,
    quat_prev: np.ndarray,
    pos_curr: np.ndarray,
    quat_curr: np.ndarray,
    half_extents: np.ndarray,
    body_diag: np.ndarray,
    active_mask: np.ndarray,
    initial_total_energy: float,
    initial_total_volume: float,
    alpha_energy: float = DEFAULT_INTER_STEADY_METRIC_ENERGY_SCALE,
    beta_pose: float = DEFAULT_INTER_STEADY_METRIC_POSE_SCALE,
    lambda_trans: float = DEFAULT_INTER_STEADY_METRIC_TRANS_SCALE,
    lambda_rot: float = DEFAULT_INTER_STEADY_METRIC_ROT_SCALE,
) -> dict[str, Any]:
    """Vertical inter-steady disturbance between two stable pile snapshots.

    v3 metric semantics are fixed-scale and scene-independent:

    ``metric_i = volume_i(scaled^3) * max(z_prev_i - z_curr_i, 0)(scaled)``.

    Only downward motion contributes to the energy term. Upward motion is not
    counted as unload damage.

    ``scaled_length = length_m * METRIC_LENGTH_SCALE``. With the current
    v3 setting, a 4cm edge stored as ``0.04`` is evaluated as ``0.4``.

    The legacy pose fields are returned as zero-valued compatibility
    diagnostics, but pose/rotation no longer contribute to the metric.
    """

    active = _active_mask_bool(active_mask)
    pos_prev_arr = np.asarray(pos_prev, dtype=np.float64)
    pos_curr_arr = np.asarray(pos_curr, dtype=np.float64)
    half = np.asarray(half_extents, dtype=np.float64).reshape(-1, 3)
    n = int(active.shape[0])
    if pos_prev_arr.shape != (n, 3) or pos_curr_arr.shape != (n, 3):
        raise ValueError(
            f"pos shapes expect ({n}, 3); got {pos_prev_arr.shape}, {pos_curr_arr.shape}."
        )
    if half.shape != (n, 3):
        raise ValueError(f"half_extents shape expects ({n}, 3); got {half.shape}.")
    vols_m3 = box_volumes_from_half_extents(half_extents=half)
    length_scale = float(DEFAULT_INTER_STEADY_METRIC_LENGTH_SCALE)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError(f"METRIC_LENGTH_SCALE must be positive and finite, got {length_scale}.")
    volume_scaled_per_box = vols_m3 * (length_scale ** 3)
    z_prev_m = pos_prev_arr[:, 2]
    z_curr_m = pos_curr_arr[:, 2]
    dz_m_per_box = z_curr_m - z_prev_m
    z_prev_scaled = z_prev_m * length_scale
    z_curr_scaled = z_curr_m * length_scale
    dz_scaled_per_box = dz_m_per_box * length_scale

    energy_prev_per_box = np.zeros_like(volume_scaled_per_box, dtype=np.float64)
    energy_curr_per_box = np.zeros_like(volume_scaled_per_box, dtype=np.float64)
    energy_prev_per_box[active] = volume_scaled_per_box[active] * z_prev_scaled[active]
    energy_curr_per_box[active] = volume_scaled_per_box[active] * z_curr_scaled[active]
    energy_prev = float(np.sum(energy_prev_per_box))
    energy_curr = float(np.sum(energy_curr_per_box))
    energy_delta_signed = float(energy_prev - energy_curr)
    energy_term_per_box = np.zeros_like(volume_scaled_per_box, dtype=np.float64)
    dz_down_scaled_per_box = np.maximum(-dz_scaled_per_box, 0.0)
    energy_term_per_box[active] = (
        volume_scaled_per_box[active] * dz_down_scaled_per_box[active]
    )
    energy_term = float(np.sum(energy_term_per_box))
    energy_delta_abs = float(energy_term)

    pose_term_per_box = np.zeros_like(energy_term_per_box, dtype=np.float64)
    pose_trans_per_box = np.zeros_like(energy_term_per_box, dtype=np.float64)
    pose_rot_per_box = np.zeros_like(energy_term_per_box, dtype=np.float64)
    metric_per_box = (
        float(alpha_energy) * energy_term_per_box
        + float(beta_pose) * pose_term_per_box
    )
    metric_total = float(np.sum(metric_per_box))
    return {
        "metric_total": float(metric_total),
        "energy_term": float(energy_term),
        "pose_term": 0.0,
        "energy_prev": float(energy_prev),
        "energy_curr": float(energy_curr),
        "energy_drop_raw": float(energy_delta_signed),
        "energy_delta_abs": float(energy_delta_abs),
        "pose_term_trans_only": 0.0,
        "pose_term_rot_only": 0.0,
        "active_volume_current": float(np.sum(vols_m3[active])),
        "metric_length_scale": float(length_scale),
        "metric_length_unit": "scaled",
        "metric_mode": "volume_scaled3_downward_dz_scaled",
        "_active_mask": active,
        "_metric_per_box": metric_per_box,
        "_energy_term_per_box": energy_term_per_box,
        "_pose_term_per_box": pose_term_per_box,
        "_pose_term_trans_only_per_box": pose_trans_per_box,
        "_pose_term_rot_only_per_box": pose_rot_per_box,
        "_height_delta_m_per_box": dz_m_per_box,
        "_volume_m3_per_box": vols_m3,
        "_height_delta_scaled_per_box": dz_scaled_per_box,
        "_height_down_delta_scaled_per_box": dz_down_scaled_per_box,
        "_volume_scaled_per_box": volume_scaled_per_box,
    }


class InterSteadyMetricTracker:
    """Accumulate inter-steady metrics over repeated unload steps."""

    def __init__(
        self,
        *,
        half_extents: np.ndarray,
        alpha_energy: float = DEFAULT_INTER_STEADY_METRIC_ENERGY_SCALE,
        beta_pose: float = DEFAULT_INTER_STEADY_METRIC_POSE_SCALE,
        lambda_trans: float = DEFAULT_INTER_STEADY_METRIC_TRANS_SCALE,
        lambda_rot: float = DEFAULT_INTER_STEADY_METRIC_ROT_SCALE,
        metric_step_deadband: float = DEFAULT_INTER_STEADY_METRIC_STEP_DEADBAND,
    ) -> None:
        self.half_extents = np.asarray(half_extents, dtype=np.float64).reshape(-1, 3)
        self.body_diag = body_diag_from_half_extents(half_extents=self.half_extents)
        self.box_volumes = box_volumes_from_half_extents(half_extents=self.half_extents)
        self.alpha_energy = float(alpha_energy)
        self.beta_pose = float(beta_pose)
        self.lambda_trans = float(lambda_trans)
        self.lambda_rot = float(lambda_rot)
        self.metric_step_deadband = max(0.0, float(metric_step_deadband))
        self.reset()

    def _clear_last_step_per_box(self) -> None:
        n = int(self.half_extents.shape[0])
        self.last_metric_per_box = np.zeros((n,), dtype=np.float64)
        self.last_metric_kept_mask = np.zeros((n,), dtype=bool)

    def reset(self) -> None:
        self.pos_prev: np.ndarray | None = None
        self.quat_prev: np.ndarray | None = None
        self.initial_total_energy: float | None = None
        self.initial_total_volume: float | None = None
        self.metric_total = 0.0
        self.energy_total = 0.0
        self.pose_total = 0.0
        self.metric_total_raw = 0.0
        self.energy_total_raw = 0.0
        self.pose_total_raw = 0.0
        self.steps: list[dict[str, Any]] = []
        self._clear_last_step_per_box()

    @property
    def has_baseline(self) -> bool:
        return self.pos_prev is not None and self.quat_prev is not None

    def set_baseline(
        self,
        *,
        pos: np.ndarray,
        quat: np.ndarray,
        active_mask: np.ndarray,
    ) -> None:
        pos_arr = np.asarray(pos, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)
        active = _active_mask_bool(active_mask)
        n = int(self.half_extents.shape[0])
        if pos_arr.shape != (n, 3):
            raise ValueError(f"pos shape expects ({n}, 3); got {pos_arr.shape}.")
        if quat_arr.shape != (n, 4):
            raise ValueError(f"quat shape expects ({n}, 4); got {quat_arr.shape}.")
        if active.shape != (n,):
            raise ValueError(f"active_mask shape expects ({n},); got {active.shape}.")

        self.pos_prev = pos_arr.copy()
        self.quat_prev = quat_arr.copy()
        self.initial_total_energy = potential_energy_proxy_sum(
            pos=pos_arr,
            half_extents=self.half_extents,
            active_mask=active,
        )
        self.initial_total_volume = float(np.sum(self.box_volumes[active]))
        self._clear_last_step_per_box()

    def set_baseline_from_mujoco(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        n_boxes: int,
        active_mask: np.ndarray,
    ) -> None:
        pos, quat = collect_box_poses_mujoco(model=model, data=data, n_boxes=int(n_boxes))
        self.set_baseline(pos=pos, quat=quat, active_mask=active_mask)

    def advance_reference(
        self,
        *,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> None:
        """Move the previous steady-state reference without accumulating metric."""

        if self.pos_prev is None or self.quat_prev is None:
            raise RuntimeError("InterSteadyMetricTracker baseline has not been set.")
        pos_arr = np.asarray(pos, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)
        n = int(self.half_extents.shape[0])
        if pos_arr.shape != (n, 3):
            raise ValueError(f"pos shape expects ({n}, 3); got {pos_arr.shape}.")
        if quat_arr.shape != (n, 4):
            raise ValueError(f"quat shape expects ({n}, 4); got {quat_arr.shape}.")
        self.pos_prev = pos_arr.copy()
        self.quat_prev = quat_arr.copy()
        self._clear_last_step_per_box()

    def advance_reference_from_mujoco(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        n_boxes: int,
    ) -> None:
        pos, quat = collect_box_poses_mujoco(model=model, data=data, n_boxes=int(n_boxes))
        self.advance_reference(pos=pos, quat=quat)

    def update(
        self,
        *,
        pos: np.ndarray,
        quat: np.ndarray,
        active_mask: np.ndarray,
    ) -> dict[str, Any]:
        if self.pos_prev is None or self.quat_prev is None:
            raise RuntimeError("InterSteadyMetricTracker baseline has not been set.")
        if self.initial_total_energy is None or self.initial_total_volume is None:
            raise RuntimeError("InterSteadyMetricTracker initial normalization is missing.")

        pos_curr = np.asarray(pos, dtype=np.float64)
        quat_curr = np.asarray(quat, dtype=np.float64)
        stats = inter_steady_metric_new(
            pos_prev=self.pos_prev,
            quat_prev=self.quat_prev,
            pos_curr=pos_curr,
            quat_curr=quat_curr,
            half_extents=self.half_extents,
            body_diag=self.body_diag,
            active_mask=active_mask,
            initial_total_energy=float(self.initial_total_energy),
            initial_total_volume=float(self.initial_total_volume),
            alpha_energy=self.alpha_energy,
            beta_pose=self.beta_pose,
            lambda_trans=self.lambda_trans,
            lambda_rot=self.lambda_rot,
        )
        active = np.asarray(stats.pop("_active_mask"), dtype=bool).reshape(-1)
        metric_per_box = np.asarray(stats.pop("_metric_per_box"), dtype=np.float64).reshape(-1)
        energy_per_box = np.asarray(stats.pop("_energy_term_per_box"), dtype=np.float64).reshape(-1)
        pose_per_box = np.asarray(stats.pop("_pose_term_per_box"), dtype=np.float64).reshape(-1)
        pose_trans_per_box = np.asarray(
            stats.pop("_pose_term_trans_only_per_box"),
            dtype=np.float64,
        ).reshape(-1)
        pose_rot_per_box = np.asarray(
            stats.pop("_pose_term_rot_only_per_box"),
            dtype=np.float64,
        ).reshape(-1)

        raw_metric_total = float(np.sum(metric_per_box[active]))
        raw_energy_term = float(np.sum(energy_per_box[active]))
        raw_pose_term = float(np.sum(pose_per_box[active]))
        deadband = float(self.metric_step_deadband)
        kept = active & (metric_per_box > deadband)
        deadbanded = active & ~kept
        deadbanded_count = int(np.count_nonzero(deadbanded))
        active_count = int(np.count_nonzero(active))
        kept_count = int(np.count_nonzero(kept))
        stats["metric_total"] = float(np.sum(metric_per_box[kept]))
        stats["energy_term"] = float(np.sum(energy_per_box[kept]))
        stats["pose_term"] = float(np.sum(pose_per_box[kept]))
        stats["pose_term_trans_only"] = float(np.sum(pose_trans_per_box[kept]))
        stats["pose_term_rot_only"] = float(np.sum(pose_rot_per_box[kept]))
        stats["metric_total_raw"] = raw_metric_total
        stats["energy_term_raw"] = raw_energy_term
        stats["pose_term_raw"] = raw_pose_term
        stats["metric_step_deadband"] = deadband
        stats["metric_deadband_mode"] = "per_box"
        stats["metric_deadbanded"] = 1.0 if deadbanded_count > 0 else 0.0
        stats["metric_deadbanded_box_count"] = float(deadbanded_count)
        stats["metric_active_box_count"] = float(active_count)
        stats["metric_kept_box_count"] = float(kept_count)
        stats["metric_max_box_raw"] = (
            0.0 if active_count == 0 else float(np.max(metric_per_box[active]))
        )
        self.last_metric_per_box = metric_per_box.copy()
        self.last_metric_kept_mask = kept.copy()
        self.metric_total += float(stats["metric_total"])
        self.energy_total += float(stats["energy_term"])
        self.pose_total += float(stats["pose_term"])
        self.metric_total_raw += raw_metric_total
        self.energy_total_raw += raw_energy_term
        self.pose_total_raw += raw_pose_term
        self.steps.append(dict(stats))
        self.pos_prev = pos_curr.copy()
        self.quat_prev = quat_curr.copy()
        return stats

    def update_from_mujoco(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        active_mask: np.ndarray,
    ) -> dict[str, Any]:
        pos, quat = collect_box_poses_mujoco(
            model=model,
            data=data,
            n_boxes=int(self.half_extents.shape[0]),
        )
        return self.update(pos=pos, quat=quat, active_mask=active_mask)

    def summary(self) -> dict[str, Any]:
        return {
            "metric_total": float(self.metric_total),
            "energy_total": float(self.energy_total),
            "pose_total": float(self.pose_total),
            "metric_total_raw": float(self.metric_total_raw),
            "energy_total_raw": float(self.energy_total_raw),
            "pose_total_raw": float(self.pose_total_raw),
            "metric_step_deadband": float(self.metric_step_deadband),
            "metric_deadbanded_steps": int(
                sum(int(float(step.get("metric_deadbanded", 0.0)) > 0.5) for step in self.steps)
            ),
            "metric_deadbanded_box_count": int(
                sum(int(float(step.get("metric_deadbanded_box_count", 0.0))) for step in self.steps)
            ),
            "metric_kept_box_count": int(
                sum(int(float(step.get("metric_kept_box_count", 0.0))) for step in self.steps)
            ),
            "step_count": int(len(self.steps)),
            "metric_steps": [float(step["metric_total"]) for step in self.steps],
            "metric_steps_raw": [
                float(step.get("metric_total_raw", step["metric_total"])) for step in self.steps
            ],
        }
