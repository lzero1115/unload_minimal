from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_MODULE_ROOT = Path(__file__).resolve().parent
_TASK_ROOT = _MODULE_ROOT.parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
_UTILS_ROOT = _TASK_ROOT / "utils"
for _p in (_MODULE_ROOT, _CONFIG_ROOT, _UTILS_ROOT, _TASK_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import phys_configs as pc
from scene_utils import quat_wxyz_to_rot
from sim_wrapper import SimWrapper

_TASK_ROOT = Path(__file__).resolve().parents[1]
_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_FACE_CORNER_INSET_M = 0.005
_FACE_CORNER_INSET_FRACTION = 0.20


def _empty_targets() -> dict[str, np.ndarray]:
    empty_f32 = np.zeros((0, 3), dtype=np.float32)
    return {
        "positions_w": empty_f32.copy(),
        "quats_wxyz_w": np.zeros((0, 4), dtype=np.float32),
        "contact_positions_w": empty_f32.copy(),
        "contact_quats_wxyz_w": np.zeros((0, 4), dtype=np.float32),
        "tip_positions_w": empty_f32.copy(),
        "face_normals_w": empty_f32.copy(),
        "box_indices": np.zeros((0,), dtype=np.int32),
        "face_axes": np.zeros((0,), dtype=np.int32),
        "face_signs": np.zeros((0,), dtype=np.int32),
        "roll_indices": np.zeros((0,), dtype=np.int32),
        "corner_pass_counts": np.zeros((0,), dtype=np.int32),
        "openness_scores": np.zeros((0,), dtype=np.int32),
        "face_areas": np.zeros((0,), dtype=np.float32),
    }


_LOCAL_FACE_SPECS = (
    (0, 1, np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    (0, -1, np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
    (1, 1, np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    (1, -1, np.array([0.0, -1.0, 0.0], dtype=np.float64)),
    (2, 1, np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    (2, -1, np.array([0.0, 0.0, -1.0], dtype=np.float64)),
)


def _face_sign_index(face_sign: int) -> int:
    return 1 if int(face_sign) > 0 else 0


def _ray_obb_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    rot: np.ndarray,
    half: np.ndarray,
) -> float | None:
    p_local = np.asarray(rot, dtype=np.float64).reshape(3, 3).T @ (
        np.asarray(origin, dtype=np.float64).reshape(3)
        - np.asarray(center, dtype=np.float64).reshape(3)
    )
    d_local = np.asarray(rot, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        direction,
        dtype=np.float64,
    ).reshape(3)
    h = np.asarray(half, dtype=np.float64).reshape(3)
    t_min = -np.inf
    t_max = np.inf
    eps = 1e-12

    for axis in range(3):
        if abs(float(d_local[axis])) < eps:
            if p_local[axis] < -h[axis] or p_local[axis] > h[axis]:
                return None
            continue
        t1 = (-h[axis] - p_local[axis]) / d_local[axis]
        t2 = (h[axis] - p_local[axis]) / d_local[axis]
        t_near = min(float(t1), float(t2))
        t_far = max(float(t1), float(t2))
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return None

    if t_max < 0.0:
        return None
    return max(float(t_min), 0.0)


def _box_joint_addrs(*, model: mujoco.MjModel, box_index: int) -> tuple[int, int]:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"box_{int(box_index)}_j",
    )
    if jid < 0:
        raise RuntimeError(f"MuJoCo model is missing joint box_{int(box_index)}_j.")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _box_body_id(*, model: mujoco.MjModel, box_index: int) -> int:
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"box_{int(box_index)}",
    )
    if bid < 0:
        raise RuntimeError(f"MuJoCo model is missing body box_{int(box_index)}.")
    return int(bid)


def _box_geom_id_for_index(*, model: mujoco.MjModel, box_index: int) -> int:
    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"box_{int(box_index)}_g",
    )
    if gid < 0:
        raise RuntimeError(f"MuJoCo model is missing geom box_{int(box_index)}_g.")
    return int(gid)


def _write_box_root_pose_vel(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_index: int,
    pos: np.ndarray,
    quat_wxyz: np.ndarray = _IDENTITY_QUAT_WXYZ,
    zero_vel: bool = True,
) -> None:
    qadr, dadr = _box_joint_addrs(model=model, box_index=int(box_index))
    data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    data.qpos[qadr + 3 : qadr + 7] = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    if bool(zero_vel):
        data.qvel[dadr : dadr + 6] = 0.0


def _disable_box_collision(*, model: mujoco.MjModel, box_index: int) -> None:
    gid = _box_geom_id_for_index(model=model, box_index=int(box_index))
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0


def _quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= np.finfo(np.float64).eps:
        return _IDENTITY_QUAT_WXYZ.copy()
    return quat / norm


def _zero_box_root_velocity(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_index: int,
) -> None:
    _qadr, dadr = _box_joint_addrs(model=model, box_index=int(box_index))
    data.qvel[dadr : dadr + 6] = 0.0


def _zero_active_box_root_velocities(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    active_mask: np.ndarray,
) -> None:
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    for i in np.flatnonzero(active):
        _zero_box_root_velocity(model=model, data=data, box_index=int(i))


def _compute_active_box_motion_status(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    unsettled = np.zeros((active.shape[0],), dtype=np.int32)
    max_lin = 0.0
    max_ang = 0.0
    for i in np.flatnonzero(active):
        bid = _box_body_id(model=model, box_index=int(i))
        cvel = np.asarray(data.cvel[bid], dtype=np.float64)
        ang = float(np.linalg.norm(cvel[:3]))
        lin = float(np.linalg.norm(cvel[3:]))
        max_lin = max(max_lin, lin)
        max_ang = max(max_ang, ang)
        if lin > float(pc.SETTLE_LINEAR_SPEED_MPS) or ang > float(pc.SETTLE_ANGULAR_SPEED_RAD):
            unsettled[int(i)] = 1
    return unsettled, max_lin, max_ang


def _collect_box_geometry(sim: SimWrapper) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(sim.n_boxes)
    centers = np.zeros((n, 3), dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    half_extents = np.asarray(sim.half_extents, dtype=np.float64).reshape(n, 3)
    for box_i in range(n):
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{box_i}")
        if bid < 0:
            raise RuntimeError(f"MuJoCo model is missing body box_{box_i}.")
        centers[box_i] = np.asarray(sim.data.xpos[bid], dtype=np.float64)
        rotations[box_i] = quat_wxyz_to_rot(np.asarray(sim.data.xquat[bid], dtype=np.float64))
    return centers, rotations, half_extents


def _collect_static_obstacles(sim: SimWrapper) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    obstacles: list[tuple[np.ndarray, np.ndarray]] = [
        (
            np.asarray(sim.table_pos, dtype=np.float64).reshape(3),
            np.asarray(sim.table_half_extents, dtype=np.float64).reshape(3),
        )
    ]
    for _wall_index, wall_pos, wall_half in sim.wall_defs:
        obstacles.append(
            (
                np.asarray(wall_pos, dtype=np.float64).reshape(3),
                np.asarray(wall_half, dtype=np.float64).reshape(3),
            )
        )
    return tuple(obstacles)


def _face_center_and_normal(
    *,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extent: np.ndarray,
    face_axis: int,
    face_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_normal = np.zeros((3,), dtype=np.float64)
    local_normal[int(face_axis)] = float(face_sign)
    normal_w = np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ local_normal
    normal_w = normal_w / max(float(np.linalg.norm(normal_w)), np.finfo(np.float64).eps)
    local_offset = np.zeros((3,), dtype=np.float64)
    local_offset[int(face_axis)] = float(face_sign) * float(half_extent[int(face_axis)])
    face_center = np.asarray(center, dtype=np.float64).reshape(3) + (
        np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ local_offset
    )
    return face_center, normal_w


def _face_corners(
    *,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extent: np.ndarray,
    face_axis: int,
    face_sign: int,
) -> np.ndarray:
    axis = int(face_axis)
    sign = int(face_sign)
    half = np.asarray(half_extent, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    axes = [idx for idx in range(3) if idx != axis]
    in_plane_half = {}
    for in_plane_axis in axes:
        axis_half = float(half[in_plane_axis])
        inset = min(float(_FACE_CORNER_INSET_M), max(0.0, axis_half * float(_FACE_CORNER_INSET_FRACTION)))
        in_plane_half[int(in_plane_axis)] = max(0.0, axis_half - inset)
    corners = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            local = np.zeros((3,), dtype=np.float64)
            local[axis] = float(sign) * half[axis]
            local[axes[0]] = float(s0) * in_plane_half[int(axes[0])]
            local[axes[1]] = float(s1) * in_plane_half[int(axes[1])]
            corners.append(np.asarray(center, dtype=np.float64).reshape(3) + rot @ local)
    return np.asarray(corners, dtype=np.float64)


def _face_area(*, half_extent: np.ndarray, face_axis: int) -> float:
    half = np.asarray(half_extent, dtype=np.float64).reshape(3)
    axes = [idx for idx in range(3) if idx != int(face_axis)]
    return float(4.0 * half[axes[0]] * half[axes[1]])


def _face_segment_blocked(
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    selected_box: int,
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
    active_mask: np.ndarray,
) -> bool:
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    for other_i in np.flatnonzero(active):
        if int(other_i) == int(selected_box):
            continue
        hit_distance = _ray_obb_distance(
            origin=origin,
            direction=direction,
            center=centers[int(other_i)],
            rot=rotations[int(other_i)],
            half=half_extents[int(other_i)],
        )
        if hit_distance is not None and float(hit_distance) <= float(max_distance):
            return True
    return False


def _static_segment_blocked(
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    obstacles: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> bool:
    rot = np.eye(3, dtype=np.float64)
    for center, half in obstacles:
        hit_distance = _ray_obb_distance(
            origin=origin,
            direction=direction,
            center=np.asarray(center, dtype=np.float64).reshape(3),
            rot=rot,
            half=np.asarray(half, dtype=np.float64).reshape(3),
        )
        if hit_distance is not None and float(hit_distance) <= float(max_distance):
            return True
    return False


def _proxy_face_candidates(
    *,
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
    active_mask: np.ndarray,
    ray_face_mask: np.ndarray,
    clearance: float,
    origin_margin: float,
    min_openness_score: int = 1,
    static_obstacles: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
    candidate_eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    n = int(centers.shape[0])
    active = np.asarray(active_mask, dtype=bool).reshape(n)
    eligible = (
        active.copy()
        if candidate_eligible_mask is None
        else np.asarray(candidate_eligible_mask, dtype=bool).reshape(n)
    )
    eligible &= active
    face_mask = np.asarray(ray_face_mask, dtype=bool).reshape(n, 3, 2)
    keep = np.zeros((n, 3, 2), dtype=bool)
    corner_pass_counts = np.zeros((n, 3, 2), dtype=np.int32)
    openness_scores = np.zeros((n, 3, 2), dtype=np.int32)
    face_areas = np.zeros((n, 3, 2), dtype=np.float32)
    tested = 0
    blocked = 0
    blocked_by_box = 0
    blocked_by_static = 0
    corner_tested = 0
    corner_blocked = 0
    corner_blocked_by_box = 0
    corner_blocked_by_static = 0
    failed_corner_threshold = 0
    clearance_dist = max(0.0, float(clearance))
    margin = max(0.0, float(origin_margin))
    min_score = max(1, min(5, int(min_openness_score)))
    required_corner_pass_count = max(0, int(min_score) - 1)

    for box_i in range(n):
        if not bool(eligible[box_i]):
            continue
        for face_axis, face_sign, _local_normal in _LOCAL_FACE_SPECS:
            sign_index = _face_sign_index(int(face_sign))
            if not bool(face_mask[box_i, int(face_axis), sign_index]):
                continue
            tested += 1
            face_center, normal_w = _face_center_and_normal(
                center=centers[box_i],
                rotation=rotations[box_i],
                half_extent=half_extents[box_i],
                face_axis=int(face_axis),
                face_sign=int(face_sign),
            )
            origin = face_center + normal_w * margin
            is_blocked = _face_segment_blocked(
                origin=origin,
                direction=normal_w,
                max_distance=clearance_dist,
                selected_box=box_i,
                centers=centers,
                rotations=rotations,
                half_extents=half_extents,
                active_mask=active,
            )
            if is_blocked:
                blocked += 1
                blocked_by_box += 1
                continue
            is_blocked = _static_segment_blocked(
                origin=origin,
                direction=normal_w,
                max_distance=clearance_dist,
                obstacles=static_obstacles,
            )
            if is_blocked:
                blocked += 1
                blocked_by_static += 1
                continue
            pass_count = 0
            corners = _face_corners(
                center=centers[box_i],
                rotation=rotations[box_i],
                half_extent=half_extents[box_i],
                face_axis=int(face_axis),
                face_sign=int(face_sign),
            )
            for corner_i, corner in enumerate(corners):
                corner_tested += 1
                corner_origin = np.asarray(corner, dtype=np.float64).reshape(3) + normal_w * margin
                corner_is_blocked = _face_segment_blocked(
                    origin=corner_origin,
                    direction=normal_w,
                    max_distance=clearance_dist,
                    selected_box=box_i,
                    centers=centers,
                    rotations=rotations,
                    half_extents=half_extents,
                    active_mask=active,
                )
                if corner_is_blocked:
                    corner_blocked += 1
                    corner_blocked_by_box += 1
                else:
                    corner_is_blocked = _static_segment_blocked(
                        origin=corner_origin,
                        direction=normal_w,
                        max_distance=clearance_dist,
                        obstacles=static_obstacles,
                    )
                    if corner_is_blocked:
                        corner_blocked += 1
                        corner_blocked_by_static += 1
                if corner_is_blocked:
                    remaining_corners = 4 - int(corner_i + 1)
                    if pass_count + remaining_corners < required_corner_pass_count:
                        break
                    continue
                pass_count += 1

            openness_score = 1 + int(pass_count)
            corner_pass_counts[box_i, int(face_axis), sign_index] = int(pass_count)
            openness_scores[box_i, int(face_axis), sign_index] = int(openness_score)
            face_areas[box_i, int(face_axis), sign_index] = float(
                _face_area(half_extent=half_extents[box_i], face_axis=int(face_axis))
            )
            if int(openness_score) < int(min_score):
                failed_corner_threshold += 1
                continue
            keep[box_i, int(face_axis), sign_index] = True

    box_mask = np.any(keep, axis=(1, 2)) & eligible
    center_pass_mask = openness_scores > 0
    corner_values = corner_pass_counts[center_pass_mask]
    corner_hist = (
        np.bincount(corner_values.reshape(-1), minlength=5)[:5]
        if corner_values.size > 0
        else np.zeros((5,), dtype=np.int64)
    )
    face_metadata = {
        "corner_pass_counts": corner_pass_counts,
        "openness_scores": openness_scores,
        "face_areas": face_areas,
    }
    debug = {
        "tested_face_count": int(tested),
        "blocked_face_count": int(blocked),
        "blocked_by_box_count": int(blocked_by_box),
        "blocked_by_static_count": int(blocked_by_static),
        "corner_tested_count": int(corner_tested),
        "corner_blocked_count": int(corner_blocked),
        "corner_blocked_by_box_count": int(corner_blocked_by_box),
        "corner_blocked_by_static_count": int(corner_blocked_by_static),
        "center_pass_face_count": int(np.count_nonzero(center_pass_mask)),
        "failed_corner_threshold_count": int(failed_corner_threshold),
        "min_openness_score": int(min_score),
        "required_corner_pass_count": int(required_corner_pass_count),
        "corner_pass_histogram": corner_hist.astype(np.int32),
        "kept_face_count": int(np.count_nonzero(keep)),
        "candidate_count": int(np.count_nonzero(box_mask)),
        "eligible_count": int(np.count_nonzero(eligible)),
        "clearance": float(clearance_dist),
        "origin_margin": float(margin),
        "static_obstacle_count": int(len(static_obstacles)),
    }
    return box_mask, keep, face_metadata, debug


def _generate_proxy_frame_targets(
    *,
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
    face_mask: np.ndarray,
    surface_clearance: float,
    face_metadata: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    box_indices: list[int] = []
    face_axes: list[int] = []
    face_signs: list[int] = []
    roll_indices: list[int] = []
    corner_pass_counts: list[int] = []
    openness_scores: list[int] = []
    face_areas: list[float] = []
    quats_wxyz = []

    n = int(centers.shape[0])
    mask = np.asarray(face_mask, dtype=bool).reshape(n, 3, 2)
    meta = {} if face_metadata is None else dict(face_metadata)
    meta_corner = np.asarray(
        meta.get("corner_pass_counts", np.zeros((n, 3, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(n, 3, 2)
    meta_open = np.asarray(
        meta.get("openness_scores", np.zeros((n, 3, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(n, 3, 2)
    meta_area = np.asarray(
        meta.get("face_areas", np.zeros((n, 3, 2), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(n, 3, 2)
    for box_i in range(n):
        for face_axis, face_sign, _local_normal in _LOCAL_FACE_SPECS:
            sign_index = _face_sign_index(int(face_sign))
            if not bool(mask[box_i, int(face_axis), sign_index]):
                continue
            face_center, normal_w = _face_center_and_normal(
                center=centers[box_i],
                rotation=rotations[box_i],
                half_extent=half_extents[box_i],
                face_axis=int(face_axis),
                face_sign=int(face_sign),
            )
            positions.append(face_center + normal_w * float(surface_clearance))
            normals.append(normal_w)
            box_indices.append(int(box_i))
            face_axes.append(int(face_axis))
            face_signs.append(int(face_sign))
            roll_indices.append(0)
            fallback_area = _face_area(half_extent=half_extents[box_i], face_axis=int(face_axis))
            corner_pass = int(meta_corner[box_i, int(face_axis), sign_index])
            openness = int(meta_open[box_i, int(face_axis), sign_index])
            area = float(meta_area[box_i, int(face_axis), sign_index])
            corner_pass_counts.append(corner_pass)
            openness_scores.append(openness if openness > 0 else 1)
            face_areas.append(area if area > 0.0 else float(fallback_area))
            quats_wxyz.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))

    if not positions:
        return _empty_targets()

    pos = np.asarray(positions, dtype=np.float32)
    quat = np.asarray(quats_wxyz, dtype=np.float32)
    return {
        "positions_w": pos,
        "quats_wxyz_w": quat,
        "contact_positions_w": pos.copy(),
        "contact_quats_wxyz_w": quat.copy(),
        "tip_positions_w": pos.copy(),
        "face_normals_w": np.asarray(normals, dtype=np.float32),
        "box_indices": np.asarray(box_indices, dtype=np.int32),
        "face_axes": np.asarray(face_axes, dtype=np.int32),
        "face_signs": np.asarray(face_signs, dtype=np.int32),
        "roll_indices": np.asarray(roll_indices, dtype=np.int32),
        "corner_pass_counts": np.asarray(corner_pass_counts, dtype=np.int32),
        "openness_scores": np.asarray(openness_scores, dtype=np.int32),
        "face_areas": np.asarray(face_areas, dtype=np.float32),
    }

