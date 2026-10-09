from __future__ import annotations

import math
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

from scene_utils import quat_wxyz_to_rot


DEFAULT_RAY_GRID_STEP_M = 0.02
DEFAULT_MIN_RAY_HITS = 8


def _grid_axis_centers(lo: float, hi: float, step: float) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    step = float(step)
    if step <= 0.0 or not math.isfinite(step):
        raise ValueError(f"grid_step must be finite and > 0, got {step!r}")
    if hi - lo < 1e-12:
        return np.array([(lo + hi) * 0.5], dtype=np.float64)
    n = max(1, int(math.ceil((hi - lo) / step)))
    width = (hi - lo) / n
    return lo + (np.arange(n, dtype=np.float64) + 0.5) * width


def _box_vertices(center: np.ndarray, rot: np.ndarray, half: np.ndarray) -> np.ndarray:
    h = np.asarray(half, dtype=np.float64).reshape(3)
    local = np.array(
        [
            [-h[0], -h[1], -h[2]],
            [h[0], -h[1], -h[2]],
            [h[0], h[1], -h[2]],
            [-h[0], h[1], -h[2]],
            [-h[0], -h[1], h[2]],
            [h[0], -h[1], h[2]],
            [h[0], h[1], h[2]],
            [-h[0], h[1], h[2]],
        ],
        dtype=np.float64,
    )
    return local @ np.asarray(rot, dtype=np.float64).reshape(3, 3).T + np.asarray(
        center,
        dtype=np.float64,
    ).reshape(3)


def _ray_obb_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    rot: np.ndarray,
    half: np.ndarray,
) -> tuple[float, int, int] | None:
    p = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    h = np.asarray(half, dtype=np.float64).reshape(3)

    p_local = r.T @ (p - c)
    d_local = r.T @ d
    t_min = -np.inf
    t_max = np.inf
    near_axis = -1
    near_sign = 0
    far_axis = -1
    far_sign = 0
    eps = 1e-12

    for axis in range(3):
        if abs(float(d_local[axis])) < eps:
            if p_local[axis] < -h[axis] or p_local[axis] > h[axis]:
                return None
            continue
        t1 = (-h[axis] - p_local[axis]) / d_local[axis]
        t2 = (h[axis] - p_local[axis]) / d_local[axis]
        if float(t1) <= float(t2):
            t_near = float(t1)
            t_far = float(t2)
            entry_sign = -1
            exit_sign = 1
        else:
            t_near = float(t2)
            t_far = float(t1)
            entry_sign = 1
            exit_sign = -1
        if t_near > t_min:
            t_min = t_near
            near_axis = int(axis)
            near_sign = int(entry_sign)
        if t_far < t_max:
            t_max = t_far
            far_axis = int(axis)
            far_sign = int(exit_sign)
        if t_min > t_max:
            return None

    if t_max < 0.0:
        return None
    if t_min >= 0.0 and near_axis >= 0:
        return float(t_min), int(near_axis), int(near_sign)
    if far_axis >= 0:
        return max(float(t_max), 0.0), int(far_axis), int(far_sign)
    return max(float(t_min), 0.0), 0, 1


def _first_hit_box(
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[int, int, int]:
    best_i = -1
    best_dist = np.inf
    best_axis = -1
    best_sign = 0
    for i in range(int(centers.shape[0])):
        if not bool(active_mask[i]):
            continue
        hit = _ray_obb_distance(
            origin,
            direction,
            centers[i],
            rotations[i],
            half_extents[i],
        )
        if hit is None:
            continue
        dist, face_axis, face_sign = hit
        if dist < best_dist:
            best_dist = float(dist)
            best_i = int(i)
            best_axis = int(face_axis)
            best_sign = int(face_sign)
    return best_i, best_axis, best_sign


def _face_sign_index(face_sign: int) -> int:
    return 1 if int(face_sign) > 0 else 0


def _side_ray_spec(side_wall: int, mn: np.ndarray, mx: np.ndarray, margin: float) -> tuple[int, float, np.ndarray]:
    wall = int(side_wall)
    if wall == 0:
        return 0, float(mn[0] - margin), np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if wall == 1:
        return 0, float(mx[0] + margin), np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    if wall == 2:
        return 1, float(mn[1] - margin), np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if wall == 3:
        return 1, float(mx[1] + margin), np.array([0.0, -1.0, 0.0], dtype=np.float64)
    raise ValueError(f"side_wall must be in 0..3, got {side_wall}")


def _side_wall_indices(side_wall: int) -> tuple[int, ...]:
    wall = int(side_wall)
    if wall < 0:
        return (0, 1, 2, 3)
    if wall in (0, 1, 2, 3):
        return (wall,)
    raise ValueError(f"side_wall must be -1 or in 0..3, got {side_wall}")


def _normalize_side_walls(
    side_wall: int,
    side_walls: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if side_walls is None:
        return _side_wall_indices(int(side_wall))
    walls = tuple(sorted(set(int(v) for v in side_walls)))
    bad = [int(v) for v in walls if int(v) not in (0, 1, 2, 3)]
    if bad:
        raise ValueError(f"side_walls must contain only 0..3, got {bad}")
    return walls


def compute_box_candidate_mask_rays(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_boxes: int,
    half_extents: np.ndarray,
    side_wall: int = -1,
    side_walls: tuple[int, ...] | list[int] | None = None,
    active_mask: np.ndarray | None = None,
    candidate_eligible_mask: np.ndarray | None = None,
    grid_step: float = DEFAULT_RAY_GRID_STEP_M,
    plane_margin: float = 0.05,
    min_ray_hits: int = DEFAULT_MIN_RAY_HITS,
    return_debug: bool = False,
    assume_forwarded: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Box-level ray first-hit candidate mask using top and open-side directions.

    This intentionally ignores table, walls, and robot. It only answers whether a
    box is on the visible outer shell of the current active box pile. If
    ``candidate_eligible_mask`` is provided, all active boxes still occlude rays,
    but only eligible boxes can be returned as candidates. If side_walls is
    provided, those wall directions are used; otherwise a negative side_wall uses
    all four horizontal side directions and sums side hit counts.

    A box is a candidate when it is the first active box hit by at least
    ``min_ray_hits`` rays across the top and side ray sets.
    """

    n = int(n_boxes)
    half = np.asarray(half_extents, dtype=np.float64).reshape(n, 3)
    active = (
        np.ones((n,), dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool).reshape(-1)
    )
    if active.shape[0] != n:
        raise ValueError("active_mask length must match n_boxes")
    eligible = (
        active.copy()
        if candidate_eligible_mask is None
        else np.asarray(candidate_eligible_mask, dtype=bool).reshape(-1)
    )
    if eligible.shape[0] != n:
        raise ValueError("candidate_eligible_mask length must match n_boxes")
    eligible &= active

    if int(np.count_nonzero(active)) == 0:
        empty = np.zeros((n,), dtype=np.uint8)
        return (empty, {}) if return_debug else empty

    if not bool(assume_forwarded):
        mujoco.mj_forward(model, data)

    centers = np.zeros((n, 3), dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    mn = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    mx = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for i in range(n):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
        if bid < 0:
            raise RuntimeError(f"MuJoCo model is missing body box_{i}.")
        centers[i] = np.asarray(data.xpos[bid], dtype=np.float64)
        rotations[i] = quat_wxyz_to_rot(np.asarray(data.xquat[bid], dtype=np.float64))
        if bool(active[i]):
            verts = _box_vertices(centers[i], rotations[i], half[i])
            mn = np.minimum(mn, verts.min(axis=0))
            mx = np.maximum(mx, verts.max(axis=0))

    step = float(grid_step)
    margin = float(plane_margin)
    xs = _grid_axis_centers(float(mn[0]), float(mx[0]), step)
    ys = _grid_axis_centers(float(mn[1]), float(mx[1]), step)
    zs = _grid_axis_centers(float(mn[2]), float(mx[2]), step)

    top_count = np.zeros((n,), dtype=np.int32)
    top_face_hit_count = np.zeros((n, 3, 2), dtype=np.int32)
    z0 = float(mx[2] + margin)
    d_top = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for x in xs:
        for y in ys:
            hit, face_axis, face_sign = _first_hit_box(
                origin=np.array([float(x), float(y), z0], dtype=np.float64),
                direction=d_top,
                centers=centers,
                rotations=rotations,
                half_extents=half,
                active_mask=active,
            )
            if hit >= 0:
                top_count[hit] += 1
                if face_axis >= 0:
                    top_face_hit_count[hit, face_axis, _face_sign_index(face_sign)] += 1

    side_walls = _normalize_side_walls(int(side_wall), side_walls)
    side_count = np.zeros((n,), dtype=np.int32)
    side_face_hit_count = np.zeros((n, 3, 2), dtype=np.int32)
    side_counts_by_wall: dict[int, np.ndarray] = {}
    face_hit_counts_by_wall: dict[int, np.ndarray] = {}
    for wall in side_walls:
        wall_count = np.zeros((n,), dtype=np.int32)
        wall_face_count = np.zeros((n, 3, 2), dtype=np.int32)
        side_axis, side_coord, d_side = _side_ray_spec(int(wall), mn, mx, margin)
        if side_axis == 0:
            for y in ys:
                for z in zs:
                    hit, face_axis, face_sign = _first_hit_box(
                        origin=np.array([side_coord, float(y), float(z)], dtype=np.float64),
                        direction=d_side,
                        centers=centers,
                        rotations=rotations,
                        half_extents=half,
                        active_mask=active,
                    )
                    if hit >= 0:
                        wall_count[hit] += 1
                        if face_axis >= 0:
                            wall_face_count[hit, face_axis, _face_sign_index(face_sign)] += 1
        else:
            for x in xs:
                for z in zs:
                    hit, face_axis, face_sign = _first_hit_box(
                        origin=np.array([float(x), side_coord, float(z)], dtype=np.float64),
                        direction=d_side,
                        centers=centers,
                        rotations=rotations,
                        half_extents=half,
                        active_mask=active,
                    )
                    if hit >= 0:
                        wall_count[hit] += 1
                        if face_axis >= 0:
                            wall_face_count[hit, face_axis, _face_sign_index(face_sign)] += 1
        side_counts_by_wall[int(wall)] = wall_count
        face_hit_counts_by_wall[int(wall)] = wall_face_count.copy()
        side_count += wall_count
        side_face_hit_count += wall_face_count

    min_hits = max(1, int(min_ray_hits))
    hit_count = top_count + side_count
    face_hit_count = top_face_hit_count + side_face_hit_count
    candidate = hit_count >= min_hits
    candidate = (candidate & eligible).astype(np.uint8)
    used_active_fallback = False
    eligible_has_ray_hit = bool(np.any(hit_count[eligible] > 0))
    active_has_ray_hit = bool(np.any(hit_count[active] > 0))
    if (
        int(np.count_nonzero(candidate)) == 0
        and int(np.count_nonzero(eligible)) > 0
        and (eligible_has_ray_hit or not active_has_ray_hit)
    ):
        candidate = eligible.astype(np.uint8)
        used_active_fallback = True
    if used_active_fallback:
        face_candidate_mask = np.zeros((n, 3, 2), dtype=bool)
        face_candidate_mask[candidate.astype(bool), :, :] = True
    else:
        face_candidate_mask = (face_hit_count > 0) & candidate.astype(bool).reshape(n, 1, 1)
        fallback_boxes = candidate.astype(bool) & (face_candidate_mask.sum(axis=(1, 2)) == 0)
        if np.any(fallback_boxes):
            face_candidate_mask[fallback_boxes, :, :] = True

    if not return_debug:
        return candidate
    debug = {
        "grid_step": float(step),
        "plane_margin": float(margin),
        "side_wall": int(side_wall),
        "side_walls": np.asarray(side_walls, dtype=np.int32),
        "ray_grid_shape": np.array([len(xs), len(ys), len(zs)], dtype=np.int32),
        "top_hit_count": top_count,
        "side_hit_count": side_count,
        "side_hit_counts_by_wall": {
            int(wall): counts.copy() for wall, counts in side_counts_by_wall.items()
        },
        "top_face_hit_count": top_face_hit_count,
        "side_face_hit_count": side_face_hit_count,
        "face_hit_count": face_hit_count,
        "face_hit_counts_by_wall": {
            int(wall): counts.copy() for wall, counts in face_hit_counts_by_wall.items()
        },
        "face_candidate_mask": face_candidate_mask,
        "hit_count": hit_count,
        "min_ray_hits": int(min_hits),
        "candidate_count": int(np.count_nonzero(candidate)),
        "eligible_count": int(np.count_nonzero(eligible)),
        "face_candidate_count": int(np.count_nonzero(face_candidate_mask)),
        "eligible_has_ray_hit": bool(eligible_has_ray_hit),
        "active_has_ray_hit": bool(active_has_ray_hit),
        "used_active_fallback": bool(used_active_fallback),
    }
    return candidate, debug
