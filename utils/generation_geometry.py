from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _quat_wxyz_to_xyzw(q):
    return np.asarray(q, dtype=np.float64).reshape(4)[[1, 2, 3, 0]]


def _quat_xyzw_to_wxyz(q):
    return np.asarray(q, dtype=np.float64).reshape(4)[[3, 0, 1, 2]]


def _box_vertices_xyzw(pos: np.ndarray, quat_xyzw: np.ndarray, half: np.ndarray) -> np.ndarray:
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
    rot = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    return local @ rot.T + np.asarray(pos, dtype=np.float64).reshape(3)


def _aabb_from_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    return np.min(v, axis=0), np.max(v, axis=0)


def _box_aabb_xyzw(pos, quat_xyzw, half):
    return _aabb_from_vertices(_box_vertices_xyzw(pos, quat_xyzw, half))


def _count_unsettled(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_boxes: int,
    lin_thr: float,
    ang_thr: float,
    active_mask: np.ndarray | None = None,
) -> tuple[int, float, float, np.ndarray]:
    unsettled = np.zeros(int(n_boxes), dtype=np.int32)
    max_lin = 0.0
    max_ang = 0.0
    if active_mask is None:
        active_mask = np.ones(int(n_boxes), dtype=bool)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    for i in range(int(n_boxes)):
        if i >= active_mask.shape[0] or not bool(active_mask[i]):
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
        if bid < 0:
            continue
        cvel = np.asarray(data.cvel[bid], dtype=np.float64)
        ang = float(np.linalg.norm(cvel[:3]))
        lin = float(np.linalg.norm(cvel[3:]))
        max_lin = max(max_lin, lin)
        max_ang = max(max_ang, ang)
        if lin > float(lin_thr) or ang > float(ang_thr):
            unsettled[i] = 1
    return int(np.count_nonzero(unsettled)), max_lin, max_ang, unsettled

