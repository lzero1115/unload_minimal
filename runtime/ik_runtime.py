from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from curobo.types.math import Pose

_MODULE_ROOT = Path(__file__).resolve().parent
_TASK_ROOT = _MODULE_ROOT.parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
_UTILS_ROOT = _TASK_ROOT / "utils"
for _path in (_MODULE_ROOT, _CONFIG_ROOT, _UTILS_ROOT, _TASK_ROOT):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

import motion_configs as mc
from .ik_candidates import (
    get_curobo_tensor_args,
    load_local_curobo_robot_config,
    make_ik_solver,
    query_to_curobo_world,
    suction_target_poses_from_query,
)


DEFAULT_IK_BATCH_SIZE = 256
DEFAULT_ROLL_ANGLES = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)


def _synchronize_device(tensor_args: Any) -> None:
    device = torch.device(tensor_args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _prepare_query_batch(
    query: dict[str, Any],
    *,
    roll_angles: tuple[float, ...],
) -> dict[str, np.ndarray | int]:
    targets = dict(query["frame_targets"])
    local_frame_count = int(
        np.asarray(targets["box_indices"], dtype=np.int32).reshape(-1).shape[0]
    )
    poses = suction_target_poses_from_query(query, roll_angles=roll_angles)
    parent_frames = np.asarray(
        poses["parent_target_indices"],
        dtype=np.int32,
    ).reshape(-1)
    pose_rolls = np.asarray(poses["roll_indices"], dtype=np.int32).reshape(-1)
    roll_count = int(len(roll_angles))

    positions_b: list[np.ndarray] = []
    quats_b: list[np.ndarray] = []
    positions_w: list[np.ndarray] = []
    quats_w: list[np.ndarray] = []
    local_indices: list[int] = []
    box_indices: list[int] = []
    face_axes: list[int] = []
    face_signs: list[int] = []
    for frame_i in range(local_frame_count):
        rows = np.flatnonzero(parent_frames == int(frame_i))
        if rows.size == 0:
            continue
        if int(rows.size) != roll_count:
            raise RuntimeError(
                f"Frame {frame_i} produced {rows.size} roll targets; "
                f"expected {roll_count}."
            )
        rows = rows[np.argsort(pose_rolls[rows])]
        positions_b.append(np.asarray(poses["positions_b"][rows], dtype=np.float32))
        quats_b.append(np.asarray(poses["quats_wxyz_b"][rows], dtype=np.float32))
        positions_w.append(np.asarray(poses["positions_w"][rows], dtype=np.float32))
        quats_w.append(np.asarray(poses["quats_wxyz_w"][rows], dtype=np.float32))
        local_indices.append(int(frame_i))
        box_indices.append(int(poses["parent_box_indices"][rows[0]]))
        face_axes.append(int(poses["parent_face_axes"][rows[0]]))
        face_signs.append(int(poses["parent_face_signs"][rows[0]]))

    if not positions_b:
        return {
            "problem_count": 0,
            "local_frame_count": int(local_frame_count),
            "positions_b": np.zeros((0, roll_count, 3), dtype=np.float32),
            "quats_b": np.zeros((0, roll_count, 4), dtype=np.float32),
            "positions_w": np.zeros((0, roll_count, 3), dtype=np.float32),
            "quats_w": np.zeros((0, roll_count, 4), dtype=np.float32),
            "local_indices": np.zeros((0,), dtype=np.int32),
            "box_indices": np.zeros((0,), dtype=np.int32),
            "face_axes": np.zeros((0,), dtype=np.int32),
            "face_signs": np.zeros((0,), dtype=np.int32),
        }
    return {
        "problem_count": int(len(positions_b)),
        "local_frame_count": int(local_frame_count),
        "positions_b": np.ascontiguousarray(np.stack(positions_b), dtype=np.float32),
        "quats_b": np.ascontiguousarray(np.stack(quats_b), dtype=np.float32),
        "positions_w": np.ascontiguousarray(np.stack(positions_w), dtype=np.float32),
        "quats_w": np.ascontiguousarray(np.stack(quats_w), dtype=np.float32),
        "local_indices": np.asarray(local_indices, dtype=np.int32),
        "box_indices": np.asarray(box_indices, dtype=np.int32),
        "face_axes": np.asarray(face_axes, dtype=np.int32),
        "face_signs": np.asarray(face_signs, dtype=np.int32),
    }


class PersistentCuroboIKCandidateSolver:
    """Persistent cuRobo solver for per-step box candidate filtering."""

    def __init__(
        self,
        *,
        batch_size: int = DEFAULT_IK_BATCH_SIZE,
        roll_angles: tuple[float, ...] = DEFAULT_ROLL_ANGLES,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}.")
        if not roll_angles:
            raise ValueError("At least one suction roll angle is required.")
        if bool(mc.CUROBO_USE_CUDA_GRAPH):
            raise RuntimeError(
                "Persistent variable-size IK batches require CUROBO_USE_CUDA_GRAPH=False."
            )
        self.batch_size = int(batch_size)
        self.roll_angles = tuple(float(value) for value in roll_angles)
        self.tensor_args = get_curobo_tensor_args()
        self.robot = load_local_curobo_robot_config()
        self.solver = None
        self.obb_cache_capacity = 0

    def solve_query(self, query: dict[str, Any]) -> dict[str, Any]:
        total_start = time.perf_counter()
        world, obstacle_count = query_to_curobo_world(query)
        batch = _prepare_query_batch(query, roll_angles=self.roll_angles)
        problem_count = int(batch["problem_count"])
        local_frame_count = int(batch["local_frame_count"])
        n_boxes = int(query["n_boxes"])
        frame_success = np.zeros((local_frame_count,), dtype=bool)
        box_success_mask = np.zeros((n_boxes,), dtype=bool)
        selected_roll_indices = np.full((local_frame_count,), -1, dtype=np.int32)
        selected_positions_w = np.full((local_frame_count, 3), np.nan, dtype=np.float32)
        selected_quats_w = np.full((local_frame_count, 4), np.nan, dtype=np.float32)

        solver_init_s = 0.0
        world_update_s = 0.0
        ik_solve_s = 0.0
        solver_batch_count = 0
        statuses: list[str] = []
        if problem_count > 0 and self.solver is None:
            init_start = time.perf_counter()
            self.solver = make_ik_solver(
                robot_config=self.robot["robot_config"],
                tensor_args=self.tensor_args,
                world_cfg=[world] * self.batch_size,
            )
            _synchronize_device(self.tensor_args)
            solver_init_s = float(time.perf_counter() - init_start)
            self.obb_cache_capacity = int(obstacle_count)
        elif problem_count > 0 and int(obstacle_count) > self.obb_cache_capacity:
            update_start = time.perf_counter()
            self.solver.world_coll_checker.load_batch_collision_model(
                [world] * self.batch_size
            )
            _synchronize_device(self.tensor_args)
            world_update_s += float(time.perf_counter() - update_start)
            self.obb_cache_capacity = int(obstacle_count)

        for chunk_start in range(0, problem_count, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, problem_count)
            chunk_count = int(chunk_end - chunk_start)
            update_start = time.perf_counter()
            for env_i in range(chunk_count):
                self.solver.world_coll_checker.load_collision_model(
                    world,
                    env_idx=int(env_i),
                    fix_cache_reference=True,
                )
            _synchronize_device(self.tensor_args)
            world_update_s += float(time.perf_counter() - update_start)

            goal = Pose(
                position=self.tensor_args.to_device(batch["positions_b"][chunk_start:chunk_end]),
                quaternion=self.tensor_args.to_device(batch["quats_b"][chunk_start:chunk_end]),
            )
            _synchronize_device(self.tensor_args)
            solve_start = time.perf_counter()
            solve_result = self.solver.solve_batch_env_goalset(goal)
            _synchronize_device(self.tensor_args)
            ik_solve_s += float(time.perf_counter() - solve_start)
            solver_batch_count += 1
            statuses.append(str(getattr(solve_result, "status", "ok")))

            success = (
                solve_result.success.detach()
                .cpu()
                .numpy()
                .astype(bool)
                .reshape(chunk_count, -1)[:, 0]
            )
            selected_rolls = np.zeros((chunk_count,), dtype=np.int32)
            if solve_result.goalset_index is not None:
                selected_rolls = (
                    solve_result.goalset_index.detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                    .reshape(chunk_count, -1)[:, 0]
                )
            selected_rolls = np.clip(selected_rolls, 0, len(self.roll_angles) - 1)
            selected_rolls[~success] = -1

            for chunk_i in range(chunk_count):
                problem_i = int(chunk_start + chunk_i)
                local_i = int(batch["local_indices"][problem_i])
                box_i = int(batch["box_indices"][problem_i])
                if not bool(success[chunk_i]):
                    continue
                roll_i = int(selected_rolls[chunk_i])
                frame_success[local_i] = True
                box_success_mask[box_i] = True
                selected_roll_indices[local_i] = roll_i
                selected_positions_w[local_i] = batch["positions_w"][problem_i, roll_i]
                selected_quats_w[local_i] = batch["quats_w"][problem_i, roll_i]

        proxy_mask = np.asarray(query["proxy_action_mask"], dtype=bool).reshape(n_boxes)
        if np.any(box_success_mask & ~proxy_mask):
            raise RuntimeError("IK candidate mask escaped the proxy action mask.")
        box_success_mask &= proxy_mask
        return {
            "box_success_mask": box_success_mask,
            "frame_success": frame_success,
            "selected_roll_indices": selected_roll_indices,
            "selected_positions_w": selected_positions_w,
            "selected_quats_wxyz_w": selected_quats_w,
            "ik_problem_count": int(problem_count),
            "goal_pose_count": int(problem_count * len(self.roll_angles)),
            "solver_batch_count": int(solver_batch_count),
            "world_obstacle_count": int(obstacle_count),
            "status": "no_targets" if not statuses else ",".join(dict.fromkeys(statuses)),
            "timing": {
                "solver_init_s": float(solver_init_s),
                "world_update_s": float(world_update_s),
                "ik_solve_s": float(ik_solve_s),
                "total_s": float(time.perf_counter() - total_start),
            },
        }


__all__ = [
    "DEFAULT_IK_BATCH_SIZE",
    "DEFAULT_ROLL_ANGLES",
    "PersistentCuroboIKCandidateSolver",
]
