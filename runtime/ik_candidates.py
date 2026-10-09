from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

_MODULE_ROOT = Path(__file__).resolve().parent
_TASK_ROOT = _MODULE_ROOT.parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
_UTILS_ROOT = _TASK_ROOT / "utils"
for _p in (_MODULE_ROOT, _CONFIG_ROOT, _UTILS_ROOT, _TASK_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import motion_configs as mc
import scene_configs as sc
from scene_utils import quat_wxyz_to_rot


LOCAL_ROBOT_CONFIG_NAME = "franka_suction_new.yml"


def get_curobo_tensor_args() -> TensorDeviceType:
    if torch.cuda.is_available():
        return TensorDeviceType()
    return TensorDeviceType(device=torch.device("cpu"))


def _resolve_local_config_path(path_str: str) -> Path:
    candidates = [
        Path(sc.ROBOT_ROOT_PATH) / "configs" / path_str,
        Path(sc.ROBOT_ROOT_PATH) / path_str,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(Path(sc.ROBOT_ROOT_PATH).resolve()) and resolved.is_file():
            return resolved
    tried = "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(f"Could not resolve local robot config path: {path_str}\nTried:\n{tried}")


def _resolve_local_asset_path(path_str: str) -> Path:
    candidates = [
        Path(sc.ROBOT_ASSET_ROOT_PATH) / path_str,
        Path(sc.ROBOT_ROOT_PATH) / path_str,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(Path(sc.ROBOT_ROOT_PATH).resolve()) and resolved.exists():
            return resolved
    tried = "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(f"Could not resolve local robot asset path: {path_str}\nTried:\n{tried}")


def load_local_curobo_robot_config(config_name: str = LOCAL_ROBOT_CONFIG_NAME) -> dict[str, Any]:
    config_path = _resolve_local_config_path(str(config_name))
    robot_config = copy.deepcopy(load_yaml(str(config_path)))
    kin = robot_config["robot_cfg"]["kinematics"]

    urdf_path = _resolve_local_asset_path(str(kin["urdf_path"]))
    asset_root_path = _resolve_local_asset_path(str(kin["asset_root_path"]))
    collision_spheres = kin.get("collision_spheres")
    if isinstance(collision_spheres, str):
        sphere_cfg = load_yaml(str(_resolve_local_config_path(collision_spheres)))
        kin["collision_spheres"] = sphere_cfg["collision_spheres"]

    kin["urdf_path"] = str(urdf_path)
    kin["asset_root_path"] = str(asset_root_path)
    return {
        "config_path": str(config_path),
        "robot_config": robot_config,
        "urdf_path": str(urdf_path),
        "asset_root_path": str(asset_root_path),
        "base_link": kin["base_link"],
        "ee_link": kin["ee_link"],
    }


def _base_from_world_rotation(query: dict[str, Any]) -> np.ndarray:
    return quat_wxyz_to_rot(np.asarray(query["robot_base_quat_wxyz"], dtype=np.float64)).T


def world_to_base_position(query: dict[str, Any], pos_w: np.ndarray) -> np.ndarray:
    base_pos = np.asarray(query["robot_base_pos_w"], dtype=np.float64).reshape(3)
    return _base_from_world_rotation(query) @ (
        np.asarray(pos_w, dtype=np.float64).reshape(3) - base_pos
    )


def world_to_base_quat_wxyz(query: dict[str, Any], quat_xyzw_w: np.ndarray) -> np.ndarray:
    rot_bw = _base_from_world_rotation(query)
    rot_wo = Rotation.from_quat(np.asarray(quat_xyzw_w, dtype=np.float64).reshape(4)).as_matrix()
    quat_xyzw_b = Rotation.from_matrix(rot_bw @ rot_wo).as_quat()
    return quat_xyzw_b[[3, 0, 1, 2]]


def _add_cuboid(
    cuboids: dict[str, dict[str, list[float]]],
    *,
    query: dict[str, Any],
    name: str,
    center_w: np.ndarray,
    half_extents: np.ndarray,
    quat_wxyz_w: np.ndarray | None = None,
) -> None:
    if quat_wxyz_w is None:
        quat_xyzw_w = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        q = np.asarray(quat_wxyz_w, dtype=np.float64).reshape(4)
        quat_xyzw_w = q[[1, 2, 3, 0]]
    pos_b = world_to_base_position(query, center_w)
    quat_wxyz_b = world_to_base_quat_wxyz(query, quat_xyzw_w)
    half = np.asarray(half_extents, dtype=np.float64).reshape(3)
    cuboids[str(name)] = {
        "dims": [float(2.0 * half[0]), float(2.0 * half[1]), float(2.0 * half[2])],
        "pose": [
            float(pos_b[0]),
            float(pos_b[1]),
            float(pos_b[2]),
            float(quat_wxyz_b[0]),
            float(quat_wxyz_b[1]),
            float(quat_wxyz_b[2]),
            float(quat_wxyz_b[3]),
        ],
    }


def query_to_curobo_world(query: dict[str, Any]) -> tuple[WorldConfig, int]:
    cuboids: dict[str, dict[str, list[float]]] = {}
    _add_cuboid(
        cuboids,
        query=query,
        name="table",
        center_w=np.asarray(query["table_pos_w"], dtype=np.float64),
        half_extents=np.asarray(query["table_half_extents"], dtype=np.float64),
    )
    for wall_i, pos, half in tuple(query.get("wall_defs", ())):
        _add_cuboid(
            cuboids,
            query=query,
            name=f"wall_{int(wall_i)}",
            center_w=np.asarray(pos, dtype=np.float64),
            half_extents=np.asarray(half, dtype=np.float64),
        )

    active = np.asarray(query["active_mask"], dtype=bool).reshape(-1)
    centers = np.asarray(query["box_centers_w"], dtype=np.float64).reshape(-1, 3)
    quats = np.asarray(query["box_quats_wxyz"], dtype=np.float64).reshape(-1, 4)
    half_extents = np.asarray(query["box_half_extents"], dtype=np.float64).reshape(-1, 3)
    for box_i in range(int(centers.shape[0])):
        if not bool(active[box_i]):
            continue
        _add_cuboid(
            cuboids,
            query=query,
            name=f"box_{box_i}",
            center_w=centers[box_i],
            half_extents=half_extents[box_i],
            quat_wxyz_w=quats[box_i],
        )
    return WorldConfig.from_dict({"cuboid": cuboids}), len(cuboids)


def make_ik_solver(
    *,
    robot_config: dict[str, Any],
    tensor_args: TensorDeviceType,
    world_cfg: WorldConfig | list[WorldConfig],
) -> IKSolver:
    cfg = IKSolverConfig.load_from_robot_config(
        robot_config,
        world_cfg,
        rotation_threshold=float(mc.CUROBO_IK_ROTATION_THRESHOLD),
        position_threshold=float(mc.CUROBO_IK_POSITION_THRESHOLD),
        num_seeds=int(mc.CUROBO_IK_NUM_SEEDS),
        self_collision_check=bool(mc.CUROBO_SELF_COLLISION_CHECK),
        self_collision_opt=bool(mc.CUROBO_SELF_COLLISION_OPT),
        tensor_args=tensor_args,
        use_cuda_graph=bool(mc.CUROBO_USE_CUDA_GRAPH),
    )
    return IKSolver(cfg)


def _synchronize_device(tensor_args: TensorDeviceType) -> None:
    device = torch.device(tensor_args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def suction_target_poses_from_query(
    query: dict[str, Any],
    *,
    roll_angles: tuple[float, ...] = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi),
) -> dict[str, np.ndarray]:
    if not roll_angles:
        raise ValueError("At least one suction roll angle is required.")
    targets = dict(query["frame_targets"])
    tip_positions_w = np.asarray(targets["tip_positions_w"], dtype=np.float64).reshape(-1, 3)
    face_normals_w = np.asarray(targets["face_normals_w"], dtype=np.float64).reshape(-1, 3)
    box_indices = np.asarray(targets["box_indices"], dtype=np.int32).reshape(-1)
    face_axes = np.asarray(targets["face_axes"], dtype=np.int32).reshape(-1)
    face_signs = np.asarray(targets["face_signs"], dtype=np.int32).reshape(-1)
    box_quats_wxyz = np.asarray(
        query["box_quats_wxyz"], dtype=np.float64
    ).reshape(-1, 4)

    positions_b: list[np.ndarray] = []
    quats_wxyz_b: list[np.ndarray] = []
    positions_w: list[np.ndarray] = []
    quats_wxyz_w: list[np.ndarray] = []
    parent_target_indices: list[int] = []
    parent_box_indices: list[int] = []
    parent_face_axes: list[int] = []
    parent_face_signs: list[int] = []
    roll_indices: list[int] = []

    for target_i, (tip_w, normal_w) in enumerate(zip(tip_positions_w, face_normals_w, strict=True)):
        normal = np.asarray(normal_w, dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= np.finfo(np.float64).eps:
            continue
        normal /= normal_norm
        box_index = int(box_indices[target_i])
        face_axis = int(face_axes[target_i])
        face_sign = int(face_signs[target_i])
        if box_index < 0 or box_index >= box_quats_wxyz.shape[0]:
            raise ValueError(
                f"Frame target {target_i} references invalid box index {box_index}."
            )
        if face_axis not in (0, 1, 2):
            raise ValueError(
                f"Frame target {target_i} has invalid face axis {face_axis}."
            )
        if face_sign not in (-1, 1):
            raise ValueError(
                f"Frame target {target_i} has invalid face sign {face_sign}."
            )
        box_rotation = quat_wxyz_to_rot(box_quats_wxyz[box_index])
        box_normal = float(face_sign) * box_rotation[:, face_axis]
        if float(np.dot(normal, box_normal)) < 0.999:
            raise ValueError(
                f"Frame target {target_i} face normal disagrees with box_{box_index} "
                f"axis={face_axis} sign={face_sign}."
            )
        normal = box_normal
        approach_axis = -normal
        tangent_axes = tuple(axis for axis in range(3) if axis != face_axis)
        # Tie the full suction frame directly to the cuboid principal axes.
        x_seed = np.asarray(box_rotation[:, tangent_axes[0]], dtype=np.float64)
        x_seed /= max(float(np.linalg.norm(x_seed)), np.finfo(np.float64).eps)
        y_seed = np.cross(approach_axis, x_seed)
        y_seed /= max(float(np.linalg.norm(y_seed)), np.finfo(np.float64).eps)

        for roll_i, roll_angle in enumerate(tuple(float(v) for v in roll_angles)):
            roll_rot = Rotation.from_rotvec(float(roll_angle) * approach_axis)
            x_axis = roll_rot.apply(x_seed)
            y_axis = roll_rot.apply(y_seed)
            ee_rot_w = np.column_stack((x_axis, y_axis, approach_axis))
            quat_xyzw_w = Rotation.from_matrix(ee_rot_w).as_quat()
            positions_w.append(np.asarray(tip_w, dtype=np.float64).reshape(3))
            quats_wxyz_w.append(quat_xyzw_w[[3, 0, 1, 2]])
            positions_b.append(world_to_base_position(query, tip_w))
            quats_wxyz_b.append(world_to_base_quat_wxyz(query, quat_xyzw_w))
            parent_target_indices.append(int(target_i))
            parent_box_indices.append(int(box_indices[target_i]))
            parent_face_axes.append(int(face_axes[target_i]))
            parent_face_signs.append(int(face_signs[target_i]))
            roll_indices.append(int(roll_i))

    if not positions_b:
        return {
            "positions_b": np.zeros((0, 3), dtype=np.float32),
            "quats_wxyz_b": np.zeros((0, 4), dtype=np.float32),
            "positions_w": np.zeros((0, 3), dtype=np.float32),
            "quats_wxyz_w": np.zeros((0, 4), dtype=np.float32),
            "parent_target_indices": np.zeros((0,), dtype=np.int32),
            "parent_box_indices": np.zeros((0,), dtype=np.int32),
            "parent_face_axes": np.zeros((0,), dtype=np.int32),
            "parent_face_signs": np.zeros((0,), dtype=np.int32),
            "roll_indices": np.zeros((0,), dtype=np.int32),
        }
    return {
        "positions_b": np.asarray(positions_b, dtype=np.float32),
        "quats_wxyz_b": np.asarray(quats_wxyz_b, dtype=np.float32),
        "positions_w": np.asarray(positions_w, dtype=np.float32),
        "quats_wxyz_w": np.asarray(quats_wxyz_w, dtype=np.float32),
        "parent_target_indices": np.asarray(parent_target_indices, dtype=np.int32),
        "parent_box_indices": np.asarray(parent_box_indices, dtype=np.int32),
        "parent_face_axes": np.asarray(parent_face_axes, dtype=np.int32),
        "parent_face_signs": np.asarray(parent_face_signs, dtype=np.int32),
        "roll_indices": np.asarray(roll_indices, dtype=np.int32),
    }
