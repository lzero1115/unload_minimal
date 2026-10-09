from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytorch_kinematics as pk
import torch

_UTILS_ROOT = Path(__file__).resolve().parent
_TASK_ROOT = _UTILS_ROOT.parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
for _p in (_UTILS_ROOT, _CONFIG_ROOT, _TASK_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from robot_naming import robot_joint_name
from scene_utils import quat_wxyz_to_rot
from viewer_utils import (
    build_mesh_records,
    compute_fk_data,
    load_visual_meshes_from_urdf,
)


def load_robot_visual_model(
    *,
    urdf_path: str | Path,
    asset_roots: list[str | Path],
) -> tuple[Any, list[dict[str, Any]]]:
    urdf_path = Path(urdf_path)
    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()
    chain = pk.build_chain_from_urdf(urdf_text)
    link_meshes = load_visual_meshes_from_urdf(urdf_path, asset_roots=asset_roots)
    return chain, build_mesh_records(chain, link_meshes)


def mujoco_chain_q_tensor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    chain: Any,
    *,
    robot_index: int = 0,
) -> torch.Tensor:
    values: list[float] = []
    for name in chain.get_joint_parameter_names():
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            robot_joint_name(int(robot_index), str(name)),
        )
        if joint_id < 0:
            values.append(0.0)
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        values.append(float(data.qpos[qadr]))
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def robot_base_transform(
    *,
    base_pos: tuple[float, float, float],
    base_quat_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = quat_wxyz_to_rot(np.asarray(base_quat_wxyz, dtype=np.float64))
    tf[:3, 3] = np.asarray(base_pos, dtype=np.float64).reshape(3)
    return tf


def robot_link_transforms(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    chain: Any,
    base_pos: tuple[float, float, float],
    base_quat_wxyz: tuple[float, float, float, float],
    robot_index: int = 0,
) -> dict[str, np.ndarray]:
    link_transforms, _frame_names, _frames = compute_fk_data(
        chain,
        mujoco_chain_q_tensor(model, data, chain, robot_index=int(robot_index)),
    )
    base_tf = robot_base_transform(base_pos=base_pos, base_quat_wxyz=base_quat_wxyz)
    return {name: base_tf @ tf for name, tf in link_transforms.items()}
