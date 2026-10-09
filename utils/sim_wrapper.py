from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import mujoco
import numpy as np

_UTILS_ROOT = Path(__file__).resolve().parent
_TASK_ROOT = _UTILS_ROOT.parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
for _p in (_UTILS_ROOT, _CONFIG_ROOT, _TASK_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import motion_configs as mc
import phys_configs as pc
import scene_configs as sc
from robot_naming import robot_actuator_name, robot_joint_name
from scene_utils import build_mjcf, merge_robot_into_mjcf, wall_definitions


def _npz_required(npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in npz.files:
        raise KeyError(f"Scene npz is missing required field: {name}")
    return np.asarray(npz[name])


def _metadata_from_npz(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in npz.files:
        return {}
    raw = np.asarray(npz["metadata_json"])
    text = str(raw.item() if raw.shape == () else raw)
    try:
        return dict(json.loads(text))
    except json.JSONDecodeError:
        return {"metadata_json_parse_failed": text}


def _optional_friction(
    npz: np.lib.npyio.NpzFile,
    name: str,
) -> tuple[float, float, float] | None:
    if name not in npz.files:
        return None
    return _friction_tuple(name, npz[name])


def _friction_tuple(name: str, values: Any) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values, got {values}")
    return tuple(float(v) for v in values)


def _resolve_scene_friction(
    snapshot: Mapping[str, Any],
    *,
    physics_config,
    friction_overrides: Mapping[str, tuple[float, float, float]] | None,
) -> tuple[
    dict[str, tuple[float, float, float]],
    dict[str, str],
]:
    overrides = {} if friction_overrides is None else dict(friction_overrides)
    unknown_overrides = set(overrides) - {"table", "wall", "box"}
    if unknown_overrides:
        raise ValueError(
            "friction_overrides keys must be table, wall, or box; got "
            f"{sorted(unknown_overrides)}"
        )

    resolved: dict[str, tuple[float, float, float]] = {}
    sources: dict[str, str] = {}
    for name in ("table", "wall", "box"):
        values = overrides.get(name)
        source = "override"
        if values is None:
            values = snapshot[f"{name}_friction"]
            source = "npz"
        if values is None:
            values = getattr(physics_config, f"{name.upper()}_FRICTION")
            source = "physics_config"
        resolved[name] = _friction_tuple(f"resolved {name}_friction", values)
        sources[name] = source
    return resolved, sources


def load_scene_npz(path: Path, *, scene_config=sc) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scene npz does not exist: {path}")

    with np.load(path, allow_pickle=False) as npz:
        box_centers = _npz_required(npz, "box_centers").astype(np.float64)
        box_quats_xyzw = _npz_required(npz, "box_quats_xyzw").astype(np.float64)
        box_half_extents = _npz_required(npz, "box_half_extents").astype(np.float64)

        if box_centers.ndim != 2 or box_centers.shape[1] != 3:
            raise ValueError(f"box_centers must have shape (N, 3), got {box_centers.shape}")
        if box_quats_xyzw.shape != (box_centers.shape[0], 4):
            raise ValueError(
                f"box_quats_xyzw must have shape ({box_centers.shape[0]}, 4), "
                f"got {box_quats_xyzw.shape}"
            )
        if box_half_extents.shape != (box_centers.shape[0], 3):
            raise ValueError(
                f"box_half_extents must have shape ({box_centers.shape[0]}, 3), "
                f"got {box_half_extents.shape}"
            )

        dims = (
            np.asarray(npz["partition_dims"], dtype=np.float64).reshape(3)
            if "partition_dims" in npz.files
            else np.asarray(scene_config.PARTITION_DIMS, dtype=np.float64)
        )
        table_pos = (
            np.asarray(npz["table_pos"], dtype=np.float64).reshape(3)
            if "table_pos" in npz.files
            else np.asarray(scene_config.TABLE_POS, dtype=np.float64)
        )
        table_half_extents = (
            np.asarray(npz["table_half_extents"], dtype=np.float64).reshape(3)
            if "table_half_extents" in npz.files
            else np.asarray(scene_config.TABLE_HALF_EXTENTS, dtype=np.float64)
        )

        walls_enabled = (
            bool(np.asarray(npz["walls_enabled"]).item())
            if "walls_enabled" in npz.files
            else bool(scene_config.WALLS_ENABLED)
        )
        walls_removed = (
            set(int(v) for v in np.asarray(npz["walls_removed"], dtype=np.int32).reshape(-1))
            if "walls_removed" in npz.files
            else set()
        )

        if walls_enabled and {"wall_ids", "wall_pos", "wall_half_extents"}.issubset(npz.files):
            wall_ids = np.asarray(npz["wall_ids"], dtype=np.int32).reshape(-1)
            wall_pos = np.asarray(npz["wall_pos"], dtype=np.float64).reshape(-1, 3)
            wall_half_extents = np.asarray(npz["wall_half_extents"], dtype=np.float64).reshape(-1, 3)
            if not (wall_ids.shape[0] == wall_pos.shape[0] == wall_half_extents.shape[0]):
                raise ValueError("wall_ids, wall_pos, and wall_half_extents lengths must match.")
            wall_defs = [
                (
                    int(wall_ids[i]),
                    tuple(float(v) for v in wall_pos[i]),
                    tuple(float(v) for v in wall_half_extents[i]),
                )
                for i in range(int(wall_ids.shape[0]))
            ]
        elif walls_enabled:
            wall_defs = wall_definitions(
                tuple(float(v) for v in dims),
                wall_scale=float(scene_config.WALL_SCALE),
                wall_thickness=float(scene_config.WALL_THICKNESS),
                walls_removed=walls_removed,
                center_xy_at_origin=bool(
                    getattr(scene_config, "PARTITION_CENTER_XY_AT_ORIGIN", False)
                ),
            )
        else:
            wall_defs = []

        metadata = _metadata_from_npz(npz)
        table_friction = _optional_friction(npz, "table_friction")
        wall_friction = _optional_friction(npz, "wall_friction")
        box_friction = _optional_friction(npz, "box_friction")
        box_density = float(np.asarray(npz["box_density"]).item()) if "box_density" in npz.files else None
        if box_density is not None and (not np.isfinite(box_density) or box_density <= 0):
            raise ValueError("Saved box_density must be finite and positive")
        physics_config_version = (
            int(np.asarray(npz["physics_config_version"]).reshape(()).item())
            if "physics_config_version" in npz.files
            else None
        )

    return {
        "path": path,
        "metadata": metadata,
        "box_centers": box_centers,
        "box_quats_xyzw": box_quats_xyzw,
        "box_half_extents": box_half_extents,
        "dims": tuple(float(v) for v in dims),
        "table_pos": tuple(float(v) for v in table_pos),
        "table_half_extents": tuple(float(v) for v in table_half_extents),
        "walls_enabled": walls_enabled,
        "walls_removed": walls_removed,
        "wall_defs": wall_defs,
        "table_friction": table_friction,
        "wall_friction": wall_friction,
        "box_friction": box_friction,
        "box_density": box_density,
        "physics_config_version": physics_config_version,
    }


def box_motion_status(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_boxes: int,
    physics_config=pc,
) -> tuple[np.ndarray, float, float]:
    unsettled = np.zeros(int(n_boxes), dtype=np.int32)
    max_lin = 0.0
    max_ang = 0.0
    for i in range(int(n_boxes)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
        if bid < 0:
            continue
        cvel = np.asarray(data.cvel[bid], dtype=np.float64)
        ang = float(np.linalg.norm(cvel[:3]))
        lin = float(np.linalg.norm(cvel[3:]))
        max_lin = max(max_lin, lin)
        max_ang = max(max_ang, ang)
        if (
            lin > float(physics_config.SETTLE_LINEAR_SPEED_MPS)
            or ang > float(physics_config.SETTLE_ANGULAR_SPEED_RAD)
        ):
            unsettled[i] = 1
    return unsettled, max_lin, max_ang


class SimWrapper:
    """MuJoCo scene state and default hold-home robot control.

    Viewer code should call this object instead of writing MuJoCo qpos/ctrl
    directly. The robot remains part of the MuJoCo model with the configured
    collision masks; this wrapper owns initialization, reset and arm control.
    """

    def __init__(
        self,
        scene_npz: str | Path,
        *,
        max_frames: int = 0,
        enable_box_mocaps: bool = False,
        scene_config=sc,
        physics_config=pc,
        motion_config=mc,
        robot_indices: tuple[int, ...] = (0,),
        friction_overrides: Mapping[str, tuple[float, float, float]] | None = None,
    ) -> None:
        self.scene_config = scene_config
        self.physics_config = physics_config
        self.motion_config = motion_config
        self.robot_indices = tuple(int(v) for v in robot_indices)
        if any(v < 0 for v in self.robot_indices):
            raise ValueError(f"robot_indices must be non-negative, got {self.robot_indices}.")
        if len(set(self.robot_indices)) != len(self.robot_indices):
            raise ValueError(f"robot_indices must not contain duplicates, got {self.robot_indices}.")
        self.enable_box_mocaps = bool(enable_box_mocaps)
        self.snapshot = load_scene_npz(Path(scene_npz), scene_config=scene_config)
        if self.snapshot["box_density"] is not None:
            values = {name: getattr(physics_config, name) for name in dir(physics_config) if name.isupper()}
            values["BOX_DENSITY"] = self.snapshot["box_density"]
            physics_config = SimpleNamespace(**values)
            self.physics_config = physics_config
        friction, self.friction_source = _resolve_scene_friction(
            self.snapshot,
            physics_config=physics_config,
            friction_overrides=friction_overrides,
        )
        self.table_friction = friction["table"]
        self.wall_friction = friction["wall"]
        self.box_friction = friction["box"]
        self.max_frames = int(max_frames)
        self.frame_dt = 1.0 / float(physics_config.FPS)
        self.sim_substeps = max(1, int(physics_config.SIM_SUBSTEPS))
        self.arm_home_q = tuple(float(v) for v in motion_config.ARM_HOME_Q)
        self.running = False
        self.frame_count = 0
        self.sim_time = 0.0

        self.dims = tuple(float(v) for v in self.snapshot["dims"])
        self.table_pos = tuple(float(v) for v in self.snapshot["table_pos"])
        self.table_half_extents = tuple(float(v) for v in self.snapshot["table_half_extents"])
        self.wall_defs = list(self.snapshot["wall_defs"])
        self.centers = np.asarray(self.snapshot["box_centers"], dtype=np.float64)
        self.quats_xyzw = np.asarray(self.snapshot["box_quats_xyzw"], dtype=np.float64)
        self.half_extents = np.asarray(self.snapshot["box_half_extents"], dtype=np.float64)
        self.n_boxes = int(self.centers.shape[0])
        self.last_unsettled = np.zeros(self.n_boxes, dtype=np.int32)
        self.last_max_lin = 0.0
        self.last_max_ang = 0.0

        self._build_model()

    def _build_model(self) -> None:
        mjcf = build_mjcf(
            centers=self.centers,
            quats_xyzw=self.quats_xyzw,
            half_extents=self.half_extents,
            walls_removed=set(self.snapshot["walls_removed"]),
            scene_config=self.scene_config,
            physics_config=self.physics_config,
            no_mocap=not self.enable_box_mocaps,
            box_drop_height=0.0,
            table_pos=self.table_pos,
            table_half_extents=self.table_half_extents,
            wall_defs=self.wall_defs,
            table_friction=self.table_friction,
            wall_friction=self.wall_friction,
            box_friction=self.box_friction,
        )
        for robot_index in self.robot_indices:
            mjcf = merge_robot_into_mjcf(
                mjcf,
                scene_config=self.scene_config,
                motion_config=self.motion_config,
                physics_config=self.physics_config,
                enable_gravcomp=True,
                robot_index=int(robot_index),
            )
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.set_all_arm_q(self.arm_home_q)
        self.canonicalize_dynamic_state(reset_time=True)
        self._initial_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        self._initial_qvel = np.asarray(self.data.qvel, dtype=np.float64).copy()

    def set_arm_q(self, q: tuple[float, ...], *, robot_index: int = 0) -> None:
        if len(q) != len(self.motion_config.ARM_JOINT_NAMES):
            raise ValueError("ARM_HOME_Q and ARM_JOINT_NAMES must have the same length.")

        for joint_name, joint_q in zip(self.motion_config.ARM_JOINT_NAMES, q):
            mj_joint_name = robot_joint_name(int(robot_index), str(joint_name))
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mj_joint_name)
            if joint_id < 0:
                raise RuntimeError(f"MuJoCo model is missing robot joint: {mj_joint_name}")
            qadr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qadr] = float(joint_q)

            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                robot_actuator_name(int(robot_index), str(joint_name)),
            )
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = float(joint_q)

    def set_arm_ctrl(self, q: tuple[float, ...], *, robot_index: int = 0) -> None:
        for joint_name, joint_q in zip(self.motion_config.ARM_JOINT_NAMES, q):
            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                robot_actuator_name(int(robot_index), str(joint_name)),
            )
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = float(joint_q)

    def set_all_arm_q(self, q: tuple[float, ...]) -> None:
        for robot_index in self.robot_indices:
            self.set_arm_q(q, robot_index=int(robot_index))

    def set_all_arm_ctrl(self, q: tuple[float, ...]) -> None:
        for robot_index in self.robot_indices:
            self.set_arm_ctrl(q, robot_index=int(robot_index))

    def reset(self) -> None:
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.running = False
        self.frame_count = 0
        self.sim_time = 0.0
        self.canonicalize_dynamic_state(reset_time=True)
        self.last_unsettled = np.zeros(self.n_boxes, dtype=np.int32)
        self.last_max_lin = 0.0
        self.last_max_ang = 0.0

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def canonicalize_dynamic_state(self, *, reset_time: bool = False) -> None:
        """Keep the current geometry but clear dynamic solver/history residue."""

        data = self.data
        if reset_time:
            data.time = 0.0
        else:
            data.time = float(self.sim_time)

        data.qvel[:] = 0.0
        if data.qacc.shape:
            data.qacc[:] = 0.0
        if data.qacc_warmstart.shape:
            data.qacc_warmstart[:] = 0.0
        if data.qfrc_applied.shape:
            data.qfrc_applied[:] = 0.0
        if data.xfrc_applied.shape:
            data.xfrc_applied[:] = 0.0
        if data.act.shape:
            data.act[:] = 0.0
        self.set_all_arm_ctrl(self.arm_home_q)
        mujoco.mj_forward(self.model, data)
        if data.qacc.shape:
            data.qacc[:] = 0.0
        if data.qacc_warmstart.shape:
            data.qacc_warmstart[:] = 0.0

    def step_frame(self) -> None:
        if not self.running:
            return
        if self.max_frames > 0 and self.frame_count >= self.max_frames:
            self.running = False
            return

        self.set_all_arm_ctrl(self.arm_home_q)
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)
        self.frame_count += 1
        self.sim_time += self.frame_dt
        self.last_unsettled, self.last_max_lin, self.last_max_ang = box_motion_status(
            model=self.model,
            data=self.data,
            n_boxes=self.n_boxes,
            physics_config=self.physics_config,
        )
