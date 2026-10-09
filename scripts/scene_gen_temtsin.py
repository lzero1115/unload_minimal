"""Generate one exact-count box pile with Temtsin-style mean placement.

Boxes are released one at a time above the current pile's mean XY position.
There are no physical bin walls: a four-sided virtual XY boundary removes
out-of-range boxes, and removed slots are replenished until exactly ``--nb``
stable boxes remain. The scene can be inspected in Polyscope or generated
headlessly and is saved in the project's regular scene NPZ format.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


_TASK_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_ROOT = Path(__file__).resolve().parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
_UTILS_ROOT = _TASK_ROOT / "utils"
for _path in (_SCRIPT_ROOT, _CONFIG_ROOT, _UTILS_ROOT, _TASK_ROOT):
    value = str(_path)
    if value not in sys.path:
        sys.path.insert(0, value)

import phys_configs as pc
import scene_configs as sc
import generation_configs as gc
from generation_geometry import (
    _box_aabb_xyzw,
    _count_unsettled,
    _quat_wxyz_to_xyzw,
    _quat_xyzw_to_wxyz,
)
from scene_utils import (
    COLOR_BOX_FILTERED_OUT,
    COLOR_BOX_INSTANT_SETTLED,
    COLOR_BOX_INSTANT_UNSETTLED,
    COLOR_TABLE,
    CUBE_FACES,
    axis_aligned_box_vertices,
    box_vertices_world,
    build_mjcf,
)
from settle_utils import ComDisplacementSettleTracker


STAGE_SETTLE = "settle"
STAGE_AUDIT = "audit"
STAGE_DONE = "done"
STAGE_FAILED = "failed"

SIZE_MODE_FIVE_LENGTH = "five-length"
SIZE_MODE_CURRENT = "current"
SIZE_MODE_TEMTSIN_CD = "temtsin-cd"
SIZE_MODE_TEMTSIN_UDL = "temtsin-udl"
SIZE_MODES = (
    SIZE_MODE_FIVE_LENGTH,
    SIZE_MODE_CURRENT,
    SIZE_MODE_TEMTSIN_CD,
    SIZE_MODE_TEMTSIN_UDL,
)

FIVE_LENGTH_CROSS_SECTION = np.array(gc.FIVE_LENGTH_CROSS_SECTION, dtype=np.float64)
FIVE_LENGTH_VALUES = np.array(gc.FIVE_LENGTH_VALUES, dtype=np.float64)
TEMTSIN_CD_EDGES = np.array(gc.TEMTSIN_CD_EDGES, dtype=np.float64)
TEMTSIN_UDL_LENGTHS = gc.TEMTSIN_UDL_UNIT * np.arange(1, 11, dtype=np.float64)


def _physics_config(*, box_density: float, sliding_mu: float) -> SimpleNamespace:
    if not np.isfinite(box_density) or box_density <= 0.0:
        raise ValueError("--box-density must be finite and positive")
    sliding_mu = float(sliding_mu)
    if not np.isfinite(sliding_mu) or sliding_mu < 0.0:
        raise ValueError(f"--sliding-mu must be finite and non-negative, got {sliding_mu}.")
    values = {
        name: getattr(pc, name)
        for name in dir(pc)
        if name.isupper()
    }
    values["BOX_DENSITY"] = float(box_density)
    for prefix in ("TABLE", "WALL", "BOX"):
        values[f"{prefix}_MU"] = sliding_mu
        values[f"{prefix}_FRICTION"] = (
            sliding_mu,
            float(values[f"{prefix}_MU_TORSIONAL"]),
            float(values[f"{prefix}_MU_ROLLING"]),
        )
    return SimpleNamespace(**values)


def _quat_angle_xyzw(q0: np.ndarray, q1: np.ndarray) -> float:
    lhs = np.asarray(q0, dtype=np.float64).reshape(4)
    rhs = np.asarray(q1, dtype=np.float64).reshape(4)
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= np.finfo(np.float64).eps or rhs_norm <= np.finfo(np.float64).eps:
        return 0.0
    dot = abs(float(np.dot(lhs / lhs_norm, rhs / rhs_norm)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


class TemtsinSceneGenSim:
    """One-at-a-time mean-pile generation with an exact final box count."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.target_n_boxes = int(args.nb)
        if self.target_n_boxes < 1:
            raise ValueError(f"--nb must be >= 1, got {self.target_n_boxes}.")

        self.dims = tuple(float(value) for value in args.dims)
        if any(value <= 0.0 for value in self.dims):
            raise ValueError(f"--dims components must be positive, got {self.dims}.")
        self.filter_scale = float(args.filter_scale)
        if self.filter_scale <= 0.0:
            raise ValueError(f"--filter-scale must be positive, got {self.filter_scale}.")
        self.release_clearance = float(args.release_clearance)
        if self.release_clearance < 0.0:
            raise ValueError(
                f"--release-clearance must be non-negative, got {self.release_clearance}."
            )

        self.seed = int(args.seed)
        self.rng = np.random.default_rng(self.seed)
        self.size_mode = str(args.size_mode)
        if self.size_mode not in SIZE_MODES:
            raise ValueError(f"Unknown size mode: {self.size_mode!r}.")
        self.edge_candidates = tuple(float(value) for value in args.edge_candidates)
        if not self.edge_candidates or any(value <= 0.0 for value in self.edge_candidates):
            raise ValueError(
                f"--edge-candidates must contain positive values, got {self.edge_candidates}."
            )

        self.frame_dt = 1.0 / float(pc.FPS)
        self.sim_substeps = max(1, int(pc.SIM_SUBSTEPS))
        self.lin_thr = float(pc.SETTLE_LINEAR_SPEED_MPS)
        self.ang_thr = float(pc.SETTLE_ANGULAR_SPEED_RAD)
        self.settle_check_interval = int(pc.SETTLE_CHECK_INTERVAL)
        self.settle_steps_required = int(pc.SETTLE_STEPS)
        self.enable_com_settle = bool(
            getattr(args, "enable_com_settle", pc.ENABLE_COM_SETTLE)
        )
        self.com_settle_max_displacement = float(
            getattr(
                args,
                "com_settle_max_displacement",
                pc.COM_SETTLE_MAX_DISPLACEMENT,
            )
        )
        self.com_settle_check_interval = int(
            getattr(args, "com_settle_check_interval", pc.COM_SETTLE_CHECK_INTERVAL)
        )
        self.com_settle_required_checks = int(
            getattr(args, "com_settle_required_checks", pc.COM_SETTLE_REQUIRED_CHECKS)
        )
        self.com_settle_tracker = ComDisplacementSettleTracker(
            enabled=self.enable_com_settle,
            max_displacement=self.com_settle_max_displacement,
            check_interval=self.com_settle_check_interval,
            required_checks=self.com_settle_required_checks,
        )
        self.last_settle_criterion = ""
        self.max_frames_per_drop = int(args.max_frames_per_drop)
        self.max_spawn_attempts = int(args.max_spawn_attempts)
        self.audit_frames_required = int(args.audit_frames)
        self.audit_max_pos_drift = float(args.audit_max_pos_drift)
        self.audit_max_rot_drift = float(args.audit_max_rot_drift)
        if self.max_frames_per_drop < 1 or self.max_spawn_attempts < self.target_n_boxes:
            raise ValueError(
                "--max-frames-per-drop must be positive and --max-spawn-attempts "
                "must be at least --nb."
            )
        if self.audit_frames_required < 0:
            raise ValueError("--audit-frames must be non-negative.")

        self.auto_save = not bool(args.no_auto_save)
        self.save_path = self._resolve_save_path(str(args.save_path))
        self.physics_config = _physics_config(
            box_density=float(args.box_density),
            sliding_mu=float(args.sliding_mu),
        )

        self.running = False
        self.done = False
        self.stage = STAGE_SETTLE
        self.status_message = "init"
        self.failure_reason = ""
        self.last_saved_path: Path | None = None
        self.frame_count = 0
        self.stage_frame_count = 0
        self.sim_time = 0.0
        self.settle_consecutive = 0
        self.spawn_attempts = 0
        self.filtered_count = 0
        self.last_filtered_count = 0
        self.last_spawn_index = -1
        self.audit_frame_count = 0
        self.audit_restart_count = 0
        self.audit_max_observed_pos_drift = 0.0
        self.audit_max_observed_rot_drift = 0.0
        self._audit_reference_centers: np.ndarray | None = None
        self._audit_reference_quats: np.ndarray | None = None
        self._last_unsettled = np.zeros(self.target_n_boxes, dtype=np.int32)
        self._last_max_lin = 0.0
        self._last_max_ang = 0.0

        self._initialize_box_state()
        self._build_model()
        self._spawn_next_box()
        self._update_status()

    def _resolve_save_path(self, value: str) -> Path:
        if value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = _TASK_ROOT / path
            return path.resolve()
        return (
            _TASK_ROOT
            / "scenes"
            / f"box_{self.target_n_boxes}_{self.seed}.npz"
        ).resolve()

    def _sample_box_edges(self) -> np.ndarray:
        if self.size_mode == SIZE_MODE_FIVE_LENGTH:
            length = float(self.rng.choice(FIVE_LENGTH_VALUES))
            return np.array(
                [length, *FIVE_LENGTH_CROSS_SECTION],
                dtype=np.float64,
            )
        if self.size_mode == SIZE_MODE_TEMTSIN_CD:
            return TEMTSIN_CD_EDGES.copy()
        if self.size_mode == SIZE_MODE_TEMTSIN_UDL:
            length = float(self.rng.choice(TEMTSIN_UDL_LENGTHS))
            return np.array([length, 0.09, 0.045], dtype=np.float64)

        catalog = np.asarray(
            list(combinations_with_replacement(self.edge_candidates, 3)),
            dtype=np.float64,
        )
        edges = catalog[int(self.rng.integers(0, len(catalog)))].copy()
        return edges[self.rng.permutation(3)]

    def _initialize_box_state(self) -> None:
        n_boxes = self.target_n_boxes
        self.centers = np.vstack([self._graveyard_pos(i) for i in range(n_boxes)])
        self.quats_xyzw = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            (n_boxes, 1),
        )
        self.half_extents = np.vstack(
            [0.5 * self._sample_box_edges() for _ in range(n_boxes)]
        ).astype(np.float64)
        self.active_mask = np.zeros(n_boxes, dtype=bool)

    def _graveyard_pos(self, slot: int) -> np.ndarray:
        base = np.asarray(self.args.graveyard_pos, dtype=np.float64).reshape(3)
        spacing = float(self.args.graveyard_spacing)
        return base + np.array(
            [spacing * (int(slot) % 20), spacing * (int(slot) // 20), 0.0],
            dtype=np.float64,
        )

    def _build_model(self) -> None:
        mjcf = build_mjcf(
            dims=self.dims,
            centers=self.centers,
            quats_xyzw=self.quats_xyzw,
            half_extents=self.half_extents,
            walls_removed={0, 1, 2, 3},
            wall_defs=[],
            scene_config=sc,
            physics_config=self.physics_config,
            box_drop_height=0.0,
        )
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        for box_index in range(self.target_n_boxes):
            self._disable_box_collision(box_index)
        self._pin_inactive_boxes()
        mujoco.mj_forward(self.model, self.data)

    def _box_joint_addrs(self, box_index: int) -> tuple[int, int]:
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"box_{int(box_index)}_j",
        )
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo model is missing box_{box_index}_j.")
        return (
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def _set_box_pose(
        self,
        box_index: int,
        center: np.ndarray,
        quat_xyzw: np.ndarray,
    ) -> None:
        qpos_addr, dof_addr = self._box_joint_addrs(box_index)
        self.data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(
            center,
            dtype=np.float64,
        ).reshape(3)
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = _quat_xyzw_to_wxyz(
            np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        )
        self.data.qvel[dof_addr : dof_addr + 6] = 0.0

    def _box_geom_id(self, box_index: int) -> int:
        geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"box_{int(box_index)}_g",
        )
        if geom_id < 0:
            raise RuntimeError(f"MuJoCo model is missing box_{box_index}_g.")
        return int(geom_id)

    def _disable_box_collision(self, box_index: int) -> None:
        geom_id = self._box_geom_id(box_index)
        self.model.geom_contype[geom_id] = 0
        self.model.geom_conaffinity[geom_id] = 0

    def _enable_box_collision(self, box_index: int) -> None:
        geom_id = self._box_geom_id(box_index)
        self.model.geom_contype[geom_id] = int(pc.BOX_GEOM_CONTYPE)
        self.model.geom_conaffinity[geom_id] = int(pc.BOX_GEOM_CONAFFINITY)

    def _pin_inactive_boxes(self) -> None:
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        for box_index in np.flatnonzero(~self.active_mask):
            self._set_box_pose(
                int(box_index),
                self._graveyard_pos(int(box_index)),
                identity,
            )

    def _extract_box_state(self) -> None:
        for box_index in np.flatnonzero(self.active_mask):
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"box_{int(box_index)}",
            )
            if body_id < 0:
                raise RuntimeError(f"MuJoCo model is missing box_{box_index}.")
            self.centers[int(box_index)] = np.asarray(
                self.data.xpos[body_id],
                dtype=np.float64,
            )
            self.quats_xyzw[int(box_index)] = _quat_wxyz_to_xyzw(
                np.asarray(self.data.xquat[body_id], dtype=np.float64)
            )

    def _active_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    def filter_limits(self) -> tuple[float, float]:
        return (
            0.5 * float(self.dims[0]) * self.filter_scale,
            0.5 * float(self.dims[1]) * self.filter_scale,
        )

    def _filter_active_boxes(self) -> int:
        self._extract_box_state()
        x_limit, y_limit = self.filter_limits()
        bad_indices = [
            int(box_index)
            for box_index in np.flatnonzero(self.active_mask)
            if abs(float(self.centers[int(box_index), 0])) > x_limit
            or abs(float(self.centers[int(box_index), 1])) > y_limit
        ]
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        for box_index in bad_indices:
            self.active_mask[box_index] = False
            self._disable_box_collision(box_index)
            self.centers[box_index] = self._graveyard_pos(box_index)
            self.quats_xyzw[box_index] = identity
            self._set_box_pose(
                box_index,
                self.centers[box_index],
                identity,
            )
        if bad_indices:
            self.filtered_count += len(bad_indices)
            self._pin_inactive_boxes()
            mujoco.mj_forward(self.model, self.data)
        self.last_filtered_count = len(bad_indices)
        return len(bad_indices)

    def _pile_mean_xy(self) -> np.ndarray:
        self._extract_box_state()
        if self._active_count() == 0:
            return np.zeros(2, dtype=np.float64)
        return np.mean(self.centers[self.active_mask, :2], axis=0)

    def _pile_top(self) -> float:
        self._extract_box_state()
        top = 0.0
        for box_index in np.flatnonzero(self.active_mask):
            _lower, upper = _box_aabb_xyzw(
                self.centers[int(box_index)],
                self.quats_xyzw[int(box_index)],
                self.half_extents[int(box_index)],
            )
            top = max(top, float(upper[2]))
        return top

    def _spawn_next_box(self) -> None:
        if self._active_count() >= self.target_n_boxes:
            self._start_audit()
            return
        if self.spawn_attempts >= self.max_spawn_attempts:
            self.mark_failed(
                f"Exceeded {self.max_spawn_attempts} spawn attempts before reaching "
                f"{self.target_n_boxes} filtered boxes."
            )
            return

        inactive = np.flatnonzero(~self.active_mask)
        if inactive.size == 0:
            raise RuntimeError("No inactive box slot is available for replenishment.")
        box_index = int(inactive[0])
        mean_xy = self._pile_mean_xy()
        top = self._pile_top()
        quat_xyzw = Rotation.random(random_state=self.rng).as_quat()
        _local_lower, local_upper = _box_aabb_xyzw(
            np.zeros(3, dtype=np.float64),
            quat_xyzw,
            self.half_extents[box_index],
        )
        vertical_half_extent = float(local_upper[2])
        center = np.array(
            [
                float(mean_xy[0]),
                float(mean_xy[1]),
                top + vertical_half_extent + self.release_clearance,
            ],
            dtype=np.float64,
        )

        self.centers[box_index] = center
        self.quats_xyzw[box_index] = quat_xyzw
        self.active_mask[box_index] = True
        self._set_box_pose(box_index, center, quat_xyzw)
        self._enable_box_collision(box_index)
        self.spawn_attempts += 1
        self.last_spawn_index = box_index
        self.stage = STAGE_SETTLE
        self.stage_frame_count = 0
        self.settle_consecutive = 0
        self.com_settle_tracker.reset()
        self.last_settle_criterion = ""
        self._last_unsettled = self.active_mask.astype(np.int32)
        mujoco.mj_forward(self.model, self.data)

    def _start_audit(self) -> None:
        self._extract_box_state()
        if self._active_count() != self.target_n_boxes:
            raise RuntimeError("Audit can only start with the exact target box count.")
        self.stage = STAGE_AUDIT
        self.stage_frame_count = 0
        self.audit_frame_count = 0
        self.settle_consecutive = 0
        self.com_settle_tracker.reset()
        self._audit_reference_centers = self.centers.copy()
        self._audit_reference_quats = self.quats_xyzw.copy()
        self.audit_max_observed_pos_drift = 0.0
        self.audit_max_observed_rot_drift = 0.0
        if self.audit_frames_required == 0:
            self._finish_audit()

    def _finish_audit(self) -> None:
        self._extract_box_state()
        self._filter_active_boxes()
        if self._active_count() < self.target_n_boxes:
            self._spawn_next_box()
            return
        if self._audit_reference_centers is None or self._audit_reference_quats is None:
            raise RuntimeError("Audit reference state is missing.")

        active_indices = np.flatnonzero(self.active_mask)
        position_drift = np.linalg.norm(
            self.centers[active_indices]
            - self._audit_reference_centers[active_indices],
            axis=1,
        )
        rotation_drift = np.array(
            [
                _quat_angle_xyzw(
                    self._audit_reference_quats[int(box_index)],
                    self.quats_xyzw[int(box_index)],
                )
                for box_index in active_indices
            ],
            dtype=np.float64,
        )
        self.audit_max_observed_pos_drift = float(np.max(position_drift, initial=0.0))
        self.audit_max_observed_rot_drift = float(np.max(rotation_drift, initial=0.0))
        stable = bool(
            self.audit_max_observed_pos_drift <= self.audit_max_pos_drift
            and self.audit_max_observed_rot_drift <= self.audit_max_rot_drift
            and int(np.count_nonzero(self._last_unsettled)) == 0
        )
        if not stable:
            self.audit_restart_count += 1
            self.stage = STAGE_SETTLE
            self.stage_frame_count = 0
            self.settle_consecutive = 0
            self.com_settle_tracker.reset()
            return

        self.stage = STAGE_DONE
        self.done = True
        self.running = False
        if self.auto_save:
            self.save()

    def _check_motion(self) -> int:
        mujoco.mj_forward(self.model, self.data)
        unsettled, max_lin, max_ang, per_box = _count_unsettled(
            model=self.model,
            data=self.data,
            n_boxes=self.target_n_boxes,
            lin_thr=self.lin_thr,
            ang_thr=self.ang_thr,
            active_mask=self.active_mask,
        )
        self._last_unsettled = per_box
        self._last_max_lin = float(max_lin)
        self._last_max_ang = float(max_ang)
        return int(unsettled)

    def step_frame(self) -> None:
        if not self.running or self.stage in (STAGE_DONE, STAGE_FAILED):
            self._update_status()
            return
        if self.stage == STAGE_SETTLE and self.stage_frame_count >= self.max_frames_per_drop:
            self.mark_failed(
                f"Box pile did not settle within {self.max_frames_per_drop} frames "
                f"after spawn attempt {self.spawn_attempts}."
            )
            return

        self._pin_inactive_boxes()
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)
            self._pin_inactive_boxes()
        self.frame_count += 1
        self.stage_frame_count += 1
        self.sim_time += self.frame_dt
        if self.stage == STAGE_AUDIT:
            self.audit_frame_count += 1

        com_settled = False
        if (
            self.stage == STAGE_SETTLE
            and self.com_settle_tracker.enabled
            and self.stage_frame_count % self.com_settle_check_interval == 0
        ):
            self._extract_box_state()
            com_settled = self.com_settle_tracker.update(
                frame_index=self.stage_frame_count,
                positions=self.centers,
                active_mask=self.active_mask,
            )

        velocity_settled = False
        if self.frame_count % self.settle_check_interval == 0:
            unsettled = self._check_motion()
            if self.stage == STAGE_SETTLE:
                if unsettled == 0:
                    self.settle_consecutive += 1
                else:
                    self.settle_consecutive = 0
                velocity_settled = (
                    self.settle_consecutive >= self.settle_steps_required
                )

        if self.stage == STAGE_SETTLE and (velocity_settled or com_settled):
            if velocity_settled and com_settled:
                self.last_settle_criterion = "velocity+com"
            elif com_settled:
                self.last_settle_criterion = "com"
            else:
                self.last_settle_criterion = "velocity"
            self._filter_active_boxes()
            if self._active_count() < self.target_n_boxes:
                self._spawn_next_box()
            else:
                self._start_audit()

        if self.stage == STAGE_AUDIT and self.audit_frame_count >= self.audit_frames_required:
            self._check_motion()
            self._finish_audit()
        self._update_status()

    def start(self) -> None:
        if self.stage not in (STAGE_DONE, STAGE_FAILED):
            self.running = True

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.args.seed = int(seed)
        replacement = TemtsinSceneGenSim(self.args)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)

    def mark_failed(self, reason: str) -> None:
        self.failure_reason = str(reason)
        self.stage = STAGE_FAILED
        self.done = False
        self.running = False
        self._update_status()

    def active_snapshot(self) -> dict[str, np.ndarray]:
        self._extract_box_state()
        active = self.active_mask.copy()
        return {
            "box_centers": self.centers[active].copy(),
            "box_quats_xyzw": self.quats_xyzw[active].copy(),
            "box_half_extents": self.half_extents[active].copy(),
        }

    def save(self, path: Path | None = None) -> Path:
        if not self.done or self._active_count() != self.target_n_boxes:
            raise RuntimeError("Only a completed exact-count scene can be saved.")
        destination = self.save_path if path is None else Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.active_snapshot()
        x_limit, y_limit = self.filter_limits()
        np.savez_compressed(
            destination,
            box_centers=np.asarray(snapshot["box_centers"], dtype=np.float32),
            box_quats_xyzw=np.asarray(snapshot["box_quats_xyzw"], dtype=np.float32),
            box_half_extents=np.asarray(snapshot["box_half_extents"], dtype=np.float32),
            box_density=np.asarray(self.physics_config.BOX_DENSITY, dtype=np.float64),
            metadata_json=np.asarray(json.dumps({
                "generator": "temtsin_mean_pile",
                "partition_seed": self.seed,
                "generator_args": vars(self.args),
                "physics_config": vars(self.physics_config),
            })),
            partition_dims=np.asarray(self.dims, dtype=np.float32),
            table_pos=np.asarray(sc.TABLE_POS, dtype=np.float32),
            table_half_extents=np.asarray(sc.TABLE_HALF_EXTENTS, dtype=np.float32),
            table_friction=np.asarray(
                self.physics_config.TABLE_FRICTION,
                dtype=np.float32,
            ),
            wall_friction=np.asarray(
                self.physics_config.WALL_FRICTION,
                dtype=np.float32,
            ),
            box_friction=np.asarray(
                self.physics_config.BOX_FRICTION,
                dtype=np.float32,
            ),
            physics_config_version=np.asarray(
                self.physics_config.PHYSICS_CONFIG_VERSION,
                dtype=np.int32,
            ),
            walls_enabled=np.asarray(False, dtype=np.bool_),
            walls_removed=np.asarray([0, 1, 2, 3], dtype=np.int32),
            generator=np.asarray("temtsin_mean_pile"),
            generator_seed=np.asarray(self.seed, dtype=np.int64),
            generator_size_mode=np.asarray(self.size_mode),
            generator_spawn_attempts=np.asarray(self.spawn_attempts, dtype=np.int32),
            generator_filtered_count=np.asarray(self.filtered_count, dtype=np.int32),
            generator_target_box_count=np.asarray(
                self.target_n_boxes,
                dtype=np.int32,
            ),
            generator_release_clearance=np.asarray(
                self.release_clearance,
                dtype=np.float32,
            ),
            generator_enable_com_settle=np.asarray(
                self.enable_com_settle,
                dtype=np.bool_,
            ),
            generator_com_settle_max_displacement=np.asarray(
                self.com_settle_max_displacement,
                dtype=np.float32,
            ),
            generator_com_settle_check_interval=np.asarray(
                self.com_settle_check_interval,
                dtype=np.int32,
            ),
            generator_com_settle_required_checks=np.asarray(
                self.com_settle_required_checks,
                dtype=np.int32,
            ),
            filter_xy_limits=np.asarray([x_limit, y_limit], dtype=np.float32),
            filter_scale=np.asarray(self.filter_scale, dtype=np.float32),
        )
        self.last_saved_path = destination
        return destination

    def _update_status(self) -> None:
        x_limit, y_limit = self.filter_limits()
        self.status_message = (
            f"stage={self.stage} running={self.running} "
            f"active={self._active_count()}/{self.target_n_boxes} "
            f"attempts={self.spawn_attempts}/{self.max_spawn_attempts} "
            f"filtered={self.filtered_count} last_filtered={self.last_filtered_count} "
            f"frame={self.stage_frame_count} "
            f"stable={self.settle_consecutive}/{self.settle_steps_required} "
            f"com={self.com_settle_tracker.consecutive}/"
            f"{self.com_settle_required_checks} "
            f"audit={self.audit_frame_count}/{self.audit_frames_required} "
            f"max|v|={self._last_max_lin:.5f} max|w|={self._last_max_ang:.5f} "
            f"limits=x:+/-{x_limit:.3f},y:+/-{y_limit:.3f}"
        )


def _filter_boundary_geometry(
    sim: TemtsinSceneGenSim,
) -> tuple[np.ndarray, np.ndarray]:
    x_limit, y_limit = sim.filter_limits()
    z_lower = 0.0
    z_upper = float(sim.dims[2])
    nodes = np.array(
        [
            [-x_limit, -y_limit, z_lower],
            [x_limit, -y_limit, z_lower],
            [x_limit, y_limit, z_lower],
            [-x_limit, y_limit, z_lower],
            [-x_limit, -y_limit, z_upper],
            [x_limit, -y_limit, z_upper],
            [x_limit, y_limit, z_upper],
            [-x_limit, y_limit, z_upper],
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    return nodes, edges


def run_polyscope(sim: TemtsinSceneGenSim) -> None:
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError as exc:
        raise ImportError(
            "Polyscope is required for interactive visualization; use --headless "
            "to generate without a viewer."
        ) from exc

    ps.init()
    ps.set_program_name("Temtsin mean-pile scene generator")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.look_at((1.1, -1.4, 0.9), (0.0, 0.0, 0.25))

    box_handles: list[object] = []

    def register_structures() -> None:
        nonlocal box_handles
        ps.remove_all_structures()
        x_limit, y_limit = sim.filter_limits()
        floor = ps.register_surface_mesh(
            "filter_floor",
            axis_aligned_box_vertices(
                (0.0, 0.0, float(sc.TABLE_POS[2])),
                (x_limit, y_limit, 0.002),
            ),
            CUBE_FACES,
        )
        floor.set_color(COLOR_TABLE)
        floor.set_transparency(0.65)

        boundary_nodes, boundary_edges = _filter_boundary_geometry(sim)
        boundary = ps.register_curve_network(
            "filter_boundary",
            boundary_nodes,
            boundary_edges,
        )
        boundary.set_color((0.95, 0.45, 0.08))
        boundary.set_radius(0.0025, relative=False)

        box_handles = []
        for box_index in range(sim.target_n_boxes):
            vertices = box_vertices_world(
                sim.data.xpos[
                    mujoco.mj_name2id(
                        sim.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"box_{box_index}",
                    )
                ],
                sim.data.xquat[
                    mujoco.mj_name2id(
                        sim.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"box_{box_index}",
                    )
                ],
                sim.half_extents[box_index],
            )
            mesh = ps.register_surface_mesh(
                f"box_{box_index}",
                vertices,
                CUBE_FACES,
            )
            mesh.set_color(COLOR_BOX_INSTANT_SETTLED)
            mesh.set_enabled(bool(sim.active_mask[box_index]))
            box_handles.append(mesh)

    register_structures()

    def callback() -> None:
        psim.TextUnformatted(sim.status_message)
        psim.TextUnformatted(
            f"seed={sim.seed} size_mode={sim.size_mode} density={float(sim.args.box_density):.1f}"
        )
        if sim.failure_reason:
            psim.TextUnformatted(f"failure={sim.failure_reason}")
        if sim.last_saved_path is not None:
            psim.TextUnformatted(f"saved={sim.last_saved_path}")
        psim.Separator()

        if psim.Button("Reset"):
            sim.reset(seed=sim.seed)
            register_structures()
        psim.SameLine()
        if psim.Button("New seed"):
            sim.reset(seed=sim.seed + 1)
            register_structures()
        psim.SameLine()
        if psim.Button("Pause / run"):
            sim.running = not sim.running
        psim.SameLine()
        step_once = bool(psim.Button("Step once"))
        if sim.done and psim.Button("Save scene"):
            sim.save()

        if step_once:
            was_running = sim.running
            sim.running = True
            sim.step_frame()
            sim.running = was_running
        else:
            sim.step_frame()

        mujoco.mj_forward(sim.model, sim.data)
        for box_index, mesh in enumerate(box_handles):
            active = bool(sim.active_mask[box_index])
            mesh.set_enabled(active)
            if not active:
                mesh.set_color(COLOR_BOX_FILTERED_OUT)
                continue
            body_id = mujoco.mj_name2id(
                sim.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"box_{box_index}",
            )
            vertices = box_vertices_world(
                sim.data.xpos[body_id],
                sim.data.xquat[body_id],
                sim.half_extents[box_index],
            )
            mesh.update_vertex_positions(vertices)
            if (
                box_index < len(sim._last_unsettled)
                and int(sim._last_unsettled[box_index]) != 0
                and sim.running
            ):
                mesh.set_color(COLOR_BOX_INSTANT_UNSETTLED)
            else:
                mesh.set_color(COLOR_BOX_INSTANT_SETTLED)

    ps.set_user_callback(callback)
    ps.show()


def run_headless(sim: TemtsinSceneGenSim, *, max_total_frames: int) -> None:
    sim.start()
    while sim.stage not in (STAGE_DONE, STAGE_FAILED):
        if max_total_frames > 0 and sim.frame_count >= max_total_frames:
            sim.mark_failed(f"Exceeded --max-total-frames={max_total_frames}.")
            break
        sim.step_frame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one exact-count scene with Temtsin et al.'s one-at-a-time "
            "mean-pile placement and a four-sided virtual XY filter."
        )
    )
    parser.add_argument("--nb", "-n", type=int, default=gc.NUM_BOXES)
    parser.add_argument("--seed", type=int, default=sc.PARTITION_SEED)
    parser.add_argument(
        "--size-mode",
        choices=SIZE_MODES,
        default=gc.SIZE_MODE,
        help=(
            "Box dimension catalog. Default five-length uses a fixed 0.04 x "
            "0.06 m cross-section and lengths 0.08, 0.10, 0.12, 0.14, 0.16 m."
        ),
    )
    parser.add_argument(
        "--edge-candidates",
        type=float,
        nargs="+",
        default=list(gc.EDGE_CANDIDATES),
    )
    parser.add_argument(
        "--dims",
        type=float,
        nargs=3,
        default=list(sc.PARTITION_DIMS),
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--filter-scale", type=float, default=gc.FILTER_SCALE)
    parser.add_argument("--release-clearance", type=float, default=gc.RELEASE_CLEARANCE)
    parser.add_argument("--box-density", type=float, default=pc.BOX_DENSITY)
    parser.add_argument(
        "--sliding-mu",
        type=float,
        default=pc.BOX_MU,
        help="Shared sliding friction for boxes, table, and walls.",
    )
    parser.add_argument(
        "--enable-com-settle",
        action=argparse.BooleanOptionalAction,
        default=bool(pc.ENABLE_COM_SETTLE),
        help=(
            "Enable bounded COM displacement as an alternative settle criterion "
            "(default: enabled; use --no-enable-com-settle to disable)."
        ),
    )
    parser.add_argument(
        "--com-settle-max-displacement",
        type=float,
        default=float(pc.COM_SETTLE_MAX_DISPLACEMENT),
    )
    parser.add_argument(
        "--com-settle-check-interval",
        type=int,
        default=int(pc.COM_SETTLE_CHECK_INTERVAL),
    )
    parser.add_argument(
        "--com-settle-required-checks",
        type=int,
        default=int(pc.COM_SETTLE_REQUIRED_CHECKS),
    )
    parser.add_argument("--max-spawn-attempts", type=int, default=gc.MAX_SPAWN_ATTEMPTS)
    parser.add_argument("--max-frames-per-drop", type=int, default=gc.MAX_FRAMES_PER_DROP)
    parser.add_argument("--audit-frames", type=int, default=gc.AUDIT_FRAMES)
    parser.add_argument("--audit-max-pos-drift", type=float, default=gc.AUDIT_MAX_POS_DRIFT)
    parser.add_argument("--audit-max-rot-drift", type=float, default=gc.AUDIT_MAX_ROT_DRIFT)
    parser.add_argument("--graveyard-pos", type=float, nargs=3, default=list(gc.GRAVEYARD_POS))
    parser.add_argument("--graveyard-spacing", type=float, default=gc.GRAVEYARD_SPACING)
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Default: scenes/box_<nb>_<seed>.npz (relative paths use project root).",
    )
    parser.add_argument("--no-auto-save", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-total-frames",
        type=int,
        default=0,
        help="Headless safety cap. Zero disables the global cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    sim = TemtsinSceneGenSim(args)
    print(
        f"[scene_gen_temtsin] target={sim.target_n_boxes} seed={sim.seed} "
        f"size_mode={sim.size_mode} filter_scale={sim.filter_scale} "
        f"sliding_mu={float(args.sliding_mu)} "
        f"save_path={sim.save_path}",
        flush=True,
    )
    if bool(args.headless):
        run_headless(sim, max_total_frames=int(args.max_total_frames))
        print(
            f"[scene_gen_temtsin] stage={sim.stage} "
            f"active={sim._active_count()}/{sim.target_n_boxes} "
            f"attempts={sim.spawn_attempts} filtered={sim.filtered_count} "
            f"frames={sim.frame_count} wall_seconds={time.perf_counter() - start:.2f} "
            f"saved={sim.last_saved_path} failure={sim.failure_reason!r}",
            flush=True,
        )
        if sim.stage == STAGE_FAILED:
            raise RuntimeError(sim.failure_reason)
        return
    run_polyscope(sim)


if __name__ == "__main__":
    main()
