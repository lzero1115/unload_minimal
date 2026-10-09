from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import motion_configs as mc
import phys_configs as pc
import scene_configs as sc
from configs.env_configs import DEFAULT_ENV_CONFIG
from runtime.environment import UnloadEnvironment
from runtime.geometry import _write_box_root_pose_vel
from robot_naming import robot_actuator_name, robot_joint_name, robot_site_name
from settle_utils import ComDisplacementSettleTracker

from .task_types import MotionPlan, TaskPhase


def _transform(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(
        np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    ).as_matrix()
    out[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return out


@dataclass
class ExecutorStats:
    completed_actions: int = 0
    metric_total: float = 0.0
    last_metric_delta: float = 0.0
    settle_frames: int = 0


class UnloadTaskExecutor:
    """Execute one user-selected action; settle and measure before the next."""

    def __init__(self, scene_path, *, config=DEFAULT_ENV_CONFIG) -> None:
        self.env = UnloadEnvironment(scene_path, config)
        self.phase = TaskPhase.IDLE
        self.status_message = "Adjust the pile, then click Settle pile and compute IK."
        self.failure_reason = ""
        self.selected_action: int | None = None
        self.initializing = True
        self.plan: MotionPlan | None = None
        self.path_cursor = 0
        self.stats = ExecutorStats()
        self._settle_consecutive = 0
        self._com_settle_tracker = ComDisplacementSettleTracker(
            enabled=self.env.enable_com_settle,
            max_displacement=self.env.com_settle_max_displacement,
            check_interval=self.env.com_settle_check_interval,
            required_checks=self.env.com_settle_required_checks,
        )
        self._attached_box = -1
        self._box_in_ee: np.ndarray | None = None
        self._last_carry_q: np.ndarray | None = None
        self._candidate_mask = np.zeros((self.sim.n_boxes,), dtype=bool)
        self._proxy_candidate_count = 0

    def close(self):
        self._release_weld_if_needed()
        self.env.close()

    def set_initial_pile_poses(self, *, positions_w, quats_wxyz):
        if self.stats.completed_actions or self.selected_action is not None:
            raise RuntimeError("Pile placement is locked after the first unload starts")
        if self.phase not in (TaskPhase.IDLE, TaskPhase.DONE):
            raise RuntimeError("Wait until physics has settled")
        positions = np.asarray(positions_w, dtype=np.float64)
        quats = np.asarray(quats_wxyz, dtype=np.float64)
        if positions.shape != (self.sim.n_boxes, 3) or quats.shape != (self.sim.n_boxes, 4):
            raise ValueError("Pile poses do not match the scene")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quats)):
            raise ValueError("Pile poses must be finite")
        for i in np.flatnonzero(self.active_mask):
            _write_box_root_pose_vel(
                model=self.sim.model, data=self.sim.data, box_index=int(i),
                pos=positions[i], quat_wxyz=quats[i], zero_vel=True,
            )
        self.sim.canonicalize_dynamic_state()
        self.env._metric().reset()
        self.initializing = True
        self.plan = None
        self.status_message = "Pile placement changed; click Settle pile and compute IK."

    def start_initial_settle(self):
        if self.stats.completed_actions or self.selected_action is not None:
            raise RuntimeError("Cannot reset the metric baseline after unloading starts")
        if self.phase not in (TaskPhase.IDLE, TaskPhase.DONE):
            raise RuntimeError("Initial settling is already in progress")
        self.env._metric().reset()
        self.initializing = True
        self.stats.settle_frames = 0
        self._settle_consecutive = 0
        self._com_settle_tracker.reset()
        self.failure_reason = ""
        self.phase = TaskPhase.POST_SETTLE
        self.status_message = "Initial pile settling (not scored); requested IK follows."

    def _begin_action(self, box_index):
        if self.phase not in (TaskPhase.IDLE, TaskPhase.DONE) or self.initializing:
            raise RuntimeError("Wait for the scene to settle before removing a box")
        if not 0 <= box_index < self.sim.n_boxes or not self.active_mask[box_index]:
            raise ValueError(f"Box {box_index} is not active")
        self.selected_action = int(box_index)
        self.stats.settle_frames = 0
        self._settle_consecutive = 0
        self._com_settle_tracker.reset()
        self.path_cursor = 0
        self.failure_reason = ""

    def start_motion(self, plan: MotionPlan):
        for path in (plan.forward_q_path, plan.contact_q_path, plan.unload_q_path, plan.return_q_path):
            if path.ndim != 2 or len(path) == 0 or not np.all(np.isfinite(path)):
                raise ValueError("The motion plan contains an empty or invalid path")
        self._begin_action(plan.box_index)
        self.plan = plan
        self.phase = TaskPhase.MOVE_TO_APPROACH
        self.status_message = f"approach box={plan.box_index}"

    def start_direct(self, box_index: int):
        self._begin_action(box_index)
        self.env._deactivate_box(box_index)
        self.plan = None
        self.phase = TaskPhase.POST_SETTLE
        self.status_message = f"direct remove box={box_index}; settling"

    def _finish_settle(self):
        self.stats.settle_frames = 0
        self._settle_consecutive = 0
        self._com_settle_tracker.reset()
        self.plan = None
        self.path_cursor = 0
        self._last_carry_q = None
        self._candidate_mask.fill(False)
        self.phase = TaskPhase.DONE
        self.status_message = (
            f"stable actions={self.stats.completed_actions} "
            f"metric={self.stats.metric_total:.6f}"
        )

    @property
    def sim(self):
        return self.env._sim()

    @property
    def current_action(self) -> int | None:
        return self.selected_action

    @property
    def attached_box(self) -> int | None:
        return None if self._attached_box < 0 else int(self._attached_box)

    @property
    def active_mask(self) -> np.ndarray:
        return np.asarray(self.env._scene_active_mask(), dtype=bool).copy()

    @property
    def candidate_mask(self) -> np.ndarray:
        return np.asarray(self._candidate_mask, dtype=bool).copy()

    def _joint_handles(self, logical_name: str) -> tuple[int, int, int]:
        model = self.sim.model
        joint_name = robot_joint_name(0, logical_name)
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo model is missing robot joint {joint_name}.")
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            robot_actuator_name(0, logical_name),
        )
        return (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
            int(actuator_id),
        )

    def current_arm_q(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.sim.data.qpos[self._joint_handles(name)[0]])
                for name in mc.ARM_JOINT_NAMES
            ],
            dtype=np.float64,
        )

    def _plan_q_to_arm_order(self, q: np.ndarray) -> np.ndarray:
        if self.plan is None:
            raise RuntimeError("No active motion plan.")
        values = np.asarray(q, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(self.plan.joint_names):
            raise ValueError("Motion path width does not match its joint names.")
        by_name = {
            name: float(values[i]) for i, name in enumerate(self.plan.joint_names)
        }
        missing = [name for name in mc.ARM_JOINT_NAMES if name not in by_name]
        if missing:
            raise ValueError(f"Motion plan is missing arm joints: {missing}")
        return np.asarray([by_name[name] for name in mc.ARM_JOINT_NAMES])

    def _force_arm_state(
        self,
        q: np.ndarray,
        qvel: np.ndarray | None = None,
    ) -> None:
        arm_q = self._plan_q_to_arm_order(q)
        arm_qvel = (
            np.zeros_like(arm_q)
            if qvel is None
            else self._plan_q_to_arm_order(qvel)
        )
        for joint_i, name in enumerate(mc.ARM_JOINT_NAMES):
            qadr, dadr, aid = self._joint_handles(name)
            self.sim.data.qpos[qadr] = float(arm_q[joint_i])
            self.sim.data.qvel[dadr] = float(arm_qvel[joint_i])
            if aid >= 0:
                self.sim.data.ctrl[aid] = float(arm_q[joint_i])

    def _return_arm_to_rest(self) -> None:
        for joint_i, name in enumerate(mc.ARM_JOINT_NAMES):
            qadr, dadr, aid = self._joint_handles(name)
            target = float(mc.ARM_HOME_Q[joint_i])
            self.sim.data.qpos[qadr] = target
            self.sim.data.qvel[dadr] = 0.0
            if aid >= 0:
                self.sim.data.ctrl[aid] = target
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def _ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        site_name = robot_site_name(0, str(sc.ROBOT_EE_LINK))
        site_id = mujoco.mj_name2id(
            self.sim.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise RuntimeError(f"MuJoCo model is missing EE site {site_name}.")
        mujoco.mj_forward(self.sim.model, self.sim.data)
        position = np.asarray(
            self.sim.data.site_xpos[site_id],
            dtype=np.float64,
        ).copy()
        rotation = np.asarray(
            self.sim.data.site_xmat[site_id],
            dtype=np.float64,
        ).reshape(3, 3)
        return position, Rotation.from_matrix(rotation).as_quat()

    def _box_pose(self, box_index: int) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(
            self.sim.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"box_{int(box_index)}",
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo model is missing box_{int(box_index)}.")
        mujoco.mj_forward(self.sim.model, self.sim.data)
        position = np.asarray(
            self.sim.data.xpos[body_id],
            dtype=np.float64,
        ).copy()
        quat_wxyz = np.asarray(
            self.sim.data.xquat[body_id],
            dtype=np.float64,
        ).copy()
        return position, quat_wxyz[[1, 2, 3, 0]]

    def _set_box_gravity_compensation(
        self,
        box_index: int,
        enabled: bool,
    ) -> None:
        body_id = mujoco.mj_name2id(
            self.sim.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"box_{int(box_index)}",
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo model is missing box_{int(box_index)}.")
        if enabled:
            gravity = np.asarray(self.sim.model.opt.gravity, dtype=np.float64)
            self.sim.data.xfrc_applied[body_id, :3] = (
                -float(self.sim.model.body_mass[body_id]) * gravity
            )
            self.sim.data.xfrc_applied[body_id, 3:] = 0.0
        else:
            self.sim.data.xfrc_applied[body_id, :] = 0.0

    def _attach(self) -> None:
        box_index = self.current_action
        if box_index is None:
            raise RuntimeError("No confirmed action is available for attachment.")
        box_position, box_quat_xyzw = self._box_pose(box_index)
        ee_position, ee_quat_xyzw = self._ee_pose()
        self._box_in_ee = np.linalg.inv(
            _transform(ee_position, ee_quat_xyzw)
        ) @ _transform(box_position, box_quat_xyzw)
        self._attached_box = int(box_index)
        self.env._set_box_mocap_pose(
            int(box_index),
            box_position,
            box_quat_xyzw[[3, 0, 1, 2]],
        )
        self.env._set_box_weld_active(int(box_index), True)
        self._set_box_gravity_compensation(int(box_index), True)
        self._update_carry_mocap()
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def _update_carry_mocap(self) -> None:
        if self._attached_box < 0 or self._box_in_ee is None:
            return
        ee_position, ee_quat_xyzw = self._ee_pose()
        box_transform = (
            _transform(ee_position, ee_quat_xyzw) @ self._box_in_ee
        )
        box_quat_xyzw = Rotation.from_matrix(box_transform[:3, :3]).as_quat()
        self.env._set_box_mocap_pose(
            self._attached_box,
            box_transform[:3, 3],
            box_quat_xyzw[[3, 0, 1, 2]],
        )

    def _release_weld_if_needed(self) -> None:
        if self._attached_box < 0 or self.env.sim is None:
            return
        try:
            self.env._set_box_weld_active(self._attached_box, False)
            self._set_box_gravity_compensation(self._attached_box, False)
        finally:
            self._attached_box = -1
            self._box_in_ee = None

    def _advance_arm_without_physics(self, q: np.ndarray) -> None:
        self._force_arm_state(q)
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def _advance_carry_with_physics(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
    ) -> None:
        q0 = np.asarray(q_start, dtype=np.float64)
        q1 = np.asarray(q_end, dtype=np.float64)
        frame_dt = max(float(self.sim.frame_dt), np.finfo(np.float64).eps)
        qvel = (q1 - q0) / frame_dt
        substeps = max(1, int(self.sim.sim_substeps))
        for substep_i in range(substeps):
            alpha = float(substep_i) / float(substeps)
            next_alpha = float(substep_i + 1) / float(substeps)
            q_substep = (1.0 - alpha) * q0 + alpha * q1
            q_next = (1.0 - next_alpha) * q0 + next_alpha * q1
            self._force_arm_state(q_substep, qvel)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self._update_carry_mocap()
            mujoco.mj_step(self.sim.model, self.sim.data)
            self._force_arm_state(q_next, qvel)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self._update_carry_mocap()
            self.env._pin_inactive_boxes()
        self._force_arm_state(q1)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self._update_carry_mocap()
        self.sim.frame_count += 1
        self.sim.sim_time += float(self.sim.frame_dt)

    def _step_settle(self) -> None:
        self.sim.set_arm_ctrl(tuple(float(value) for value in mc.ARM_HOME_Q))
        for _ in range(int(self.sim.sim_substeps)):
            mujoco.mj_step(self.sim.model, self.sim.data)
            self.env._pin_inactive_boxes()
        self.sim.frame_count += 1
        self.sim.sim_time += float(self.sim.frame_dt)
        self.stats.settle_frames += 1
        self.sim.forward()
        if not np.all(np.isfinite(self.sim.data.qpos)) or not np.all(np.isfinite(self.sim.data.qvel)):
            raise RuntimeError("Non-finite simulation state; this step was not scored")

        com_settled = False
        if self._com_settle_tracker.enabled:
            com_settled = self._com_settle_tracker.update(
                frame_index=self.stats.settle_frames,
                positions=self.env._box_com_positions(),
                active_mask=self.env._scene_active_mask(),
            )
        if self.stats.settle_frames > int(self.env.max_settle_frames):
            raise RuntimeError(
                f"Settling exceeded {self.env.max_settle_frames} frames; "
                f"max linear={self.sim.last_max_lin:.6f}, "
                f"max angular={self.sim.last_max_ang:.6f}; this step was not scored."
            )

        velocity_check = (
            int(self.sim.frame_count) % int(pc.SETTLE_CHECK_INTERVAL) == 0
        )
        if not velocity_check and not com_settled:
            return

        velocity_settled = False
        if velocity_check:
            unsettled, max_linear, max_angular = (
                self.env._active_box_motion_status()
            )
            self.sim.last_unsettled = unsettled
            self.sim.last_max_lin = float(max_linear)
            self.sim.last_max_ang = float(max_angular)
            if int(np.count_nonzero(unsettled)) != 0:
                self._settle_consecutive = 0
            else:
                self._settle_consecutive += 1
                velocity_settled = self._settle_consecutive >= int(pc.SETTLE_STEPS)
        if not velocity_settled and not com_settled:
            return

        self.sim.canonicalize_dynamic_state()
        metric_tracker = self.env._metric()
        if self.initializing:
            metric_tracker.set_baseline_from_mujoco(
                model=self.sim.model, data=self.sim.data,
                n_boxes=self.sim.n_boxes, active_mask=self.active_mask,
            )
            self.initializing = False
            self._finish_settle()
            return
        if metric_tracker.pos_prev is None or metric_tracker.quat_prev is None:
            raise RuntimeError("Metric tracker has no previous steady-state snapshot.")
        metric_stats = metric_tracker.update_from_mujoco(
            model=self.sim.model,
            data=self.sim.data,
            active_mask=self.env._scene_active_mask(),
        )
        self.stats.last_metric_delta = float(metric_stats.get("metric_total", 0.0))
        self.stats.metric_total = float(metric_tracker.metric_total)
        self.stats.completed_actions += 1
        self._finish_settle()

    def _fail(self, exc: Exception) -> None:
        self.failure_reason = f"{type(exc).__name__}: {exc}"
        self.status_message = f"failed: {self.failure_reason}"
        self.phase = TaskPhase.FAILED
        self._release_weld_if_needed()

    def tick(self) -> TaskPhase:
        if self.phase in {TaskPhase.IDLE, TaskPhase.DONE, TaskPhase.FAILED}:
            return self.phase
        try:
            if self.phase == TaskPhase.MOVE_TO_APPROACH:
                if self.plan is None:
                    raise RuntimeError("MOVE_TO_APPROACH has no motion plan.")
                path = self.plan.forward_q_path
                self._advance_arm_without_physics(
                    path[min(self.path_cursor, len(path) - 1)]
                )
                self.path_cursor += 1
                self.status_message = (
                    f"approach box={self.plan.box_index} "
                    f"{min(self.path_cursor, len(path))}/{len(path)}"
                )
                if self.path_cursor >= len(path):
                    self.path_cursor = 1
                    self.phase = TaskPhase.APPROACH_TO_CONTACT

            elif self.phase == TaskPhase.APPROACH_TO_CONTACT:
                if self.plan is None:
                    raise RuntimeError("APPROACH_TO_CONTACT has no motion plan.")
                path = self.plan.contact_q_path
                if self.path_cursor >= len(path):
                    self.path_cursor = 0
                    self.phase = TaskPhase.ATTACH
                else:
                    self._advance_arm_without_physics(path[self.path_cursor])
                    self.path_cursor += 1
                    self.status_message = (
                        f"contact box={self.plan.box_index} "
                        f"{min(self.path_cursor, len(path))}/{len(path)}"
                    )
                    if self.path_cursor >= len(path):
                        self.path_cursor = 0
                        self.phase = TaskPhase.ATTACH

            elif self.phase == TaskPhase.ATTACH:
                if self.plan is None:
                    raise RuntimeError("ATTACH has no motion plan.")
                self._attach()
                self._last_carry_q = self.plan.contact_q_path[-1].copy()
                self.path_cursor = 1
                self.phase = TaskPhase.UNLOAD_WITH_PHYSICS
                self.status_message = (
                    f"attached box={self.plan.box_index}; unloading along "
                    f"selected direction"
                )

            elif self.phase == TaskPhase.UNLOAD_WITH_PHYSICS:
                if self.plan is None or self._last_carry_q is None:
                    raise RuntimeError("UNLOAD_WITH_PHYSICS has no motion plan.")
                path = self.plan.unload_q_path
                if self.path_cursor >= len(path):
                    self._last_carry_q = path[-1].copy()
                    self.path_cursor = 1
                    self.phase = TaskPhase.RETURN_TO_REST_WITH_PHYSICS
                else:
                    q_next = path[self.path_cursor]
                    self._advance_carry_with_physics(self._last_carry_q, q_next)
                    self._last_carry_q = q_next.copy()
                    self.path_cursor += 1
                    self.status_message = (
                        f"unload box={self.plan.box_index} "
                        f"{min(self.path_cursor, len(path))}/{len(path)}"
                    )
                    if self.path_cursor >= len(path):
                        self.path_cursor = 1
                        self.phase = TaskPhase.RETURN_TO_REST_WITH_PHYSICS

            elif self.phase == TaskPhase.RETURN_TO_REST_WITH_PHYSICS:
                if self.plan is None or self._last_carry_q is None:
                    raise RuntimeError(
                        "RETURN_TO_REST_WITH_PHYSICS has no motion plan."
                    )
                path = self.plan.return_q_path
                if self.path_cursor >= len(path):
                    self.phase = TaskPhase.RELEASE_REMOVE
                else:
                    q_next = path[self.path_cursor]
                    self._advance_carry_with_physics(self._last_carry_q, q_next)
                    self._last_carry_q = q_next.copy()
                    self.path_cursor += 1
                    self.status_message = (
                        f"return box={self.plan.box_index} "
                        f"{min(self.path_cursor, len(path))}/{len(path)}"
                    )
                    if self.path_cursor >= len(path):
                        self.phase = TaskPhase.RELEASE_REMOVE

            elif self.phase == TaskPhase.RELEASE_REMOVE:
                box_index = self.current_action
                if box_index is None:
                    raise RuntimeError("No confirmed action is available for removal.")
                self._release_weld_if_needed()
                self.env._deactivate_box(int(box_index))
                self._return_arm_to_rest()
                self.stats.settle_frames = 0
                self._settle_consecutive = 0
                self._com_settle_tracker.reset()
                self.phase = TaskPhase.POST_SETTLE
                self.status_message = (
                    f"removed box={box_index} at robot rest; settling"
                )

            elif self.phase == TaskPhase.POST_SETTLE:
                self._step_settle()
        except Exception as exc:
            self._fail(exc)
        return self.phase
