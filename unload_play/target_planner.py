from __future__ import annotations

from typing import Any

import numpy as np
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

import motion_configs as mc
import task_configs as tc
from runtime.ik_candidates import (
    get_curobo_tensor_args,
    load_local_curobo_robot_config,
    make_ik_solver,
    query_to_curobo_world,
    world_to_base_position,
    world_to_base_quat_wxyz,
)

from .target_types import (
    SelectedTarget,
    UnloadAdjustment,
    normalized,
    unload_direction_w,
    unload_target_position_w,
)
from .task_types import MotionPlan, MotionPlanUnavailableError


def _device_sync(tensor_args: Any) -> None:
    device = torch.device(tensor_args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _success_scalar(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].detach().cpu().item())
    return bool(np.asarray(value).reshape(-1)[0])


class SelectedTargetMotionPlanner:
    """Plan a complete selected grasp, unload, and return-to-rest motion."""

    def __init__(self) -> None:
        self.tensor_args = get_curobo_tensor_args()
        self.robot = load_local_curobo_robot_config()
        self.motion_gen: MotionGen | None = None
        self.contact_solver = None

    def _make_motion_gen(self, world_cfg) -> MotionGen:
        cfg = MotionGenConfig.load_from_robot_config(
            self.robot["robot_config"],
            world_cfg,
            tensor_args=self.tensor_args,
            num_ik_seeds=int(mc.CUROBO_MOTION_NUM_IK_SEEDS),
            num_batch_ik_seeds=int(mc.CUROBO_MOTION_NUM_IK_SEEDS),
            num_trajopt_seeds=int(mc.CUROBO_MOTION_NUM_TRAJOPT_SEEDS),
            num_batch_trajopt_seeds=int(mc.CUROBO_MOTION_NUM_TRAJOPT_SEEDS),
            num_graph_seeds=int(mc.CUROBO_MOTION_NUM_GRAPH_SEEDS),
            interpolation_dt=float(mc.FRAME_DT),
            use_cuda_graph=bool(mc.CUROBO_MOTION_USE_CUDA_GRAPH),
            self_collision_check=bool(mc.CUROBO_SELF_COLLISION_CHECK),
            self_collision_opt=bool(mc.CUROBO_SELF_COLLISION_OPT),
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            collision_activation_distance=float(
                mc.CUROBO_MOTION_COLLISION_ACTIVATION_DISTANCE
            ),
        )
        motion_gen = MotionGen(cfg)
        if bool(mc.CUROBO_MOTION_WARMUP):
            motion_gen.warmup()
        return motion_gen

    @staticmethod
    def _plan_config(*, check_start_validity: bool = True) -> MotionGenPlanConfig:
        return MotionGenPlanConfig(
            max_attempts=int(mc.CUROBO_MOTION_ATTEMPTS),
            timeout=float(mc.CUROBO_MOTION_TIMEOUT),
            enable_graph=bool(mc.CUROBO_MOTION_ENABLE_GRAPH),
            enable_graph_attempt=(
                1 if bool(mc.CUROBO_MOTION_ENABLE_GRAPH) else None
            ),
            enable_opt=True,
            partial_ik_opt=False,
            success_ratio=float(mc.CUROBO_MOTION_SUCCESS_RATIO),
            check_start_validity=bool(check_start_validity),
        )

    def _update_world(self, world_cfg) -> None:
        if self.motion_gen is None:
            self.motion_gen = self._make_motion_gen(world_cfg)
        else:
            self.motion_gen.update_world(world_cfg)
        if self.contact_solver is None:
            self.contact_solver = make_ik_solver(
                robot_config=self.robot["robot_config"],
                tensor_args=self.tensor_args,
                world_cfg=world_cfg,
            )
        else:
            self.contact_solver.update_world(world_cfg)
        _device_sync(self.tensor_args)

    def update_scene(self, query: dict[str, Any]) -> int:
        """Synchronize MotionGen and contact IK with one stable scene query."""
        world_cfg, obstacle_count = query_to_curobo_world(query)
        self._update_world(world_cfg)
        return int(obstacle_count)

    @staticmethod
    def _q_in_names(
        q: np.ndarray,
        *,
        source_names: tuple[str, ...],
        target_names: tuple[str, ...],
    ) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(source_names):
            raise ValueError("Current arm q length does not match source joint names.")
        by_name = {name: float(values[i]) for i, name in enumerate(source_names)}
        missing = [name for name in target_names if name not in by_name]
        if missing:
            raise ValueError(f"Current arm state is missing cuRobo joints: {missing}")
        return np.asarray([by_name[name] for name in target_names], dtype=np.float64)

    def _joint_state(
        self,
        q: np.ndarray,
        joint_names: tuple[str, ...],
    ) -> JointState:
        values = np.ascontiguousarray(
            np.asarray(q, dtype=np.float32).reshape(1, -1)
        )
        return JointState.from_position(
            self.tensor_args.to_device(values),
            joint_names=list(joint_names),
        )

    @staticmethod
    def _path_from_result(result, *, label: str) -> np.ndarray:
        try:
            path = (
                result.get_interpolated_plan()
                .position.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            path = path.reshape(-1, path.shape[-1])
        except Exception as exc:
            raise MotionPlanUnavailableError(
                f"MotionGen returned no usable {label} path: {exc}"
            ) from exc
        if path.shape[0] < 2:
            raise MotionPlanUnavailableError(
                f"MotionGen returned a degenerate {label} path."
            )
        return np.ascontiguousarray(path)

    def _contact_path(
        self,
        *,
        contact_position_b: np.ndarray,
        contact_quat_wxyz_b: np.ndarray,
        q_approach: np.ndarray,
        joint_names: tuple[str, ...],
    ) -> tuple[np.ndarray | None, str]:
        solver_names = tuple(
            str(name)
            for name in getattr(self.contact_solver, "joint_names", joint_names)
        )
        q_solver = self._q_in_names(
            q_approach,
            source_names=joint_names,
            target_names=solver_names,
        ).astype(np.float32)
        goal = Pose(
            position=self.tensor_args.to_device(
                np.ascontiguousarray(
                    np.asarray(contact_position_b, dtype=np.float32).reshape(1, 3)
                )
            ),
            quaternion=self.tensor_args.to_device(
                np.ascontiguousarray(
                    np.asarray(contact_quat_wxyz_b, dtype=np.float32).reshape(1, 4)
                )
            ),
        )
        result = self.contact_solver.solve_single(
            goal,
            retract_config=self.tensor_args.to_device(q_solver.reshape(1, -1)),
            seed_config=self.tensor_args.to_device(q_solver.reshape(1, 1, -1)),
            return_seeds=1,
            num_seeds=max(1, int(mc.CUROBO_IK_NUM_SEEDS)),
            use_nn_seed=False,
        )
        _device_sync(self.tensor_args)
        status = str(getattr(result, "status", "contact_ik"))
        if not _success_scalar(result.success):
            return None, status

        raw = (
            result.solution.reshape(-1, result.solution.shape[-1])[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        by_name = {name: float(raw[i]) for i, name in enumerate(solver_names)}
        q_contact = np.asarray([by_name[name] for name in joint_names], dtype=np.float64)
        limit_index = {name: i for i, name in enumerate(mc.ARM_JOINT_NAMES)}
        adjusted = np.zeros_like(q_contact)
        for joint_i, name in enumerate(joint_names):
            if name not in limit_index:
                return None, f"missing joint limit for {name}"
            cfg_i = int(limit_index[name])
            lower = float(mc.ARM_LOWER_LIMITS[cfg_i])
            upper = float(mc.ARM_UPPER_LIMITS[cfg_i])
            raw_q = float(q_contact[joint_i])
            reference_q = float(q_approach[joint_i])
            k_min = int(np.ceil((lower - raw_q) / (2.0 * np.pi)))
            k_max = int(np.floor((upper - raw_q) / (2.0 * np.pi)))
            if k_min > k_max:
                return None, f"no limited contact branch for {name}"
            candidates = raw_q + 2.0 * np.pi * np.arange(k_min, k_max + 1)
            adjusted[joint_i] = candidates[
                int(np.argmin(np.abs(candidates - reference_q)))
            ]

        steps = max(2, int(tc.APPROACH_CONTACT_INTERPOLATION_STEPS))
        alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64).reshape(-1, 1)
        path = (
            (1.0 - alpha) * np.asarray(q_approach, dtype=np.float64).reshape(1, -1)
            + alpha * adjusted.reshape(1, -1)
        )
        return np.ascontiguousarray(path), status

    def plan_target(
        self,
        *,
        query: dict[str, Any],
        target: SelectedTarget,
        unload_adjustment: UnloadAdjustment,
        current_q: np.ndarray,
        current_joint_names: tuple[str, ...],
    ) -> MotionPlan:
        box_index = int(target.box_index)
        proxy_mask = np.asarray(query["proxy_action_mask"], dtype=bool).reshape(-1)
        if box_index < 0 or box_index >= proxy_mask.shape[0]:
            raise ValueError(f"box_{box_index} is outside the current scene.")
        if not bool(proxy_mask[box_index]):
            raise MotionPlanUnavailableError(
                f"box_{box_index} left the current proxy action mask."
            )
        if float(unload_adjustment.distance_m) <= 0.0:
            raise ValueError("Unload distance must be positive.")

        world_cfg, _obstacle_count = query_to_curobo_world(query)
        self._update_world(world_cfg)
        if self.motion_gen is None:
            raise RuntimeError("MotionGen failed to initialize.")

        joint_names = tuple(str(name) for name in self.motion_gen.joint_names)
        start_q = self._q_in_names(
            current_q,
            source_names=current_joint_names,
            target_names=joint_names,
        ).astype(np.float32)
        start_state = self._joint_state(start_q, joint_names)

        contact_pos_w = np.asarray(target.position_w, dtype=np.float64).reshape(3)
        contact_quat_wxyz = np.asarray(target.quat_wxyz_w, dtype=np.float64).reshape(4)
        normal_w = normalized(target.normal_w, name="selected target normal")
        approach_pos_w = contact_pos_w + normal_w * float(
            tc.GRASP_APPROACH_CLEARANCE
        )
        quat_wxyz_b = world_to_base_quat_wxyz(
            query,
            contact_quat_wxyz[[1, 2, 3, 0]],
        ).astype(np.float32)
        approach_goal = Pose(
            position=self.tensor_args.to_device(
                np.ascontiguousarray(
                    world_to_base_position(query, approach_pos_w).reshape(1, 3),
                    dtype=np.float32,
                )
            ),
            quaternion=self.tensor_args.to_device(
                np.ascontiguousarray(quat_wxyz_b.reshape(1, 4), dtype=np.float32)
            ),
        )
        approach_result = self.motion_gen.plan_single(
            start_state,
            approach_goal,
            self._plan_config(),
        )
        _device_sync(self.tensor_args)
        approach_status = str(getattr(approach_result, "status", "approach"))
        if not _success_scalar(approach_result.success):
            raise MotionPlanUnavailableError(
                f"Approach planning rejected box_{box_index}: {approach_status}"
            )
        forward_path = self._path_from_result(
            approach_result,
            label="approach",
        )

        contact_path, contact_status = self._contact_path(
            contact_position_b=world_to_base_position(query, contact_pos_w),
            contact_quat_wxyz_b=quat_wxyz_b,
            q_approach=forward_path[-1],
            joint_names=joint_names,
        )
        if contact_path is None:
            raise MotionPlanUnavailableError(
                f"Contact IK rejected box_{box_index}: {contact_status}"
            )

        unload_direction = unload_direction_w(target, unload_adjustment)
        unload_target_w = unload_target_position_w(target, unload_adjustment)
        unload_goal = Pose(
            position=self.tensor_args.to_device(
                np.ascontiguousarray(
                    world_to_base_position(query, unload_target_w).reshape(1, 3),
                    dtype=np.float32,
                )
            ),
            quaternion=self.tensor_args.to_device(
                np.ascontiguousarray(quat_wxyz_b.reshape(1, 4), dtype=np.float32)
            ),
        )
        contact_state = self._joint_state(contact_path[-1], joint_names)
        attached = False
        try:
            attached = bool(
                self.motion_gen.attach_objects_to_robot(
                    contact_state,
                    [f"box_{box_index}"],
                    surface_sphere_radius=0.001,
                    remove_obstacles_from_world_config=False,
                )
            )
            if not attached:
                raise MotionPlanUnavailableError(
                    f"Could not attach box_{box_index} to the cuRobo robot model."
                )

            unload_result = self.motion_gen.plan_single(
                contact_state,
                unload_goal,
                self._plan_config(check_start_validity=False),
            )
            _device_sync(self.tensor_args)
            unload_status = str(getattr(unload_result, "status", "unload"))
            if not _success_scalar(unload_result.success):
                raise MotionPlanUnavailableError(
                    f"Unload target rejected box_{box_index}: {unload_status}"
                )
            unload_path = self._path_from_result(unload_result, label="unload")

            return_start = self._joint_state(unload_path[-1], joint_names)
            home_q = self._q_in_names(
                np.asarray(mc.ARM_HOME_Q, dtype=np.float64),
                source_names=tuple(str(name) for name in mc.ARM_JOINT_NAMES),
                target_names=joint_names,
            )
            home_state = self._joint_state(home_q, joint_names)
            return_result = self.motion_gen.plan_single_js(
                return_start,
                home_state,
                self._plan_config(),
            )
            _device_sync(self.tensor_args)
            return_status = str(getattr(return_result, "status", "return"))
            if not _success_scalar(return_result.success):
                raise MotionPlanUnavailableError(
                    f"Return-to-rest rejected box_{box_index}: {return_status}"
                )
            return_path = self._path_from_result(
                return_result,
                label="return-to-rest",
            )
        finally:
            try:
                if attached:
                    self.motion_gen.detach_object_from_robot()
            finally:
                self.motion_gen.world_coll_checker.enable_obstacle(
                    name=f"box_{box_index}",
                    enable=True,
                )
                _device_sync(self.tensor_args)

        adjustment = target.adjustment
        status = (
            f"frame={target.candidate.frame_index} roll={target.candidate.roll_index} "
            f"approach={approach_status} contact={contact_status} "
            f"unload={unload_status} return={return_status} "
            f"unload_distance_m={float(unload_adjustment.distance_m):.4f}"
        )
        return MotionPlan(
            box_index=box_index,
            frame_index=int(target.candidate.frame_index),
            roll_index=int(target.candidate.roll_index),
            joint_names=joint_names,
            forward_q_path=forward_path,
            contact_q_path=contact_path,
            unload_q_path=unload_path,
            return_q_path=return_path,
            contact_position_w=contact_pos_w.copy(),
            contact_quat_wxyz_w=contact_quat_wxyz.copy(),
            unload_direction_w=unload_direction.copy(),
            unload_distance_m=float(unload_adjustment.distance_m),
            unload_target_position_w=unload_target_w.copy(),
            unload_target_quat_wxyz_w=contact_quat_wxyz.copy(),
            attempted_targets=1,
            status=(
                f"{status} position_offsets_m=({adjustment.tangent_u_m:.4f},"
                f"{adjustment.tangent_v_m:.4f},{adjustment.normal_m:.4f})"
            ),
        )


__all__ = ["SelectedTargetMotionPlanner"]
