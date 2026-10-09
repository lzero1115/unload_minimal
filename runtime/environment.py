"""Scene state, geometric candidates and metrics, without a training API."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import mujoco
import numpy as np

from configs import scene_configs as sc, task_configs as tc
from configs.env_configs import DEFAULT_ENV_CONFIG, UnloadEnvConfig
from candidate_mask import compute_box_candidate_mask_rays
from sim_wrapper import SimWrapper
from unload_metric import InterSteadyMetricTracker
from .geometry import (
    _box_body_id, _compute_active_box_motion_status, _disable_box_collision,
    _write_box_root_pose_vel, _quat_wxyz_normalize, _collect_box_geometry,
    _collect_static_obstacles, _proxy_face_candidates, _generate_proxy_frame_targets,
)


class UnloadEnvironment:
    def __init__(self, scene_path: str | Path, config: UnloadEnvConfig = DEFAULT_ENV_CONFIG):
        if config.max_settle_frames < 1:
            raise ValueError("max_settle_frames must be positive")
        self.config = config
        for field in fields(config):
            setattr(self, field.name, getattr(config, field.name))
        self._scene_path = Path(scene_path).expanduser().resolve()
        self.sim = SimWrapper(self._scene_path, enable_box_mocaps=True)
        if self.sim.n_boxes < 1:
            raise ValueError("The scene must contain at least one box")
        self._active_mask = np.ones(self.sim.n_boxes, dtype=bool)
        self._graveyard_pos_arr = np.asarray(config.graveyard_pos, dtype=np.float64)
        self.metric = InterSteadyMetricTracker(
            half_extents=self.sim.half_extents,
            alpha_energy=tc.METRIC_ALPHA_ENERGY, beta_pose=tc.METRIC_BETA_POSE,
            lambda_trans=tc.METRIC_LAMBDA_TRANS, lambda_rot=tc.METRIC_LAMBDA_ROT,
        )

    def _sim(self):
        return self.sim

    def _metric(self):
        return self.metric

    def _scene_active_mask(self):
        return self._active_mask

    def _pin_inactive_boxes(self):
        for i in np.flatnonzero(~self._active_mask):
            _write_box_root_pose_vel(
                model=self.sim.model, data=self.sim.data, box_index=int(i),
                pos=self._graveyard_pos_arr, quat_wxyz=np.array([1., 0., 0., 0.]), zero_vel=True,
            )

    def _active_box_motion_status(self):
        return _compute_active_box_motion_status(
            model=self.sim.model, data=self.sim.data, active_mask=self._active_mask,
        )

    def _box_com_positions(self):
        return np.asarray([
            self.sim.data.xpos[_box_body_id(model=self.sim.model, box_index=i)]
            for i in range(self.sim.n_boxes)
        ], dtype=np.float64)

    def _set_box_mocap_pose(self, box_index, pos, quat_wxyz):
        bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{box_index}_anchor")
        if bid < 0 or self.sim.model.body_mocapid[bid] < 0:
            raise RuntimeError(f"Missing mocap anchor for box {box_index}")
        mid = self.sim.model.body_mocapid[bid]
        self.sim.data.mocap_pos[mid] = pos
        self.sim.data.mocap_quat[mid] = _quat_wxyz_normalize(quat_wxyz)

    def _set_box_weld_active(self, box_index, active):
        eqid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"box_{box_index}_pin")
        if eqid < 0:
            raise RuntimeError(f"Missing weld for box {box_index}")
        self.sim.data.eq_active[eqid] = bool(active)
        self.sim.forward()

    def _deactivate_box(self, box_index):
        sim = self.sim
        if not 0 <= box_index < self.sim.n_boxes or not self._active_mask[box_index]:
            raise ValueError(f"Box {box_index} is not active")
        self._set_box_weld_active(box_index, False)
        bid = _box_body_id(model=self.sim.model, box_index=box_index)
        self.sim.data.xfrc_applied[bid] = 0.0
        self._active_mask[box_index] = False
        _disable_box_collision(model=sim.model, box_index=box_index)
        self._pin_inactive_boxes()
        self.sim.forward()

    def _active_ray_side_walls(self):
        mode = self.ray_side_wall
        if mode in (0, 1, 2, 3):
            return (mode,)
        if mode == -1:
            return (0, 1, 2, 3)
        if mode != -2:
            raise ValueError("ray_side_wall must be -2, -1 or 0..3")
        removed = tuple(sorted(self.sim.snapshot["walls_removed"]))
        return removed or tuple(sc.DEFAULT_REMOVED_WALLS) or (2,)

    def _compute_candidates(self):
        sim = self.sim
        sim.forward()
        active = self._active_mask
        centers, rotations, half_extents = _collect_box_geometry(sim)
        self.ray_mask, self.ray_debug = compute_box_candidate_mask_rays(
            model=sim.model, data=sim.data, n_boxes=sim.n_boxes, half_extents=half_extents,
            side_wall=self.ray_side_wall, side_walls=self._active_ray_side_walls(),
            active_mask=active, candidate_eligible_mask=active,
            grid_step=self.ray_grid_step, plane_margin=self.ray_plane_margin,
            min_ray_hits=self.ray_min_hits, return_debug=True, assume_forwarded=True,
        )
        ray_faces = self.ray_debug.get("face_candidate_mask")
        if ray_faces is None:
            ray_faces = np.zeros((sim.n_boxes, 3, 2), dtype=bool)
            ray_faces[np.asarray(self.ray_mask, dtype=bool)] = True
        self._action_mask, face_mask, metadata, debug = _proxy_face_candidates(
            centers=centers, rotations=rotations, half_extents=half_extents,
            active_mask=active, ray_face_mask=ray_faces,
            static_obstacles=_collect_static_obstacles(sim),
            clearance=self.proxy_face_clearance, origin_margin=self.proxy_face_origin_margin,
            min_openness_score=self.proxy_min_openness_score, candidate_eligible_mask=active,
        )
        self.ray_debug["proxy"] = debug
        self.frame_targets = _generate_proxy_frame_targets(
            centers=centers, rotations=rotations, half_extents=half_extents,
            face_mask=face_mask, surface_clearance=self.surface_clearance, face_metadata=metadata,
        )

    def export_candidate_frame_query(self):
        sim = self.sim
        centers, _, _ = _collect_box_geometry(sim)
        quats = np.asarray([
            sim.data.xquat[_box_body_id(model=sim.model, box_index=i)]
            for i in range(sim.n_boxes)
        ], dtype=np.float64)
        return {
            "scene_npz": str(self._scene_path), "n_boxes": sim.n_boxes,
            "robot_base_pos_w": np.asarray(sc.ROBOT_BASE_POS, dtype=np.float64),
            "robot_base_quat_wxyz": np.asarray(sc.ROBOT_BASE_QUAT_WXYZ, dtype=np.float64),
            "table_pos_w": np.asarray(sim.table_pos), "table_half_extents": np.asarray(sim.table_half_extents),
            "wall_defs": tuple((i, np.asarray(p), np.asarray(h)) for i, p, h in sim.wall_defs),
            "box_centers_w": centers.copy(), "box_quats_wxyz": quats.copy(),
            "box_half_extents": sim.half_extents.copy(), "active_mask": self._active_mask.copy(),
            "proxy_action_mask": self._action_mask.copy(), "ray_mask": self.ray_mask.copy(),
            "frame_targets": {k: v.copy() for k, v in self.frame_targets.items()},
            "ray_debug": dict(self.ray_debug),
        }

    def close(self):
        self.sim.running = False
