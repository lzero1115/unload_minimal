from __future__ import annotations

import re

import mujoco
import numpy as np
import polyscope as ps
from scipy.spatial.transform import Rotation

from configs import scene_configs as sc
from robot_naming import robot_site_name

COLOR_CONFIRMED = (0.12, 0.80, 0.26)
COLOR_FRAME = (0.10, 0.75, 0.95)
COLOR_FRAME_UNREACHABLE = (0.52, 0.56, 0.60)
COLOR_FRAME_SELECTED = (0.95, 0.72, 0.08)
COLOR_UNLOAD_PATH = (0.92, 0.22, 0.55)
ROBOT_PALE_TURQUOISE = (187.0 / 255.0, 1.0, 1.0)
_FRAME_NAME = re.compile(
    r"^target_frame_(\d+)(?:_axis_(?:normal|tangent_u|tangent_v))?$"
)
_MAX_BOX_FACE_FRAMES = 6
_FRAME_AXIS_RADIUS = 0.0002
_UNLOAD_PATH_RADIUS = 0.00008
_FRAME_NORMAL_LENGTH_M = 0.03
_FRAME_TANGENT_LENGTH_M = 0.03


class PoseFrameOverlay:
    """A right-handed XYZ triad drawn from the actual rotation matrix columns."""

    def __init__(self, name, *, opacity, radius):
        self.axes = []
        for axis_name, color in zip(
            ("x", "y", "z"), ((0.95, 0.15, 0.12), (0.15, 0.80, 0.25), (0.15, 0.35, 1.0)),
        ):
            handle = ps.register_curve_network(
                f"{name}_{axis_name}", np.zeros((4, 3)),
                np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int32),
                enabled=False, color=color,
            )
            handle.set_radius(radius, relative=False)
            handle.set_transparency(opacity)
            self.axes.append(handle)
        self.origin = ps.register_point_cloud(
            f"{name}_origin", np.zeros((1, 3)), enabled=False,
            color=(0.95, 0.95, 0.95), point_render_mode="sphere",
        )
        self.origin.set_radius(radius * 1.5, relative=False)
        self.origin.set_transparency(opacity)

    def set_enabled(self, enabled):
        self.origin.set_enabled(enabled)
        for handle in self.axes:
            handle.set_enabled(enabled)

    def update(self, position, rotation, *, length, opacity):
        position = np.asarray(position, dtype=np.float64).reshape(3)
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.origin.update_point_positions(position.reshape(1, 3))
        self.origin.set_transparency(opacity)
        for axis, handle in enumerate(self.axes):
            direction = rotation[:, axis]
            tangent = rotation[:, (axis + 1) % 3]
            tip = position + length * direction
            shoulder = tip - 0.20 * length * direction
            nodes = np.stack((position, tip, shoulder + 0.08 * length * tangent,
                              shoulder - 0.08 * length * tangent))
            handle.update_node_positions(nodes)
            handle.set_transparency(opacity)
        self.set_enabled(True)


class EndEffectorFrameOverlay:
    """Compare the live MuJoCo EE with the selected contact goal used by planning."""

    def __init__(self):
        self.actual = PoseFrameOverlay("actual_ee_frame", opacity=1.0, radius=0.0008)
        self.target = PoseFrameOverlay("contact_ee_target_frame", opacity=0.45, radius=0.0012)
        self.show_actual = True
        self.show_target = True
        self.target_opacity = 0.45
        self.axis_length = 0.06

    def target_visible(self, controller):
        box = controller.selected_box
        return bool(not controller.replay_active and self.show_target and controller.selected_target is not None
                    and box is not None and controller.executor.active_mask[box])

    def update(self, controller, *, data=None):
        # The renderer has refreshed live dynamics or recorded-pose kinematics.
        # Both use the executor's EE site without an additional hand/tip offset.
        sim = controller.executor.sim
        pose_data = sim.data if data is None else data
        if self.show_actual:
            site_name = robot_site_name(0, str(sc.ROBOT_EE_LINK))
            site_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise RuntimeError(f"MuJoCo model is missing EE site {site_name}")
            self.actual.update(pose_data.site_xpos[site_id], pose_data.site_xmat[site_id],
                               length=self.axis_length, opacity=1.0)
        else:
            self.actual.set_enabled(False)
        if self.target_visible(controller):
            target = controller.selected_target
            # This is already the EE quaternion (including IK roll and manual
            # edits). Do not flip Z again or substitute the outward face normal.
            quat_wxyz = np.asarray(target.quat_wxyz_w, dtype=np.float64)
            rotation = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
            self.target.update(target.position_w, rotation,
                               length=self.axis_length, opacity=self.target_opacity)
        else:
            self.target.set_enabled(False)


class TargetFrameOverlay:
    """Pickable grasp frames plus the independently edited unload target."""

    def __init__(self) -> None:
        self.show_face_axes = False
        self.handles = []
        self.axis_handles = []
        for slot in range(_MAX_BOX_FACE_FRAMES):
            handle = ps.register_point_cloud(
                f"target_frame_{slot}",
                np.zeros((1, 3), dtype=np.float64),
                enabled=False,
                radius=0.0003,
                point_render_mode="sphere",
                color=COLOR_FRAME,
            )
            self.handles.append(handle)
            axes = (
                ps.register_curve_network(
                    f"target_frame_{slot}_axis_normal",
                    np.zeros((2, 3), dtype=np.float64),
                    np.asarray([[0, 1]], dtype=np.int32),
                    enabled=False,
                    radius=_FRAME_AXIS_RADIUS,
                    color=(0.92, 0.20, 0.16),
                ),
                ps.register_curve_network(
                    f"target_frame_{slot}_axis_tangent_u",
                    np.zeros((2, 3), dtype=np.float64),
                    np.asarray([[0, 1]], dtype=np.int32),
                    enabled=False,
                    radius=_FRAME_AXIS_RADIUS,
                    color=(0.15, 0.75, 0.28),
                ),
                ps.register_curve_network(
                    f"target_frame_{slot}_axis_tangent_v",
                    np.zeros((2, 3), dtype=np.float64),
                    np.asarray([[0, 1]], dtype=np.int32),
                    enabled=False,
                    radius=_FRAME_AXIS_RADIUS,
                    color=(0.16, 0.35, 0.95),
                ),
            )
            self.axis_handles.append(axes)
        self.unload_path_handle = ps.register_curve_network(
            "selected_unload_path",
            np.zeros((2, 3), dtype=np.float64),
            np.asarray([[0, 1]], dtype=np.int32),
            enabled=False,
            radius=_UNLOAD_PATH_RADIUS,
            color=COLOR_UNLOAD_PATH,
        )
        self.unload_target_handle = ps.register_point_cloud(
            "selected_unload_target",
            np.zeros((1, 3), dtype=np.float64),
            enabled=False,
            radius=0.00045,
            point_render_mode="sphere",
            color=COLOR_UNLOAD_PATH,
        )

    def update(self, controller) -> None:
        visible = bool(
            controller.interaction_available and controller.selected_box is not None
        )
        frames = controller.selected_frames if visible else ()
        for slot, handle in enumerate(self.handles):
            if slot >= len(frames):
                handle.set_enabled(False)
                for axis_handle in self.axis_handles[slot]:
                    axis_handle.set_enabled(False)
                continue
            candidate = frames[slot]
            if controller.selected_frame_slot == slot:
                position = candidate.adjusted_position(controller.adjustment)
                normal = candidate.adjusted_normal(controller.adjustment)
                tangent_u = candidate.adjusted_tangent_u(controller.adjustment)
                tangent_v = candidate.adjusted_tangent_v(controller.adjustment)
                color = COLOR_FRAME_SELECTED
            else:
                position = np.asarray(candidate.position_w, dtype=np.float64)
                normal = np.asarray(candidate.normal_w, dtype=np.float64)
                tangent_u = np.asarray(candidate.tangent_u_w, dtype=np.float64)
                tangent_v = np.asarray(candidate.tangent_v_w, dtype=np.float64)
                color = (
                    COLOR_FRAME
                    if candidate.initially_ik_reachable
                    else COLOR_FRAME_UNREACHABLE
                )
            handle.update_point_positions(position.reshape(1, 3))
            handle.set_color(color)
            handle.set_enabled(True)
            axis_specs = (
                (normal, _FRAME_NORMAL_LENGTH_M),
                (tangent_u, _FRAME_TANGENT_LENGTH_M),
                (tangent_v, _FRAME_TANGENT_LENGTH_M),
            )
            for axis_handle, (direction, length_m) in zip(
                self.axis_handles[slot], axis_specs
            ):
                endpoint = position + np.asarray(direction, dtype=np.float64) * float(
                    length_m
                )
                axis_handle.update_node_positions(np.stack((position, endpoint)))
                axis_handle.set_enabled(self.show_face_axes)

        selected_target = controller.selected_target if visible else None
        unload_target = (
            controller.current_unload_target_position_w
            if selected_target is not None
            else None
        )
        if selected_target is None or unload_target is None:
            self.unload_path_handle.set_enabled(False)
            self.unload_target_handle.set_enabled(False)
            return
        origin = np.asarray(selected_target.position_w, dtype=np.float64)
        endpoint = np.asarray(unload_target, dtype=np.float64)
        self.unload_path_handle.update_node_positions(np.stack((origin, endpoint)))
        self.unload_path_handle.set_enabled(True)
        self.unload_target_handle.update_point_positions(endpoint.reshape(1, 3))
        self.unload_target_handle.set_enabled(True)


def picked_frame_slot(pick_result) -> int | None:
    if not bool(getattr(pick_result, "is_hit", False)):
        return None
    match = _FRAME_NAME.fullmatch(str(getattr(pick_result, "structure_name", "")))
    return None if match is None else int(match.group(1))
