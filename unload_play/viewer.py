from __future__ import annotations

import re
import time
from pathlib import Path

import mujoco
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from configs import scene_configs as sc
from robot_visual import load_robot_visual_model, robot_link_transforms
from scene_utils import (
    CUBE_FACES, axis_aligned_box_vertices, box_vertices_world,
    COLOR_TABLE, COLOR_WALL, COLOR_BOX_INSTANT_SETTLED, COLOR_BOX_INSTANT_UNSETTLED,
)
from viewer_utils import register_robot_meshes, update_robot_meshes
from .overlay import EndEffectorFrameOverlay, TargetFrameOverlay, picked_frame_slot
from .target_types import DecisionPhase

BLUE = (0.12, 0.42, 0.95)
GREEN = (0.12, 0.80, 0.26)
ROBOT_WHITE = (0.92, 0.93, 0.94)
ROBOT_PALE_TURQUOISE = (187.0 / 255.0, 1.0, 1.0)


class PlayRenderer:
    def __init__(self, controller):
        self.controller = controller
        sim = controller.executor.sim
        table = ps.register_surface_mesh("table", axis_aligned_box_vertices(sim.table_pos, sim.table_half_extents), CUBE_FACES)
        table.set_color(COLOR_TABLE)
        table.set_transparency(0.55)
        for i, pos, half in sim.wall_defs:
            wall = ps.register_surface_mesh(f"wall_{i}", axis_aligned_box_vertices(pos, half), CUBE_FACES)
            wall.set_color(COLOR_WALL)
            wall.set_transparency(0.45)
        self.box_handles = []
        for i in range(sim.n_boxes):
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
            vertices = box_vertices_world(sim.data.xpos[bid], sim.data.xquat[bid], sim.half_extents[i])
            self.box_handles.append(ps.register_surface_mesh(f"box_{i}", vertices, CUBE_FACES))
        urdf = Path(sc.ROBOT_URDF_PATH)
        self.chain, self.robot_meshes = load_robot_visual_model(
            urdf_path=urdf, asset_roots=[sc.ROBOT_ASSET_ROOT_PATH, sc.ROBOT_ROOT_PATH, urdf.parent],
        )
        register_robot_meshes(self.robot_meshes, self._robot_transforms())
        for record in self.robot_meshes:
            match = re.search(r"panda_link(\d+)", str(record.get("link_name", "")).lower())
            color = ROBOT_WHITE if match is not None and int(match.group(1)) % 2 == 1 else ROBOT_PALE_TURQUOISE
            if record.get("handle") is not None:
                record["handle"].set_color(color)
        self.overlay = TargetFrameOverlay()
        self.ee_frames = EndEffectorFrameOverlay()
        self.transparent_selected_box = True
        self._last_qpos = None
        self._last_replay_frame = None
        self.update()

    def _robot_transforms(self, data=None):
        sim = self.controller.executor.sim
        return robot_link_transforms(
            model=sim.model, data=sim.data if data is None else data, chain=self.chain,
            base_pos=sc.ROBOT_BASE_POS, base_quat_wxyz=sc.ROBOT_BASE_QUAT_WXYZ, robot_index=0,
        )

    def robot_visual_world_transforms(self, qpos):
        """Evaluate recorded joints without modifying the live MuJoCo data."""
        sim = self.controller.executor.sim
        if getattr(self, "_export_model", None) is not sim.model:
            self._export_model = sim.model
            self._export_data = mujoco.MjData(sim.model)
        self._export_data.qpos[:] = qpos
        transforms = robot_link_transforms(
            model=sim.model, data=self._export_data, chain=self.chain,
            base_pos=sc.ROBOT_BASE_POS, base_quat_wxyz=sc.ROBOT_BASE_QUAT_WXYZ, robot_index=0,
        )
        return np.stack([transforms[record["link_name"]] for record in self.robot_meshes])

    def update(self):
        controller = self.controller
        if controller.replay_active:
            self._update_replay()
            return
        self._last_replay_frame = None
        executor = controller.executor
        sim = executor.sim
        sim.forward()
        active = executor.active_mask
        changed = self._last_qpos is None or not np.array_equal(self._last_qpos, sim.data.qpos)
        unsettled, _, _ = executor.env._active_box_motion_status()
        for i, mesh in enumerate(self.box_handles):
            mesh.set_enabled(bool(active[i]))
            if not active[i]:
                continue
            if changed:
                bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
                mesh.update_vertex_positions(box_vertices_world(sim.data.xpos[bid], sim.data.xquat[bid], sim.half_extents[i]))
            color = COLOR_BOX_INSTANT_SETTLED
            if controller.phase == DecisionPhase.SETTLING and unsettled[i]:
                color = COLOR_BOX_INSTANT_UNSETTLED
            elif controller.stable_for_input and controller.ik_mask[i]:
                color = BLUE
            if controller.selected_box == i and controller.phase != DecisionPhase.SETTLING:
                color = GREEN
            mesh.set_color(color)
            transparent = (self.transparent_selected_box and controller.selected_box == i
                           and self.ee_frames.target_visible(controller))
            mesh.set_transparency(0.35 if transparent else 1.0)
        if changed:
            update_robot_meshes(self.robot_meshes, self._robot_transforms())
            self._last_qpos = sim.data.qpos.copy()
        self.overlay.update(controller)
        self.ee_frames.update(controller)

    def _update_replay(self):
        controller = self.controller
        sim = controller.executor.sim
        frame = controller.replay_frame
        if getattr(self, "_replay_model", None) is not sim.model:
            self._replay_model = sim.model
            self._replay_data = mujoco.MjData(sim.model)
            self._last_replay_frame = None
        if self._last_replay_frame is not frame:
            for i, mesh in enumerate(self.box_handles):
                mesh.set_enabled(bool(frame.active_mask[i]))
                if not frame.active_mask[i]:
                    continue
                mesh.update_vertex_positions(box_vertices_world(
                    frame.box_positions_w[i], frame.box_quats_wxyz[i], sim.half_extents[i],
                ))
                carrying = (frame.current_action == i and frame.phase in {
                    "move_to_approach", "approach_to_contact", "attach",
                    "unload_with_physics", "return_to_rest_with_physics", "release_remove",
                })
                mesh.set_color(GREEN if carrying else COLOR_BOX_INSTANT_SETTLED)
                mesh.set_transparency(1.0)
            self._replay_data.qpos[:] = frame.qpos
            # Only update a separate display state's kinematics; never step or
            # overwrite the live simulation to seek through recorded frames.
            mujoco.mj_kinematics(sim.model, self._replay_data)
            update_robot_meshes(self.robot_meshes, self._robot_transforms(self._replay_data))
            self._last_replay_frame = frame
        self._last_qpos = None  # Restore every live mesh when leaving replay.
        self.overlay.update(controller)  # Hide live selection markers.
        self.ee_frames.update(controller, data=self._replay_data)


def _draw_replay_controls(controller):
    psim.Separator()
    if controller.can_replay and psim.Button("Replay recorded motion"):
        controller.enter_replay()
        ps.reset_selection()
    if not controller.replay_active:
        return
    psim.TextUnformatted("Recorded motion (read-only)")
    changed, value = psim.SliderInt(
        "Replay frame", controller.replay_index + 1, 1, len(controller.history),
    )
    if changed:
        controller.set_replay_index(value - 1)
    if psim.Button("Previous frame"):
        controller.set_replay_index(controller.replay_index - 1)
    psim.SameLine()
    if psim.Button("Next frame"):
        controller.set_replay_index(controller.replay_index + 1)
    if psim.Button("Play replay"):
        controller.play_replay()
    psim.SameLine()
    if psim.Button("Pause replay"):
        controller.pause_replay()
    frame = controller.replay_frame
    elapsed = controller.replay_index * controller.executor.sim.frame_dt
    psim.TextUnformatted(
        f"Frame {controller.replay_index + 1}/{len(controller.history)}; "
        f"timeline={elapsed:.3f}s; phase={frame.phase}"
    )
    psim.TextUnformatted(
        f"Recorded steps: {frame.completed_actions}; "
        f"remaining: {np.count_nonzero(frame.active_mask)}; metric: {frame.metric_total:.6f}"
    )
    psim.TextUnformatted("Playing" if controller.replay_playing else "Paused")
    if psim.Button("Exit replay"):
        controller.exit_replay()
        ps.reset_selection()


def _draw_frame_controls(renderer):
    frames = renderer.ee_frames
    psim.Separator()
    psim.TextUnformatted("EE frames: X red / Y green / Z blue")
    psim.TextUnformatted("Solid = actual EE; translucent = adjusted contact target")
    _, frames.show_actual = psim.Checkbox("Show actual EE frame", frames.show_actual)
    _, frames.show_target = psim.Checkbox("Show contact EE target", frames.show_target)
    _, renderer.overlay.show_face_axes = psim.Checkbox(
        "Show face N/U/V helpers", renderer.overlay.show_face_axes,
    )
    _, renderer.transparent_selected_box = psim.Checkbox(
        "Transparent selected box", renderer.transparent_selected_box,
    )
    changed, value = psim.SliderFloat("Target opacity", frames.target_opacity, 0.10, 0.80)
    if changed:
        frames.target_opacity = float(value)
    changed, value = psim.SliderFloat("EE axis length (m)", frames.axis_length, 0.02, 0.15)
    if changed:
        frames.axis_length = float(value)


def _draw_pile_controls(controller, translation_step_m, yaw_step_deg):
    psim.TextUnformatted("Initial pile placement")
    changed, value = psim.InputFloat(
        "Pile translation increment (m)", translation_step_m,
        step=0.001, step_fast=0.01, format="%.4f",
    )
    if changed and np.isfinite(value):
        translation_step_m = float(np.clip(value, 0.0001, 1.0))
    changed, value = psim.InputFloat(
        "Pile yaw increment (deg)", yaw_step_deg,
        step=0.5, step_fast=5.0, format="%.1f",
    )
    if changed and np.isfinite(value):
        yaw_step_deg = float(np.clip(value, 0.1, 180.0))
    values = {}
    for label, field, step, limit in (
        ("Pile X offset (m)", "translation_x_m", translation_step_m, 1.0),
        ("Pile Y offset (m)", "translation_y_m", translation_step_m, 1.0),
        ("Pile yaw about center (deg)", "yaw_deg", yaw_step_deg, 180.0),
    ):
        changed, value = psim.InputFloat(
            label, float(getattr(controller.pile_adjustment, field)),
            step=step, step_fast=step * 10.0, format="%.4f",
        )
        if changed:
            values[field] = float(np.clip(value, -limit, limit))
    if values:
        controller.set_pile_adjustment(**values)
    if psim.Button("Reset pile offsets"):
        controller.set_pile_adjustment(translation_x_m=0.0, translation_y_m=0.0, yaw_deg=0.0)
    psim.TextUnformatted("Placement changes clear candidates and plans; initial settling is not scored.")
    return translation_step_m, yaw_step_deg


def _draw_adjustments(controller):
    candidate = controller.selected_candidate
    if candidate is None:
        return
    psim.TextUnformatted("Grasp offsets: meters; rotations: degrees")
    for label, field, step in (
        ("Tangent U (m)", "tangent_u_m", 0.001),
        ("Tangent V (m)", "tangent_v_m", 0.001),
        ("Normal offset (m)", "normal_m", 0.001),
        ("Rotate U (deg)", "rotation_u_deg", 1.0),
        ("Rotate V (deg)", "rotation_v_deg", 1.0),
        ("Rotate normal (deg)", "rotation_normal_deg", 1.0),
    ):
        changed, value = psim.InputFloat(label, float(getattr(controller.adjustment, field)),
                                        step=step, step_fast=step * 10, format="%.4f")
        if changed:
            controller.set_adjustment(**{field: value})
    if psim.Button("Reset grasp offsets"):
        controller.select_frame(controller.selected_frame_slot)
    psim.TextUnformatted("Unload direction and distance")
    for label, field, step in (
        ("Unload tilt U (deg)", "tilt_u_deg", 1.0),
        ("Unload tilt V (deg)", "tilt_v_deg", 1.0),
        ("Unload distance (m)", "distance_m", 0.005),
    ):
        changed, value = psim.InputFloat(label, float(getattr(controller.unload_adjustment, field)),
                                        step=step, step_fast=step * 10, format="%.4f")
        if changed:
            controller.set_unload_adjustment(**{field: value})


def run_polyscope(controller, *, frames_per_render=1, export_dir=None, export_frame_stride=2):
    ps.init()
    ps.set_program_name("unload_minimal / unload_play")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_transparency_mode("pretty")
    ps.look_at((1.1, -1.4, 0.9), (0.0, 0.0, 0.2))
    renderer = PlayRenderer(controller)
    paused = False
    pile_translation_step_m = 0.001
    pile_yaw_step_deg = 1.0
    last_tick = time.perf_counter()
    if export_dir is None:
        export_dir = Path(__file__).resolve().parents[1] / "exports"
    export_stride = max(1, int(export_frame_stride))
    export_status = ""

    def callback():
        nonlocal paused, last_tick, pile_translation_step_m, pile_yaw_step_deg
        nonlocal export_stride, export_status
        now = time.perf_counter()
        if controller.replay_active:
            controller.advance_replay()
        elif not paused and now - last_tick >= controller.executor.sim.frame_dt:
            last_tick = now
            for _ in range(frames_per_render):
                controller.tick()
                if controller.phase not in (DecisionPhase.EXECUTING, DecisionPhase.SETTLING):
                    break
        stats = controller.executor.stats
        mode = "DIRECT REMOVE (IK candidates retained)" if controller.direct_remove else "MOTIONGEN"
        psim.TextUnformatted(f"Mode: {mode}")
        psim.TextUnformatted(f"Live state: {controller.phase.value}")
        psim.TextUnformatted(controller.executor.status_message)
        psim.TextUnformatted(f"Live steps: {stats.completed_actions}; remaining: {np.count_nonzero(controller.executor.active_mask)}")
        psim.TextUnformatted(f"Live step metric: {stats.last_metric_delta:.6f}; total: {stats.metric_total:.6f}")
        if controller._pending_job is not None:
            psim.TextUnformatted(f"Pending: {controller._pending_job} (planning can take time)")
        if controller.warning:
            psim.TextUnformatted(controller.warning)
        if not controller.replay_active:
            if psim.Button("Pause / resume"):
                paused = not paused
            psim.SameLine()
            psim.TextUnformatted("PAUSED" if paused else "running")

        try:
            _draw_replay_controls(controller)
            _draw_frame_controls(renderer)
            if controller.stable_for_input:
                psim.Separator()
                if controller.can_adjust_pile:
                    pile_translation_step_m, pile_yaw_step_deg = _draw_pile_controls(
                        controller, pile_translation_step_m, pile_yaw_step_deg,
                    )
                else:
                    psim.TextUnformatted("Initial pile placement locked. Reset the scene to adjust it again.")
                if controller.pile_scene_dirty or controller.executor.initializing:
                    candidate_label = "Settle pile and compute IK"
                else:
                    candidate_label = "Compute IK candidates" if controller.snapshot is None else "Recompute IK candidates"
                if psim.Button(candidate_label):
                    controller.refresh_candidates()
                if controller.snapshot is None:
                    psim.TextUnformatted("IK runs only when requested; use the button above.")
                psim.Separator()
                psim.TextUnformatted("Click a blue box in the scene to select it.")
                if controller.selected_box is not None:
                    psim.TextUnformatted(f"Selected box: {controller.selected_box}")
                if not controller.direct_remove:
                    psim.TextUnformatted("Click a face marker in the scene to select the grasp frame.")
                    frame = controller.selected_candidate
                    if frame is not None:
                        status = "IK" if frame.initially_ik_reachable else "unverified"
                        psim.TextUnformatted(
                            f"Selected face: {controller.selected_frame_slot}; "
                            f"axis={frame.face_axis} sign={frame.face_sign:+d}; {status}"
                        )
                if controller.direct_remove:
                    psim.TextUnformatted("Direct mode removes the selected box; grasp offsets are unused.")
                else:
                    _draw_adjustments(controller)
                    if controller.selected_target is not None and psim.Button("Validate with MotionGen"):
                        controller.request_plan()
                if controller.selected_box is not None:
                    label = "Remove selected box" if controller.direct_remove else "Execute validated motion"
                    if psim.Button(label):
                        controller.execute()
                io = psim.GetIO()
                if bool(io.MouseClicked[0]) and not bool(getattr(io, "WantCaptureMouse", False)):
                    pick = ps.pick(screen_coords=io.MousePos)
                    slot = picked_frame_slot(pick)
                    if slot is not None:
                        controller.select_frame(slot)
                    elif bool(getattr(pick, "is_hit", False)):
                        match = re.fullmatch(r"box_(\d+)", str(getattr(pick, "structure_name", "")))
                        if match:
                            controller.select_box(int(match.group(1)))
                    ps.reset_selection()

            psim.Separator()
            if controller.can_undo:
                if psim.Button("Undo last unload"):
                    controller.undo_last_unload()
                    renderer._last_qpos = None
            elif controller.checkpoints:
                psim.TextUnformatted("Undo requires live mode with execution / settling stopped.")
            psim.TextUnformatted(f"Undo steps available: {len(controller.checkpoints)}")

            psim.Separator()
            psim.TextUnformatted(f"Recorded motion frames (memory): {len(controller.history)}")
            changed, stride = psim.InputInt("Export frame stride", export_stride)
            if changed:
                export_stride = max(1, int(stride))
            if controller.can_export:
                if psim.Button("Export recorded motion"):
                    from .exporter import export_recorded_motion
                    controller.pause_replay()
                    export_status = ""
                    result = export_recorded_motion(
                        controller=controller, renderer=renderer,
                        output_root=export_dir, frame_stride=export_stride,
                    )
                    export_status = f"Exported {result.exported_frame_count} frames: {result.output_dir}"
            else:
                psim.TextUnformatted("Export is available after settling, while idle or complete.")
            if export_status:
                psim.TextUnformatted(export_status)

            if psim.Button("Reset to saved scene"):
                controller.reset()
                renderer._last_qpos = None
                paused = False
                export_status = ""
        except Exception as exc:
            controller.warning = f"{type(exc).__name__}: {exc}"
            print(f"[ui] {controller.warning}", flush=True)

        if controller.records:
            psim.Separator()
            psim.TextUnformatted("Recent step metrics (session only)")
            for record in controller.records[-8:]:
                psim.TextUnformatted(f"#{record['step']} box={record['box']} "
                                     f"delta={record['metric_delta']:.6f} total={record['metric_total']:.6f}")
        renderer.update()

    ps.set_user_callback(callback)
    ps.show()
