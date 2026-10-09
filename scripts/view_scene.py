from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

_TASK_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_ROOT = Path(__file__).resolve().parent
_CONFIG_ROOT = _TASK_ROOT / "configs"
_UTILS_ROOT = _TASK_ROOT / "utils"
for _p in (_SCRIPT_ROOT, _CONFIG_ROOT, _UTILS_ROOT, _TASK_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import scene_configs as sc
import phys_configs as pc
from robot_visual import load_robot_visual_model, robot_link_transforms
from scene_utils import (
    COLOR_BOX_INSTANT_SETTLED,
    COLOR_BOX_INSTANT_UNSETTLED,
    COLOR_TABLE,
    COLOR_WALL,
    CUBE_FACES,
    axis_aligned_box_vertices,
    box_vertices_world,
)
from sim_wrapper import SimWrapper
from settle_utils import ComDisplacementSettleTracker
from viewer_utils import register_robot_meshes, update_robot_meshes


class SceneViewer:
    """Polyscope presentation for a MuJoCo scene.

    Simulation state and robot control live in SimWrapper. This class only owns
    visual assets and UI actions that call the wrapper.
    """

    def __init__(
        self,
        sim: SimWrapper,
        *,
        enable_com_settle: bool = bool(pc.ENABLE_COM_SETTLE),
        com_settle_max_displacement: float = float(pc.COM_SETTLE_MAX_DISPLACEMENT),
        com_settle_check_interval: int = int(pc.COM_SETTLE_CHECK_INTERVAL),
        com_settle_required_checks: int = int(pc.COM_SETTLE_REQUIRED_CHECKS),
    ) -> None:
        self.sim = sim
        self.velocity_settle_consecutive = 0
        self.velocity_settled = False
        self._last_velocity_check_frame = -1
        self.scene_settled = False
        self.settle_criterion = ""
        self.com_settle_tracker = ComDisplacementSettleTracker(
            enabled=enable_com_settle,
            max_displacement=com_settle_max_displacement,
            check_interval=com_settle_check_interval,
            required_checks=com_settle_required_checks,
        )
        urdf_path = Path(sc.ROBOT_URDF_PATH)
        asset_roots = [
            Path(sc.ROBOT_ASSET_ROOT_PATH),
            Path(sc.ROBOT_ROOT_PATH),
            urdf_path.parent,
        ]
        self.robot_chain, self.robot_mesh_records = load_robot_visual_model(
            urdf_path=urdf_path,
            asset_roots=asset_roots,
        )

    def robot_link_transforms(self) -> dict[str, np.ndarray]:
        return robot_link_transforms(
            model=self.sim.model,
            data=self.sim.data,
            chain=self.robot_chain,
            base_pos=tuple(float(v) for v in sc.ROBOT_BASE_POS),
            base_quat_wxyz=tuple(float(v) for v in sc.ROBOT_BASE_QUAT_WXYZ),
            robot_index=0,
        )

    def reset_settle_diagnostics(self) -> None:
        self.velocity_settle_consecutive = 0
        self.velocity_settled = False
        self._last_velocity_check_frame = -1
        self.scene_settled = False
        self.settle_criterion = ""
        self.com_settle_tracker.reset()

    def update_settle_diagnostics(self) -> None:
        frame = int(self.sim.frame_count)
        if (
            frame > 0
            and frame % int(pc.SETTLE_CHECK_INTERVAL) == 0
            and frame != self._last_velocity_check_frame
        ):
            self._last_velocity_check_frame = frame
            if int(np.count_nonzero(self.sim.last_unsettled)) != 0:
                self.velocity_settle_consecutive = 0
            else:
                self.velocity_settle_consecutive += 1
            self.velocity_settled = (
                self.velocity_settle_consecutive >= int(pc.SETTLE_STEPS)
            )

        if self.com_settle_tracker.enabled:
            positions = np.asarray(
                [
                    self.sim.data.xpos[
                        mujoco.mj_name2id(
                            self.sim.model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            f"box_{box_index}",
                        )
                    ]
                    for box_index in range(int(self.sim.n_boxes))
                ],
                dtype=np.float64,
            )
            self.com_settle_tracker.update(
                frame_index=frame,
                positions=positions,
            )
        if self.scene_settled:
            return
        if not self.velocity_settled and not self.com_settle_tracker.settled:
            return

        self.scene_settled = True
        if self.velocity_settled and self.com_settle_tracker.settled:
            self.settle_criterion = "velocity+com"
        elif self.com_settle_tracker.settled:
            self.settle_criterion = "com"
        else:
            self.settle_criterion = "velocity"
        self.sim.canonicalize_dynamic_state()
        self.sim.running = False
        self.sim.forward()


def run_polyscope(viewer: SceneViewer) -> None:
    sim = viewer.sim

    ps.init()
    ps.set_program_name("view_scene")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    pile_center = np.mean(sim.centers, axis=0)
    ps.look_at((1.1, -1.5, 1.0), tuple(float(v) for v in pile_center))

    table_mesh = ps.register_surface_mesh(
        "table",
        axis_aligned_box_vertices(sim.table_pos, sim.table_half_extents),
        CUBE_FACES,
    )
    table_mesh.set_color(COLOR_TABLE)
    table_mesh.set_transparency(0.55)

    for wi, pos, half in sim.wall_defs:
        wall_mesh = ps.register_surface_mesh(
            f"wall_{wi}",
            axis_aligned_box_vertices(pos, half),
            CUBE_FACES,
        )
        wall_mesh.set_color(COLOR_WALL)
        wall_mesh.set_transparency(0.45)

    box_handles = []
    for i in range(sim.n_boxes):
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
        verts = box_vertices_world(sim.data.xpos[bid], sim.data.xquat[bid], sim.half_extents[i])
        mesh = ps.register_surface_mesh(f"box_{i}", verts, CUBE_FACES)
        mesh.set_color(COLOR_BOX_INSTANT_SETTLED)
        box_handles.append(mesh)

    register_robot_meshes(viewer.robot_mesh_records, viewer.robot_link_transforms())

    def callback() -> None:
        status = "running" if sim.running else "paused"
        psim.TextUnformatted(
            f"status={status} boxes={sim.n_boxes} frame={sim.frame_count}/{sim.max_frames} "
            f"sim_time={sim.sim_time:.3f}s"
        )
        psim.TextUnformatted(
            f"max|v|={sim.last_max_lin:.5f} max|w|={sim.last_max_ang:.5f}"
        )
        com_tracker = viewer.com_settle_tracker
        psim.TextUnformatted(
            f"velocity stable={viewer.velocity_settle_consecutive}/{pc.SETTLE_STEPS} "
            f"com={'on' if com_tracker.enabled else 'off'} "
            f"stable={com_tracker.consecutive}/{com_tracker.required_checks} "
            f"max_drift={com_tracker.last_max_displacement:.5f}m "
            f"result={'stable' if viewer.scene_settled else 'waiting'}"
        )
        if viewer.scene_settled:
            psim.TextUnformatted(
                f"settled_by={viewer.settle_criterion}; simulation frozen"
            )
        psim.TextUnformatted(f"scene={sim.snapshot['path']}")
        if sim.snapshot["metadata"]:
            meta = sim.snapshot["metadata"]
            psim.TextUnformatted(
                f"seed={meta.get('partition_seed', 'n/a')} raw={meta.get('raw_partition_count', 'n/a')} "
                f"length_filter={meta.get('length_filter', 'n/a')}"
            )
        psim.Separator()

        if psim.Button("Reset"):
            sim.reset()
            viewer.reset_settle_diagnostics()
        if psim.Button("Step once"):
            if not viewer.scene_settled:
                was_running = sim.running
                sim.running = True
                sim.step_frame()
                sim.running = was_running
        if psim.Button("Pause / run"):
            if not viewer.scene_settled:
                sim.running = not sim.running

        sim.step_frame()
        sim.forward()
        viewer.update_settle_diagnostics()

        for i, mesh in enumerate(box_handles):
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
            verts = box_vertices_world(sim.data.xpos[bid], sim.data.xquat[bid], sim.half_extents[i])
            mesh.set_color(
                COLOR_BOX_INSTANT_SETTLED
                if viewer.scene_settled or int(sim.last_unsettled[i]) == 0
                else COLOR_BOX_INSTANT_UNSETTLED
            )
            mesh.update_vertex_positions(verts)

        update_robot_meshes(viewer.robot_mesh_records, viewer.robot_link_transforms())

    ps.set_user_callback(callback)
    ps.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a settled scene npz with the local Franka suction robot in MuJoCo."
    )
    parser.add_argument("scene_npz", type=str, help="Path to a saved scene .npz file.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum viewer frames to simulate after pressing run. 0 means unlimited.",
    )
    parser.add_argument(
        "--enable-com-settle",
        action=argparse.BooleanOptionalAction,
        default=bool(pc.ENABLE_COM_SETTLE),
        help="Enable bounded COM displacement settling (default: enabled).",
    )
    parser.add_argument(
        "--com-settle-max-displacement",
        type=float,
        default=float(pc.COM_SETTLE_MAX_DISPLACEMENT),
        help="Maximum COM displacement from the fixed reference state in meters.",
    )
    parser.add_argument(
        "--com-settle-check-interval",
        type=int,
        default=int(pc.COM_SETTLE_CHECK_INTERVAL),
        help="Frames between COM displacement checks.",
    )
    parser.add_argument(
        "--com-settle-required-checks",
        type=int,
        default=int(pc.COM_SETTLE_REQUIRED_CHECKS),
        help="Consecutive COM checks required after recording the reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = SimWrapper(args.scene_npz, max_frames=int(args.max_frames))
    viewer = SceneViewer(
        sim,
        enable_com_settle=bool(args.enable_com_settle),
        com_settle_max_displacement=float(args.com_settle_max_displacement),
        com_settle_check_interval=int(args.com_settle_check_interval),
        com_settle_required_checks=int(args.com_settle_required_checks),
    )
    print(
        f"[view_scene] scene={sim.snapshot['path']} boxes={sim.n_boxes} "
        f"nbody={sim.model.nbody} nq={sim.model.nq} nv={sim.model.nv} nu={sim.model.nu} "
        f"com_settle={bool(args.enable_com_settle)}"
    )
    run_polyscope(viewer)


if __name__ == "__main__":
    main()
