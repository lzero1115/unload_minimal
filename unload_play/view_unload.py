"""Load one scene and manually choose every removal."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "configs", ROOT / "utils"):
    sys.path.insert(0, str(directory))

from configs.env_configs import DEFAULT_ENV_CONFIG
from configs import motion_configs as mc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_npz", type=Path)
    parser.add_argument("--direct-remove", action="store_true",
                        help="Remove a selected IK candidate immediately; skip MotionGen, then settle and score.")
    parser.add_argument("--max-settle-frames", type=int, default=DEFAULT_ENV_CONFIG.max_settle_frames)
    parser.add_argument("--ik-batch-size", type=int, default=mc.CUROBO_IK_TARGET_BATCH_SIZE)
    parser.add_argument("--frames-per-render", type=int, default=1,
                        help="Simulation/control frames per display tick (default 1 at 30 Hz).")
    parser.add_argument("--export-dir", type=Path, default=ROOT / "exports",
                        help="Manual Blender export directory; relative paths use the project root.")
    parser.add_argument("--export-frame-stride", type=int, default=2,
                        help="Export every Nth recorded frame, retaining event boundaries; 1 keeps all.")
    args = parser.parse_args()
    if min(args.max_settle_frames, args.ik_batch_size, args.frames_per_render, args.export_frame_stride) < 1:
        parser.error("Frame limits and IK batch size must be positive")
    scene = args.scene_npz.expanduser().resolve()
    if not scene.is_file():
        parser.error(f"Scene not found: {scene}")
    export_dir = args.export_dir.expanduser()
    if not export_dir.is_absolute():
        export_dir = ROOT / export_dir

    from unload_play.controller import PlayController
    from unload_play.viewer import run_polyscope
    controller = PlayController(
        scene, direct_remove=args.direct_remove,
        config=replace(DEFAULT_ENV_CONFIG, max_settle_frames=args.max_settle_frames),
        ik_batch_size=args.ik_batch_size,
    )
    print(f"[unload_play] scene={scene}; mode={'direct-remove' if args.direct_remove else 'motiongen'}", flush=True)
    print("[unload_play] No input sequence. No automatic session save.", flush=True)
    print(f"[physics] table={controller.executor.sim.table_friction} "
          f"box={controller.executor.sim.box_friction}; waiting for manual pile placement / IK.", flush=True)
    try:
        run_polyscope(controller, frames_per_render=args.frames_per_render,
                      export_dir=export_dir, export_frame_stride=args.export_frame_stride)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
