from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from scene_utils import COLOR_TABLE, COLOR_WALL
from .history import DynamicFrameHistory, RecordedFrame


CARDBOARD_BROWN = (0.72, 0.48, 0.23, 1.0)
SELECTED_GREEN = (0.12, 0.80, 0.26, 1.0)
_SELECTED_BOX_PHASES = {
    "move_to_approach",
    "approach_to_contact",
    "attach",
    "unload_with_physics",
    "return_to_rest_with_physics",
    "release_remove",
}


@dataclass(frozen=True)
class RecordedMotionExport:
    output_dir: Path
    source_frame_count: int
    exported_frame_count: int
    source_frame_indices: tuple[int, ...]


def select_export_frame_indices(
    history: DynamicFrameHistory,
    *,
    frame_stride: int,
) -> tuple[int, ...]:
    """Sample replay history while preserving phase boundaries and endpoints."""
    stride = int(frame_stride)
    if stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}.")
    frame_count = len(history)
    if frame_count == 0:
        return ()

    selected = set(range(0, frame_count, stride))
    selected.update((0, frame_count - 1))
    for frame_index in range(1, frame_count):
        previous, current = history[frame_index - 1], history[frame_index]
        if (previous.phase == current.phase
                and previous.current_action == current.current_action
                and previous.completed_actions == current.completed_actions
                and previous.metric_total == current.metric_total
                and np.array_equal(previous.active_mask, current.active_mask)):
            continue
        selected.update((frame_index - 1, frame_index))
    return tuple(sorted(selected))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "recorded_motion"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _display_style(
    handle: Any,
    *,
    color: tuple[float, float, float],
    alpha: float = 1.0,
) -> tuple[tuple[float, float, float], float]:
    if handle is None:
        return color, float(alpha)
    try:
        displayed_color = tuple(
            float(value)
            for value in np.asarray(handle.get_color(), dtype=np.float64).reshape(3)
        )
    except Exception:
        displayed_color = color
    try:
        displayed_alpha = float(handle.get_transparency())
    except Exception:
        displayed_alpha = float(alpha)
    return displayed_color, displayed_alpha


def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    homogeneous = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)],
        axis=1,
    )
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _export_assets(
    renderer,
    output_path: Path,
) -> tuple[list[str], list[str], list[list[float]]]:
    """Write geometry without a file-format coordinate-system conversion."""
    box_names = [
        f"box_{box_index}" for box_index in range(int(renderer.controller.executor.sim.n_boxes))
    ]
    arrays: dict[str, np.ndarray] = {
        "box_half_extents": np.asarray(
            renderer.controller.executor.sim.half_extents,
            dtype=np.float32,
        )
    }
    robot_names: list[str] = []
    robot_colors: list[list[float]] = []
    for robot_index, record in enumerate(renderer.robot_meshes):
        name = _safe_name(str(record["name"]))
        robot_names.append(name)
        color, alpha = _display_style(
            record.get("handle"),
            color=(0.82, 0.86, 0.88),
        )
        robot_colors.append([*color, float(alpha)])

        # Bake the static URDF visual origin and mesh scale into local vertices.
        # Animation matrices can then remain pure world-link rigid transforms.
        local_transform = np.asarray(
            record["visual_tf"],
            dtype=np.float64,
        ) @ np.asarray(record["scale_tf"], dtype=np.float64)
        arrays[f"robot_{robot_index:03d}_vertices"] = np.asarray(
            _transform_vertices(record["vertices"], local_transform),
            dtype=np.float32,
        )
        arrays[f"robot_{robot_index:03d}_faces"] = np.asarray(
            record["faces"],
            dtype=np.int32,
        )

    _atomic_savez(output_path, **arrays)
    return box_names, robot_names, robot_colors


def _frame_manifest_row(
    frame: RecordedFrame,
    *,
    source_frame_index: int,
) -> dict:
    return {
        "source_frame_index": int(source_frame_index),
        "blender_frame": int(source_frame_index) + 1,
        "phase": str(frame.phase),
        "sim_frame": int(frame.sim_frame),
        "sim_time": float(frame.sim_time),
        "completed_actions": int(frame.completed_actions),
        "current_action": frame.current_action,
        "metric_total": float(frame.metric_total),
        "active_box_indices": [
            int(index) for index, active in enumerate(frame.active_mask) if bool(active)
        ],
    }


def export_recorded_motion(
    *,
    controller,
    renderer,
    output_root: str | Path,
    frame_stride: int,
) -> RecordedMotionExport:
    """Export one asset scene plus sampled object transforms for Blender."""
    if not controller.can_export:
        raise RuntimeError(
            "Finish initial settling and wait for a stable state or completion before exporting."
        )
    indices = select_export_frame_indices(
        controller.history,
        frame_stride=int(frame_stride),
    )
    if not indices:
        raise RuntimeError("No recorded physical frames are available for export.")

    exported_utc = datetime.now(timezone.utc)
    batch_name = exported_utc.strftime("export_%Y%m%dT%H%M%S_%fZ")
    scene_name = _safe_name(Path(controller.scene_path).stem)
    run_dir = (
        Path(output_root).expanduser().resolve()
        / scene_name
        / batch_name
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    box_names, robot_names, robot_colors = _export_assets(
        renderer,
        run_dir / "assets.npz",
    )
    sampled_frames = [controller.history[index] for index in indices]
    source_indices = np.asarray(indices, dtype=np.int32)
    box_positions = np.stack(
        [
            np.asarray(frame.box_positions_w, dtype=np.float32)
            for frame in sampled_frames
        ]
    )
    box_quats = np.stack(
        [
            np.asarray(frame.box_quats_wxyz, dtype=np.float32)
            for frame in sampled_frames
        ]
    )
    box_active = np.stack(
        [np.asarray(frame.active_mask, dtype=bool) for frame in sampled_frames]
    )
    # Keep hidden boxes at their last visible pose. Otherwise interpolation
    # toward MuJoCo's off-screen storage position makes them fly away before
    # their visibility keyframe, especially when exporting with a stride.
    for sample_index in range(1, len(sampled_frames)):
        hidden = ~box_active[sample_index]
        box_positions[sample_index, hidden] = box_positions[sample_index - 1, hidden]
        box_quats[sample_index, hidden] = box_quats[sample_index - 1, hidden]
    actual_actions = [int(record["box"]) for record in controller.records]
    box_green = np.zeros_like(box_active, dtype=bool)
    for sample_index, frame in enumerate(sampled_frames):
        if str(frame.phase) not in _SELECTED_BOX_PHASES:
            continue
        if frame.current_action is None:
            continue
        selected_box = int(frame.current_action)
        if 0 <= selected_box < box_green.shape[1] and bool(
            box_active[sample_index, selected_box]
        ):
            box_green[sample_index, selected_box] = True

    robot_transforms: list[np.ndarray] = []
    for export_index, (source_index, frame) in enumerate(
        zip(indices, sampled_frames),
        start=1,
    ):
        robot_transforms.append(
            np.asarray(
                renderer.robot_visual_world_transforms(frame.qpos),
                dtype=np.float32,
            )
        )
        if (
            export_index == 1
            or export_index == len(indices)
            or export_index % 50 == 0
        ):
            print(
                "[unload_play_export] "
                f"transform={export_index}/{len(indices)} "
                f"source_index={source_index} phase={frame.phase}",
                flush=True,
            )

    _atomic_savez(
        run_dir / "animation.npz",
        source_frame_indices=source_indices,
        blender_frames=source_indices + 1,
        sim_times=np.asarray(
            [frame.sim_time for frame in sampled_frames],
            dtype=np.float32,
        ),
        metric_totals=np.asarray(
            [frame.metric_total for frame in sampled_frames],
            dtype=np.float32,
        ),
        box_positions_w=box_positions,
        box_quats_wxyz=box_quats,
        box_active=box_active,
        box_green=box_green,
        robot_visual_transforms=np.stack(robot_transforms),
    )

    importer_source = Path(__file__).with_name("blender_import_recorded_motion.py")
    shutil.copyfile(importer_source, run_dir / importer_source.name)
    video_source = Path(__file__).with_name("blender_video.py")
    shutil.copyfile(video_source, run_dir / video_source.name)

    metric_total = float(controller.executor.stats.metric_total)
    sim = controller.executor.sim
    manifest = {
        "format": "unload_replay_transform_animation_v2",
        "exported_utc": exported_utc.isoformat(),
        "scene_npz": str(controller.scene_path),
        "mode": "direct-remove" if controller.direct_remove else "motiongen",
        "phase_at_export": str(controller.phase.value),
        "complete": str(controller.phase.value) == "complete",
        "actual_actions": actual_actions,
        "pile_adjustment": asdict(controller.pile_adjustment),
        "actual_metric_total": metric_total,
        "source_replay_fps": 1.0 / float(sim.frame_dt),
        "source_frame_count": len(controller.history),
        "frame_stride": int(frame_stride),
        "exported_frame_count": len(indices),
        "source_frame_indices": list(indices),
        "files": {
            "assets": "assets.npz",
            "animation": "animation.npz",
            "blender_importer": importer_source.name,
            "blender_video": video_source.name,
        },
        "box_object_names": box_names,
        "robot_visual_object_names": robot_names,
        "robot_visual_colors": robot_colors,
        "cardboard_box_color": list(CARDBOARD_BROWN),
        "selected_box_color": list(SELECTED_GREEN),
        "static_boxes": [
            {"name": "table", "position": list(sim.table_pos),
             "half_extents": list(sim.table_half_extents), "color": [*COLOR_TABLE, 1.0]},
            *[{"name": f"wall_{index}", "position": list(position),
               "half_extents": list(half), "color": [*COLOR_WALL, 1.0]}
              for index, position, half in sim.wall_defs
              if index not in sim.snapshot["walls_removed"]],
        ],
        "frames": [
            _frame_manifest_row(frame, source_frame_index=source_index)
            for source_index, frame in zip(indices, sampled_frames)
        ],
        "execution_records": _json_value(controller.records),
    }
    _atomic_write_text(
        run_dir / "replay_manifest.json",
        json.dumps(_json_value(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    metric_lines = [
        f"scene_npz={controller.scene_path}",
        f"phase_at_export={controller.phase.value}",
        f"source_frame_count={len(controller.history)}",
        f"exported_frame_count={len(indices)}",
        f"frame_stride={int(frame_stride)}",
        f"actual_actions={actual_actions}",
        f"pile_adjustment={asdict(controller.pile_adjustment)}",
        f"actual_metric_total={metric_total:.12g}",
        "metric_deltas="
        + str([float(record["metric_delta"]) for record in controller.records]),
        "metric_cumulative="
        + str([float(record["metric_total"]) for record in controller.records]),
    ]
    _atomic_write_text(run_dir / "metric.txt", "\n".join(metric_lines) + "\n")
    print(
        "[unload_play_export] complete "
        f"source_frames={len(controller.history)} exported={len(indices)} "
        f"metric={metric_total:.6f} output={run_dir}",
        flush=True,
    )
    return RecordedMotionExport(
        output_dir=run_dir,
        source_frame_count=len(controller.history),
        exported_frame_count=len(indices),
        source_frame_indices=indices,
    )


__all__ = [
    "RecordedMotionExport",
    "export_recorded_motion",
    "select_export_frame_indices",
]
