"""Prepare or batch-render compact unload replays as MP4 or PNG sequences.

Edit the configuration block, open this file in Blender's Text Editor, and
press Run Script. Run once in ``prepare`` mode, adjust the camera manually,
then render from Blender, rerun in ``render`` mode, or use ``batch_render`` to
apply that camera to several inputs. This script has no dependency on cuRobo or
the unload runtime.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import traceback

import bpy
from mathutils import Matrix, Vector
import numpy as np


# INPUT_DIR must contain assets.npz, animation.npz, and replay_manifest.json.
# Relative paths use the saved .blend directory, or this script's directory when
# the .blend file has not been saved yet.
INPUT_DIR = ""  # Empty uses the export directory beside this copied script.
# Used only by MODE="batch_render". Each entry is one directory containing a
# compact replay export. The currently adjusted Blender camera is reused for
# every entry.
BATCH_INPUT_DIRS = (
    # "/path/to/exports/box_10_42/export_...",
)
OUTPUT_DIR = ""  # Empty uses <INPUT_DIR>/render_video or render_frames.
MODE = "prepare"  # "prepare", "render", or "batch_render".
OUTPUT_FORMAT = "mp4"  # "mp4" or "png"
VIDEO_FILENAME = "unload_replay.mp4"

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_SAMPLES = 16  # Used only by Cycles.
RENDER_ENGINE = "eevee"  # "eevee" is fast; use "cycles" for final quality.
TRANSPARENT_BACKGROUND = False  # MP4/H.264 has no alpha channel.
RENDER_NOW = False
OVERWRITE_EXISTING = False
FRAME_START = None  # None uses the complete imported timeline.
FRAME_END = None

SHOW_METRIC_OVERLAY = True
METRIC_LABEL = "Disturbance"
METRIC_DECIMALS = 3
METRIC_TEXT_COLOR = (1.0, 1.0, 1.0, 1.0)
METRIC_TEXT_HEIGHT_FRACTION = 0.052
METRIC_MARGIN_FRACTION = 0.045

CAMERA_LOCATION = (1.508657, 0.494216, 1.506888)
CAMERA_ROTATION_DEG = (54.726839, 0.000049, 121.769623)
CAMERA_LENS_MM = 55.0

KEY_LIGHT_ENERGY = 50.0
FILL_LIGHT_ENERGY = 50.0
FRONT_LIGHT_ENERGY = 0.0
BACK_LIGHT_ENERGY = 0.0
WORLD_LIGHT_STRENGTH = 0.08
CAMERA_BACKGROUND_STRENGTH = 1.0
BACKGROUND_GRAY = (0.18, 0.20, 0.22, 1.0)
AMBIENT_LIGHT_WHITE = (1.0, 1.0, 1.0, 1.0)

_BATCH_STATE_KEY = "unload_replay_batch_renderer"


def _script_directory() -> Path:
    script_path = globals().get("__file__")
    if not script_path:
        raise RuntimeError(
            "Open blender_video.py from disk so Blender defines __file__."
        )
    return Path(script_path).resolve().parent


def _base_directory() -> Path:
    if bpy.data.filepath:
        return Path(bpy.data.filepath).resolve().parent
    return _script_directory()


def _resolve_path(path_like: str, *, default: Path | None = None) -> Path:
    if not str(path_like).strip():
        if default is None:
            raise ValueError("A required path is empty.")
        return default.resolve()
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = _base_directory() / path
    return path.resolve()


def _validate_input(input_dir: Path) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    missing = [
        name
        for name in ("assets.npz", "animation.npz", "replay_manifest.json")
        if not (input_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Compact replay directory {input_dir} is missing {missing}."
        )


def _load_animation(input_dir: Path) -> None:
    importer_path = _script_directory() / "blender_import_recorded_motion.py"
    if not importer_path.is_file():
        copied_importer = input_dir / "blender_import_recorded_motion.py"
        if not copied_importer.is_file():
            raise FileNotFoundError(
                "No blender_import_recorded_motion.py found beside "
                f"blender_video.py or under {input_dir}."
            )
        importer_path = copied_importer

    spec = importlib.util.spec_from_file_location(
        "unload_replay_blender_animation_importer",
        importer_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Blender importer from {importer_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUT_DIR = str(input_dir)
    module.CLEAR_SCENE = True
    module.main()


def _mesh_objects() -> list[bpy.types.Object]:
    # The large ground slab must not determine the lighting scale.
    return [obj for obj in bpy.context.scene.objects
            if obj.type == "MESH" and not obj.get("unload_static_environment", False)]


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    if not points:
        raise RuntimeError("Imported animation contains no mesh bounds.")
    lower = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return lower, upper


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _add_area_light(
    name: str,
    location: Vector,
    target: Vector,
    *,
    energy: float,
    size: float,
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = float(energy)
    data.shape = "DISK"
    data.size = float(size)
    data.color = (1.0, 1.0, 1.0)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    _look_at(obj, target)


def _capture_camera_state(camera: bpy.types.Object) -> dict:
    data = camera.data
    return {
        "matrix_world": [list(row) for row in camera.matrix_world],
        "type": str(data.type),
        "lens": float(data.lens),
        "sensor_width": float(data.sensor_width),
        "sensor_height": float(data.sensor_height),
        "sensor_fit": str(data.sensor_fit),
        "shift_x": float(data.shift_x),
        "shift_y": float(data.shift_y),
        "clip_start": float(data.clip_start),
        "clip_end": float(data.clip_end),
    }


def _setup_camera_and_lighting(*, camera_state: dict | None = None) -> None:
    lower, upper = _world_bounds(_mesh_objects())
    center = 0.5 * (lower + upper)
    extent = upper - lower
    span = max(float(extent.x), float(extent.y), float(extent.z), 0.45)
    center.z = lower.z + 0.48 * float(extent.z)

    camera_data = bpy.data.cameras.new("VideoCamera")
    camera_data.lens = float(CAMERA_LENS_MM)
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = False
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("VideoCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    if camera_state is None:
        camera.location = Vector(CAMERA_LOCATION)
        camera.rotation_euler = tuple(
            math.radians(float(value)) for value in CAMERA_ROTATION_DEG
        )
    else:
        camera.matrix_world = Matrix(camera_state["matrix_world"])
        camera_data.type = str(camera_state["type"])
        camera_data.lens = float(camera_state["lens"])
        camera_data.sensor_width = float(camera_state["sensor_width"])
        camera_data.sensor_height = float(camera_state["sensor_height"])
        camera_data.sensor_fit = str(camera_state["sensor_fit"])
        camera_data.shift_x = float(camera_state["shift_x"])
        camera_data.shift_y = float(camera_state["shift_y"])
        camera_data.clip_start = float(camera_state["clip_start"])
        camera_data.clip_end = float(camera_state["clip_end"])
    bpy.context.scene.camera = camera

    _add_area_light(
        "KeyLight",
        center + Vector((1.2, -1.4, 2.2)) * span,
        center,
        size=1.5 * span,
        energy=KEY_LIGHT_ENERGY,
    )
    _add_area_light(
        "FillLight",
        center + Vector((-1.2, 1.4, 2.2)) * span,
        center,
        size=1.5 * span,
        energy=FILL_LIGHT_ENERGY,
    )
    _add_area_light(
        "FrontLight",
        center + Vector((0.0, -1.8, 1.6)) * span,
        center,
        size=1.5 * span,
        energy=FRONT_LIGHT_ENERGY,
    )
    _add_area_light(
        "BackLight",
        center + Vector((0.0, 1.8, 1.6)) * span,
        center,
        size=1.5 * span,
        energy=BACK_LIGHT_ENERGY,
    )


def _configure_world() -> None:
    world = bpy.context.scene.world
    world.color = BACKGROUND_GRAY[:3]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    ambient = nodes.new("ShaderNodeBackground")
    ambient.name = "WorldAmbient"
    ambient.inputs["Color"].default_value = AMBIENT_LIGHT_WHITE
    ambient.inputs["Strength"].default_value = float(WORLD_LIGHT_STRENGTH)

    camera_background = nodes.new("ShaderNodeBackground")
    camera_background.name = "CameraBackground"
    camera_background.inputs["Color"].default_value = BACKGROUND_GRAY
    camera_background.inputs["Strength"].default_value = float(
        CAMERA_BACKGROUND_STRENGTH
    )

    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(ambient.outputs["Background"], mix.inputs[1])
    links.new(camera_background.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def _remove_metric_overlay() -> None:
    for handler in list(bpy.app.handlers.frame_change_pre):
        if bool(getattr(handler, "_unload_metric_overlay", False)):
            bpy.app.handlers.frame_change_pre.remove(handler)
    overlay = bpy.data.objects.get("DisturbanceMetricOverlay")
    if overlay is not None:
        bpy.data.objects.remove(overlay, do_unlink=True)


def _metric_text(value: float) -> str:
    return f"{METRIC_LABEL}: {float(value):.{int(METRIC_DECIMALS)}f}"


def _setup_metric_overlay(input_dir: Path) -> None:
    _remove_metric_overlay()
    if not bool(SHOW_METRIC_OVERLAY):
        return
    with np.load(input_dir / "animation.npz", allow_pickle=False) as animation:
        blender_frames = np.asarray(
            animation["blender_frames"], dtype=np.int32
        ).reshape(-1)
        metric_totals = np.asarray(
            animation["metric_totals"], dtype=np.float64
        ).reshape(-1)
    if blender_frames.size == 0 or metric_totals.shape != blender_frames.shape:
        raise ValueError("Metric samples do not match the exported animation frames.")

    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("Cannot place the metric overlay without an active camera.")

    text_data = bpy.data.curves.new("DisturbanceMetricText", type="FONT")
    text_data.align_x = "RIGHT"
    text_data.align_y = "BOTTOM"
    text_data.extrude = 0.0
    text_data.bevel_depth = 0.0
    text_data.body = _metric_text(metric_totals[0])

    overlay = bpy.data.objects.new("DisturbanceMetricOverlay", text_data)
    bpy.context.collection.objects.link(overlay)
    overlay.parent = camera
    overlay.rotation_euler = (0.0, 0.0, 0.0)

    depth = max(float(camera.data.clip_start) * 4.0, 0.25)
    view_corners = [
        Vector(corner) * (depth / abs(float(corner.z)))
        for corner in camera.data.view_frame(scene=scene)
    ]
    left = min(float(corner.x) for corner in view_corners)
    right = max(float(corner.x) for corner in view_corners)
    bottom = min(float(corner.y) for corner in view_corners)
    top = max(float(corner.y) for corner in view_corners)
    view_width = right - left
    view_height = top - bottom
    margin = float(METRIC_MARGIN_FRACTION) * min(view_width, view_height)
    overlay.location = (right - margin, bottom + margin, -depth)
    text_data.size = float(METRIC_TEXT_HEIGHT_FRACTION) * view_height

    material = bpy.data.materials.new(name="DisturbanceMetricMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = METRIC_TEXT_COLOR
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    text_data.materials.append(material)

    def update_metric_text(current_scene, _depsgraph=None) -> None:
        frame = int(current_scene.frame_current)
        sample_index = int(np.searchsorted(blender_frames, frame, side="right") - 1)
        sample_index = max(0, min(sample_index, int(metric_totals.size) - 1))
        text_data.body = _metric_text(metric_totals[sample_index])

    update_metric_text._unload_metric_overlay = True
    bpy.app.handlers.frame_change_pre.append(update_metric_text)
    update_metric_text(scene)


def _configure_render(output_dir: Path) -> Path:
    scene = bpy.context.scene
    engine = str(RENDER_ENGINE).lower()
    if engine not in {"eevee", "cycles"}:
        raise ValueError(
            f"RENDER_ENGINE must be 'eevee' or 'cycles', got {RENDER_ENGINE!r}."
        )
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(RENDER_SAMPLES)
        scene.cycles.use_denoising = True
    else:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = int(RENDER_WIDTH)
    scene.render.resolution_y = int(RENDER_HEIGHT)
    scene.render.resolution_percentage = 100
    output_format = str(OUTPUT_FORMAT).strip().lower()
    if output_format not in {"mp4", "png"}:
        raise ValueError(
            f"OUTPUT_FORMAT must be 'mp4' or 'png', got {OUTPUT_FORMAT!r}."
        )
    if output_format == "mp4":
        if bool(TRANSPARENT_BACKGROUND):
            raise ValueError(
                "MP4/H.264 does not preserve transparency; set "
                "TRANSPARENT_BACKGROUND=False or use OUTPUT_FORMAT='png'."
            )
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        output_path = output_dir / str(VIDEO_FILENAME)
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = (
            "RGBA" if TRANSPARENT_BACKGROUND else "RGB"
        )
        output_path = output_dir / "frame_"
    scene.render.film_transparent = bool(TRANSPARENT_BACKGROUND)
    scene.render.filepath = str(output_path)
    scene.render.use_file_extension = True
    scene.render.use_overwrite = bool(OVERWRITE_EXISTING)

    if FRAME_START is not None:
        scene.frame_start = max(scene.frame_start, int(FRAME_START))
    if FRAME_END is not None:
        scene.frame_end = min(scene.frame_end, int(FRAME_END))
    if scene.frame_start > scene.frame_end:
        raise ValueError(
            f"Empty render range {scene.frame_start}:{scene.frame_end}."
        )
    scene.frame_set(scene.frame_start)
    # AgX maps a linear white world below display white. Standard keeps the
    # opaque MP4 background visually white for paper/video composition.
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except (TypeError, ValueError):
        pass
    return output_path


def _default_output_name() -> str:
    return "render_video" if str(OUTPUT_FORMAT).strip().lower() == "mp4" else "render_frames"


def _output_directory(input_dir: Path, *, batch: bool) -> Path:
    if not str(OUTPUT_DIR).strip():
        return (input_dir / _default_output_name()).resolve()
    root = _resolve_path(OUTPUT_DIR)
    if not batch:
        return root
    import json

    manifest = json.loads(
        (input_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    scene_name = Path(str(manifest.get("scene_npz", "scene"))).stem
    sequence_name = Path(str(manifest.get("source_sequence", input_dir.name))).stem
    return (root / scene_name / sequence_name / _default_output_name()).resolve()


def _print_render_setup(*, mode: str, input_dir: Path, output_path: Path) -> None:
    scene = bpy.context.scene
    print(
        "[blender_video] prepared "
        f"mode={mode} input={input_dir} output={output_path} "
        f"format={str(OUTPUT_FORMAT).strip().lower()} "
        f"frames={scene.frame_start}:{scene.frame_end} fps={scene.render.fps} "
        f"engine={scene.render.engine} resolution="
        f"{scene.render.resolution_x}x{scene.render.resolution_y}",
        flush=True,
    )


def _batch_inputs() -> list[Path]:
    if not BATCH_INPUT_DIRS:
        raise ValueError(
            "MODE='batch_render' requires at least one BATCH_INPUT_DIRS entry."
        )
    inputs = [_resolve_path(value) for value in BATCH_INPUT_DIRS]
    if len(set(inputs)) != len(inputs):
        raise ValueError("BATCH_INPUT_DIRS contains duplicate directories.")
    for input_dir in inputs:
        _validate_input(input_dir)
    return inputs


class _AsyncBatchRenderer:
    """Render one animation at a time without blocking Blender's UI loop."""

    def __init__(self, *, inputs: list[Path], camera_state: dict) -> None:
        self.inputs = list(inputs)
        self.camera_state = dict(camera_state)
        self.index = 0
        self.active = True
        self.current_output: Path | None = None
        self._timer_pending = False
        self._timer_callback = self._start_next
        self._complete_callback = self._on_render_complete
        self._cancel_callback = self._on_render_cancel

    def start(self) -> None:
        bpy.app.handlers.render_complete.append(self._complete_callback)
        bpy.app.handlers.render_cancel.append(self._cancel_callback)
        try:
            bpy.context.window_manager.progress_begin(0, len(self.inputs))
            bpy.context.window_manager.progress_update(0)
        except Exception:
            pass
        self._schedule_next()

    def _schedule_next(self) -> None:
        if not self.active or self._timer_pending:
            return
        self._timer_pending = True
        bpy.app.timers.register(self._timer_callback, first_interval=0.1)

    def _start_next(self):
        if not self.active:
            self._timer_pending = False
            return None
        if bpy.app.is_job_running("RENDER"):
            return 0.25
        self._timer_pending = False
        if self.index >= len(self.inputs):
            self._finish()
            return None

        input_dir = self.inputs[self.index]
        try:
            print(
                f"[blender_video] batch_start={self.index + 1}/"
                f"{len(self.inputs)} input={input_dir}",
                flush=True,
            )
            _remove_metric_overlay()
            _load_animation(input_dir)
            _setup_camera_and_lighting(camera_state=self.camera_state)
            _configure_world()
            output_dir = _output_directory(input_dir, batch=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.current_output = _configure_render(output_dir)
            _setup_metric_overlay(input_dir)
            _print_render_setup(
                mode="batch_render",
                input_dir=input_dir,
                output_path=self.current_output,
            )
            result = bpy.ops.render.render("INVOKE_DEFAULT", animation=True)
            if "CANCELLED" in result:
                raise RuntimeError("Blender rejected the animation render request.")
            print(
                f"[blender_video] batch_render_started={self.index + 1}/"
                f"{len(self.inputs)}",
                flush=True,
            )
        except Exception:
            print(
                f"[blender_video] batch_failed={self.index + 1}/"
                f"{len(self.inputs)} input={input_dir}",
                flush=True,
            )
            traceback.print_exc()
            self._finish()
        return None

    def _on_render_complete(self, _scene, *_args) -> None:
        if not self.active:
            return
        print(
            f"[blender_video] batch_complete={self.index + 1}/"
            f"{len(self.inputs)} output={self.current_output}",
            flush=True,
        )
        self.index += 1
        try:
            bpy.context.window_manager.progress_update(self.index)
        except Exception:
            pass
        if self.index >= len(self.inputs):
            self._finish()
        else:
            self._schedule_next()

    def _on_render_cancel(self, _scene, *_args) -> None:
        if not self.active:
            return
        print(
            f"[blender_video] batch_cancelled={self.index + 1}/"
            f"{len(self.inputs)}; remaining inputs were not started",
            flush=True,
        )
        self._finish()

    def _finish(self) -> None:
        if not self.active:
            return
        self.active = False
        if self._complete_callback in bpy.app.handlers.render_complete:
            bpy.app.handlers.render_complete.remove(self._complete_callback)
        if self._cancel_callback in bpy.app.handlers.render_cancel:
            bpy.app.handlers.render_cancel.remove(self._cancel_callback)
        try:
            bpy.context.window_manager.progress_end()
        except Exception:
            pass
        if bpy.app.driver_namespace.get(_BATCH_STATE_KEY) is self:
            bpy.app.driver_namespace.pop(_BATCH_STATE_KEY, None)
        if self.index >= len(self.inputs):
            print(
                f"[blender_video] batch_all_complete={len(self.inputs)}",
                flush=True,
            )


def _render_batch() -> None:
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError(
            "Batch rendering needs the manually adjusted camera. Run "
            "MODE='prepare' first, adjust VideoCamera, then rerun with "
            "MODE='batch_render'."
        )
    if bpy.app.is_job_running("RENDER"):
        raise RuntimeError("A Blender render job is already running.")
    if not bool(RENDER_NOW):
        raise ValueError("MODE='batch_render' requires RENDER_NOW=True.")
    existing = bpy.app.driver_namespace.get(_BATCH_STATE_KEY)
    if existing is not None and bool(getattr(existing, "active", False)):
        raise RuntimeError("An unload replay batch render is already active.")

    renderer = _AsyncBatchRenderer(
        inputs=_batch_inputs(),
        camera_state=_capture_camera_state(scene.camera),
    )
    bpy.app.driver_namespace[_BATCH_STATE_KEY] = renderer
    renderer.start()
    print(
        f"[blender_video] batch_queued={len(renderer.inputs)}; "
        "Blender UI remains available while each animation renders",
        flush=True,
    )


def main() -> None:
    mode = str(MODE).strip().lower()
    if mode not in {"prepare", "render", "batch_render"}:
        raise ValueError(
            "MODE must be 'prepare', 'render', or 'batch_render', "
            f"got {MODE!r}."
        )
    if mode == "batch_render":
        _render_batch()
        return

    input_dir = _resolve_path(INPUT_DIR, default=_script_directory())
    _validate_input(input_dir)
    output_dir = _output_directory(input_dir, batch=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "prepare":
        _load_animation(input_dir)
        _setup_camera_and_lighting()
    elif bpy.context.scene.camera is None:
        raise RuntimeError(
            "MODE='render' preserves the current Blender scene, but it has no "
            "active camera. Run MODE='prepare' first and adjust VideoCamera."
        )
    _configure_world()
    output_path = _configure_render(output_dir)
    _setup_metric_overlay(input_dir)

    _print_render_setup(mode=mode, input_dir=input_dir, output_path=output_path)
    if not RENDER_NOW:
        print(
            "[blender_video] RENDER_NOW=False; adjust VideoCamera and render "
            "from Blender, or set MODE='render' and RENDER_NOW=True.",
            flush=True,
        )
        return
    result = bpy.ops.render.render("INVOKE_DEFAULT", animation=True)
    if "CANCELLED" in result:
        raise RuntimeError("Blender rejected the animation render request.")
    print(f"[blender_video] render_started output={output_path}", flush=True)


if __name__ == "__main__":
    main()
