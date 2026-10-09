"""Build a Blender keyframe animation from a compact unload replay export."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion
import numpy as np


# Leave empty when running the copy stored beside assets.npz and animation.npz.
INPUT_DIR = ""
CLEAR_SCENE = True
VALIDATE_IMPORTED_TRANSFORMS = True
TRANSFORM_VALIDATION_ATOL = 2.0e-5

_FORMAT = "unload_replay_transform_animation_v2"
_BOX_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def _input_directory() -> Path:
    if INPUT_DIR:
        return Path(INPUT_DIR).expanduser().resolve()
    script_path = globals().get("__file__")
    if not script_path:
        raise RuntimeError("Set INPUT_DIR when the script has no file path.")
    return Path(script_path).resolve().parent


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _mesh_object(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray | tuple[tuple[int, ...], ...],
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    mesh.from_pydata(
        np.asarray(vertices, dtype=np.float64).tolist(),
        [],
        np.asarray(faces, dtype=np.int32).tolist(),
    )
    mesh.update()
    obj = bpy.data.objects.new(str(name), mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _box_vertices(half_extents: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(half_extents, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            (-x, -y, -z),
            (x, -y, -z),
            (x, y, -z),
            (-x, y, -z),
            (-x, -y, z),
            (x, -y, z),
            (x, y, z),
            (-x, y, z),
        ],
        dtype=np.float64,
    )


def _material(
    name: str,
    color: tuple[float, float, float, float],
):
    material = bpy.data.materials.new(name=f"{name}_material")
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    if node is None:
        raise RuntimeError("Blender material has no Principled BSDF node.")
    node.inputs["Base Color"].default_value = color
    node.inputs["Metallic"].default_value = 0.0
    node.inputs["Roughness"].default_value = 0.72
    material.diffuse_color = color
    return material, node.inputs["Base Color"]


def _continuous_quaternion(
    quaternion: Quaternion,
    previous: Quaternion | None,
) -> Quaternion:
    quaternion.normalize()
    if previous is not None and quaternion.dot(previous) < 0.0:
        quaternion.negate()
    return quaternion


def _keyframe_matrix_transform(
    obj: bpy.types.Object,
    transform: np.ndarray,
    frame_number: int,
    previous_quaternion: Quaternion | None,
) -> Quaternion:
    location, quaternion, scale = Matrix(
        np.asarray(transform, dtype=np.float64).tolist()
    ).decompose()
    quaternion = _continuous_quaternion(quaternion, previous_quaternion)
    obj.location = location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = quaternion
    obj.scale = scale
    obj.keyframe_insert(data_path="location", frame=int(frame_number))
    obj.keyframe_insert(data_path="rotation_quaternion", frame=int(frame_number))
    obj.keyframe_insert(data_path="scale", frame=int(frame_number))
    return quaternion.copy()


def _set_interpolation(owner, interpolation: str) -> None:
    animation = getattr(owner, "animation_data", None)
    action = None if animation is None else animation.action
    if action is None:
        return
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = str(interpolation)


def _validate_robot_keyframes(
    *,
    blender_frames: np.ndarray,
    robot_objects: list[bpy.types.Object],
    robot_transforms: np.ndarray,
) -> float:
    """Verify that Blender reconstructs every exported robot world matrix."""
    max_error = 0.0
    worst_frame = -1
    worst_object = ""
    for sample_index, frame_number_value in enumerate(blender_frames):
        frame_number = int(frame_number_value)
        bpy.context.scene.frame_set(frame_number)
        bpy.context.view_layer.update()
        for robot_index, obj in enumerate(robot_objects):
            actual = np.asarray(obj.matrix_world, dtype=np.float64)
            expected = robot_transforms[sample_index, robot_index]
            error = float(np.max(np.abs(actual - expected)))
            if error > max_error:
                max_error = error
                worst_frame = frame_number
                worst_object = str(obj.name)

    tolerance = float(TRANSFORM_VALIDATION_ATOL)
    if not np.isfinite(max_error) or max_error > tolerance:
        raise RuntimeError(
            "Imported robot transform validation failed: "
            f"max_abs_error={max_error:.6g} tolerance={tolerance:.6g} "
            f"frame={worst_frame} object={worst_object!r}."
        )
    print(
        "[blender_import_recorded_motion] robot_transform_validation "
        f"passed=True max_abs_error={max_error:.6g} "
        f"tolerance={tolerance:.6g} samples={len(blender_frames)} "
        f"robot_visuals={len(robot_objects)}",
        flush=True,
    )
    return max_error


def main() -> None:
    input_dir = _input_directory()
    manifest = json.loads(
        (input_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    if str(manifest.get("format")) != _FORMAT:
        raise RuntimeError(
            "This importer requires a v2 compact replay. Re-export the recorded "
            f"motion; found format={manifest.get('format')!r}."
        )
    with np.load(input_dir / "animation.npz", allow_pickle=False) as animation:
        blender_frames = np.asarray(animation["blender_frames"], dtype=np.int32)
        box_positions = np.asarray(animation["box_positions_w"], dtype=np.float64)
        box_quats = np.asarray(animation["box_quats_wxyz"], dtype=np.float64)
        box_active = np.asarray(animation["box_active"], dtype=bool)
        box_green = np.asarray(animation["box_green"], dtype=bool)
        robot_transforms = np.asarray(
            animation["robot_visual_transforms"],
            dtype=np.float64,
        )

    box_names = [str(value) for value in manifest["box_object_names"]]
    robot_names = [
        str(value) for value in manifest["robot_visual_object_names"]
    ]
    robot_colors = [
        tuple(float(component) for component in color)
        for color in manifest["robot_visual_colors"]
    ]
    if box_positions.shape[:2] != (len(blender_frames), len(box_names)):
        raise ValueError("Box transform arrays do not match the manifest.")
    if robot_transforms.shape[:2] != (len(blender_frames), len(robot_names)):
        raise ValueError("Robot transform arrays do not match the manifest.")
    if len(robot_colors) != len(robot_names):
        raise ValueError("Robot colors do not match the manifest.")

    if CLEAR_SCENE:
        _clear_scene()
    with np.load(input_dir / "assets.npz", allow_pickle=False) as assets:
        half_extents = np.asarray(assets["box_half_extents"], dtype=np.float64)
        if half_extents.shape != (len(box_names), 3):
            raise ValueError("Box asset dimensions do not match the manifest.")
        box_objects = [
            _mesh_object(name, _box_vertices(half_extents[index]), _BOX_FACES)
            for index, name in enumerate(box_names)
        ]
        robot_objects = [
            _mesh_object(
                name,
                assets[f"robot_{index:03d}_vertices"],
                assets[f"robot_{index:03d}_faces"],
            )
            for index, name in enumerate(robot_names)
        ]

    for record in manifest.get("static_boxes", []):
        obj = _mesh_object(record["name"], _box_vertices(record["half_extents"]), _BOX_FACES)
        obj.location = record["position"]
        obj["unload_static_environment"] = True
        material, _ = _material(record["name"], tuple(record["color"]))
        obj.data.materials.append(material)

    cardboard_color = tuple(
        float(value) for value in manifest["cardboard_box_color"]
    )
    selected_color = tuple(
        float(value) for value in manifest["selected_box_color"]
    )
    box_color_inputs = []
    for name, obj in zip(box_names, box_objects):
        material, color_input = _material(name, cardboard_color)
        obj.data.materials.append(material)
        box_color_inputs.append(color_input)
    for name, color, obj in zip(robot_names, robot_colors, robot_objects):
        material, _ = _material(name, color)
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    box_previous_quaternions: list[Quaternion | None] = [None] * len(box_names)
    robot_previous_quaternions: list[Quaternion | None] = [None] * len(robot_names)
    for sample_index, frame_number_value in enumerate(blender_frames):
        frame_number = int(frame_number_value)
        for box_index, obj in enumerate(box_objects):
            quaternion = Quaternion(
                tuple(float(value) for value in box_quats[sample_index, box_index])
            )
            quaternion = _continuous_quaternion(
                quaternion,
                box_previous_quaternions[box_index],
            )
            box_previous_quaternions[box_index] = quaternion.copy()
            obj.location = box_positions[sample_index, box_index]
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = quaternion
            obj.keyframe_insert(data_path="location", frame=frame_number)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

            hidden = not bool(box_active[sample_index, box_index])
            obj.hide_viewport = hidden
            obj.hide_render = hidden
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
            obj.keyframe_insert(data_path="hide_render", frame=frame_number)

            color = (
                selected_color
                if bool(box_green[sample_index, box_index])
                else cardboard_color
            )
            color_input = box_color_inputs[box_index]
            color_input.default_value = color
            color_input.keyframe_insert(
                data_path="default_value",
                frame=frame_number,
            )

        for robot_index, obj in enumerate(robot_objects):
            robot_previous_quaternions[robot_index] = _keyframe_matrix_transform(
                obj,
                robot_transforms[sample_index, robot_index],
                frame_number,
                robot_previous_quaternions[robot_index],
            )

    for obj in (*box_objects, *robot_objects):
        _set_interpolation(obj, "LINEAR")
        animation_data = obj.animation_data
        if animation_data is None or animation_data.action is None:
            continue
        for curve in animation_data.action.fcurves:
            if curve.data_path in {"hide_viewport", "hide_render"}:
                for point in curve.keyframe_points:
                    point.interpolation = "CONSTANT"
    for color_input in box_color_inputs:
        _set_interpolation(color_input.node.id_data, "CONSTANT")

    scene = bpy.context.scene
    scene.render.fps = int(round(float(manifest["source_replay_fps"])))
    scene.frame_start = int(blender_frames.min())
    scene.frame_end = int(blender_frames.max())
    if VALIDATE_IMPORTED_TRANSFORMS:
        _validate_robot_keyframes(
            blender_frames=blender_frames,
            robot_objects=robot_objects,
            robot_transforms=robot_transforms,
        )
    scene.frame_set(scene.frame_start)
    print(
        "[blender_import_recorded_motion] "
        f"frames={len(blender_frames)} range={scene.frame_start}:{scene.frame_end} "
        f"boxes={len(box_objects)} robot_visuals={len(robot_objects)} "
        f"input={input_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
