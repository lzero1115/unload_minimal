from __future__ import annotations

import copy
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robot_naming import (
    robot_actuator_name,
    robot_base_body_name,
    robot_body_name,
    robot_joint_name,
    robot_name,
    robot_site_name,
)


CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [6, 5, 4],
        [7, 6, 4],
        [1, 5, 6],
        [1, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [3, 2, 6],
        [3, 6, 7],
        [1, 0, 4],
        [1, 4, 5],
    ],
    dtype=np.int32,
)

COLOR_TABLE = (0.72, 0.70, 0.64)
COLOR_WALL = (0.55, 0.55, 0.55)
COLOR_BOX_INSTANT_SETTLED = (0.50, 0.50, 0.50)
COLOR_BOX_INSTANT_UNSETTLED = (0.85, 0.20, 0.18)
COLOR_BOX_FILTERED_OUT = (0.95, 0.78, 0.15)


def convert_solref(ke: float, kd: float) -> tuple[float, float]:
    if ke > 0.0 and kd > 0.0:
        timeconst = 2.0 / kd
        dampratio = kd / (2.0 * math.sqrt(ke))
        return timeconst, dampratio
    return 0.02, 1.0


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in np.asarray(q_xyzw, dtype=np.float64).reshape(4))
    return np.array([w, x, y, z], dtype=np.float64)


def quat_xyzw_to_mjcf(q_xyzw: np.ndarray) -> str:
    w, x, y, z = (float(v) for v in quat_xyzw_to_wxyz(q_xyzw))
    return f"{w} {x} {y} {z}"


def quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q_wxyz, dtype=np.float64).reshape(4))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def box_vertices_world(pos: np.ndarray, quat_wxyz: np.ndarray, half: np.ndarray) -> np.ndarray:
    hx, hy, hz = (float(v) for v in np.asarray(half, dtype=np.float64).reshape(3))
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    return local @ quat_wxyz_to_rot(quat_wxyz).T + np.asarray(pos, dtype=np.float64).reshape(3)


def axis_aligned_box_vertices(
    pos: tuple[float, float, float],
    half: tuple[float, float, float],
) -> np.ndarray:
    return box_vertices_world(
        np.asarray(pos, dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray(half, dtype=np.float64),
    )


def wall_definitions(
    dims: tuple[float, float, float],
    *,
    wall_scale: float,
    wall_thickness: float,
    walls_removed: set[int],
    center_xy_at_origin: bool = False,
) -> list[tuple[int, tuple[float, float, float], tuple[float, float, float]]]:
    dx, dy, dz = (float(v) for v in dims)
    wall_h = dz * 1.1
    whz = wall_h / 2.0
    if bool(center_xy_at_origin):
        cx, cy = 0.0, 0.0
    else:
        cx, cy = dx / 2.0, dy / 2.0
    chx = dx * float(wall_scale) / 2.0
    chy = dy * float(wall_scale) / 2.0
    wt = float(wall_thickness)
    all_defs = [
        (0, (cx - chx - wt, cy, whz), (wt, chy + wt, whz)),
        (1, (cx + chx + wt, cy, whz), (wt, chy + wt, whz)),
        (2, (cx, cy - chy - wt, whz), (chx + wt, wt, whz)),
        (3, (cx, cy + chy + wt, whz), (chx + wt, wt, whz)),
    ]
    return [item for item in all_defs if item[0] not in walls_removed]


def friction_str(values: tuple[float, float, float]) -> str:
    return " ".join(str(float(v)) for v in values)


def solimp_str(values: tuple[float, ...]) -> str:
    return " ".join(str(float(v)) for v in values)


def build_mjcf(
    *,
    centers: np.ndarray,
    quats_xyzw: np.ndarray,
    half_extents: np.ndarray,
    walls_removed: set[int],
    scene_config,
    physics_config,
    no_mocap: bool = True,
    box_drop_height: float | None = None,
    table_pos: tuple[float, float, float] | np.ndarray | None = None,
    table_half_extents: tuple[float, float, float] | np.ndarray | None = None,
    wall_defs: list[tuple[int, tuple[float, float, float], tuple[float, float, float]]] | None = None,
    dims: tuple[float, float, float] | None = None,
    table_friction: tuple[float, float, float] | None = None,
    wall_friction: tuple[float, float, float] | None = None,
    box_friction: tuple[float, float, float] | None = None,
) -> str:
    """Build the MuJoCo XML for a table + optional walls + dynamic boxes scene."""

    tc, dr = convert_solref(
        float(physics_config.MUJOCO_CONTACT_KE),
        float(physics_config.MUJOCO_CONTACT_KD),
    )
    solimp = solimp_str(tuple(float(v) for v in physics_config.MUJOCO_SOLIMP))
    gravity = " ".join(str(float(v)) for v in physics_config.MJ_GRAVITY)

    opt = (
        f'<option timestep="{float(physics_config.MJ_TIMESTEP):.10f}" '
        f'gravity="{gravity}" '
        f'integrator="{str(physics_config.MJ_INTEGRATOR).lower()}" '
        f'solver="{physics_config.MJ_SOLVER}" '
        f'iterations="{int(physics_config.MJ_ITERATIONS)}" '
        f'ls_iterations="{int(physics_config.MJ_LS_ITERATIONS)}" '
        f'cone="{physics_config.MJ_CONE}" '
        f'impratio="{float(physics_config.MJ_IMPRATIO)}" '
        f'tolerance="{float(physics_config.MJ_TOLERANCE)}" '
        f'ls_tolerance="{float(physics_config.MJ_LS_TOLERANCE)}" '
        f'noslip_iterations="{int(physics_config.MJ_NOSLIP_ITERATIONS)}" '
        f'noslip_tolerance="{float(physics_config.MJ_NOSLIP_TOLERANCE)}"/>'
    )

    table_pos = tuple(
        float(v) for v in (scene_config.TABLE_POS if table_pos is None else table_pos)
    )
    table_half = tuple(
        float(v)
        for v in (
            scene_config.TABLE_HALF_EXTENTS
            if table_half_extents is None
            else table_half_extents
        )
    )
    table_friction = tuple(
        float(v)
        for v in (
            physics_config.TABLE_FRICTION
            if table_friction is None
            else table_friction
        )
    )
    wall_friction = tuple(
        float(v)
        for v in (
            physics_config.WALL_FRICTION
            if wall_friction is None
            else wall_friction
        )
    )
    box_friction = tuple(
        float(v)
        for v in (
            physics_config.BOX_FRICTION
            if box_friction is None
            else box_friction
        )
    )

    lines: list[str] = [
        '<mujoco model="unload_partition">',
        '  <compiler angle="radian" inertiafromgeom="auto"/>',
        f"  {opt}",
        '  <statistic meansize="0.05"/>',
        "  <worldbody>",
        (
            f'    <geom name="table_g" type="box" pos="{table_pos[0]} {table_pos[1]} {table_pos[2]}" '
            f'size="{table_half[0]} {table_half[1]} {table_half[2]}" '
            f'contype="{physics_config.SCENE_GEOM_CONTYPE}" '
            f'conaffinity="{physics_config.SCENE_GEOM_CONAFFINITY}" '
            f'friction="{friction_str(table_friction)}" '
            f'solref="{tc} {dr}" solimp="{solimp}"/>'
        ),
    ]

    if wall_defs is None:
        wall_defs = []
        if bool(scene_config.WALLS_ENABLED):
            if dims is None:
                raise ValueError("dims is required when wall_defs is not provided and walls are enabled.")
            wall_defs = wall_definitions(
                dims,
                wall_scale=float(scene_config.WALL_SCALE),
                wall_thickness=float(scene_config.WALL_THICKNESS),
                walls_removed=walls_removed,
                center_xy_at_origin=bool(
                    getattr(scene_config, "PARTITION_CENTER_XY_AT_ORIGIN", False)
                ),
            )

    for wi, pos, half in wall_defs:
        lines.append(
            f'    <geom name="wall_{wi}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'size="{half[0]} {half[1]} {half[2]}" '
            f'contype="{physics_config.SCENE_GEOM_CONTYPE}" '
            f'conaffinity="{physics_config.SCENE_GEOM_CONAFFINITY}" '
            f'friction="{friction_str(wall_friction)}" '
            f'solref="{tc} {dr}" solimp="{solimp}"/>'
        )

    drop = float(scene_config.DROP_HEIGHT if box_drop_height is None else box_drop_height)
    density = float(physics_config.BOX_DENSITY)
    use_box_mocaps = not bool(no_mocap)
    box_weld_lines: list[str] = []
    weld_solref = " ".join(str(float(v)) for v in getattr(physics_config, "WELD_SOLREF", (0.005, 1.0)))
    weld_solimp = " ".join(
        str(float(v)) for v in getattr(physics_config, "WELD_SOLIMP", (0.95, 0.99, 0.001))
    )
    weld_torquescale = float(getattr(physics_config, "WELD_TORQUE_SCALE", 1.0))
    for i in range(int(half_extents.shape[0])):
        c = np.asarray(centers[i], dtype=np.float64).reshape(3)
        q = np.asarray(quats_xyzw[i], dtype=np.float64).reshape(4)
        h = np.asarray(half_extents[i], dtype=np.float64).reshape(3)
        mass = density * float(8.0 * h[0] * h[1] * h[2])
        box_pos = (float(c[0]), float(c[1]), float(c[2] + drop))
        box_quat = quat_xyzw_to_mjcf(q)
        if use_box_mocaps:
            lines.append(
                f'    <body name="box_{i}_anchor" mocap="true" '
                f'pos="{box_pos[0]} {box_pos[1]} {box_pos[2]}" quat="{box_quat}">'
                f'      <geom name="box_{i}_anchor_g" type="sphere" size="0.001" '
                f'rgba="0 0 0 0" contype="0" conaffinity="0"/>'
                f"    </body>"
            )
            box_weld_lines.append(
                f'    <weld name="box_{i}_pin" body1="box_{i}_anchor" body2="box_{i}" '
                f'relpose="0 0 0 1 0 0 0" active="false" '
                f'solref="{weld_solref}" solimp="{weld_solimp}" '
                f'torquescale="{weld_torquescale}"/>'
            )
        lines.append(
            f'    <body name="box_{i}" pos="{box_pos[0]} {box_pos[1]} {box_pos[2]}" '
            f'quat="{box_quat}">'
            f'      <freejoint name="box_{i}_j"/>'
            f'      <geom name="box_{i}_g" type="box" mass="{mass}" '
            f'size="{float(h[0])} {float(h[1])} {float(h[2])}" '
            f'contype="{physics_config.BOX_GEOM_CONTYPE}" '
            f'conaffinity="{physics_config.BOX_GEOM_CONAFFINITY}" '
            f'friction="{friction_str(box_friction)}" '
            f'margin="{float(physics_config.BOX_CONTACT_MARGIN)}" '
            f'solref="{tc} {dr}" solimp="{solimp}"/>'
            f"    </body>"
        )

    lines.append("  </worldbody>")
    if box_weld_lines:
        lines.append("  <equality>")
        lines.extend(box_weld_lines)
        lines.append("  </equality>")
    lines.append("</mujoco>")
    return "\n".join(lines)


def _sanitize_urdf_inertials(urdf_text: str) -> str:
    root = ET.fromstring(urdf_text)

    mujoco_tag = root.find("mujoco")
    if mujoco_tag is None:
        mujoco_tag = ET.SubElement(root, "mujoco")
    compiler_tag = mujoco_tag.find("compiler")
    if compiler_tag is None:
        compiler_tag = ET.SubElement(mujoco_tag, "compiler")
    compiler_tag.set("strippath", "false")
    compiler_tag.set("balanceinertia", "true")

    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue

        mass_tag = inertial.find("mass")
        if mass_tag is not None:
            try:
                if float(mass_tag.get("value", "0")) <= 0.0:
                    mass_tag.set("value", "1e-6")
            except ValueError:
                mass_tag.set("value", "1e-6")

        inertia_tag = inertial.find("inertia")
        if inertia_tag is None:
            continue

        ixx = float(inertia_tag.get("ixx", "0"))
        ixy = float(inertia_tag.get("ixy", "0"))
        ixz = float(inertia_tag.get("ixz", "0"))
        iyy = float(inertia_tag.get("iyy", "0"))
        iyz = float(inertia_tag.get("iyz", "0"))
        izz = float(inertia_tag.get("izz", "0"))
        inertia = np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
            dtype=np.float64,
        )
        inertia = 0.5 * (inertia + inertia.T)
        eigvals, eigvecs = np.linalg.eigh(inertia)
        eps = max(1.0e-8, 1.0e-6 * float(np.max(np.abs(np.diag(inertia)))))
        changed = False
        if float(np.min(eigvals)) <= eps:
            eigvals = np.maximum(eigvals, eps)
            changed = True
        if float(eigvals[2]) > float(eigvals[0] + eigvals[1]):
            eigvals[2] = float(eigvals[0] + eigvals[1])
            changed = True
        if not changed:
            continue

        fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        inertia_tag.set("ixx", f"{float(fixed[0, 0]):.12g}")
        inertia_tag.set("ixy", f"{float(fixed[0, 1]):.12g}")
        inertia_tag.set("ixz", f"{float(fixed[0, 2]):.12g}")
        inertia_tag.set("iyy", f"{float(fixed[1, 1]):.12g}")
        inertia_tag.set("iyz", f"{float(fixed[1, 2]):.12g}")
        inertia_tag.set("izz", f"{float(fixed[2, 2]):.12g}")

    return ET.tostring(root, encoding="unicode")


def _load_robot_mjcf_xml(robot_path: Path) -> str:
    path = Path(robot_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Robot model file does not exist: {path}")

    if path.suffix.lower() != ".urdf":
        return path.read_text(encoding="utf-8")

    import mujoco

    sanitized = _sanitize_urdf_inertials(path.read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        delete=False,
        encoding="utf-8",
        dir=str(path.parent),
    ) as tf:
        tf.write(sanitized)
        tmp_path = Path(tf.name)

    try:
        spec = mujoco.MjSpec.from_file(str(tmp_path))
        spec.compile()
        return spec.to_xml()
    finally:
        tmp_path.unlink(missing_ok=True)


def _insert_top_level_before_worldbody(root: ET.Element, element: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        root.append(element)
        return
    root.insert(list(root).index(worldbody), element)


def _merge_top_level_children(
    *,
    scene_root: ET.Element,
    robot_root: ET.Element,
    tag: str,
) -> None:
    robot_section = robot_root.find(tag)
    if robot_section is None:
        return

    scene_section = scene_root.find(tag)
    if scene_section is None:
        _insert_top_level_before_worldbody(scene_root, copy.deepcopy(robot_section))
        return

    existing_keys = {
        _top_level_child_key(child)
        for child in list(scene_section)
        if _top_level_child_key(child) is not None
    }
    for child in list(robot_section):
        key = _top_level_child_key(child)
        if key is not None and key in existing_keys:
            continue
        scene_section.append(copy.deepcopy(child))
        if key is not None:
            existing_keys.add(key)


def _top_level_child_key(child: ET.Element) -> tuple[str, str, str] | None:
    for attr in ("name", "class"):
        value = child.get(attr)
        if value is not None:
            return child.tag, attr, str(value)
    return None


def _ensure_actuator_section(root: ET.Element) -> ET.Element:
    actuator = root.find("actuator")
    if actuator is not None:
        return actuator
    actuator = ET.Element("actuator")
    root.append(actuator)
    return actuator


def _inject_arm_position_actuators(root: ET.Element, motion_config, *, robot_index: int) -> None:
    joint_names = tuple(str(v) for v in motion_config.ARM_JOINT_NAMES)
    kp = tuple(float(v) for v in motion_config.ARM_ACTUATOR_KP)
    kv = tuple(float(v) for v in motion_config.ARM_ACTUATOR_KV)
    lower = tuple(float(v) for v in motion_config.ARM_LOWER_LIMITS)
    upper = tuple(float(v) for v in motion_config.ARM_UPPER_LIMITS)
    if not (len(joint_names) == len(kp) == len(kv) == len(lower) == len(upper)):
        raise ValueError("Arm joint, gain, and limit config lengths must match.")

    actuator = _ensure_actuator_section(root)
    existing_names = {child.get("name") for child in list(actuator)}
    ctrl_limited = "true" if bool(motion_config.ARM_ACTUATOR_CTRL_LIMITED) else "false"

    for i, joint_name in enumerate(joint_names):
        act_name = robot_actuator_name(robot_index, joint_name)
        if act_name in existing_names:
            continue
        ET.SubElement(
            actuator,
            "position",
            {
                "name": act_name,
                "joint": robot_joint_name(robot_index, joint_name),
                "kp": str(kp[i]),
                "kv": str(kv[i]),
                "ctrllimited": ctrl_limited,
                "ctrlrange": f"{lower[i]} {upper[i]}",
            },
        )


def _merge_actuator_children(
    *,
    scene_root: ET.Element,
    robot_root: ET.Element,
    robot_index: int,
) -> None:
    robot_actuator = robot_root.find("actuator")
    if robot_actuator is None:
        return
    scene_actuator = _ensure_actuator_section(scene_root)
    existing_names = {child.get("name") for child in list(scene_actuator)}
    for child in list(robot_actuator):
        child_copy = copy.deepcopy(child)
        name = child_copy.get("name")
        if name:
            child_copy.set("name", robot_name(robot_index, name))
        joint = child_copy.get("joint")
        if joint:
            child_copy.set("joint", robot_joint_name(robot_index, joint))
        if child_copy.get("name") in existing_names:
            continue
        scene_actuator.append(child_copy)
        if child_copy.get("name") is not None:
            existing_names.add(child_copy.get("name"))


def _make_robot_mesh_paths_absolute(robot_root: ET.Element, robot_model_path: Path) -> None:
    mesh_base = Path(robot_model_path).expanduser().resolve().parent
    for mesh in robot_root.findall(".//mesh"):
        file_attr = mesh.get("file")
        if file_attr is None or not file_attr.strip():
            continue

        mesh_path = Path(file_attr).expanduser()
        if mesh_path.is_absolute():
            continue

        resolved = (mesh_base / mesh_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Robot MJCF mesh path could not be resolved: {file_attr!r} -> {resolved}"
            )
        mesh.set("file", str(resolved))


def _set_robot_body_and_geom_properties(
    *,
    robot_anchor: ET.Element,
    physics_config,
    enable_gravcomp: bool,
) -> None:
    for body in robot_anchor.iter("body"):
        if enable_gravcomp:
            body.set("gravcomp", "1")

    contype = str(getattr(physics_config, "ROBOT_GEOM_CONTYPE", 2))
    conaffinity = str(getattr(physics_config, "ROBOT_GEOM_CONAFFINITY", 1))
    for geom in robot_anchor.iter("geom"):
        geom.set("contype", contype)
        geom.set("conaffinity", conaffinity)


def _prefix_robot_worldbody_names(robot_entries: list[ET.Element], *, robot_index: int) -> None:
    tags_with_physical_names = {"body", "joint", "geom", "site", "camera", "light"}
    for entry in robot_entries:
        for elem in entry.iter():
            name = elem.get("name")
            if name is None or elem.tag not in tags_with_physical_names:
                continue
            elem.set("name", robot_name(robot_index, name))


def _ensure_robot_ee_site(
    robot_anchor: ET.Element,
    scene_config,
    *,
    robot_index: int,
) -> None:
    ee_link = str(getattr(scene_config, "ROBOT_EE_LINK", "ee_link"))
    ee_site = robot_site_name(robot_index, ee_link)
    if robot_anchor.find(f'.//site[@name="{ee_site}"]') is not None:
        return

    ee_body = robot_anchor.find(f'.//body[@name="{robot_body_name(robot_index, ee_link)}"]')
    if ee_body is not None:
        ET.SubElement(
            ee_body,
            "site",
            {
                "name": ee_site,
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "size": "0.001 0.001 0.001",
                "rgba": "0 0 0 0",
            },
        )
        return

    # MuJoCo's URDF compiler can flatten fixed links. Re-create the configured
    # EE site on its parent body so Jacobian/FK consumers see a stable frame.
    parent_link = str(getattr(scene_config, "ROBOT_EE_SITE_PARENT_LINK", "panda_link7"))
    parent_body = robot_anchor.find(f'.//body[@name="{robot_body_name(robot_index, parent_link)}"]')
    if parent_body is None:
        return
    site_pos = tuple(float(v) for v in getattr(scene_config, "ROBOT_EE_SITE_POS", (0.0, 0.0, 0.0)))
    site_quat = tuple(
        float(v) for v in getattr(scene_config, "ROBOT_EE_SITE_QUAT_WXYZ", (1.0, 0.0, 0.0, 0.0))
    )
    ET.SubElement(
        parent_body,
        "site",
        {
            "name": ee_site,
            "pos": f"{site_pos[0]} {site_pos[1]} {site_pos[2]}",
            "quat": f"{site_quat[0]} {site_quat[1]} {site_quat[2]} {site_quat[3]}",
            "size": "0.001 0.001 0.001",
            "rgba": "0 0 0 0",
        },
    )


def _robot_base_pose_from_config(
    scene_config,
    *,
    robot_index: int,
    base_pos: tuple[float, float, float] | None,
    base_quat_wxyz: tuple[float, float, float, float] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    suffix = "" if int(robot_index) == 0 else f"_{int(robot_index) + 1}"
    pos_name = f"ROBOT_BASE_POS{suffix}"
    quat_name = f"ROBOT_BASE_QUAT_WXYZ{suffix}"
    resolved_pos = base_pos if base_pos is not None else getattr(scene_config, pos_name)
    resolved_quat = base_quat_wxyz if base_quat_wxyz is not None else getattr(scene_config, quat_name)
    return (
        tuple(float(v) for v in resolved_pos),
        tuple(float(v) for v in resolved_quat),
    )


def merge_robot_into_mjcf(
    mjcf: str,
    *,
    scene_config,
    motion_config,
    physics_config=None,
    enable_gravcomp: bool = True,
    robot_index: int = 0,
    base_pos: tuple[float, float, float] | None = None,
    base_quat_wxyz: tuple[float, float, float, float] | None = None,
) -> str:
    """Merge the configured local robot into an existing scene MJCF.

    This is an initialization-time XML operation. Call it before
    ``mujoco.MjModel.from_xml_string``.
    """

    scene_root = ET.fromstring(mjcf)
    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is None:
        raise ValueError("Scene MJCF has no <worldbody> tag.")

    robot_model_path = Path(scene_config.ROBOT_URDF_PATH)
    robot_xml = _load_robot_mjcf_xml(robot_model_path)
    robot_root = ET.fromstring(robot_xml)
    _make_robot_mesh_paths_absolute(robot_root, robot_model_path)
    robot_worldbody = robot_root.find("worldbody")
    if robot_worldbody is None:
        raise ValueError("Robot MJCF has no <worldbody> tag.")

    _merge_top_level_children(scene_root=scene_root, robot_root=robot_root, tag="asset")
    _merge_top_level_children(scene_root=scene_root, robot_root=robot_root, tag="default")
    _merge_actuator_children(scene_root=scene_root, robot_root=robot_root, robot_index=int(robot_index))

    robot_children = [copy.deepcopy(child) for child in list(robot_worldbody)]
    robot_bodies = [child for child in robot_children if child.tag == "body"]
    if not robot_bodies:
        raise RuntimeError("Robot MJCF worldbody has no top-level <body> entries.")
    # Fixed base-link collisions can compile to root-level geoms instead of bodies.
    robot_worldbody_entries = [
        child
        for child in robot_children
        if child.tag in {"body", "geom", "site"}
    ]
    _prefix_robot_worldbody_names(robot_worldbody_entries, robot_index=int(robot_index))

    base_pos, base_quat = _robot_base_pose_from_config(
        scene_config,
        robot_index=int(robot_index),
        base_pos=base_pos,
        base_quat_wxyz=base_quat_wxyz,
    )
    robot_anchor = ET.Element(
        "body",
        {
            "name": robot_base_body_name(int(robot_index)),
            "pos": f"{base_pos[0]} {base_pos[1]} {base_pos[2]}",
            "quat": f"{base_quat[0]} {base_quat[1]} {base_quat[2]} {base_quat[3]}",
        },
    )
    for child in robot_worldbody_entries:
        robot_anchor.append(child)

    _set_robot_body_and_geom_properties(
        robot_anchor=robot_anchor,
        physics_config=physics_config,
        enable_gravcomp=enable_gravcomp,
    )
    _ensure_robot_ee_site(robot_anchor, scene_config, robot_index=int(robot_index))

    scene_worldbody.append(robot_anchor)
    _inject_arm_position_actuators(scene_root, motion_config, robot_index=int(robot_index))
    return ET.tostring(scene_root, encoding="unicode")
