import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
import polyscope as ps



FRAME_POINT_RADIUS = 0.004
FRAME_AXIS_RADIUS = 0.0012
FRAME_AXIS_LENGTH = 0.06
POSE_FRAME_POINT_RADIUS = 0.005

# <origin xyz="0 0 0.1" rpy="0 0 1.57"/> in urdf, string to vec3
def parse_vec3(text, default):
    if text is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in text.split()], dtype=np.float64)

# extrinsic roll, pitch yaw to matrix
def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )

    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )

    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rz @ ry @ rx

# first rotation second translation
def make_transform(xyz, rpy):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz
    transform[:3, :3] = rpy_to_matrix(rpy)
    return transform

# find mesh path, default use, not robust enough
def resolve_mesh_path(filename, urdf_path, asset_roots=None):
    urdf_dir = Path(urdf_path).parent
    asset_roots = [] if asset_roots is None else [Path(p) for p in asset_roots]

    candidates = []

    if filename.startswith("package://"):
        rel = filename[len("package://"):]
        parts = rel.split("/", 1)

        if len(parts) == 2:
            package_name, package_rel = parts

            candidates.extend(
                [
                    urdf_dir / package_rel,
                    urdf_dir / rel,
                ]
            )

            for root in asset_roots:
                candidates.extend(
                    [
                        root / "robot" / package_name / package_rel,
                        root / package_name / package_rel,
                        root / rel,
                    ]
                )
        else:
            candidates.append(urdf_dir / rel)
            for root in asset_roots:
                candidates.append(root / rel)

    elif filename.startswith("file://"):
        candidates.append(Path(filename[len("file://"):]))

    else:
        path = Path(filename)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(urdf_dir / path)
            for root in asset_roots:
                candidates.append(root / path)

    for path in candidates:
        resolved = path.resolve()
        project_root = Path(__file__).resolve().parents[1]
        if resolved.is_relative_to(project_root) and resolved.is_file():
            return str(resolved)

    tried = "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(
        f"Could not find mesh file: {filename}\n"
        f"Tried:\n{tried}"
    )


def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")

    if isinstance(mesh, trimesh.Trimesh):
        return mesh

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
        if isinstance(mesh, trimesh.Trimesh):
            return mesh

    raise RuntimeError(f"Could not load mesh as Trimesh: {mesh_path}")


def make_scale_transform(scale):
    scale_tf = np.eye(4, dtype=np.float64)
    scale_tf[0, 0] = scale[0]
    scale_tf[1, 1] = scale[1]
    scale_tf[2, 2] = scale[2]
    return scale_tf


def require_geometry_attr(tag, attr, geometry_type):
    value = tag.get(attr)
    if value is None:
        raise ValueError(f"URDF {geometry_type} visual geometry is missing required '{attr}' attribute.")
    return value


def load_visual_geometry(geometry, urdf_path, asset_roots=None):
    mesh_tag = geometry.find("mesh")
    if mesh_tag is not None:
        filename = mesh_tag.get("filename")
        if filename is None:
            return None

        scale = parse_vec3(mesh_tag.get("scale"), [1.0, 1.0, 1.0])
        mesh_path = resolve_mesh_path(filename, urdf_path, asset_roots)
        mesh = load_mesh(mesh_path) # local coordinate
        return mesh, make_scale_transform(scale), mesh_path

    box_tag = geometry.find("box")
    if box_tag is not None:
        size = parse_vec3(require_geometry_attr(box_tag, "size", "box"), [1.0, 1.0, 1.0])
        mesh = trimesh.creation.box(extents=size)
        return mesh, np.eye(4, dtype=np.float64), "primitive:box"

    cylinder_tag = geometry.find("cylinder")
    if cylinder_tag is not None:
        radius = float(require_geometry_attr(cylinder_tag, "radius", "cylinder"))
        length = float(require_geometry_attr(cylinder_tag, "length", "cylinder"))
        mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=48)
        return mesh, np.eye(4, dtype=np.float64), "primitive:cylinder"

    sphere_tag = geometry.find("sphere")
    if sphere_tag is not None:
        radius = float(require_geometry_attr(sphere_tag, "radius", "sphere"))
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        return mesh, np.eye(4, dtype=np.float64), "primitive:sphere"

    return None


def load_visual_meshes_from_urdf(urdf_path, asset_roots=None):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_meshes = {}

    for link in root.findall("link"):
        link_name = link.get("name")
        visuals = []

        for visual in link.findall("visual"):
            origin = visual.find("origin")

            if origin is not None:
                xyz = parse_vec3(origin.get("xyz"), [0.0, 0.0, 0.0])
                rpy = parse_vec3(origin.get("rpy"), [0.0, 0.0, 0.0])
            else:
                xyz = np.zeros(3, dtype=np.float64)
                rpy = np.zeros(3, dtype=np.float64)

            visual_tf = make_transform(xyz, rpy) # transformation respect to parent link frame

            geometry = visual.find("geometry")
            if geometry is None:
                continue

            visual_geometry = load_visual_geometry(geometry, urdf_path, asset_roots)
            if visual_geometry is None:
                continue
            mesh, scale_tf, mesh_path = visual_geometry

            visuals.append(
                {
                    "mesh": mesh,
                    "visual_tf": visual_tf,
                    "scale_tf": scale_tf,
                    "mesh_path": mesh_path,
                }
            )

        link_meshes[link_name] = visuals

    return link_meshes

# rotation range of each joint
def read_joint_limits(urdf_path, joint_names):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    lower = []
    upper = []

    for name in joint_names:
        joint = joints.get(name)
        limit = joint.find("limit") if joint is not None else None

        if limit is None:
            lower.append(-np.pi)
            upper.append(np.pi)
        else:
            lower.append(float(limit.get("lower", -np.pi)))
            upper.append(float(limit.get("upper", np.pi)))

    return np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)

# random initialization of q
def sample_q(lower, upper):
    alpha = np.random.rand(len(lower)).astype(np.float32)
    return lower + alpha * (upper - lower)


def make_q_tensor(q):
    return torch.tensor(q, dtype=torch.float32).unsqueeze(0)

# transformation of link frame respect to world frame
def matrix_from_fk(fk_result, link_name):
    mat = fk_result[link_name].get_matrix()
    if mat.ndim == 3:
        mat = mat[0]
    return mat.detach().cpu().numpy().reshape(4, 4)


def transform_vertices(vertices, transform):
    ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
    vertices_h = np.concatenate([vertices, ones], axis=1) # homogeneous
    transformed_h = (transform @ vertices_h.T).T
    return transformed_h[:, :3]


# def transform_points(points, transform):
#     ones = np.ones((points.shape[0], 1), dtype=points.dtype)
#     points_h = np.concatenate([points, ones], axis=1)
#     transformed_h = (transform @ points_h.T).T
#     return transformed_h[:, :3]


# link frame in world coordinates, chain is from pk.build_serial_chain_from_urdf
def compute_fk_data(chain, q_tensor):
    try:
        fk = chain.forward_kinematics(q_tensor, end_only=False)
    except TypeError:
        fk = chain.forward_kinematics(q_tensor)

    link_transforms = {}
    frame_names = []
    frames = []

    for link in chain.get_links():
        link_name = link.name

        if link_name not in fk:
            continue

        tf = matrix_from_fk(fk, link_name)
        link_transforms[link_name] = tf
        frame_names.append(link_name)
        frames.append(tf)

    return link_transforms, frame_names, frames

# for initialization or reset
def build_mesh_records(chain, link_meshes):
    records = []

    for link in chain.get_links():
        link_name = link.name

        for visual_id, visual in enumerate(link_meshes.get(link_name, [])):
            mesh = visual["mesh"]

            records.append(
                {
                    "name": f"{link_name}_visual_{visual_id}",
                    "link_name": link_name,
                    "vertices": np.asarray(mesh.vertices, dtype=np.float64).copy(),
                    "faces": np.asarray(mesh.faces).copy(),
                    "visual_tf": visual["visual_tf"],
                    "scale_tf": visual["scale_tf"],
                    "mesh_path": visual["mesh_path"],
                    "handle": None,
                    "last_vertices": None,
                }
            )

    return records

# once
def register_robot_meshes(mesh_records, link_transforms):
    for record in mesh_records:
        link_name = record["link_name"]

        if link_name in link_transforms:
            world_link_tf = link_transforms[link_name]
            world_mesh_tf = world_link_tf @ record["visual_tf"] @ record["scale_tf"]
            vertices = transform_vertices(record["vertices"], world_mesh_tf)
        else:
            vertices = record["vertices"]

        record["last_vertices"] = vertices

        record["handle"] = ps.register_surface_mesh(
            record["name"],
            vertices,
            record["faces"],
            color=(0.85, 0.85, 0.85),
            smooth_shade=True,
        )
        

# for updating
def update_robot_meshes(mesh_records, link_transforms):
    all_vertices = []

    for record in mesh_records:
        link_name = record["link_name"]

        if link_name not in link_transforms:
            continue

        world_link_tf = link_transforms[link_name]
        world_mesh_tf = world_link_tf @ record["visual_tf"] @ record["scale_tf"]

        new_vertices = transform_vertices(record["vertices"], world_mesh_tf)
        record["last_vertices"] = new_vertices
        record["handle"].update_vertex_positions(new_vertices)

        all_vertices.append(new_vertices)

    if len(all_vertices) == 0:
        return None, None

    all_vertices = np.concatenate(all_vertices, axis=0)
    return all_vertices.min(axis=0), all_vertices.max(axis=0)
