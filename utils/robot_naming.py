from __future__ import annotations


def robot_prefix(robot_index: int) -> str:
    index = int(robot_index)
    if index < 0:
        raise ValueError(f"robot_index must be >= 0, got {index}.")
    return f"robot{index}_"


def robot_name(robot_index: int, logical_name: str) -> str:
    name = str(logical_name)
    prefix = robot_prefix(robot_index)
    if name.startswith(prefix):
        return name
    return f"{prefix}{name}"


def robot_body_name(robot_index: int, logical_body_name: str) -> str:
    return robot_name(robot_index, logical_body_name)


def robot_joint_name(robot_index: int, logical_joint_name: str) -> str:
    return robot_name(robot_index, logical_joint_name)


def robot_geom_name(robot_index: int, logical_geom_name: str) -> str:
    return robot_name(robot_index, logical_geom_name)


def robot_site_name(robot_index: int, logical_site_name: str) -> str:
    return robot_name(robot_index, logical_site_name)


def robot_base_body_name(robot_index: int) -> str:
    return robot_name(robot_index, "robot_base")


def robot_actuator_name(robot_index: int, logical_joint_name: str) -> str:
    return robot_name(robot_index, f"act_{logical_joint_name}")
