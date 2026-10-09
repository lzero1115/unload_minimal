from __future__ import annotations

from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_ROBOT_ROOT = _TASK_ROOT / "robots" / "curobo_franka"


# Partition geometry for the initial box pile.
PARTITION_TARGET_NUM_BOXES = 50
PARTITION_DIMS = (0.5, 0.5, 0.6)
PARTITION_SEED = 42
# The low-level partition kernel emits local coordinates in
# [0, dx] x [0, dy] x [0, dz]. Scene data is stored in world coordinates with the
# bin centered horizontally at the world origin:
# [-dx/2, dx/2] x [-dy/2, dy/2] x [0, dz].
PARTITION_CENTER_XY_AT_ORIGIN = True


# Support surface. This task uses the table itself as the support surface, not
# an extra infinite ground plane.
DROP_HEIGHT = 0.03
TABLE_POS = (0.0, 0.0, -0.025)
TABLE_HALF_EXTENTS = (10.0, 10.0, 0.025)


# Optional bin walls around the partition region.
WALLS_ENABLED = True
WALL_THICKNESS = 0.02
WALL_SCALE = 1.2
# Default opening is wall 2, the centered scene's -y side.
DEFAULT_REMOVED_WALLS = (2,)


# Post-settlement scene validity filter for the default -y opening. Boxes whose
# final center moves past the original bin boundary are considered fallen out
# and are excluded from saved npz snapshots.
FILTER_SETTLED_BOXES_BEYOND_NEG_Y = True


# Local Franka suction resources.
ROBOT_ROOT_PATH = _LOCAL_ROBOT_ROOT
ROBOT_URDF_PATH = (
    _LOCAL_ROBOT_ROOT
    / "assets"
    / "robot"
    / "franka_description"
    / "franka_suction_new.urdf"
)
ROBOT_ASSET_ROOT_PATH = _LOCAL_ROBOT_ROOT / "assets"
ROBOT_EE_LINK = "ee_link"
# Fixed-link ee sites can be flattened by MuJoCo's URDF compiler. These values
# re-create the suction-tip frame from panda_link7 when that happens.
ROBOT_EE_SITE_PARENT_LINK = "panda_link7"
ROBOT_EE_SITE_POS = (0.0, 0.0, 0.143)
ROBOT_EE_SITE_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


# Robot placement in the MuJoCo world frame.
# Preserves the old relative placement to the bin center after centering the
# partition/walls at world xy=(0, 0).
ROBOT_BASE_POS = (0.0, -0.45, 0.0)
ROBOT_BASE_QUAT_WXYZ = (0.7071067812, 0.0, 0.0, 0.7071067812)
