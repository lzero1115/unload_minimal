from __future__ import annotations

# Time integration.
FPS = 30
SIM_SUBSTEPS = 4
MJ_TIMESTEP = 1.0 / (FPS * SIM_SUBSTEPS)
MJ_GRAVITY = (0.0, 0.0, -9.81)
MJ_INTEGRATOR = "implicitfast"
MJ_SOLVER = "Newton"
MJ_CONE = "pyramidal"
MJ_ITERATIONS = 80
MJ_LS_ITERATIONS = 50
MJ_TOLERANCE = 1.0e-7
MJ_LS_TOLERANCE = 0.01
MJ_IMPRATIO = 5.0
MJ_NOSLIP_ITERATIONS = 5
MJ_NOSLIP_TOLERANCE = 1.0e-6


# Collision masks. The robot follows a prescribed task trajectory in MuJoCo; it
# is not a passive body that box contacts should perturb. Robot/world collision
# is handled by cuRobo during planning, while runtime pile physics is driven by
# boxes, walls, table, and the selected welded box.
ROBOT_GEOM_CONTYPE = 0
ROBOT_GEOM_CONAFFINITY = 0
SCENE_GEOM_CONTYPE = 1
SCENE_GEOM_CONAFFINITY = 3
BOX_GEOM_CONTYPE = SCENE_GEOM_CONTYPE
BOX_GEOM_CONAFFINITY = SCENE_GEOM_CONAFFINITY


# Contact/material defaults for physical scene objects. The support surface is
# a finite table, not an infinite ground plane.
PHYSICS_CONFIG_VERSION = 1

TABLE_MU = 0.7
TABLE_MU_TORSIONAL = 0.04
TABLE_MU_ROLLING = 0.03
TABLE_FRICTION = (TABLE_MU, TABLE_MU_TORSIONAL, TABLE_MU_ROLLING)

WALL_MU = 0.7
WALL_MU_TORSIONAL = 0.04
WALL_MU_ROLLING = 0.015
WALL_FRICTION = (WALL_MU, WALL_MU_TORSIONAL, WALL_MU_ROLLING)

BOX_MU = 0.7
BOX_MU_TORSIONAL = 0.06
BOX_MU_ROLLING = 0.05
BOX_FRICTION = (BOX_MU, BOX_MU_TORSIONAL, BOX_MU_ROLLING)

MUJOCO_CONTACT_KE = 3e4
MUJOCO_CONTACT_KD = 2e3
MUJOCO_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
BOX_CONTACT_MARGIN = 0.0001


# Box mass model.
BOX_DENSITY = 250.0


# Passive pile settlement criteria.
SETTLE_LINEAR_SPEED_MPS = 0.005
SETTLE_ANGULAR_SPEED_RAD = 0.07
SETTLE_CHECK_INTERVAL = 3
SETTLE_STEPS = 20

# Alternative passive-settle criterion. Velocity settling remains enabled;
# either criterion may accept a steady state.
ENABLE_COM_SETTLE = True
COM_SETTLE_MAX_DISPLACEMENT = 0.005
COM_SETTLE_CHECK_INTERVAL = 10
COM_SETTLE_REQUIRED_CHECKS = 15


# Mocap weld attachment defaults used after suction contact is accepted.
WELD_SOLREF = (0.005, 1.0)
WELD_SOLIMP = (0.95, 0.99, 0.001)
WELD_TORQUE_SCALE = 1.0
