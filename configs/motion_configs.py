from __future__ import annotations

# Franka arm only. There is no gripper control in the suction setup.
ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
ARM_HOME_Q = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
ARM_LOWER_LIMITS = (
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
)
ARM_UPPER_LIMITS = (
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
)


# MuJoCo position actuators for the 7-DoF arm.
ARM_ACTUATOR_KP = (4500.0, 4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0)
ARM_ACTUATOR_KV = (450.0, 450.0, 450.0, 350.0, 350.0, 200.0, 200.0)
ARM_ACTUATOR_CTRL_LIMITED = True


# Runtime Jacobian IK used for target tracking inside MuJoCo.
DIFF_IK_ENABLED = True
DIFF_IK_ITERATIONS = 24
DIFF_IK_DAMPING = 0.1
DIFF_IK_MAX_JOINT_STEP = 0.12
DIFF_IK_POSITION_TOL = 0.01
DIFF_IK_ROTATION_TOL_DEG = 5.0


# cuRobo IK settings for candidate filtering/planning.
CUROBO_IK_NUM_SEEDS = 32
CUROBO_IK_POSITION_THRESHOLD = 0.005
CUROBO_IK_ROTATION_THRESHOLD = 0.05
CUROBO_IK_TARGET_BATCH_SIZE = 256
CUROBO_SELF_COLLISION_CHECK = True
CUROBO_SELF_COLLISION_OPT = True
CUROBO_USE_CUDA_GRAPH = False


# cuRobo MotionGen settings for candidate filtering/planning.
CUROBO_MOTION_NUM_IK_SEEDS = 32
CUROBO_MOTION_NUM_TRAJOPT_SEEDS = 4
CUROBO_MOTION_NUM_GRAPH_SEEDS = 4
CUROBO_MOTION_ATTEMPTS = 2
CUROBO_MOTION_TIMEOUT = 10.0
CUROBO_MOTION_SUCCESS_RATIO = 1.0
CUROBO_MOTION_ENABLE_GRAPH = False
CUROBO_MOTION_WARMUP = False
CUROBO_MOTION_USE_CUDA_GRAPH = False
CUROBO_MOTION_COLLISION_ACTIVATION_DISTANCE = 0.0
CUROBO_MOTION_TARGET_BATCH_SIZE = 32


# Target tracking cadence. Physics substeps live in phys_configs.py.
CONTROL_FPS = 30
FRAME_DT = 1.0 / CONTROL_FPS
MAX_IK_RESIDUAL = 0.05
