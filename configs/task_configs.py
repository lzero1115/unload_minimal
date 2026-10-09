from __future__ import annotations


# Geometry from franka_suction_new.urdf. The cuRobo ee_link is already the
# suction-tip frame, so grasp targets do not need a hand-to-tip back-offset.
SUCTION_MOUNT_Z_FROM_LINK7 = 0.107
SUCTION_NECK_RADIUS = 0.006
SUCTION_NECK_LENGTH = 0.024
SUCTION_HEAD_RADIUS = 0.016
SUCTION_HEAD_LENGTH = 0.012
SUCTION_TOOL_LENGTH = SUCTION_NECK_LENGTH + SUCTION_HEAD_LENGTH
SUCTION_TIP_Z_FROM_LINK7 = SUCTION_MOUNT_Z_FROM_LINK7 + SUCTION_TOOL_LENGTH
SUCTION_DIAMETER = 2.0 * SUCTION_HEAD_RADIUS


# Suction-tip grasp semantics. Generate targets directly for ee_link.
GRASP_APPROACH_CLEARANCE = 2.0 * SUCTION_TOOL_LENGTH
GRASP_REFINE_CLEARANCE = 0.0
GRASP_CONTACT_CLEARANCE = 0.0
GRASP_PREFER_TOP_FACES = True
GRASP_MIN_FACE_EDGE_LENGTH = SUCTION_DIAMETER
APPROACH_CONTACT_INTERPOLATION_STEPS = 8


# MotionGen retreat execution policy. This is applied to the reversed
# MotionGen path, so the prefix starts at the grasp/contact pose.
MOTIONGEN_RETREAT_SLOW_PREFIX_RATIO = 0.25
MOTIONGEN_RETREAT_SLOW_PREFIX_DURATION = 1.2


# Inter-steady unload metric weights.
# v3 uses a fixed-scale vertical disturbance metric. MuJoCo state is stored in
# meters, but metric length values are scaled before volume*height is computed.
# With METRIC_LENGTH_SCALE=10, a 4cm box edge stored as 0.04 is evaluated as 0.4.
#
#   scaled_length = length_m * METRIC_LENGTH_SCALE
#   energy_term_i = scaled_volume_i * abs(scaled_delta_z_i)
#   metric_i = alpha_energy * energy_term_i
#
# Pose terms are kept as zero-valued compatibility diagnostics; they are not
# part of the v3 disturbance metric.
METRIC_ALPHA_ENERGY = 10.0
METRIC_BETA_POSE = 0.0
METRIC_LAMBDA_TRANS = 1.0
METRIC_LAMBDA_ROT = 1.0
METRIC_LENGTH_SCALE = 10.0

# Per-box metric contributions at or below this threshold are treated as
# numerical residue and omitted from the accumulated disturbance metric.
# The threshold is applied after METRIC_ALPHA_ENERGY scaling.
METRIC_STEP_DEADBAND = 0.1
