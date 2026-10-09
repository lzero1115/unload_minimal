# Physics and Metrics

## Configuration Files

| File | Responsibility |
| --- | --- |
| `configs/phys_configs.py` | Timestep, contacts, friction, density, solver, settling, and attachment weld |
| `configs/scene_configs.py` | Table/walls, robot assets, and robot base pose |
| `configs/motion_configs.py` | Joint settings, IK/MotionGen seeds, tolerances, and planning limits |
| `configs/task_configs.py` | Suction geometry, approach clearance, interpolation, and metrics |
| `configs/env_configs.py` | Settling limits, removal location, ray tests, and face filtering |
| `configs/generation_configs.py` | Single-scene Temtsin generation defaults |

## Default Physics

The default simulation settings are:

| Setting | Value |
| --- | --- |
| Control rate / physics substeps | 30 Hz / 4; physics timestep 1/120 s |
| Gravity | `(0, 0, -9.81)` m/s^2 |
| Integrator / solver / friction cone | `implicitfast` / `Newton` / `pyramidal` |
| Solver / line-search iterations | 80 / 50 |
| Table friction | `(0.7, 0.04, 0.03)` |
| Wall friction | `(0.7, 0.04, 0.015)` |
| Box friction | `(0.7, 0.06, 0.05)` |
| Box density | 250 kg/m^3 |
| Contact KE / KD | `3e4` / `2e3`, converted to `solref` |
| Velocity settling | Linear <= 0.005 m/s and angular <= 0.07 rad/s, checked every 3 frames for 20 consecutive checks |
| Alternative COM settling | Displacement <= 0.005 m, checked every 10 frames for 15 consecutive checks |
| Settling limit | 600 control frames, approximately 20 simulated seconds |
| Candidate ray spacing / minimum hits | 0.02 m / 8 |
| Face clearance distance / minimum openness score | 0.10 m / 3 |

Either settling criterion is sufficient. `--max-settle-frames` changes the
timeout, and `--frames-per-render` changes display advancement without changing
the timestep or settling thresholds.

Friction tuples specify sliding, torsional, and rolling friction. The MJCF does
not explicitly override `condim`; supplying a friction tuple does not itself
enable six-dimensional contacts. Robot MuJoCo collision masks are zero. cuRobo
handles robot/world collision constraints during planning, while MuJoCo handles
contacts between boxes, the table, walls, and the attached box.

Approach/contact phases apply planned joint positions without advancing pile
physics. Extraction and return advance MuJoCo with the carried box.

## Attachment

Each box has a non-colliding mocap anchor and an initially inactive weld equality
constraint. At attachment, the executor stores the box transform relative to
the EE, activates the weld, and updates the anchor every physics substep to
follow the EE. `xfrc_applied` compensates the attached box's gravity. Releasing
disables the weld and clears this compensation.

This is a constrained attachment model; vacuum pressure, seal quality, and
suction break limits are not simulated.

## Scene Generation

Temtsin generation uses no physical side walls by default. Boxes are dropped
above the current mean pile XY position. After settling, boxes outside virtual
XY boundaries are removed and replaced until the requested count is reached,
followed by a stability review.

The default `--dims 0.5 0.5 0.6` and `--filter-scale 1.3` place XY center limits
at +/-0.325 m. The Z dimension does not define a closed container. Default boxes
have a 0.04 by 0.06 m cross section, with lengths sampled from 0.08, 0.10, 0.12,
0.14, and 0.16 m.

```bash
python scripts/scene_gen_temtsin.py --nb 30 --seed 43 --headless \
  --dims 0.5 0.5 0.6 --filter-scale 1.3 \
  --sliding-mu 0.7 --box-density 250 --save-path scenes/my_scene.npz
```

NPZ scenes store box poses, half extents, boundaries, friction, density,
generation arguments, and physics metadata. Loading uses saved geometry,
friction, and density, with local defaults for missing legacy material fields.
Solver, timestep, and settling settings come from this project's configuration;
saved physics metadata provides provenance rather than overriding them.

## Disturbance Metric

The v3 metric measures height changes of boxes remaining after each removal,
between consecutive stable states:

```text
raw_i = alpha * (volume_m3 * length_scale^3) * (abs(delta_z_m) * length_scale)
step_metric = sum(raw_i for each remaining box if raw_i > deadband)
```

Defaults are `alpha = 10`, `length_scale = 10`, and a per-box `deadband = 0.1`.
Removed boxes are excluded. Horizontal translation and rotation do not contribute
to the v3 total. This is a scaled vertical disturbance score, not physical
energy in joules. Lower values indicate less height change among remaining
boxes. Both removal modes use the same metric, but can produce different
disturbances.

## Runtime Organization

`runtime/environment.py` owns scene state, active/candidate masks, attachment
helpers, and metric tracking. `runtime/geometry.py` provides geometric queries;
`ik_candidates.py` constructs target poses and solver inputs, and `ik_runtime.py`
reuses the batched cuRobo IK solver between explicit requests.

`unload_play/controller.py` manages placement, selection, validation, execution,
undo, and recording. `task_executor.py` implements approach, contact, attachment,
extraction, return, removal, and post-removal settling. Candidate filtering
operates over all active boxes. The runtime manages scene state and manual
interaction without training observation/reward interfaces.
