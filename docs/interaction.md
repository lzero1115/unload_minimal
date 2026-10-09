# Interactive Unloading

## Initial Placement

Opening a scene leaves physics idle and does not run IK. Use `Initial pile
placement` to translate the pile with `Pile X/Y offset` or rotate it with
`Pile yaw`. Offsets are relative to the loaded scene; rotation is about the
original pile center, and relative box poses are preserved. Default increments
are 1 mm and 1 degree, with adjustable increment fields.

Click `Settle pile and compute IK` to confirm placement. This immediately locks
the pile, settles it, establishes the metric baseline, and computes candidates.
Initial placement and settling are not scored. Only `Reset to saved scene`
unlocks pile placement. Recomputing IK, a failed plan, and undo do not unlock it.
Individual grasp targets remain editable.

## Selecting and Executing

1. Click a blue box in the scene. Blue indicates that the box passed geometric
   filtering and has at least one successful IK target. The selected box turns
   green. Candidate boxes are not listed as UI buttons.
2. Click a face marker on the selected box to choose a grasp frame. All six faces
   remain available; `IK` means that face passed the initial IK calculation,
   while `unverified` requires validation after selection or adjustment.
   The first reachable face is selected initially. Frames are not listed as
   UI buttons.
3. Adjust position with `Tangent U`, `Tangent V`, and `Normal offset`; adjust
   orientation with `Rotate U`, `Rotate V`, and `Rotate normal`. Translation
   values are in meters and rotations are in degrees. Default increments are
   1 mm and 1 degree. `Reset grasp offsets` restores the selected face target.
4. Set `Unload tilt U/V` and `Unload distance` independently to control the
   extraction direction and distance.
5. Click `Validate with MotionGen`. Once the complete plan succeeds, click
   `Execute validated motion`. Selecting another box or face, or editing a
   target parameter, invalidates the previous plan.
6. The robot approaches, attaches, extracts the box, returns, and removes it.
   The remaining pile settles and the step metric is calculated. Click
   `Compute IK candidates` to start the next decision; it is never automatic.

In `--direct-remove` mode, select a blue box and click `Remove selected box`.
The box is hidden and its collisions are disabled immediately, then the pile
settles and is scored. Grasp offsets do not affect this operation. This mode
still uses cuRobo IK for candidates, but does not construct MotionGen.

IK success only establishes target reachability. Full motion validation is a
separate step. Planning runs synchronously and may temporarily block the UI;
the terminal reports progress.

## End-Effector Frames

Both displayed EE frames use red X, green Y, and blue Z axes:

| Frame | Meaning |
| --- | --- |
| Solid `actual_ee_frame` | Current world pose of the MuJoCo `ee_link` site |
| Translucent `contact_ee_target_frame` | Adjusted contact pose supplied to IK/MotionGen |

The target includes the chosen roll and manual orientation edits. Before manual
tilt, its +Z points into the box, opposite the outward surface normal. The
visualization uses the target quaternion directly without another flip.

The target updates immediately when edited, without running MotionGen. It
represents a fixed world-space contact goal, not the approach, extraction, or
return pose. Displaying it does not imply a validated plan. The actual frame
follows the robot during execution; removing the selected box or clearing the
selection hides the target. Direct removal can display the target for inspection
without moving the robot.

Default target opacity is 0.45 and axis length is 0.06 m. Both are adjustable.
`Transparent selected box` makes the inward-pointing target Z axis easier to see.
`Show face N/U/V helpers` enables the optional surface helper axes, which are
hidden by default; face pick markers remain available.

Robot colors alternate by link: odd-numbered links are white, even-numbered
links and the suction tool are pale turquoise.

## Pause, Undo, and Reset

- `Pause / resume` pauses or resumes advancement.
- `Recompute IK candidates` explicitly refreshes the candidates for the current
  stable scene.
- `Undo last unload` restores the state before the most recent completed or
  failed attempt. Repeated undo is supported in both removal modes.
- `Reset to saved scene` reloads the source scene, clears metrics, history,
  selections, and checkpoints, and unlocks initial pile placement.

Undo restores box and robot poses, collision and attachment state, simulation
time, metric baselines, accumulated metrics, and completed step count. It also
trims the corresponding animation history and execution records. It does not
play a reverse trajectory, run IK, or settle the pile again. It is unavailable
while execution or settling is in progress, even when paused.

Checkpoints and animation samples are held in memory. Closing the window does
not save them. An execution or settling failure stops the workflow without
scoring the unfinished step; undo the failed unload or reset the scene. Having
no candidates does not end the session. `complete` means all boxes were removed.

## Timeline Replay

After a step settles, or after all boxes are removed, click `Replay recorded
motion` to enter read-only playback. Replay is also available after initial
settling, when the history may contain only the initial frame.

- Drag `Replay frame` to seek to any recorded frame; seeking pauses playback.
- Use `Previous frame` / `Next frame` for single-frame navigation.
- `Play replay` advances at the control frame rate and stops at the final frame.
  Playing from the end restarts at the beginning.
- `Pause replay` holds the displayed frame.
- `Exit replay` restores the current live display and its selected box, target
  adjustments, and validated plan.

The replay panel shows frame index, timeline time, execution phase, completed
steps, remaining boxes, and cumulative metric for the displayed frame. The
header's `Live` values refer to the actual session. Recorded frames represent
control/execution ticks, not individual MuJoCo physics substeps. Export stride
does not reduce the interactive replay history.

Robot and box geometry are displayed using recorded poses and visibility. The
actual EE frame follows the replayed robot pose. Live candidate markers and the
contact target frame are hidden during replay. Replaying uses a separate display
state and never changes live physics, collision masks, attachment constraints,
metrics, or history. Live execution, selection, planning, and undo are blocked
until replay ends. Reset exits replay and clears the session as usual.

Both motion and direct-removal recordings can be replayed. Undo in live mode
trims the timeline. Export during replay pauses playback and exports the entire
retained history, not just the displayed frame or a prefix of the timeline.

## Blender Export

After initial placement settles, the application records box poses, robot joint
states, box visibility, execution phases, and cumulative metrics in memory.
Every execution/settling tick is recorded, including intermediate ticks when
`--frames-per-render` is greater than one. The first frame is the stable scene
before unloading. Initial placement/settling, idle UI time, IK/planning time,
and helper overlays are excluded.

Click `Export recorded motion` after a step settles or the session completes.
`Export frame stride` defaults to 2; set it to 1 to retain every recorded frame.
Sampling preserves endpoints and both sides of phase, visibility, and metric
changes. Original frame indices retain the animation timing. In direct mode,
the animation contains disappearance and settling without a fabricated robot
trajectory.

Each export creates `exports/<scene>/export_<UTC timestamp>/` containing:

| File | Contents |
| --- | --- |
| `assets.npz` | Box dimensions and robot visual meshes with baked visual origins/scales |
| `animation.npz` | Box world poses/visibility, robot link transforms, frame indices, and metrics |
| `replay_manifest.json` | Actual removal order, target edits, step records, and scene information |
| `metric.txt` | Human-readable per-step and total metrics |
| `blender_import_recorded_motion.py` | Geometry, materials, static environment, and animation import |
| `blender_video.py` | Camera, lighting, metric overlay, and MP4/PNG rendering |

The export uses the `unload_replay_transform_animation_v2` format.
All geometry needed for Blender travels with the directory; the source NPZ,
cuRobo, and MuJoCo are not required for import. Robot colors are preserved.
Boxes use cardboard colors, with the current motion target shown in green;
selection transparency and EE overlays are not exported.

Undo trims future exports, but never modifies directories already exported.
Reset clears the in-memory recording. Exporting is always an explicit action.

### Rendering in Blender

1. Open the exported copy of `blender_video.py` in Blender's Text Editor.
2. Leave `INPUT_DIR = ""`, `MODE = "prepare"`, and `RENDER_NOW = False`, then run
   the script to load the adjacent data and prepare the camera and lights.
3. Adjust `VideoCamera`, then render the animation from Blender. Alternatively,
   set `MODE = "render"` and `RENDER_NOW = True`, and run the script again to
   render using the current scene and camera.

Run the exported `blender_import_recorded_motion.py` alone to import only the
geometry and animation. Import and prepare operations replace the current
Blender scene; render mode preserves it. Blender is a separate application
and is not installed by this project's Python dependency specification.
