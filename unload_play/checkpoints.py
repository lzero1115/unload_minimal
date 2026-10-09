"""In-memory checkpoints of stable scenes before manual unload attempts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .task_types import TaskPhase


_DATA_ARRAYS = (
    "qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "act", "qfrc_applied",
    "mocap_pos", "mocap_quat", "eq_active", "xfrc_applied",
)
_MODEL_ARRAYS = ("geom_contype", "geom_conaffinity")
_CANDIDATE_FIELDS = ("_action_mask", "ray_mask", "ray_debug", "frame_targets")


@dataclass(frozen=True)
class UnloadCheckpoint:
    data_arrays: dict[str, np.ndarray]
    model_arrays: dict[str, np.ndarray]
    data_time: float
    frame_count: int
    sim_time: float
    last_unsettled: np.ndarray
    last_max_lin: float
    last_max_ang: float
    active_mask: np.ndarray
    candidate_state: dict[str, Any]
    metric: Any
    stats: Any
    selected_action: int | None
    record_count: int
    history_frame_count: int


def capture_checkpoint(executor, *, record_count: int, history_frame_count: int) -> UnloadCheckpoint:
    if executor.initializing or executor.phase not in (TaskPhase.IDLE, TaskPhase.DONE):
        raise RuntimeError("An unload checkpoint requires a settled scene")
    if executor.attached_box is not None:
        raise RuntimeError("Cannot checkpoint a scene while carrying a box")
    sim = executor.sim
    env = executor.env
    return UnloadCheckpoint(
        data_arrays={name: getattr(sim.data, name).copy() for name in _DATA_ARRAYS},
        model_arrays={name: getattr(sim.model, name).copy() for name in _MODEL_ARRAYS},
        data_time=float(sim.data.time), frame_count=int(sim.frame_count), sim_time=float(sim.sim_time),
        last_unsettled=sim.last_unsettled.copy(),
        last_max_lin=float(sim.last_max_lin), last_max_ang=float(sim.last_max_ang),
        active_mask=env._active_mask.copy(),
        candidate_state={name: deepcopy(getattr(env, name)) for name in _CANDIDATE_FIELDS if hasattr(env, name)},
        # The metric is a plain Python/NumPy object; this includes its baseline,
        # per-box diagnostics and complete accumulated step history.
        metric=deepcopy(env._metric()), stats=deepcopy(executor.stats),
        selected_action=executor.selected_action, record_count=int(record_count),
        history_frame_count=int(history_frame_count),
    )


def restore_checkpoint(executor, checkpoint: UnloadCheckpoint) -> None:
    sim = executor.sim
    env = executor.env
    # Check compatibility before changing any live state.
    for name, value in checkpoint.data_arrays.items():
        if getattr(sim.data, name).shape != value.shape:
            raise ValueError(f"Checkpoint does not match this scene: data.{name}")
    for name, value in checkpoint.model_arrays.items():
        if getattr(sim.model, name).shape != value.shape:
            raise ValueError(f"Checkpoint does not match this scene: model.{name}")
    executor._release_weld_if_needed()
    for name, value in checkpoint.model_arrays.items():
        getattr(sim.model, name)[:] = value
    for name, value in checkpoint.data_arrays.items():
        getattr(sim.data, name)[:] = value
    sim.data.time = checkpoint.data_time
    sim.frame_count = checkpoint.frame_count
    sim.sim_time = checkpoint.sim_time
    sim.running = False
    sim.last_unsettled = checkpoint.last_unsettled.copy()
    sim.last_max_lin = checkpoint.last_max_lin
    sim.last_max_ang = checkpoint.last_max_ang
    env._active_mask = checkpoint.active_mask.copy()
    for name in _CANDIDATE_FIELDS:
        if name in checkpoint.candidate_state:
            setattr(env, name, deepcopy(checkpoint.candidate_state[name]))
        elif hasattr(env, name):
            delattr(env, name)
    env.metric = deepcopy(checkpoint.metric)
    executor.stats = deepcopy(checkpoint.stats)
    executor.selected_action = checkpoint.selected_action
    executor.initializing = False
    executor.plan = None
    executor.path_cursor = 0
    executor._attached_box = -1
    executor._box_in_ee = None
    executor._last_carry_q = None
    executor._settle_consecutive = 0
    executor._com_settle_tracker.reset()
    executor._candidate_mask.fill(False)
    executor._proxy_candidate_count = 0
    executor.failure_reason = ""
    executor.phase = TaskPhase.DONE
    executor.status_message = "Undo complete; click Compute IK candidates to choose again."
    sim.forward()
    # forward() refreshes derived geometry and dynamics. Restore the saved solver
    # history afterward, so undo itself does not advance or re-settle the scene.
    sim.data.qacc[:] = checkpoint.data_arrays["qacc"]
    sim.data.qacc_warmstart[:] = checkpoint.data_arrays["qacc_warmstart"]
