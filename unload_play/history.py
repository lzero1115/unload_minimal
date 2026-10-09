"""In-memory animation samples, independent of UI and planning time."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class RecordedFrame:
    phase: str
    sim_frame: int
    sim_time: float
    completed_actions: int
    metric_total: float
    current_action: int | None
    qpos: np.ndarray
    box_positions_w: np.ndarray
    box_quats_wxyz: np.ndarray
    active_mask: np.ndarray


def capture_frame(executor, *, phase: str | None = None) -> RecordedFrame:
    sim = executor.sim
    # Executor updates already refresh geometry. Read it without touching
    # dynamics or the solver state cleared at steady-state checkpoints.
    body_ids = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}")
                for i in range(sim.n_boxes)]
    if any(body_id < 0 for body_id in body_ids):
        raise RuntimeError("Cannot record a scene with missing box bodies")
    return RecordedFrame(
        phase=executor.phase.value if phase is None else phase,
        sim_frame=int(sim.frame_count), sim_time=float(sim.sim_time),
        completed_actions=int(executor.stats.completed_actions),
        metric_total=float(executor.stats.metric_total),
        current_action=executor.current_action,
        qpos=sim.data.qpos.copy(),
        box_positions_w=sim.data.xpos[body_ids].copy(),
        box_quats_wxyz=sim.data.xquat[body_ids].copy(),
        active_mask=executor.active_mask,
    )


class DynamicFrameHistory:
    def __init__(self):
        self._frames: list[RecordedFrame] = []

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        return self._frames[index]

    def clear(self):
        self._frames.clear()

    def reset(self, frame):
        self._frames = [frame]

    def append(self, frame):
        # Called once per executor tick, never from display/IK/target editing.
        # Keep even stationary ticks: they belong to the control timeline.
        self._frames.append(frame)

    def truncate(self, frame_count):
        if not 1 <= frame_count <= len(self._frames):
            raise ValueError("Invalid animation checkpoint length")
        del self._frames[frame_count:]
