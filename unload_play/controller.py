from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from configs import motion_configs as mc
from configs.env_configs import DEFAULT_ENV_CONFIG
from .candidate_service import StableCandidateService
from .checkpoints import capture_checkpoint, restore_checkpoint
from .history import DynamicFrameHistory, capture_frame
from .task_executor import UnloadTaskExecutor
from .task_types import TaskPhase
from .target_types import (
    DecisionPhase, FrameAdjustment, SelectedTarget, UnloadAdjustment,
    PileAdjustment, adjusted_pile_poses, unload_target_position_w,
)


class PlayController:
    def __init__(self, scene_path, *, direct_remove=False, config=DEFAULT_ENV_CONFIG,
                 ik_batch_size=mc.CUROBO_IK_TARGET_BATCH_SIZE):
        self.scene_path = scene_path
        self.config = config
        self.direct_remove = bool(direct_remove)
        self.executor = UnloadTaskExecutor(scene_path, config=config)
        self.candidates = StableCandidateService(ik_batch_size=ik_batch_size)
        self.planner = None  # Direct mode never constructs/imports MotionGen.
        self.phase = DecisionPhase.AWAIT_IK
        self.snapshot = None
        self.selected_box = None
        self.selected_frame_slot = None
        self.adjustment = FrameAdjustment()
        self.unload_adjustment = UnloadAdjustment()
        self.plan = None
        self.warning = ""
        self.records = []  # Session only; no automatic disk output.
        self.checkpoints = []
        self.history = DynamicFrameHistory()
        self.replay_active = False
        self.replay_playing = False
        self.replay_index = 0
        self._next_replay_frame_s = 0.0
        self._pending_record = None
        self._pending_job = None
        self._initialize_pile_placement()

    def _initialize_pile_placement(self):
        sim = self.executor.sim
        self.pile_adjustment = PileAdjustment()
        self.pile_placement_locked = False
        self.pile_scene_dirty = True
        self._pile_reference_positions = sim.centers.copy()
        self._pile_reference_quats = sim.quats_xyzw[:, [3, 0, 1, 2]].copy()
        self._pile_reference_active = self.executor.active_mask
        self._ik_after_initial_settle = False

    @property
    def can_undo(self):
        return not self.replay_active and bool(self.checkpoints) and self._pending_job is None and (
            self.stable_for_input or self.phase in (DecisionPhase.COMPLETE, DecisionPhase.FAILED)
        )

    def undo_last_unload(self):
        if not self.can_undo:
            raise RuntimeError("Undo is available after an unload completes or fails")
        checkpoint = self.checkpoints[-1]
        restore_checkpoint(self.executor, checkpoint)
        self.history.truncate(checkpoint.history_frame_count)
        del self.records[checkpoint.record_count:]
        self.checkpoints.pop()
        self.snapshot = None
        self._pending_record = self._pending_job = None
        self._ik_after_initial_settle = False
        self.pile_scene_dirty = False
        self._clear_selection()
        self.phase = DecisionPhase.AWAIT_IK
        self.warning = ""
        print(f"[undo] steps={self.executor.stats.completed_actions} "
              f"metric_total={self.executor.stats.metric_total:.6f} "
              f"remaining={np.count_nonzero(self.executor.active_mask)}; waiting for manual IK", flush=True)

    @property
    def can_export(self):
        return (len(self.history) > 0 and not self.pile_scene_dirty
                and not self.executor.initializing and self._pending_job is None
                and (self.stable_for_input or self.replay_active or self.phase == DecisionPhase.COMPLETE))

    @property
    def can_replay(self):
        return not self.replay_active and self.can_export

    @property
    def replay_frame(self):
        return self.history[self.replay_index] if self.replay_active else None

    def enter_replay(self):
        if not self.can_replay:
            raise RuntimeError("Replay is available after settling, while idle or complete")
        self.replay_active = True
        self.replay_playing = False
        self.replay_index = 0
        self._next_replay_frame_s = 0.0

    def set_replay_index(self, index):
        if not self.replay_active:
            raise RuntimeError("Enter replay before seeking")
        self.replay_playing = False
        self.replay_index = max(0, min(int(index), len(self.history) - 1))

    def play_replay(self):
        if not self.replay_active:
            raise RuntimeError("Enter replay before playing")
        if self.replay_index == len(self.history) - 1:
            self.replay_index = 0
        self.replay_playing = len(self.history) > 1
        self._next_replay_frame_s = time.perf_counter() + self.executor.sim.frame_dt

    def pause_replay(self):
        self.replay_playing = False

    def advance_replay(self):
        if not self.replay_active or not self.replay_playing:
            return
        now = time.perf_counter()
        if now < self._next_replay_frame_s:
            return
        self.replay_index = min(self.replay_index + 1, len(self.history) - 1)
        self._next_replay_frame_s = now + self.executor.sim.frame_dt
        if self.replay_index == len(self.history) - 1:
            self.replay_playing = False

    def exit_replay(self):
        # Live physics, selections, plans, and metrics were never modified.
        self.replay_active = False
        self.replay_playing = False
        self.replay_index = 0
        self._next_replay_frame_s = 0.0

    @property
    def can_adjust_pile(self):
        return (not self.pile_placement_locked and self.stable_for_input
                and self.executor.stats.completed_actions == 0
                and self.executor.current_action is None)

    def set_pile_adjustment(self, **values):
        if not self.can_adjust_pile:
            raise RuntimeError("Initial pile placement is locked; reset the scene to adjust it again")
        adjustment = replace(self.pile_adjustment, **values)
        if not all(np.isfinite(v) for v in (
            adjustment.translation_x_m, adjustment.translation_y_m, adjustment.yaw_deg,
        )):
            raise ValueError("Pile adjustments must be finite")
        if adjustment == self.pile_adjustment:
            return
        positions, quats = adjusted_pile_poses(
            positions_w=self._pile_reference_positions, quats_wxyz=self._pile_reference_quats,
            active_mask=self._pile_reference_active, adjustment=adjustment,
        )
        self.executor.set_initial_pile_poses(positions_w=positions, quats_wxyz=quats)
        self.pile_adjustment = adjustment
        self.pile_scene_dirty = True
        self.snapshot = None
        self._clear_selection()
        self.phase = DecisionPhase.AWAIT_IK
        self.warning = ""

    @property
    def stable_for_input(self):
        return self.phase in {
            DecisionPhase.AWAIT_IK,
            DecisionPhase.AWAIT_BOX, DecisionPhase.AWAIT_FRAME,
            DecisionPhase.AWAIT_PLAN, DecisionPhase.READY_TO_EXECUTE,
        } and self._pending_job is None and not self.replay_active

    @property
    def interaction_available(self):
        return self.stable_for_input

    @property
    def ik_mask(self):
        if self.snapshot is None:
            return np.zeros(self.executor.sim.n_boxes, dtype=bool)
        return self.snapshot.ik_mask

    @property
    def selected_frames(self):
        if self.snapshot is None or self.selected_box is None:
            return ()
        return self.snapshot.frames_by_box.get(self.selected_box, ())

    @property
    def selected_candidate(self):
        slot = self.selected_frame_slot
        return None if slot is None or slot >= len(self.selected_frames) else self.selected_frames[slot]

    @property
    def selected_target(self):
        candidate = self.selected_candidate
        return None if candidate is None else SelectedTarget(candidate, self.adjustment)

    @property
    def current_unload_target_position_w(self):
        target = self.selected_target
        return None if target is None else unload_target_position_w(target, self.unload_adjustment)

    def _clear_selection(self):
        self.selected_box = self.selected_frame_slot = None
        self.adjustment = FrameAdjustment()
        self.unload_adjustment = UnloadAdjustment()
        self.plan = None

    def _queue_candidates(self):
        self._clear_selection()
        self.snapshot = None
        if not np.any(self.executor.active_mask):
            self.phase = DecisionPhase.COMPLETE
            self.warning = ""
            return
        self.phase = DecisionPhase.COMPUTING_IK
        self._pending_job = "candidates"

    def refresh_candidates(self):
        if not self.stable_for_input or self.executor.phase not in (TaskPhase.IDLE, TaskPhase.DONE):
            raise RuntimeError("Wait until physics has settled")
        if self.pile_scene_dirty or self.executor.initializing:
            self.executor.start_initial_settle()
            # Confirmation locks placement for this session, including after
            # failed IK and undo. Only resetting the saved scene unlocks it.
            self.pile_placement_locked = True
            self._clear_selection()
            self.snapshot = None
            self.warning = ""
            self._ik_after_initial_settle = True
            self.phase = DecisionPhase.SETTLING
            return
        self._queue_candidates()

    def select_box(self, box_index):
        if not self.stable_for_input:
            return
        if not 0 <= box_index < len(self.ik_mask) or not self.ik_mask[box_index]:
            self.warning = "Select a blue IK candidate box."
            return
        self._clear_selection()
        self.selected_box = int(box_index)
        self.warning = ""
        self.phase = DecisionPhase.AWAIT_FRAME
        if self.direct_remove:
            self.phase = DecisionPhase.READY_TO_EXECUTE
        # Choose the first originally reachable face; all six remain editable.
        for slot, candidate in enumerate(self.selected_frames):
            if candidate.initially_ik_reachable:
                self.select_frame(slot)
                break

    def select_frame(self, slot):
        if not self.stable_for_input or not 0 <= slot < len(self.selected_frames):
            return
        self.selected_frame_slot = int(slot)
        self.adjustment = FrameAdjustment()
        self._invalidate_plan()

    def _invalidate_plan(self):
        self.plan = None
        self.warning = ""
        self.phase = DecisionPhase.READY_TO_EXECUTE if self.direct_remove else DecisionPhase.AWAIT_PLAN

    def set_adjustment(self, **values):
        if self.stable_for_input and self.selected_candidate is not None:
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Pose adjustments must be finite")
            self.adjustment = replace(self.adjustment, **values)
            self._invalidate_plan()

    def set_unload_adjustment(self, **values):
        if self.stable_for_input and self.selected_candidate is not None:
            updated = replace(self.unload_adjustment, **values)
            if not all(np.isfinite(v) for v in (updated.tilt_u_deg, updated.tilt_v_deg, updated.distance_m)):
                raise ValueError("Unload adjustments must be finite")
            if updated.distance_m <= 0:
                raise ValueError("Unload distance must be positive")
            self.unload_adjustment = updated
            self._invalidate_plan()

    def request_plan(self):
        if self.direct_remove or not self.stable_for_input or self.selected_target is None:
            return
        self.plan = None
        self.warning = ""
        self._pending_job = "plan"

    def _compute_plan(self):
        if self.planner is None:
            from .target_planner import SelectedTargetMotionPlanner
            self.planner = SelectedTargetMotionPlanner()
        print(f"[plan] box={self.selected_box}: validating adjusted target and complete motion", flush=True)
        self.plan = self.planner.plan_target(
            query=self.snapshot.query, target=self.selected_target,
            unload_adjustment=self.unload_adjustment,
            current_q=self.executor.current_arm_q(), current_joint_names=tuple(mc.ARM_JOINT_NAMES),
        )
        self.phase = DecisionPhase.READY_TO_EXECUTE
        print(f"[plan] ready: {self.plan.status}", flush=True)

    def execute(self):
        if not self.stable_for_input or self.selected_box is None:
            return
        if not self.direct_remove and self.plan is None:
            self.warning = "Validate this pose with MotionGen before executing."
            return
        record = {
            "box": self.selected_box,
            "mode": "direct-remove" if self.direct_remove else "motiongen",
            "target": self.selected_target,
            "unload_adjustment": self.unload_adjustment,
        }
        checkpoint = capture_checkpoint(
            self.executor, record_count=len(self.records), history_frame_count=len(self.history),
        )
        self.checkpoints.append(checkpoint)
        try:
            if self.direct_remove:
                self.executor.start_direct(self.selected_box)
                self.phase = DecisionPhase.SETTLING
            else:
                self.executor.start_motion(self.plan)
                self.phase = DecisionPhase.EXECUTING
        except Exception as exc:
            # Retain the pre-attempt checkpoint even if starting an action fails
            # after changing part of the physical state.
            self.executor._fail(exc)
            self.phase = DecisionPhase.FAILED
            self.warning = self.executor.failure_reason
            raise
        self._pending_record = record
        # Direct removal changes visibility immediately, before the next tick.
        self.history.append(capture_frame(self.executor))
        self.plan = None
        self.warning = ""

    def tick(self):
        if self.replay_active:
            return
        job = self._pending_job
        if job is not None:
            self._pending_job = None
            try:
                if job == "plan":
                    self._compute_plan()
                else:
                    print("[candidates] geometric filtering + cuRobo IK...", flush=True)
                    self.snapshot = self.candidates.compute(self.executor.env)
                    self.phase = DecisionPhase.AWAIT_BOX
                    self.warning = "" if np.any(self.ik_mask) else "No IK candidates. You can retry IK or reset the scene."
                    print(f"[candidates] proxy={np.count_nonzero(self.snapshot.proxy_mask)} "
                          f"IK={np.count_nonzero(self.ik_mask)} time={self.snapshot.ik_total_s:.3f}s", flush=True)
            except Exception as exc:
                self.plan = None
                self.phase = DecisionPhase.AWAIT_PLAN if job == "plan" else DecisionPhase.AWAIT_BOX
                self.warning = f"{type(exc).__name__}: {exc}"
                print(f"[{job}] {self.warning}", flush=True)
            return
        if self.phase not in (DecisionPhase.EXECUTING, DecisionPhase.SETTLING):
            return
        initial_settle = self.executor.initializing
        result = self.executor.tick()
        if initial_settle:
            if result == TaskPhase.DONE:
                self.history.reset(capture_frame(self.executor, phase="initial_scene"))
        else:
            self.history.append(capture_frame(self.executor))
        if result == TaskPhase.FAILED:
            self.phase = DecisionPhase.FAILED
            self.warning = self.executor.failure_reason
            print(f"[stopped] {self.warning}", flush=True)
        elif result == TaskPhase.DONE:
            if self._ik_after_initial_settle:
                self._ik_after_initial_settle = False
                self.pile_scene_dirty = False
                self._queue_candidates()  # Authorized by the user's button click.
                return
            if self._pending_record is not None:
                stats = self.executor.stats
                record = dict(self._pending_record, step=stats.completed_actions,
                              metric_delta=stats.last_metric_delta, metric_total=stats.metric_total)
                self.records.append(record)
                self._pending_record = None
                print(f"[step {record['step']}] box={record['box']} mode={record['mode']} "
                      f"metric_delta={stats.last_metric_delta:.6f} "
                      f"metric_total={stats.metric_total:.6f} remaining={np.count_nonzero(self.executor.active_mask)}",
                      flush=True)
            self._clear_selection()
            self.snapshot = None
            if np.any(self.executor.active_mask):
                self.phase = DecisionPhase.AWAIT_IK
                self.executor.status_message = "Settled and scored; click Compute IK candidates for the next step."
            else:
                self.phase = DecisionPhase.COMPLETE
        elif result == TaskPhase.POST_SETTLE:
            self.phase = DecisionPhase.SETTLING

    def reset(self):
        self.exit_replay()
        self.executor.close()
        self.executor = UnloadTaskExecutor(self.scene_path, config=self.config)
        self.snapshot = None
        self.records.clear()
        self.checkpoints.clear()
        self.history.clear()
        self._pending_record = self._pending_job = None
        self._clear_selection()
        self.phase = DecisionPhase.AWAIT_IK
        self._initialize_pile_placement()
        self.warning = ""

    def close(self):
        self.executor.close()
