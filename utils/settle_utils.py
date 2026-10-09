from __future__ import annotations

import numpy as np


class ComDisplacementSettleTracker:
    """Track whether active box COMs remain near a fixed reference state."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_displacement: float = 0.005,
        check_interval: int = 10,
        required_checks: int = 15,
    ) -> None:
        if float(max_displacement) <= 0.0:
            raise ValueError("max_displacement must be positive.")
        if int(check_interval) < 1:
            raise ValueError("check_interval must be at least 1.")
        if int(required_checks) < 1:
            raise ValueError("required_checks must be at least 1.")
        self.enabled = bool(enabled)
        self.max_displacement = float(max_displacement)
        self.check_interval = int(check_interval)
        self.required_checks = int(required_checks)
        self.reset()

    def reset(self) -> None:
        self.reference_positions: np.ndarray | None = None
        self.reference_active_mask: np.ndarray | None = None
        self.consecutive = 0
        self.last_max_displacement = 0.0
        self.settled = False
        self.last_check_frame = -1

    def update(
        self,
        *,
        frame_index: int,
        positions: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        frame = int(frame_index)
        if frame <= 0 or frame % self.check_interval != 0:
            return self.settled
        if frame == self.last_check_frame:
            return self.settled
        self.last_check_frame = frame

        current = np.asarray(positions, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {current.shape}.")
        active = (
            np.ones((current.shape[0],), dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool).reshape(-1)
        )
        if active.shape != (current.shape[0],):
            raise ValueError(
                f"active_mask must have shape ({current.shape[0]},), got {active.shape}."
            )

        reference_invalid = (
            self.reference_positions is None
            or self.reference_positions.shape != current.shape
            or self.reference_active_mask is None
            or not np.array_equal(self.reference_active_mask, active)
        )
        if reference_invalid:
            self.reference_positions = current.copy()
            self.reference_active_mask = active.copy()
            self.consecutive = 0
            self.last_max_displacement = 0.0
            self.settled = False
            return False

        displacement = np.linalg.norm(
            current[active] - self.reference_positions[active],
            axis=1,
        )
        self.last_max_displacement = float(np.max(displacement, initial=0.0))
        if self.last_max_displacement > self.max_displacement:
            self.reference_positions = current.copy()
            self.consecutive = 0
            self.last_max_displacement = 0.0
            self.settled = False
            return False

        self.consecutive += 1
        self.settled = self.consecutive >= self.required_checks
        return self.settled


__all__ = ["ComDisplacementSettleTracker"]
