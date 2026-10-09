from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from configs import task_configs as tc

from .target_types import CandidateSnapshot, FrameCandidate, normalized


class StableCandidateService:
    """Compute box-level IK candidates and expose all faces of reachable boxes."""

    def __init__(self, *, ik_batch_size: int = 256) -> None:
        if int(ik_batch_size) < 1:
            raise ValueError("ik_batch_size must be positive.")
        self.ik_batch_size = int(ik_batch_size)
        self._solver: Any | None = None

    def _get_solver(self):
        if self._solver is None:
            from runtime.ik_runtime import PersistentCuroboIKCandidateSolver

            self._solver = PersistentCuroboIKCandidateSolver(
                batch_size=self.ik_batch_size
            )
        return self._solver

    def compute(self, env) -> CandidateSnapshot:
        env._compute_candidates()
        query = env.export_candidate_frame_query()
        result = self._get_solver().solve_query(query)

        proxy_mask = np.asarray(query["proxy_action_mask"], dtype=bool).reshape(-1)
        ik_mask = np.asarray(result["box_success_mask"], dtype=bool).reshape(-1)
        if proxy_mask.shape != ik_mask.shape:
            raise RuntimeError(
                f"Proxy mask shape {proxy_mask.shape} does not match "
                f"IK mask shape {ik_mask.shape}."
            )
        if np.any(ik_mask & ~proxy_mask):
            raise RuntimeError("IK candidate mask escaped the proxy action mask.")
        ik_mask &= proxy_mask

        targets = dict(query["frame_targets"])
        frame_boxes = np.asarray(targets["box_indices"], dtype=np.int32).reshape(-1)
        face_axes = np.asarray(targets["face_axes"], dtype=np.int32).reshape(-1)
        face_signs = np.asarray(targets["face_signs"], dtype=np.int32).reshape(-1)
        normals = np.asarray(targets["face_normals_w"], dtype=np.float64).reshape(-1, 3)
        openness = np.asarray(
            targets.get("openness_scores", np.zeros_like(frame_boxes)),
            dtype=np.int32,
        ).reshape(-1)
        areas = np.asarray(
            targets.get("face_areas", np.zeros_like(frame_boxes, dtype=np.float32)),
            dtype=np.float64,
        ).reshape(-1)
        frame_success = np.asarray(result["frame_success"], dtype=bool).reshape(-1)
        roll_indices = np.asarray(
            result["selected_roll_indices"], dtype=np.int32
        ).reshape(-1)
        positions = np.asarray(
            result["selected_positions_w"], dtype=np.float64
        ).reshape(-1, 3)
        quats = np.asarray(
            result["selected_quats_wxyz_w"], dtype=np.float64
        ).reshape(-1, 4)

        frame_count = int(frame_boxes.shape[0])
        arrays = (
            face_axes,
            face_signs,
            normals,
            openness,
            areas,
            frame_success,
            roll_indices,
            positions,
            quats,
        )
        if any(int(value.shape[0]) != frame_count for value in arrays):
            raise RuntimeError("IK frame result does not match the exported frame table.")

        centers = np.asarray(query["box_centers_w"], dtype=np.float64).reshape(-1, 3)
        box_quats = np.asarray(query["box_quats_wxyz"], dtype=np.float64).reshape(-1, 4)
        half_extents = np.asarray(query["box_half_extents"], dtype=np.float64).reshape(-1, 3)
        n_boxes = int(proxy_mask.shape[0])
        if centers.shape != (n_boxes, 3):
            raise RuntimeError("IK query box centers do not match the action mask.")
        if box_quats.shape != (n_boxes, 4):
            raise RuntimeError("IK query box quaternions do not match the action mask.")
        if half_extents.shape != (n_boxes, 3):
            raise RuntimeError("IK query box extents do not match the action mask.")

        source_by_face: dict[tuple[int, int, int], int] = {}
        for frame_i in range(frame_count):
            key = (
                int(frame_boxes[frame_i]),
                int(face_axes[frame_i]),
                int(face_signs[frame_i]),
            )
            source_by_face[key] = int(frame_i)

        frames_by_box: dict[int, list[FrameCandidate]] = {}
        for box_index in np.flatnonzero(ik_mask):
            box_i = int(box_index)
            quat_wxyz = np.asarray(box_quats[box_i], dtype=np.float64)
            box_rotation = Rotation.from_quat(
                quat_wxyz[[1, 2, 3, 0]]
            ).as_matrix()
            for face_axis in range(3):
                tangent_axes = tuple(axis for axis in range(3) if axis != face_axis)
                for face_sign in (-1, 1):
                    normal = normalized(
                        float(face_sign) * box_rotation[:, face_axis],
                        name="face normal",
                    )
                    position = (
                        centers[box_i]
                        + normal
                        * (
                            float(half_extents[box_i, face_axis])
                            + float(tc.GRASP_CONTACT_CLEARANCE)
                        )
                    )
                    approach_axis = -normal
                    tangent_u = normalized(
                        box_rotation[:, tangent_axes[0]],
                        name="frame tangent u",
                    )
                    tangent_v = normalized(
                        np.cross(approach_axis, tangent_u),
                        name="frame tangent v",
                    )
                    rotation = np.column_stack(
                        (tangent_u, tangent_v, approach_axis)
                    )
                    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
                    target_quat = quat_xyzw[[3, 0, 1, 2]]

                    source_index = source_by_face.get(
                        (box_i, int(face_axis), int(face_sign))
                    )
                    reachable = bool(
                        source_index is not None
                        and frame_success[int(source_index)]
                    )
                    roll_index = 0
                    openness_score = 0
                    face_area = float(
                        4.0
                        * half_extents[box_i, tangent_axes[0]]
                        * half_extents[box_i, tangent_axes[1]]
                    )
                    frame_index = int(
                        box_i * 6
                        + face_axis * 2
                        + (1 if face_sign > 0 else 0)
                    )
                    if source_index is not None:
                        source_i = int(source_index)
                        openness_score = int(openness[source_i])
                        if float(areas[source_i]) > 0.0:
                            face_area = float(areas[source_i])
                        if reachable:
                            solved_quat = np.asarray(
                                quats[source_i], dtype=np.float64
                            )
                            solved_position = np.asarray(
                                positions[source_i], dtype=np.float64
                            )
                            if bool(np.all(np.isfinite(solved_quat))) and bool(
                                np.all(np.isfinite(solved_position))
                            ):
                                target_quat = solved_quat.copy()
                                position = solved_position.copy()
                                solved_rotation = Rotation.from_quat(
                                    solved_quat[[1, 2, 3, 0]]
                                ).as_matrix()
                                normal = normalized(
                                    -solved_rotation[:, 2], name="face normal"
                                )
                                tangent_u = normalized(
                                    solved_rotation[:, 0],
                                    name="frame tangent u",
                                )
                                tangent_v = normalized(
                                    solved_rotation[:, 1],
                                    name="frame tangent v",
                                )
                                roll_index = int(roll_indices[source_i])
                            else:
                                reachable = False

                    frames_by_box.setdefault(box_i, []).append(
                        FrameCandidate(
                            frame_index=frame_index,
                            box_index=box_i,
                            face_axis=int(face_axis),
                            face_sign=int(face_sign),
                            roll_index=int(roll_index),
                            position_w=np.asarray(position, dtype=np.float64).copy(),
                            quat_wxyz_w=np.asarray(
                                target_quat, dtype=np.float64
                            ).copy(),
                            normal_w=np.asarray(normal, dtype=np.float64).copy(),
                            tangent_u_w=np.asarray(
                                tangent_u, dtype=np.float64
                            ).copy(),
                            tangent_v_w=np.asarray(
                                tangent_v, dtype=np.float64
                            ).copy(),
                            openness_score=int(openness_score),
                            face_area=float(face_area),
                            initially_ik_reachable=bool(reachable),
                        )
                    )

        ordered = {
            box_index: tuple(
                sorted(
                    frames,
                    key=lambda frame: (
                        -int(frame.initially_ik_reachable),
                        -float(frame.normal_w[2]),
                        -int(frame.openness_score),
                        -float(frame.face_area),
                        int(frame.frame_index),
                    ),
                )
            )
            for box_index, frames in frames_by_box.items()
        }
        for box_index in np.flatnonzero(ik_mask):
            if int(box_index) not in ordered:
                raise RuntimeError(
                    f"box_{int(box_index)} is IK-reachable but has no retained frame."
                )

        timing = dict(result.get("timing", {}))
        return CandidateSnapshot(
            query=query,
            proxy_mask=proxy_mask.copy(),
            ik_mask=ik_mask.copy(),
            frames_by_box=ordered,
            ik_total_s=float(timing.get("total_s", 0.0)),
        )


__all__ = ["StableCandidateService"]
