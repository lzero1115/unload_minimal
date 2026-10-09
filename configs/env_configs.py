from dataclasses import dataclass

from configs import phys_configs as pc, task_configs as tc


@dataclass(frozen=True)
class UnloadEnvConfig:
    """The physical/candidate subset of unload_0810's environment defaults."""

    max_settle_frames: int = 600
    enable_com_settle: bool = pc.ENABLE_COM_SETTLE
    com_settle_max_displacement: float = pc.COM_SETTLE_MAX_DISPLACEMENT
    com_settle_check_interval: int = pc.COM_SETTLE_CHECK_INTERVAL
    com_settle_required_checks: int = pc.COM_SETTLE_REQUIRED_CHECKS
    graveyard_pos: tuple[float, float, float] = (0.0, 0.0, -100.0)
    surface_clearance: float = tc.GRASP_CONTACT_CLEARANCE
    proxy_face_clearance: float = 0.10
    proxy_face_origin_margin: float = 1e-4
    proxy_min_openness_score: int = 3
    ray_side_wall: int = -2
    ray_grid_step: float = 0.02
    ray_plane_margin: float = 0.05
    ray_min_hits: int = 8


DEFAULT_ENV_CONFIG = UnloadEnvConfig()
