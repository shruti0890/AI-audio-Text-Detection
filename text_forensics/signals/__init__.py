# signals package — exposes all four forensic signal functions
from .curvature import get_curvature
from .burstiness import get_burstiness
from .cliche_scanner import get_cliche_density
from .lexical_entropy import get_lexical_stats

__all__ = ["get_curvature", "get_burstiness", "get_cliche_density", "get_lexical_stats"]
