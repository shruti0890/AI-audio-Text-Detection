# signals package — exposes all six forensic signal functions
from .curvature import get_curvature
from .burstiness import get_burstiness
from .cliche_scanner import get_cliche_density
from .lexical_entropy import get_lexical_stats
from .ngram_repetition import get_ngram_repetition
from .structural_regularity import get_structural_regularity

__all__ = [
    "get_curvature",
    "get_burstiness",
    "get_cliche_density",
    "get_lexical_stats",
    "get_ngram_repetition",
    "get_structural_regularity",
]
