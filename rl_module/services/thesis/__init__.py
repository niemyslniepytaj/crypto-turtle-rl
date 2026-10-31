"""
Pakiet tezy: knoby MP, walk-forward, compare.
Styl jak core_logic — płaskie moduły, osobne entry plgrid_*.

Import ciężkich rzeczy (WalkForwardService) leniwie — żeby knobs/console
nie ciągnęły SimulationService.
"""

from rl_module.services.thesis.arms import (  # noqa: F401
    ARM_META,
    CANONICAL_ARMS,
    IMPLEMENTED_ARMS,
    PLACEHOLDER_ARMS,
)
from rl_module.services.thesis.knobs import (  # noqa: F401
    MP_FREE,
    build_optuna_config,
    default_mp_params,
)


def __getattr__(name):
    if name in (
        "WalkForwardService",
        "resolve_folds_range",
        "resolve_eval_folds",
        "expanding_train_spec",
        "rolling_train_spec",
    ):
        from rl_module.services.thesis import walkforward as _wf

        return getattr(_wf, name)
    if name in ("compare_runs", "discover_run_dirs"):
        from rl_module.services.thesis import compare as _cmp

        return getattr(_cmp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WalkForwardService",
    "resolve_folds_range",
    "compare_runs",
    "discover_run_dirs",
    "build_optuna_config",
    "default_mp_params",
    "CANONICAL_ARMS",
    "ARM_META",
    "MP_FREE",
]
