"""
Meta ramion + dispatch. Logika ciężka zostaje w WalkForwardService.
"""

CANONICAL_ARMS = ("R0", "R1", "R2", "R3", "M1", "M2", "M3")
IMPLEMENTED_ARMS = ("R0", "R1", "R2", "R3", "M1", "M2", "M3")
# R3/M3 wymagają agent_rlul + regime_lock — preflight/skip gdy brak
OPTIONAL_UL_ARMS = ("R3", "M3")
PLACEHOLDER_ARMS = ()

ARM_META = {
    "R0": {
        "plan_id": "R0",
        "label": "Buy & Hold",
        "self_improvement": "none",
        "adaptivity": "none",
        "status": "implemented",
    },
    "R1": {
        "plan_id": "R1",
        "label": "MP static (v6+mp)",
        "self_improvement": "none",
        "adaptivity": "frozen",
        "status": "implemented",
        "experiment_version": "v6",
        "sim_mode": "mp",
    },
    "R2": {
        "plan_id": "R2",
        "label": "RL frozen (v6+rl)",
        "self_improvement": "none",
        "adaptivity": "frozen",
        "status": "implemented",
        "experiment_version": "v6",
        "sim_mode": "rl",
    },
    "R3": {
        "plan_id": "R3",
        "label": "RL+UL frozen (v6+rlul)",
        "self_improvement": "none",
        "adaptivity": "frozen",
        "status": "implemented",
        "experiment_version": "v6",
        "sim_mode": "rlul",
    },
    "M1": {
        "plan_id": "M1",
        "label": "MP + Optuna rolling-3 (v6+mp)",
        "self_improvement": "optuna",
        "adaptivity": "retune_per_fold",
        "status": "implemented",
        "experiment_version": "v6",
        "sim_mode": "mp",
    },
    "M2": {
        "plan_id": "M2",
        "label": "RL continue-learn (v6+rl)",
        "self_improvement": "rl",
        "adaptivity": "adaptive",
        "status": "implemented",
        "experiment_version": "v6",
        "sim_mode": "rl",
    },
    "M3": {
        "plan_id": "M3",
        "label": "RL + UL (v6+rlul)",
        "self_improvement": "rl+ul",
        "adaptivity": "adaptive",
        "status": "implemented",
        "experiment_version": "v6",
        "sim_mode": "rlul",
    },
}

_ARM_ALIASES = {
    "r0": "R0",
    "hodl": "R0",
    "r1": "R1",
    "mp": "R1",
    "r2": "R2",
    "rl": "R2",
    "r3": "R3",
    "rlul_frozen": "R3",
    "m1": "M1",
    "optuna": "M1",
    "m2": "M2",
    "m3": "M3",
    "rlul": "M3",
}


def normalize_arm(raw):
    s = str(raw).strip()
    key = s.upper() if s.upper() in CANONICAL_ARMS else s.lower().replace("-", "_").replace(" ", "_")
    if key not in CANONICAL_ARMS:
        key = _ARM_ALIASES.get(key, key)
    if key not in CANONICAL_ARMS:
        raise ValueError(f"Nieznane ramię '{raw}' → '{key}'. Znane: {CANONICAL_ARMS}")
    return key
