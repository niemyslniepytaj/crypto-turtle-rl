"""
Zestawy hipotez tezy — konfiguracja bez twardego kodu w analysis.py (Z4).

HYPOTHESIS_SET = "intermediate" | "extended"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Domyślny zestaw (wersja pośrednia) — użytkownik może zmienić później
HYPOTHESIS_SET = "intermediate"

ALPHA = 0.05
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20241023
TREND_THRESHOLD = 0.05  # ±5% na (BTC+ETH)/2

# Reguła „najlepsza metoda” (efektywność × skuteczność)
BEST_METHOD_RULE = (
    "Pierwsza w obu rankingach (mediana Delta oraz mediana e=Delta/core_h); "
    "inaczej kompromis: ranking sumy rang (niższa suma lepsza)."
)

# Pary metod: (method_id, adaptive_arm, frozen_arm)
METHOD_PAIRS = (
    ("MP", "M1", "R1"),
    ("RL", "M2", "R2"),
    ("UL", "M3", "R3"),
)

# --- definicje zestawów -------------------------------------------------

SETS: Dict[str, Dict[str, Any]] = {
    "intermediate": {
        "label": "wersja pośrednia",
        "tests": [
            {
                "id": "H1",
                "kind": "wilcoxon_pair",
                "arm_x": "M1",
                "arm_y": "R1",
                "delta_name": "Delta_MP",
                "requires": ("M1", "R1"),
                "holm_family": None,  # bez korekty między H1–H3
                "latex_tab": "tab:testy-h1-h3",
            },
            {
                "id": "H2",
                "kind": "wilcoxon_pair",
                "arm_x": "M2",
                "arm_y": "R2",
                "delta_name": "Delta_RL",
                "requires": ("M2", "R2"),
                "holm_family": None,
                "latex_tab": "tab:testy-h1-h3",
            },
            {
                "id": "H3",
                "kind": "wilcoxon_pair",
                "arm_x": "M3",
                "arm_y": "R3",
                "delta_name": "Delta_UL",
                "requires": ("M3", "R3"),
                "holm_family": None,
                "conditional": False,  # M3/R3 realizowane (rozszerzenie R2/M2)
                "latex_tab": "tab:testy-h1-h3",
            },
            {
                "id": "H4",
                "kind": "friedman_deltas",
                "methods": ("MP", "RL", "UL"),
                "requires_any": (("M1", "R1"), ("M2", "R2"), ("M3", "R3")),
                "fallback_without_ul": "wilcoxon_delta_diff",
                "fallback_pair": ("RL", "MP"),
                "holm_posthoc": True,
                "holm_m": 3,
                "latex_tab": "tab:testy-h4",
            },
            {
                "id": "H5",
                "kind": "hodl_descriptive",
                "baseline": "R0",
                "requires": ("R0",),
                "latex_tab": "tab:h5",
            },
        ],
        # p_Holm na H1–H3 tylko jako wrażliwość (nie decyzja)
        "holm_sensitivity_on": ("H1", "H2", "H3"),
        "extended_delta_diffs": False,
    },
    "extended": {
        "label": "wersja rozszerzona",
        "tests": [
            {
                "id": "H1",
                "kind": "wilcoxon_pair",
                "arm_x": "M1",
                "arm_y": "R1",
                "delta_name": "Delta_MP",
                "requires": ("M1", "R1"),
                "holm_family": 1,
                "latex_tab": "tab:testy-rodzina1",
            },
            {
                "id": "H2",
                "kind": "wilcoxon_pair",
                "arm_x": "M2",
                "arm_y": "R2",
                "delta_name": "Delta_RL",
                "requires": ("M2", "R2"),
                "holm_family": 1,
                "latex_tab": "tab:testy-rodzina1",
            },
            {
                "id": "H3",
                "kind": "wilcoxon_pair",
                "arm_x": "M3",
                "arm_y": "R3",
                "delta_name": "Delta_UL",
                "requires": ("M3", "R3"),
                "holm_family": 2,
                "conditional": False,  # M3/R3 realizowane (rozszerzenie R2/M2)
                "latex_tab": "tab:testy-rodzina2",
            },
            {
                "id": "H4",
                "kind": "friedman_deltas",
                "methods": ("MP", "RL", "UL"),
                "requires_any": (("M1", "R1"), ("M2", "R2"), ("M3", "R3")),
                "fallback_without_ul": "wilcoxon_delta_diff",
                "fallback_pair": ("RL", "MP"),
                "holm_posthoc": True,
                "holm_m": 3,
                "latex_tab": "tab:testy-h4",
            },
            {
                "id": "H4b",
                "kind": "delta_diffs",
                "pairs": (
                    ("RL", "MP", "Delta_RL_minus_MP"),
                    ("UL", "RL", "Delta_UL_minus_RL"),
                    ("UL", "MP", "Delta_UL_minus_MP"),
                ),
                "holm_family": 2,
                "latex_tab": "tab:testy-przyrosty",
            },
            {
                "id": "H5",
                "kind": "hodl_descriptive",
                "baseline": "R0",
                "requires": ("R0",),
                "latex_tab": "tab:h5",
            },
        ],
        "holm_sensitivity_on": (),
        "extended_delta_diffs": True,
        "holm_families": {
            1: ("H1", "H2"),
            2: ("H3", "H4b"),
        },
    },
}


def get_hypothesis_set(name: Optional[str] = None) -> Dict[str, Any]:
    key = name or HYPOTHESIS_SET
    if key not in SETS:
        raise KeyError(f"Nieznany HYPOTHESIS_SET={key!r}; znane: {tuple(SETS)}")
    return SETS[key]


def method_delta_arms(method_id: str) -> Tuple[str, str]:
    for mid, ax, ay in METHOD_PAIRS:
        if mid == method_id:
            return ax, ay
    raise KeyError(method_id)
