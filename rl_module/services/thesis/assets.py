"""
Aktywa tezy: jeden portfel ETH+BTC (nie osobne runy ×3).

Knoby (decyzja na tezę)
-----------------------
Wspólne (shared) na poziomie portfela — Optuna 6F i CoreConfig:
  mp_*, wallet_percentage_limit, max_wallet_percentage_crypto, stop_loss, mincost, …

Różnicowanie per symbol:
  • MP: ten sam knob × inny sygnał (mu/sigma) → inny target
  • RL: akcje już są per symbol (obs/action ∝ n_symbols)

Osobne knoby Optuny per coin = 6×N wymiarów → odrzucone (mnoży protokół).

max_wallet_percentage_crypto = limit PER COIN.
  • clamp_wallet_cap_to_1n=True (legacy): silnik ≤ 1/N (suma capów ≤ 100%).
  • False (teza): silnik ≤ 1.0 — pozwala rotację (np. 0.75 przy N=2).
  Przy maxc=0.5 i N=2 oba tryby dają 0.5 (0.5 ≤ 1/N).

Progi tx/closed (Optuna gates): portfolio-level (bez ×N) — scoring jest
na jednym kapitale; aktywność rośnie subliniowo przy wspólnym cash.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Zamrożone ścieżki tezy (Binance spot 1m, ten sam zakres dat)
THESIS_DATA_FILES: Dict[str, str] = {
    "ETH": "ETHUSDT_1m_20250511_20260831.csv",
    "BTC": "BTCUSDT_1m_20250511_20260831.csv",
}

THESIS_MODEL_PATHS: Dict[str, str] = {
    "ETH": "LR1449_eth_model_E114_2709",
    "BTC": "LR1414_btc_model_B104_2209",
}

# Linie eksperymentu (OOS / RL / M1) — nie mylić z seedem study Optuny R1 LOCK
THESIS_SEEDS: Tuple[int, ...] = (396379, 951830, 246215)


def n_symbols(data_or_models: Optional[Mapping] = None, symbols: Optional[Sequence[str]] = None) -> int:
    if symbols is not None:
        return max(len(symbols), 1)
    if data_or_models:
        return max(len(data_or_models), 1)
    return 1


def clamp_max_wallet_per_symbol(
    max_wallet: float,
    n: int,
    *,
    clamp_to_1n: bool = True,
) -> float:
    """
    Per-coin cap.
    clamp_to_1n=True → min(raw, 1/N); False → min(raw, 1.0) (teza / rotacja).
    """
    n = max(int(n), 1)
    raw = float(max_wallet)
    if raw <= 0:
        return raw
    if clamp_to_1n:
        return min(raw, 1.0 / n)
    return min(raw, 1.0)


def max_wallet_optuna_range(n: int = 2, *, clamp_to_1n: bool = True) -> Tuple[float, float]:
    """Zakres F dla Optuny — hi ≤1/N gdy clamp_to_1n, inaczej ≤1.0."""
    n = max(int(n), 1)
    hi = (1.0 / n) if clamp_to_1n else 1.0
    lo = 0.25 if n > 1 else 0.30
    if lo >= hi:
        lo = max(0.10, hi * 0.5)
    return (lo, hi)


def format_mwpc_tag(max_wallet: float) -> str:
    """0.75 → 'mwpc075'; 0.5 → 'mwpc050'."""
    return "mwpc" + f"{float(max_wallet):.2f}".replace(".", "")


def format_sl_tag(stop_loss: float) -> str:
    """0.03 → 'sl003'; 0.06 → 'sl006'."""
    return "sl" + f"{float(stop_loss):.2f}".replace(".", "")


def equal_weight_hodl_curve(
    crypto_ref_values: Mapping[str, Sequence[float]],
) -> List[float]:
    """
    Equal-weight buy&hold z krzywych per-symbol (każda zaczyna od balance_start).
    Średnia arytmetyczna == alokacja 1/N kapitału w każdy asset.
    """
    series = [list(v) for v in crypto_ref_values.values() if v]
    if not series:
        return []
    n_ticks = min(len(s) for s in series)
    if n_ticks <= 0:
        return []
    n = len(series)
    return [sum(s[i] for s in series) / n for i in range(n_ticks)]


def hodl_return_pct_from_refs(
    crypto_ref_values: Mapping[str, Sequence[float]],
) -> Optional[float]:
    """Return % equal-weight HODL (tożsama ze średnią returnów per symbol)."""
    curve = equal_weight_hodl_curve(crypto_ref_values)
    if not curve or curve[0] == 0:
        return None
    return (curve[-1] - curve[0]) / curve[0] * 100.0
