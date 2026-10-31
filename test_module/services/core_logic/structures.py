from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import numpy as np
import time

# MODEL PACKAGE (MP) knobs — formerly ml_* (accepted as deprecated aliases)
MP_KNOB_KEYS = (
    "mp_alloc_bias",
    "mp_alloc_scale",
    "mp_confidence",
    "mp_urgency",
    "mp_signal_deadzone",
    "mp_rebalance_deadzone",
    "mp_ema_alpha_min",
    "mp_ema_alpha_span",
)
_ML_TO_MP = {k.replace("mp_", "ml_", 1): k for k in MP_KNOB_KEYS}


def coerce_sim_mode(mode: Optional[str]) -> Optional[str]:
    """Normalize sim_mode; legacy ml→mp, mlrl/mprl→rl (meta-knoby)."""
    if mode is None:
        return None
    m = str(mode)
    if m == "ml":
        return "mp"
    if m in ("mlrl", "mprl"):
        return "rl"  # legacy alias → v6 meta-knoby
    return m


# Legacy experiment ids → (v6, default sim_mode). Stare v5 zostaje osobno.
_EXPERIMENT_TO_V6 = {
    "v5mp": "mp",
    "v5ml": "mp",  # legacy
    "v5rl": "rl",  # ex-mprl / meta-knoby
}


def coerce_experiment_pair(
    experiment_version: Optional[str],
    sim_mode: Optional[str] = None,
) -> tuple:
    """
    Zwraca (experiment_version, sim_mode).
    v5mp/v5rl → v6 + domyślny sim_mode; jawny sim_mode wygrywa.
    Stare v5 zostaje osobno (klasyczny position_target tylko na v5).
    """
    ver = str(experiment_version or "v0")
    mode = coerce_sim_mode(sim_mode)
    if ver in _EXPERIMENT_TO_V6:
        if not mode:
            mode = _EXPERIMENT_TO_V6[ver]
        ver = "v6"
    if ver == "v6" and not mode:
        mode = "mp"
    return ver, mode


def is_meta_knob_mode(experiment_version: Optional[str], sim_mode: Optional[str] = None) -> bool:
    """Meta-kontroler knóbów MP: v6+rl (legacy mprl/v5rl → to samo). rlul = stub jak rl."""
    ver, mode = coerce_experiment_pair(experiment_version, sim_mode)
    return ver == "v6" and mode in ("rl", "rlul")


def normalize_mp_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer mp_* knobs; copy from deprecated ml_* if mp_* missing/None.
    Also coerces sim_mode. Does not mutate the input dict.
    """
    out = dict(kwargs)
    if "sim_mode" in out:
        out["sim_mode"] = coerce_sim_mode(out.get("sim_mode"))
    if "ml_threshold" in out and ("mp_threshold" not in out or out.get("mp_threshold") is None):
        out["mp_threshold"] = out["ml_threshold"]
    out.pop("ml_threshold", None)
    for ml_key, mp_key in _ML_TO_MP.items():
        if mp_key not in out or out.get(mp_key) is None:
            if ml_key in out:
                out[mp_key] = out[ml_key]
        out.pop(ml_key, None)
    return out


def mp_attr(obj, name: str, default=None):
    """Read mp_* with fallback to legacy ml_* on object/dict."""
    ml_name = "ml_" + name[3:] if name.startswith("mp_") else name
    if isinstance(obj, dict):
        if name in obj and obj[name] is not None:
            return obj[name]
        if ml_name in obj and obj[ml_name] is not None:
            return obj[ml_name]
        return default
    val = getattr(obj, name, None)
    if val is not None:
        return val
    return getattr(obj, ml_name, default)


@dataclass(slots=True)
class Order:
    """Zlecenie handlowe w symulacji lub realtime (status, qty, metryki egzekucji)."""

    timestamp: int
    symbol: str
    action: str
    value: float
    buy_p: float = 0.0
    hold_p: float = 0.0
    sell_p: float = 0.0
    price: float = 0.0
    rl_config: Any = None
    fraction: float = 0.0
    qty: float = 0.0
    status: str = "created"
    reason: str = ""
    slippage_price: float = 0.0
    actual_qty: float = 0.0
    actual_usdt: float = 0.0
    total_portfolio_value: float = 0.0


    # Stan portfela PRZED transakcją
    local_before_balance: float = 0.0
    local_before_holdings: float = 0.0
    local_before_avg_buy_price: float = 0.0
    # Stan po lokalnej egzekucji (przed korektą Binance)
    local_after_balance: float = 0.0
    local_after_holdings: float = 0.0
    local_after_avg_buy_price: float = 0.0
    # Wynik Binance
    binance_status: str = ""
    binance_qty: float = 0.0
    binance_executed_price: float = 0.0
    binance_transaction_value: float = 0.0
    binance_order_id: str = ""
    binance_message: str = ""
    # Stan po korekcie (ground truth)
    reconciled_balance: float = 0.0
    reconciled_holdings: float = 0.0
    reconciled_avg_buy_price: float = 0.0
    # Drift local_after vs reconciled
    drift_qty: float = 0.0
    drift_value: float = 0.0
    force_exit: bool = False  # stop-loss / flatten — omija mincost


    # , "buy:0","hold":1, "sell:2"
    def rejected(self, reason: str):
        self.status = "rejected"
        self.reason = reason

    def to_dict(self):
        d = asdict(self)
        if hasattr(self.rl_config, "tolist"):
            d['rl_config'] = self.rl_config.tolist()
        return d

@dataclass
class PendingBinanceOrder:
    """Zlecenie Binance oczekujące na potwierdzenie; snapshot lokalny do ewentualnego rollbacku."""

    order_id: str
    symbol: str
    action: str
    local_qty: float
    local_price: float
    local_balance_before: float
    local_holdings_before: float
    local_avg_buy_price_before: float
    submitted_at: float = field(default_factory=time.time)

    def age_seconds(self) -> float:
        return time.time() - self.submitted_at


@dataclass(slots=True)
class PreviousEpoch:
    """Agregaty metryk i przebiegów z zakończonego epizodu treningu (dashboard / callback)."""

    epoch: int = 1

    buys: int = 0
    holds: int = 0
    sells: int = 0
    transactions: int = 0
    closed_trades: int = 0
    episode_bars: int = 0
    tx_per_day: float = 0.0
    strat_return_pct: float = 0.0
    hodl_return_pct: float = 0.0
    excess_pct: float = 0.0
    max_dd_pct: float = 0.0
    knobs: dict = field(default_factory=dict)

    rewards: List[float] = field(default_factory=list)
    min_r: float = 0.0
    max_r: float = 0.0
    mean_r: float = 0.0
    end_r:float = 0.0

    portfolio_values: List[float] = field(default_factory=list)
    hodl_values: List[float] = field(default_factory=list)  # ta sama częstotliwość co portfolio (::50)
    min_p: float = 0.0
    max_p: float = 0.0
    mean_p: float = 0.0
    end_p: float = 0.0

    def calculate_history(self):
        """Domknięcie metryk epizodu i downsampling przebiegów portfela / nagrody."""
        self.holds = self.transactions - self.sells - self.buys

        self.min_r = np.min(self.rewards)
        self.max_r = np.max(self.rewards)
        self.mean_r = round(np.mean(self.rewards),0)
        self.end_r = self.rewards[-1]
        self.rewards = self.rewards[::50]

        self.min_p = np.min(self.portfolio_values)
        self.max_p = np.max(self.portfolio_values)
        self.mean_p = round(np.mean(self.portfolio_values),0)
        self.end_p = self.portfolio_values[-1]
        # HODL w tym samym stride co portfel — zanim obetniemy portfel
        if self.hodl_values and len(self.hodl_values) == len(self.portfolio_values):
            self.hodl_values = self.hodl_values[::50]
        elif self.hodl_values:
            # wyrównaj długość i downsample
            n = len(self.portfolio_values)
            hv = self.hodl_values
            if len(hv) != n and n > 1 and len(hv) > 1:
                idxs = [int(round(i * (len(hv) - 1) / (n - 1))) for i in range(n)]
                hv = [hv[j] for j in idxs]
            self.hodl_values = hv[::50]
        self.portfolio_values = self.portfolio_values[::50]

    def to_dict(self):
        """Serializacja epizodu do JSON (rzutowanie typów numerycznych)."""
        return {
            "epoch": int(self.epoch),
            "metrics": {
                "buys": int(self.buys),
                "sells": int(self.sells),
                "holds": int(self.holds),
                "transactions": int(self.transactions),
                "closed_trades": int(self.closed_trades),
                "episode_bars": int(self.episode_bars),
                "tx_per_day": float(self.tx_per_day),
                "strat_return_pct": float(self.strat_return_pct),
                "hodl_return_pct": float(self.hodl_return_pct),
                "excess_pct": float(self.excess_pct),
                "max_dd_pct": float(self.max_dd_pct),
                "knobs": {k: float(v) for k, v in (self.knobs or {}).items()},
            },
            "portfolio": {
                "portfolio_values": [float(x) for x in self.portfolio_values],
                "hodl_values": [float(x) for x in (self.hodl_values or [])],
                "min_p": float(self.min_p),
                "mean_p": float(self.mean_p),
                "max_p": float(self.max_p),
                "end_p": float(self.end_p),
            },
            "reward": {
                "reward_values": [float(x) for x in self.rewards],
                "min_r": float(self.min_r),
                "mean_r": float(self.mean_r),
                "max_r": float(self.max_r),
                "end_r": float(self.end_r),
            }
        }
@dataclass(slots=True)
class CoreConfig:
    """Wspólna konfiguracja silnika: tryb sim, limity portfela, knoby MP, nagroda RL."""

    sim_mode: str = ""
    balance_start: float = 0.0
    add_fee: bool = True
    symbols: List[str] = field(default_factory=list)
    rlagent: str = ""
    rlagent_deterministic: bool = True
    slippage: int = 0
    wallet_percentage_limit: float = 0.0
    max_wallet_percentage_crypto: float = 0.0
    stop_loss: float = 0.0
    stop_loss_cooldown: int = 120
    threshold_buy: float = 0.0
    threshold_sell: float = 0.0
    mincost: float = 0.0
    tx_fraction: float = 0.0
    max_fraction_when_confident: float = 0.0
    mp_alloc_bias: float = 0.5
    mp_alloc_scale: float = 0.5
    mp_confidence: float = 0.5
    mp_urgency: float = 0.5
    mp_signal_deadzone: float = 0.15
    mp_rebalance_deadzone: float = 0.25
    mp_ema_alpha_min: float = 0.02
    mp_ema_alpha_span: float = 0.08
    v5rl_knob_hold_bars: int = 1
    v5rl_trade_lock_bars: int = 15
    clamp_wallet_cap_to_1n: bool = True
    reward_mode: str = "legacy"
    experiment_version: str = "v0"

    def to_dict(self):
        d = asdict(self)
        return d

@dataclass(slots=True)
class DecisionConfig:
    """Parametry decyzji per symbol (progi, frakcje, knoby MP) kopiowane z CoreConfig."""

    threshold_buy: float = 0.5
    threshold_sell: float = 0.5
    tx_fraction: float = 0.1
    max_fraction_when_confident: float = 0.5
    clamp_model_value: bool = True
    fraction_buy: float = 0.3
    fraction_sell: float = 0.3
    mp_alloc_bias: float = 0.5
    mp_alloc_scale: float = 0.5
    mp_confidence: float = 0.5
    mp_urgency: float = 0.5
    mp_signal_deadzone: float = 0.15
    mp_rebalance_deadzone: float = 0.25
    mp_ema_alpha_min: float = 0.02
    mp_ema_alpha_span: float = 0.08
    rebalance_dead_zone: float = 0.15
    ema_alpha_min: float = 0.05
    ema_alpha_span: float = 0.20

    def set(self, output_config: CoreConfig):
        """Kopiuje pola decyzji z globalnego CoreConfig."""
        self.threshold_buy = output_config.threshold_buy
        self.threshold_sell = output_config.threshold_sell
        self.tx_fraction = output_config.tx_fraction
        self.max_fraction_when_confident = output_config.max_fraction_when_confident
        self.mp_alloc_bias = output_config.mp_alloc_bias
        self.mp_alloc_scale = output_config.mp_alloc_scale
        self.mp_confidence = output_config.mp_confidence
        self.mp_urgency = output_config.mp_urgency
        self.mp_signal_deadzone = output_config.mp_signal_deadzone
        self.mp_rebalance_deadzone = output_config.mp_rebalance_deadzone
        self.mp_ema_alpha_min = output_config.mp_ema_alpha_min
        self.mp_ema_alpha_span = output_config.mp_ema_alpha_span

@dataclass(slots=True)
class WalletConfig:
    """Limity portfela per symbol: ekspozycja, stop-loss, prowizja, slippage."""

    wallet_percentage_limit: float = 0.9
    mincost: float = 1.0
    slippage: int = 0
    stop_loss: float = 0.0
    stop_loss_cooldown: int = 120
    fee: float = 0.001
    max_wallet_percentage_crypto: float = 1.0

    def set(self, output_config: CoreConfig):
        """Kopiuje limity z CoreConfig; opcjonalnie zaciska max_wallet do 1/N symboli."""
        self.wallet_percentage_limit = output_config.wallet_percentage_limit
        self.mincost = output_config.mincost
        self.slippage = output_config.slippage
        self.stop_loss = output_config.stop_loss
        self.stop_loss_cooldown = int(getattr(output_config, "stop_loss_cooldown", 120))
        n = max(len(getattr(output_config, "symbols", None) or []), 1)
        raw = float(getattr(output_config, "max_wallet_percentage_crypto", 1.0) or 1.0)
        clamp_1n = bool(getattr(output_config, "clamp_wallet_cap_to_1n", True))
        if raw <= 0:
            self.max_wallet_percentage_crypto = raw
        elif clamp_1n:
            self.max_wallet_percentage_crypto = min(raw, 1.0 / n)
        else:
            self.max_wallet_percentage_crypto = min(raw, 1.0)
        if not output_config.add_fee: self.fee = 0.0