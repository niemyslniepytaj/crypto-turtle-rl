"""Klamp max_wallet_percentage_crypto: 1/N vs teza (False)."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.thesis.assets import clamp_max_wallet_per_symbol
from rl_module.services.thesis.knobs import CLAMP_WALLET_CAP_TO_1N
from test_module.services.core_logic.structures import CoreConfig, WalletConfig


def test_thesis_clamp_flag_is_false():
    assert CLAMP_WALLET_CAP_TO_1N is False


def test_helper_0_5_identical_with_or_without_1n():
    # 0.5 ≤ 1/2 → bez zmian w obu trybach
    assert clamp_max_wallet_per_symbol(0.5, 2, clamp_to_1n=True) == 0.5
    assert clamp_max_wallet_per_symbol(0.5, 2, clamp_to_1n=False) == 0.5


def test_helper_0_75_differs():
    assert clamp_max_wallet_per_symbol(0.75, 2, clamp_to_1n=True) == 0.5
    assert clamp_max_wallet_per_symbol(0.75, 2, clamp_to_1n=False) == 0.75


def test_wallet_config_set_legacy_clamps_to_1n():
    cfg = CoreConfig(
        symbols=["ETH", "BTC"],
        max_wallet_percentage_crypto=0.75,
        clamp_wallet_cap_to_1n=True,
        add_fee=True,
    )
    w = WalletConfig()
    w.set(cfg)
    assert w.max_wallet_percentage_crypto == 0.5


def test_wallet_config_set_thesis_allows_075():
    cfg = CoreConfig(
        symbols=["ETH", "BTC"],
        max_wallet_percentage_crypto=0.75,
        clamp_wallet_cap_to_1n=False,
        add_fee=True,
    )
    w = WalletConfig()
    w.set(cfg)
    assert w.max_wallet_percentage_crypto == 0.75


def test_wallet_config_0_5_unchanged_either_way():
    for clamp in (True, False):
        cfg = CoreConfig(
            symbols=["ETH", "BTC"],
            max_wallet_percentage_crypto=0.5,
            clamp_wallet_cap_to_1n=clamp,
            add_fee=True,
        )
        w = WalletConfig()
        w.set(cfg)
        assert w.max_wallet_percentage_crypto == 0.5
