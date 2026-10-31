"""I0.2 — HODL z prowizją wejścia."""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.thesis.knobs import HODL_ENTRY_FEE_RATE


def _hodl_curve(closes, balance, fee_rate):
    """Mirror equal-weight single-asset path z _run_hodl."""
    c0 = float(closes[0])
    crypto_usd = balance / (1.0 + fee_rate) if fee_rate > 0 else balance
    qty = crypto_usd / c0
    return [qty * float(c) for c in closes]


def test_entry_fee_reduces_start_value():
    closes = [100.0, 110.0, 120.0]
    bal = 1000.0
    no_fee = _hodl_curve(closes, bal, 0.0)
    with_fee = _hodl_curve(closes, bal, HODL_ENTRY_FEE_RATE)
    assert abs(no_fee[0] - bal) < 1e-9
    assert with_fee[0] < bal
    assert abs(with_fee[0] - bal / (1.0 + HODL_ENTRY_FEE_RATE)) < 1e-6
    # proporcjonalny wzrost ceny → ten sam % ruchu netto od wartości po fee
    r_nf = no_fee[-1] / no_fee[0] - 1
    r_f = with_fee[-1] / with_fee[0] - 1
    assert abs(r_nf - r_f) < 1e-12


def test_dual_equal_weight_fee():
    eth = np.array([100.0, 100.0])
    btc = np.array([200.0, 200.0])
    bal = 1000.0
    fee = HODL_ENTRY_FEE_RATE
    cash_per = bal / 2
    q_eth = (cash_per / (1 + fee)) / eth[0]
    q_btc = (cash_per / (1 + fee)) / btc[0]
    v0 = q_eth * eth[0] + q_btc * btc[0]
    assert abs(v0 - bal / (1 + fee)) < 1e-6
