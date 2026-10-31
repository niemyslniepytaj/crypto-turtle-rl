"""
Funkcje bindowane do Wallet w trybie testnet_v0.

Cykl: check_pending_orders (reconcile / rollback) → buy/sell z pending →
sync_after_orders (pełna synchronizacja portfela wirtualnego).
"""

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wallet import Wallet
    from .structures import Order

from .structures import PendingBinanceOrder

logger = logging.getLogger(__name__)

PENDING_ORDER_TIMEOUT_SECONDS = 150  # 2.5 minuty – po tym rollback


def _get_binance_service():
    from .binance_testnet_service import BinanceTestnetService
    if not hasattr(_get_binance_service, "_instance"):
        _get_binance_service._instance = BinanceTestnetService()
    return _get_binance_service._instance


# ---------------------------------------------------------------------------
# WALLET INIT
# ---------------------------------------------------------------------------

def testnet_wallet_init(self: "Wallet"):
    """
    Wywołaj raz przy starcie serwisu.
    1. Inicjalizuje offset w BinanceTestnetService (snapshot co jest na Binance).
    2. Zeruje lokalne holdings (symulacja zaczyna od zera).
    3. Tworzy listę pending_orders na self.
    """
    svc = _get_binance_service()
    result = svc.initialize_offset(self.symbols)
    if result["result"] != "success":
        logger.error(f"[Testnet] initialize_offset failed: {result.get('message')}")
    else:
        logger.info(
            f"[Testnet] Offset: {result['offset_usdt']:.2f} USDT, "
            f"holdings={result['offset_holdings']}"
        )

    # Zerujemy holdings – symulacja startuje od czystego stanu
    for symbol in list(self.holdings.keys()):
        self.initialize_symbol_and_holdings(symbol)

    # Lista pending zleceń (między iteracjami pętli)
    self._pending_binance_orders: list[PendingBinanceOrder] = []

    logger.info(f"[Testnet] Portfel startowy: balance={self.balance:.2f}, holdings={self.holdings}")


# ---------------------------------------------------------------------------
# CHECK PENDING ORDERS – wywołaj na początku każdej iteracji
# ---------------------------------------------------------------------------

def testnet_check_pending_orders(self: "Wallet") -> bool:
    """
    Sprawdza status wszystkich pending zleceń z poprzednich timestampów.
    Wywołaj NA POCZĄTKU każdej iteracji pętli core_logic, przed generowaniem decyzji.

    Zwraca True jeśli wszystkie pending są rozwiązane (można kontynuować normalnie).
    Zwraca False jeśli coś było pending i wykonano sync portfela z Binance.
    """
    if not hasattr(self, "_pending_binance_orders"):
        self._pending_binance_orders = []

    pending = self._pending_binance_orders
    if not pending:
        return True

    svc = _get_binance_service()
    any_resolved = False

    for po in list(pending):
        order_result = svc.get_order(symbol=po.symbol, order_id=po.order_id)

        if order_result.get("result") != "success":
            # Błąd API – nie wiemy co się stało, zostaw do następnej iteracji
            logger.warning(f"[Testnet] Nie mogę sprawdzić statusu #{po.order_id}: {order_result.get('message')}")
            continue

        simple_status = order_result["simple_status"]

        if simple_status == "filled":
            logger.info(f"[Testnet] Pending #{po.order_id} {po.symbol} {po.action} → FILLED")
            pending.remove(po)
            any_resolved = True

        elif simple_status == "rejected":
            logger.warning(f"[Testnet] Pending #{po.order_id} {po.symbol} {po.action} → REJECTED, rollback")
            _rollback_from_pending(self, po)
            pending.remove(po)
            any_resolved = True

        elif simple_status == "pending":
            if po.age_seconds() > PENDING_ORDER_TIMEOUT_SECONDS:
                logger.warning(
                    f"[Testnet] Pending #{po.order_id} timeout ({po.age_seconds():.0f}s), rollback"
                )
                # Próba anulowania na Binance (best-effort)
                try:
                    svc._get_client().cancel_order(symbol=po.symbol.upper(), orderId=int(po.order_id))
                except Exception as e:
                    logger.warning(f"[Testnet] cancel_order failed: {e}")
                _rollback_from_pending(self, po)
                pending.remove(po)
                any_resolved = True
            else:
                logger.debug(f"[Testnet] Pending #{po.order_id} still pending ({po.age_seconds():.0f}s)")

    if any_resolved:
        # Pełny sync po rozwiązaniu pending
        _full_sync_from_binance(self)

    return len(pending) == 0


# ---------------------------------------------------------------------------
# SYNC PO ITERACJI – wywołaj po wykonaniu wszystkich orderów w timestampie
# ---------------------------------------------------------------------------

def testnet_sync_after_orders(self: "Wallet", wait_seconds: float = 10.0):
    """
    Odczekaj N sekund (żeby fills zdążyły wrócić z Binance),
    następnie zsynchronizuj lokalny portfel z wirtualnym stanem Binance.
    Wywołaj PO execute_orders() dla danego timestampu.
    """
    if not hasattr(self, "_pending_binance_orders"):
        return

    # Tylko jeśli były jakieś transakcje w tej iteracji
    # (sprawdzamy czy są pending które właśnie wysłaliśmy)
    if not self._pending_binance_orders:
        return

    logger.info(f"[Testnet] Czekam {wait_seconds}s na fills z Binance...")
    time.sleep(wait_seconds)

    # Sprawdź pending po odczekaniu
    svc = _get_binance_service()
    for po in list(self._pending_binance_orders):
        order_result = svc.get_order(symbol=po.symbol, order_id=po.order_id)
        simple_status = order_result.get("simple_status", "unknown")

        if simple_status == "filled":
            logger.info(f"[Testnet] Post-wait #{po.order_id} FILLED")
            self._pending_binance_orders.remove(po)
        elif simple_status == "rejected":
            logger.warning(f"[Testnet] Post-wait #{po.order_id} REJECTED, rollback")
            _rollback_from_pending(self, po)
            self._pending_binance_orders.remove(po)
        # "pending" zostaje na liście – zostanie sprawdzone w następnej iteracji

    # Pełny sync po odczekaniu
    _full_sync_from_binance(self)


# ---------------------------------------------------------------------------
# WALLET BUY
# ---------------------------------------------------------------------------

def testnet_wallet_buy(self: "Wallet", order: "Order"):
    """
    1. Snapshot → lokalna egzekucja
    2. Wyślij do Binance → dodaj do pending
    3. Jeśli Binance od razu FILLED (typowe dla market order) → reconcile
    4. Jeśli FAILED → rollback
    5. Jeśli PENDING → zostaw na liście, reconcile nastąpi w check_pending_orders
    """
    if not hasattr(self, "_pending_binance_orders"):
        self._pending_binance_orders = []

    symbol = order.symbol
    snap = _snapshot(self, symbol)
    _annotate_before(order, snap)

    # Lokalna egzekucja
    qty = order.qty
    price = order.price
    order.slippage_price = price
    transaction_value = qty * price * (1 + self.config[order.symbol].fee)

    if transaction_value > self.balance:
        order.rejected("no_balance")
        _annotate_skipped(order, "local_rejected_no_balance", snap)
        return

    current_qty = self.holdings.get(symbol, 0.0)
    current_avg = self.avg_buy_price.get(symbol, 0.0)
    if current_qty > 0:
        self.avg_buy_price[symbol] = (
            (current_qty * current_avg) + (qty * price)
        ) / (current_qty + qty)
    else:
        self.avg_buy_price[symbol] = price

    self.holdings[symbol] += qty
    self.balance -= transaction_value
    self.buys_count += 1
    order.status = "accepted"
    _annotate_local_after(order, self)

    # Binance call
    try:
        svc = _get_binance_service()
        result = svc.buy(symbol=symbol, qty=qty)

        order.binance_status = result.get("result", "unknown")
        order.binance_qty = float(result.get("qty", 0.0))
        order.binance_executed_price = float(result.get("executed_price", 0.0))
        order.binance_transaction_value = order.binance_qty * order.binance_executed_price
        order.binance_order_id = str(result.get("orderId", ""))
        order.binance_message = result.get("message", "")

        if result.get("result") == "success":
            raw_status = result.get("raw_status", "")
            if raw_status == "FILLED" or order.binance_qty > 0:
                # Market order wykonał się od razu – reconcile
                _reconcile_buy(self, order, snap)
            else:
                # PENDING – dodaj do listy, reconcile później
                self._pending_binance_orders.append(PendingBinanceOrder(
                    order_id=order.binance_order_id,
                    symbol=symbol,
                    action="buy",
                    local_qty=qty,
                    local_price=price,
                    local_balance_before=snap["balance"],
                    local_holdings_before=snap["holdings"],
                    local_avg_buy_price_before=snap["avg_buy_price"],
                ))
                order.binance_status = "pending"
                logger.info(f"[Testnet] BUY #{order.binance_order_id} {symbol} → PENDING")
        else:
            _rollback(self, symbol, snap)
            _annotate_rollback(order, snap, order.binance_message)

    except Exception as e:
        logger.error(f"[Testnet] BUY {symbol} exception: {e}")
        _rollback(self, symbol, snap)
        order.binance_status = "error"
        order.binance_message = str(e)
        _annotate_rollback(order, snap, str(e))


# ---------------------------------------------------------------------------
# WALLET SELL
# ---------------------------------------------------------------------------

def testnet_wallet_sell(self: "Wallet", order: "Order"):
    """
    Analogicznie do testnet_wallet_buy.
    """
    if not hasattr(self, "_pending_binance_orders"):
        self._pending_binance_orders = []

    symbol = order.symbol
    snap = _snapshot(self, symbol)
    _annotate_before(order, snap)

    order.slippage_price = order.price

    if order.qty > self.holdings.get(symbol, 0.0):
        order.rejected("not_enough_holdings")
        _annotate_skipped(order, "local_rejected_not_enough_holdings", snap)
        return

    transaction_value = order.qty * order.price * (1 - self.config[order.symbol].fee)
    self.holdings[symbol] -= order.qty
    self.balance += transaction_value
    self.sells_count += 1
    order.status = "accepted"
    if self.holdings[symbol] <= 1e-9:
        self.holdings[symbol] = 0.0
        self.avg_buy_price[symbol] = 0.0
    _annotate_local_after(order, self)

    try:
        svc = _get_binance_service()
        result = svc.sell(symbol=symbol, qty=order.qty)

        order.binance_status = result.get("result", "unknown")
        order.binance_qty = float(result.get("qty", 0.0))
        order.binance_executed_price = float(result.get("executed_price", 0.0))
        order.binance_transaction_value = order.binance_qty * order.binance_executed_price
        order.binance_order_id = str(result.get("orderId", ""))
        order.binance_message = result.get("message", "")

        if result.get("result") == "success":
            raw_status = result.get("raw_status", "")
            if raw_status == "FILLED" or order.binance_qty > 0:
                _reconcile_sell(self, order, snap)
            else:
                self._pending_binance_orders.append(PendingBinanceOrder(
                    order_id=order.binance_order_id,
                    symbol=symbol,
                    action="sell",
                    local_qty=order.qty,
                    local_price=order.price,
                    local_balance_before=snap["balance"],
                    local_holdings_before=snap["holdings"],
                    local_avg_buy_price_before=snap["avg_buy_price"],
                ))
                order.binance_status = "pending"
                logger.info(f"[Testnet] SELL #{order.binance_order_id} {symbol} → PENDING")
        else:
            _rollback(self, symbol, snap)
            _annotate_rollback(order, snap, order.binance_message)

    except Exception as e:
        logger.error(f"[Testnet] SELL {symbol} exception: {e}")
        _rollback(self, symbol, snap)
        order.binance_status = "error"
        order.binance_message = str(e)
        _annotate_rollback(order, snap, str(e))


# ---------------------------------------------------------------------------
# Full sync – korekta lokalnego portfela do wirtualnego stanu Binance
# ---------------------------------------------------------------------------

def _full_sync_from_binance(wallet: "Wallet"):
    """
    Pobiera wirtualny stan portfela z Binance i nadpisuje lokalny portfel.
    Używa get_virtual_portfolio() (z offsetem), więc widzi tylko to
    co zrobiła symulacja.
    """
    svc = _get_binance_service()
    result = svc.get_virtual_portfolio(wallet.symbols)

    if result.get("result") != "success":
        logger.error(f"[Testnet] full_sync failed: {result.get('message')}")
        return

    old_balance = wallet.balance
    old_holdings = dict(wallet.holdings)

    wallet.balance = result["virtual_usdt"]
    for symbol, qty in result["virtual_holdings"].items():
        if symbol in wallet.holdings:
            wallet.holdings[symbol] = qty

    logger.info(
        f"[Testnet] Full sync | "
        f"balance: {old_balance:.2f} → {wallet.balance:.2f} | "
        f"holdings: {old_holdings} → {dict(wallet.holdings)}"
    )


# ---------------------------------------------------------------------------
# Snapshot / Rollback
# ---------------------------------------------------------------------------

def _snapshot(wallet: "Wallet", symbol: str) -> dict:
    return {
        "balance": wallet.balance,
        "holdings": wallet.holdings.get(symbol, 0.0),
        "avg_buy_price": wallet.avg_buy_price.get(symbol, 0.0),
        "buys_count": wallet.buys_count,
        "sells_count": wallet.sells_count,
    }


def _rollback(wallet: "Wallet", symbol: str, snap: dict):
    wallet.balance = snap["balance"]
    wallet.holdings[symbol] = snap["holdings"]
    wallet.avg_buy_price[symbol] = snap["avg_buy_price"]
    wallet.buys_count = snap["buys_count"]
    wallet.sells_count = snap["sells_count"]
    logger.warning(f"[Testnet] ROLLBACK {symbol} – portfel przywrócony.")


def _rollback_from_pending(wallet: "Wallet", po: "PendingBinanceOrder"):
    """Rollback na podstawie danych z PendingBinanceOrder."""
    wallet.balance = po.local_balance_before
    wallet.holdings[po.symbol] = po.local_holdings_before
    wallet.avg_buy_price[po.symbol] = po.local_avg_buy_price_before
    if po.action == "buy":
        wallet.buys_count = max(0, wallet.buys_count - 1)
    else:
        wallet.sells_count = max(0, wallet.sells_count - 1)
    logger.warning(f"[Testnet] ROLLBACK from pending #{po.order_id} {po.symbol} {po.action}")


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _reconcile_buy(wallet: "Wallet", order: "Order", snap: dict):
    symbol = order.symbol
    bqty = order.binance_qty
    bprice = order.binance_executed_price

    wallet.balance = snap["balance"]
    wallet.holdings[symbol] = snap["holdings"]
    wallet.avg_buy_price[symbol] = snap["avg_buy_price"]

    tv = bqty * bprice * (1 + wallet.config[symbol].fee)
    current_qty = snap["holdings"]
    current_avg = snap["avg_buy_price"]
    if current_qty > 0:
        wallet.avg_buy_price[symbol] = (
            (current_qty * current_avg) + (bqty * bprice)
        ) / (current_qty + bqty)
    else:
        wallet.avg_buy_price[symbol] = bprice

    wallet.holdings[symbol] += bqty
    wallet.balance -= tv
    _annotate_reconciled(order, wallet, snap)
    logger.info(
        f"[Testnet] RECONCILE BUY {symbol} | "
        f"local qty={order.local_after_holdings - snap['holdings']:.6f} @ {order.price:.4f} | "
        f"binance qty={bqty:.6f} @ {bprice:.4f} | "
        f"drift qty={order.drift_qty:+.6f} val={order.drift_value:+.4f}$"
    )


def _reconcile_sell(wallet: "Wallet", order: "Order", snap: dict):
    symbol = order.symbol
    bqty = order.binance_qty
    bprice = order.binance_executed_price

    wallet.balance = snap["balance"]
    wallet.holdings[symbol] = snap["holdings"]
    wallet.avg_buy_price[symbol] = snap["avg_buy_price"]

    tv = bqty * bprice * (1 - wallet.config[symbol].fee)
    wallet.holdings[symbol] -= bqty
    wallet.balance += tv
    if wallet.holdings[symbol] <= 1e-9:
        wallet.holdings[symbol] = 0.0
        wallet.avg_buy_price[symbol] = 0.0

    _annotate_reconciled(order, wallet, snap)
    logger.info(
        f"[Testnet] RECONCILE SELL {symbol} | "
        f"local qty={snap['holdings'] - order.local_after_holdings:.6f} @ {order.price:.4f} | "
        f"binance qty={bqty:.6f} @ {bprice:.4f} | "
        f"drift qty={order.drift_qty:+.6f} val={order.drift_value:+.4f}$"
    )


# ---------------------------------------------------------------------------
# Adnotacje order
# ---------------------------------------------------------------------------

def _annotate_before(order: "Order", snap: dict):
    order.local_before_balance = snap["balance"]
    order.local_before_holdings = snap["holdings"]
    order.local_before_avg_buy_price = snap["avg_buy_price"]


def _annotate_local_after(order: "Order", wallet: "Wallet"):
    s = order.symbol
    order.local_after_balance = wallet.balance
    order.local_after_holdings = wallet.holdings.get(s, 0.0)
    order.local_after_avg_buy_price = wallet.avg_buy_price.get(s, 0.0)


def _annotate_reconciled(order: "Order", wallet: "Wallet", snap: dict):
    s = order.symbol
    order.reconciled_balance = wallet.balance
    order.reconciled_holdings = wallet.holdings.get(s, 0.0)
    order.reconciled_avg_buy_price = wallet.avg_buy_price.get(s, 0.0)
    order.drift_qty = round(order.local_after_holdings - order.reconciled_holdings, 8)
    order.drift_value = round(order.local_after_balance - order.reconciled_balance, 4)


def _annotate_skipped(order: "Order", reason: str, snap: dict):
    order.binance_status = "skipped"
    order.binance_qty = 0.0
    order.binance_executed_price = 0.0
    order.binance_transaction_value = 0.0
    order.binance_order_id = ""
    order.binance_message = reason
    order.local_after_balance = snap["balance"]
    order.local_after_holdings = snap["holdings"]
    order.local_after_avg_buy_price = snap["avg_buy_price"]
    order.reconciled_balance = snap["balance"]
    order.reconciled_holdings = snap["holdings"]
    order.reconciled_avg_buy_price = snap["avg_buy_price"]
    order.drift_qty = 0.0
    order.drift_value = 0.0


def _annotate_rollback(order: "Order", snap: dict, message: str):
    order.reconciled_balance = snap["balance"]
    order.reconciled_holdings = snap["holdings"]
    order.reconciled_avg_buy_price = snap["avg_buy_price"]
    order.drift_qty = 0.0
    order.drift_value = 0.0
    order.status = "binance_failed"
    order.reason = message