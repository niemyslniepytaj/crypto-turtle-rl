from collections import defaultdict

from .structures import WalletConfig, CoreConfig

class Wallet:
    """Portfel symulacyjny: salda, limity ekspozycji, stop-loss oraz egzekucja zleceń."""

    def __init__(self, core_mode="sim_mp", core_config: CoreConfig = None):
        self.core_config = core_config
        self.balance = self.core_config.balance_start

        self.core_mode = core_mode

        self.symbols = self.core_config.symbols
        self.config = {}
        for symbol in self.symbols:
            self.config[symbol] = WalletConfig()
            # train_rl też bierze SL / limity z CoreConfig (wcześniej tylko sim_*)
            if core_mode in ("sim_rl", "sim_mp", "sim_ml", "train_rl"):
                self.config[symbol].set(core_config)

        self.holdings = {}
        self.last_buy_price = {}
        self.avg_buy_price = {}

        for symbol in self.symbols:
            self.initialize_symbol_and_holdings(symbol)

        self.transaction_count = 0
        self.orders = []
        self.logs = []
        self.current_prices_wallet = {}

        self.portfolio_values_history = []
        self.portfolio_holdings_history = []
        self.closed_trades = []

        # Callback Counters
        self.buys_count = 0
        self.sells_count = 0
        self.transaction_count = 0
        self.orders_by_t = defaultdict(list)


    def reset(self, balance):
        """Przywraca stan portfela do salda początkowego przy zachowaniu alokacji obiektów."""

        self.balance = float(balance)

        self.transaction_count = 0
        self.buys_count = 0
        self.sells_count = 0

        self.config = {}
        for symbol in self.symbols:
            self.config[symbol] = WalletConfig()
            if self.core_mode in ("sim_rl", "sim_mp", "sim_ml", "train_rl"):
                self.config[symbol].set(self.core_config)
        self.current_prices_wallet.clear()
        self.portfolio_values_history.clear()
        self.portfolio_holdings_history.clear()
        self.closed_trades.clear()
        self.orders.clear()
        self.orders_by_t = defaultdict(list)
        self.logs.clear()

        for symbol in self.holdings:
            self.initialize_symbol_and_holdings(symbol)

    def initialize_symbol_and_holdings(self, symbol):
        self.holdings[symbol] = 0.0
        self.last_buy_price[symbol] = 0.0
        self.avg_buy_price[symbol] = 0.0

    def get_last_t_logs(self, close_time):

        last_t_logs = [order for order in self.logs if order.timestamp == -1]

        for order in last_t_logs:
            order.timestamp = close_time

        return last_t_logs

    def update_history(self, current_orders):
        """ Przygotowuje logi dla symulacji """

        total_value = self.total_value(self.current_prices_wallet)
        self.portfolio_values_history.append(round(total_value,1))
        self.portfolio_holdings_history.append(self.total_holdings(self.current_prices_wallet))
        for order in current_orders:
            order.total_portfolio_value = total_value
            order.actual_qty = self.holdings[order.symbol]
            order.actual_usdt = self.balance
            self.logs.append(order)
            self.orders.remove(order)


    def execute_orders(self, t, current_prices):
        """Egzekwuje kolejkę zleceń w chwili t przy podanych cenach rynkowych."""
        self.current_prices_wallet = current_prices

        new_orders = self.orders_by_t.get(t, [])

        if not new_orders:
            pass
        else:
            for order in new_orders:
                if self.core_mode in ("rl_train", "sim_rl"): self.rl_config(order.rl_config)

                if order.action == "buy":  self.calculate_quantity_buy(order, current_prices)
                if order.action == "sell": self.calculate_quantity_sell(order)
        #TODO: poprawka
        current_orders = self.orders_by_t.get(t - self.core_config.slippage, [])
        if not current_orders:
            return

        for order in current_orders:
            if order.status  == "rejected":
                pass
            else:
                if order.action == "buy":  self.buy(order)
                if order.action == "sell": self.sell(order)
                if order.status != "rejected": self.transaction_count += 1

        self.update_history(current_orders)


    def rl_config(self,scaled):
        pass
        # mincost_min, mincost_max = 1.0, 5.0
        # mincost = mincost_min + scaled[4] * (mincost_max - mincost_min)
        #
        # mwpc_min, mwpc_max = 0.7, 1.0
        # max_wallet_percentage_crypto = mwpc_min + scaled[5] * (mwpc_max - mwpc_min)
        #
        # self.config.mincost = float(round(mincost,2))
        # self.config.max_wallet_percentage_crypto = float(round(max_wallet_percentage_crypto,2))
        # self.config.stop_loss = scaled_dict[6]

    def buy(self, order):
        """Realizuje kupno: aktualizuje holdings, średnią cenę nabycia i saldo USDT."""
        # TODO: Naprawić slippage
        current_price = order.price
        order.slippage_price = current_price
        qty = order.qty
        symbol = order.symbol

        usd_value = qty * current_price
        transaction_value = usd_value * (1 + self.config[symbol].fee)

        if transaction_value > self.balance:
            order.rejected("no_balance")
            return
        else:
            current_qty = self.holdings.get(symbol, 0.0)
            current_avg = self.avg_buy_price.get(symbol, 0.0)

            if current_qty > 0:
                new_avg = ((current_qty * current_avg) + (qty * current_price)) / (current_qty + qty)
                self.avg_buy_price[symbol] = new_avg
            else:
                self.avg_buy_price[symbol] = current_price
            self.holdings[symbol] += qty
            self.balance -= transaction_value
            self.buys_count +=1
            order.status = "accepted"
            return


    def sell(self, order):
        """Realizuje sprzedaż: zapisuje closed_trade, aktualizuje holdings i saldo."""
        order.slippage_price = order.price

        symbol = order.symbol

        if order.qty > self.holdings.get(symbol, 0.0):
            order.rejected('not_enough_holdings')
            return
        else:
            buy_price = self.avg_buy_price.get(symbol, 0.0)
            fee = self.config[symbol].fee
            usd_value = order.qty * order.price
            transaction_value = usd_value * (1 - fee)
            pnl_usd = (order.price - buy_price) * order.qty - usd_value * fee
            pnl_pct = ((order.price / buy_price) - 1.0) * 100 if buy_price > 0 else 0.0

            self.closed_trades.append({
                "symbol": symbol,
                "qty": order.qty,
                "buy_price": buy_price,
                "sell_price": order.price,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "fee": usd_value * fee,
                "t": getattr(order, "timestamp", None),
            })

            self.holdings[order.symbol] -= order.qty
            self.balance += transaction_value
            self.sells_count += 1
            order.status = "accepted"
            if self.holdings[symbol] <= 1e-9:
                self.holdings[symbol] = 0.0
                self.avg_buy_price[symbol] = 0.0
            return

    def calculate_quantity_buy(self, order, prices):
        """Wylicza qty kupna z limitem cash i max ekspozycji per symbol."""
        price = order.price
        symbol = order.symbol
        available_cash = self.balance * self.config[symbol].wallet_percentage_limit
        usd_amount = available_cash * order.fraction

        if usd_amount < self.config[symbol].mincost:
            order.rejected("lower_than_mincost")
            return

        qty = usd_amount / price

        crypto_qty = self.holdings.get(order.symbol, 0.0)
        current_holding_value = crypto_qty * price

        portfolio_value = self.total_value(prices)
        order.qty = qty
        if portfolio_value > 0:
            new_holding_value = current_holding_value + qty * price
            current_crypto_percentage = new_holding_value / portfolio_value
            order.qty = qty
            if current_crypto_percentage > self.config[symbol].max_wallet_percentage_crypto:
                max_allowed_holding_value = self.config[symbol].max_wallet_percentage_crypto * portfolio_value
                max_allowed_qty = (max_allowed_holding_value - current_holding_value) / price

                qty = max(0.0, min(qty, max_allowed_qty))
                order.qty = qty
                if qty == 0.0:
                    order.rejected("limit_max_wallet_percentage_crypto")
                    return
                if max_allowed_qty * price < self.config[symbol].mincost:
                    order.rejected("lower_than_mincost")
                    return
        order.qty = qty
        return

    def calculate_quantity_sell(self, order):
        if self.holdings[order.symbol] > 0:
            crypto_qty = self.holdings.get(order.symbol, 0.0)
            qty = crypto_qty * order.fraction

            usd_value = qty * order.price
            # force_exit (stop-loss): sprzedaj nawet poniżej mincost
            if usd_value < self.config[order.symbol].mincost and not getattr(order, "force_exit", False):
                order.rejected("lower_than_mincost")
                return
            order.qty = qty
        else: order.rejected("no_holdings")
        return

    def total_value(self, prices: dict):
        total = self.balance
        for s, p in prices.items():
            total += self.holdings.get(s, 0.0) * p
        #self.actual_portfolio_value = total
        return total

    def total_holding_value(self, symbol, prices):
        price = prices.get(symbol, 0.0)
        crypto_qty = self.holdings[symbol]
        return round(crypto_qty * price,4)

    def total_holdings(self, prices: dict):
        per_symbol_values = []

        for s, p in prices.items():
            value = self.holdings.get(s, 0.0) * p
            per_symbol_values.append(round(value, 4))

        return per_symbol_values

    @staticmethod
    def stop_loss_enabled(stop_loss: float) -> bool:
        """0 lub >=1 → wyłączony (stare configi z stop_loss=1.0 nic nie robią)."""
        try:
            sl = float(stop_loss)
        except (TypeError, ValueError):
            return False
        return 0.0 < sl < 1.0

    def check_stop_loss(self, symbol, market_price):
        """
        Stop od średniej ceny nabycia: price <= avg_buy * (1 - stop_loss).
        Zwraca True gdy trzeba natychmiast spłaszczyć pozycję.
        """
        sl = float(getattr(self.config.get(symbol), "stop_loss", 0.0) or 0.0)
        if not self.stop_loss_enabled(sl):
            return False
        qty = float(self.holdings.get(symbol, 0.0) or 0.0)
        avg = float(self.avg_buy_price.get(symbol, 0.0) or 0.0)
        if qty <= 1e-12 or avg <= 0.0 or market_price <= 0.0:
            return False
        return float(market_price) <= avg * (1.0 - sl)
