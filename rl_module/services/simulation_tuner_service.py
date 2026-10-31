from .rl.base_tuner_service import BaseTunerService
import optuna


class SimulationTunerService(BaseTunerService):
    """Optuna parametrów symulacji MP (bez agenta RL); ścieżka v6+mp, staged E1–E3."""

    def __init__(self, base_dir=None):
        super().__init__(base_dir=base_dir)

        self.agent_type = 'simulation'
        print(f"[SimulationTuner] Service ready.")

    def _objective(self, trial: optuna.Trial) -> float:
        """Funkcja celu Optuna dla powierzchni parametrów symulacji MP."""

        wallet_percentage_limit = self._get_param(trial, 'wallet_percentage_limit', 0.9)
        max_wallet_percentage_crypto = self._get_param(trial, 'max_wallet_percentage_crypto', 1.0)
        stop_loss = self._get_param(trial, 'stop_loss', 1.0)
        stop_loss_cooldown = int(self._get_param(trial, 'stop_loss_cooldown', 120))
        slippage = self._get_param(trial, 'slippage', 0)
        mincost = self._get_param(trial, 'mincost', 1.0)
        balance_start = self._get_param(trial, 'balance_start', 1000.0)

        threshold_buy = self._get_param(trial, 'threshold_buy', 0.5)
        threshold_sell = self._get_param(trial, 'threshold_sell', 0.5)
        tx_fraction = self._get_param(trial, 'tx_fraction', 0.01)
        max_fraction_when_confident = self._get_param(trial, 'max_fraction_when_confident', 0.5)

        mp_alloc_bias = self._get_param(trial, 'mp_alloc_bias', 0.5)
        mp_alloc_scale = self._get_param(trial, 'mp_alloc_scale', 0.5)
        mp_confidence = self._get_param(trial, 'mp_confidence', 0.5)
        mp_urgency = self._get_param(trial, 'mp_urgency', 0.5)
        mp_signal_deadzone = self._get_param(trial, 'mp_signal_deadzone', 0.15)
        mp_rebalance_deadzone = self._get_param(trial, 'mp_rebalance_deadzone', 0.25)
        mp_ema_alpha_min = self._get_param(trial, 'mp_ema_alpha_min', 0.02)
        mp_ema_alpha_span = self._get_param(trial, 'mp_ema_alpha_span', 0.08)
        from rl_module.services.thesis.knobs import CLAMP_WALLET_CAP_TO_1N
        clamp_wallet_cap_to_1n = bool(
            self._get_param(trial, 'clamp_wallet_cap_to_1n', CLAMP_WALLET_CAP_TO_1N)
        )
        window = self.get_window(trial) if hasattr(self, 'get_window') else None

        trial.set_user_attr('log_window', window)
        trial.set_user_attr('stage', getattr(self, '_stage_label', None))
        trial.set_user_attr('log_bias', mp_alloc_bias)
        trial.set_user_attr('log_scale', mp_alloc_scale)
        trial.set_user_attr('log_conf', mp_confidence)
        trial.set_user_attr('log_urg', mp_urgency)
        trial.set_user_attr('log_mincost', mincost)
        trial.set_user_attr('log_wlim', wallet_percentage_limit)
        trial.set_user_attr('log_maxcrypto', max_wallet_percentage_crypto)
        trial.set_user_attr('log_sig_dz', mp_signal_deadzone)
        trial.set_user_attr('log_reb_dz', mp_rebalance_deadzone)
        trial.set_user_attr('log_sl', stop_loss)
        trial.set_user_attr('log_sl_cd', stop_loss_cooldown)

        try:
            return self._evaluate_mp_simulation(
                trial=trial,
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell,
                wallet_percentage_limit=wallet_percentage_limit,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                stop_loss=stop_loss,
                stop_loss_cooldown=stop_loss_cooldown,
                tx_fraction=tx_fraction,
                max_fraction_when_confident=max_fraction_when_confident,
                slippage=slippage,
                mincost=mincost,
                balance_start=balance_start,
                mp_alloc_bias=mp_alloc_bias,
                mp_alloc_scale=mp_alloc_scale,
                mp_confidence=mp_confidence,
                mp_urgency=mp_urgency,
                mp_signal_deadzone=mp_signal_deadzone,
                mp_rebalance_deadzone=mp_rebalance_deadzone,
                mp_ema_alpha_min=mp_ema_alpha_min,
                mp_ema_alpha_span=mp_ema_alpha_span,
                clamp_wallet_cap_to_1n=clamp_wallet_cap_to_1n,
                window=window,
            )

        except Exception as e:
            print(f"[Optuna] Trial {trial.number} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0
