from .train_agent_dqn_service import TrainAgentDQNService
from .rl.base_tuner_service import BaseTunerService
import optuna

class DQNTunerService(BaseTunerService):
    """Optuna hiperparametrów DQN."""

    def __init__(self, base_dir=None):
        super().__init__(base_dir=base_dir)

        self.agent_type = 'dqn'
        self.dqn = TrainAgentDQNService(base_dir=base_dir)

        print(f"[DQNTuner] DQN service not yet implemented")
    # TODO: Initialize DQN service when implemented
    def _objective(self, trial: optuna.Trial) -> float:
        """Funkcja celu Optuna DQN (implementacja w toku)."""
        raise NotImplementedError("DQN training service not yet implemented")