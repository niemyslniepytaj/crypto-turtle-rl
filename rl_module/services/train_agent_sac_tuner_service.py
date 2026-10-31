import optuna
from .rl.base_tuner_service import BaseTunerService
from .train_agent_sac_service import TrainAgentSACService

class SACTunerService(BaseTunerService):
    """Optuna hiperparametrów SAC."""

    def __init__(self,base_dir=None):
        super().__init__(base_dir=base_dir)

        self.agent_type = 'sac'
        self.sac = TrainAgentSACService(base_dir=base_dir)
        print(f"[SACTuner] SAC service ready")

    def _objective(self, trial: optuna.Trial) -> float:
        """Funkcja celu Optuna: trening SAC i ewaluacja próby."""

        episode_timesteps = self._get_param(trial, 'episode_timesteps', 1024)
        learning_rate = self._get_param(trial, 'learning_rate', 3e-4)
        gamma = self._get_param(trial, 'gamma', 0.99)
        batch_size = self._get_param(trial, 'batch_size', 256)
        mp_threshold = self._get_param(trial, 'mp_threshold', 0.45)
        total_timesteps = self._get_param(trial, 'total_timesteps', 500_000)

        # SAC-specific
        buffer_size = self._get_param(trial, 'buffer_size', 100_000)
        learning_starts = self._get_param(trial, 'learning_starts', 10_000)
        tau = self._get_param(trial, 'tau', 0.005)
        train_freq = self._get_param(trial, 'train_freq', 1)
        gradient_steps = self._get_param(trial, 'gradient_steps', 1)
        ent_coef = self._get_param(trial, 'ent_coef', 'auto')

        # Print trial info
        print(f"\n{'=' * 70}")
        print(f"Trial {trial.number} - SAC")
        print(f"{'=' * 70}")
        print(f"  episode_timesteps: {episode_timesteps}")
        print(f"  learning_rate: {learning_rate:.6f}")
        print(f"  buffer_size: {buffer_size:,}")
        print(f"  learning_starts: {learning_starts:,}")
        print(f"  batch_size: {batch_size}")
        print(f"  gamma: {gamma:.4f}")
        print(f"  tau: {tau:.5f}")
        print(f"  train_freq: {train_freq}")
        print(f"  gradient_steps: {gradient_steps}")
        print(f"  ent_coef: {ent_coef}")
        print(f"  mp_threshold: {mp_threshold:.2f}")
        print(f"  total_timesteps: {total_timesteps:,}")
        print(f"{'=' * 70}\n")

        try:
            agent_result = self.sac.train_agent_sac(
                rabbitmq_msg_id=None,
                model_files=self.model_files,
                data_files=self.train_data_files,
                total_timesteps=total_timesteps,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                gamma=gamma,
                tau=tau,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                ent_coef=ent_coef,
                episode_timesteps=episode_timesteps,
                mp_threshold=mp_threshold,
                live_logging=self.live_logging,
                experiment_version=self.experiment_version
            )

            if agent_result["result"] != "success":
                raise RuntimeError(f"Training failed: {agent_result['message']}")

            agent_filename = agent_result["agent_file"]

            return self._evaluate_agent(trial, agent_filename, mp_threshold)

        except Exception as e:
            print(f"\n[SAC Trial {trial.number}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0