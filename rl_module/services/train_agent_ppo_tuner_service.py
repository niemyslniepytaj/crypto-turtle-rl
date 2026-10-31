from .train_agent_ppo_service import TrainAgentPPOService
from .rl.base_tuner_service import BaseTunerService
import optuna

class PPOTunerService(BaseTunerService):
    """Optuna hiperparametrów PPO (warsztat; poza zamrożonym pretrainem tezy)."""

    def __init__(self, base_dir=None):
        super().__init__(base_dir=base_dir)

        self.agent_type = 'ppo'
        self.ppo = TrainAgentPPOService(base_dir=base_dir)
        print(f"[PPOTuner] PPO service ready.")

    def _objective(self, trial: optuna.Trial) -> float:
        """Funkcja celu Optuna: trening PPO i ewaluacja próby."""

        episode_timesteps = self._get_param(trial, 'episode_timesteps', 1024)
        learning_rate = self._get_param(trial, 'learning_rate', 3e-4)
        gamma = self._get_param(trial, 'gamma', 0.99)
        batch_size = self._get_param(trial, 'batch_size', 64)
        mp_threshold = self._get_param(trial, 'mp_threshold', 0.45)
        total_timesteps = self._get_param(trial, 'total_timesteps', 1_000_000)

        # PPO-specific
        n_steps_multiplier = self._get_param(trial, 'n_steps_multiplier', 2)
        n_steps = episode_timesteps * n_steps_multiplier
        n_epochs = self._get_param(trial, 'n_epochs', 10)
        gae_lambda = self._get_param(trial, 'gae_lambda', 0.95)
        clip_range = self._get_param(trial, 'clip_range', 0.2)
        ent_coef = self._get_param(trial, 'ent_coef', 0.01)

        # Enforce constraint: batch_size <= n_steps
        if batch_size > n_steps:
            batch_size = n_steps // 2

        # Print trial info
        print(f"\n{'=' * 70}")
        print(f"Trial {trial.number} - PPO")
        print(f"{'=' * 70}")
        print(f"  episode_timesteps: {episode_timesteps}")
        print(f"  n_steps: {n_steps} (episode × {n_steps_multiplier})")
        print(f"  learning_rate: {learning_rate:.6f}")
        print(f"  batch_size: {batch_size}")
        print(f"  gamma: {gamma:.4f}")
        print(f"  n_epochs: {n_epochs}")
        print(f"  gae_lambda: {gae_lambda:.3f}")
        print(f"  clip_range: {clip_range:.3f}")
        print(f"  ent_coef: {ent_coef:.6f}")
        print(f"  mp_threshold: {mp_threshold:.2f}")
        print(f"  total_timesteps: {total_timesteps:,}")
        print(f"{'=' * 70}\n")

        try:
            agent_result = self.ppo.train_agent_ppo(
                rabbitmq_msg_id=None,
                model_files=self.model_files,
                data_files=self.train_data_files,
                total_timesteps=total_timesteps,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                ent_coef=ent_coef,
                n_epochs=n_epochs,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
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
            print(f"\n[PPO Trial {trial.number}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0