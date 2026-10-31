from .rl.base_train_service import BaseTrainService
import os
import time
from datetime import datetime, timedelta
from stable_baselines3 import DQN
import torch

class TrainAgentDQNService(BaseTrainService):
    """Trening agenta DQN (Stable-Baselines3) na bazie BaseTrainService."""

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(base_dir=base_dir, agent_type='dqn')

    def train_agent_dqn(self,
                    rabbitmq_msg_id=None,
                    model_files=None,
                    data_files=None,
                    total_timesteps=500_000,
                    learning_rate=0.0001,
                    buffer_size=100_000,
                    learning_starts=10_000,
                    batch_size=128,
                    gamma=0.99,
                    tau=0.01,
                    train_freq=4,
                    gradient_steps=1,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    episode_timesteps=1024,
                    mp_threshold=0.45,
                    live_logging=True,
                    experiment_version="v0",
                    dqn_bins=3):

        try:
            start_time = time.time()
            if self.models == {}: self._load_data(model_files, data_files)

            env, callback = self._setup_env(
                total_timesteps=total_timesteps,
                base_dir=self.base_dir,
                episode_timesteps=episode_timesteps,
                live_logging=live_logging,
                mp_threshold=mp_threshold,
                num_envs=4,
                gamma=gamma,
                experiment_version=experiment_version
            )

            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[128, 128]
            )

            agent = DQN(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                gamma=gamma,
                tau=tau,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                policy_kwargs=policy_kwargs,
                verbose=0
            )

            # Train
            print(f"[TrainAgentDQN] Rozpoczynanie treningu...")
            agent.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10000)

            agent_path, agent_file = self._save_agent(agent)
            env.close()

            elapsed = timedelta(seconds=time.time() - start_time)
            print(f"[TrainAgentDQN] Trening zakończony ({elapsed}).")

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "DQN training completed successfully",
                "agent_path": agent_path,
                "agent_file": agent_file,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print(f"[DQN][ERROR] {e}")
            traceback.print_exc()

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "failed",
                "message": str(e),
                "agent_file": None,
                "agent_path": None,
                "timestamp": datetime.now().isoformat()
            }