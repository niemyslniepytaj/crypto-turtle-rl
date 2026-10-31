from .rl.base_train_service import BaseTrainService
import os
import time
from datetime import datetime, timedelta
from stable_baselines3 import SAC
import torch

class TrainAgentSACService(BaseTrainService):
    """Trening agenta SAC (Stable-Baselines3) na bazie BaseTrainService."""

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(base_dir=base_dir, agent_type='sac')

    def train_agent_sac(self,
                    rabbitmq_msg_id=None,
                    model_files=None,
                    data_files=None,
                    total_timesteps=500_000,
                    learning_rate=0.0003,
                    buffer_size=100_000,
                    learning_starts=10_000,
                    batch_size=256,
                    gamma=0.99,
                    tau=0.005,
                    train_freq=1,
                    gradient_steps=1,
                    ent_coef='auto',
                    episode_timesteps=1024,
                    mp_threshold=0.45,
                    live_logging=True,
                    experiment_version="v0"):

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
                net_arch=[256, 256]
            )

            agent = SAC(
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
                ent_coef=ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=0
            )

            print(f"[TrainAgentSAC] Rozpoczynanie treningu...")
            agent.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10000)

            agent_path, agent_file = self._save_agent(agent)
            env.close()

            elapsed = timedelta(seconds=time.time() - start_time)
            print(f"[TrainAgentSAC] Trening zakończony ({elapsed}).")

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "SAC training completed successfully",
                "agent_path": agent_path,
                "agent_file": agent_file,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print(f"[SAC][ERROR] {e}")
            traceback.print_exc()

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "failed",
                "message": str(e),
                "agent_file": None,
                "agent_path": None,
                "timestamp": datetime.now().isoformat()
            }