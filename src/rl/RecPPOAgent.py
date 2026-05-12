import sys
import os
import numpy as np
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.rl.CNNFeatureExtractor import CNNFeatureExtractor


class RecPPOAgent:
    """
    LSTM-PPO agent for Active Flow Control (Li et al. 2025, Table I).
    Reward and episode progress are logged to TensorBoard automatically.
    """

    def __init__(
        self,
        env,
        use_cnn:       bool  = False,   # paper uses plain LSTM
        learning_rate: float = 5e-4,    # paper Table I
        n_steps:       int   = 128,
        batch_size:    int   = 64,
        n_epochs:      int   = 10,
        gamma:         float = 0.99,    # paper Table I
        clip_range:    float = 0.2,     # paper Table I
        tensorboard_log: str = "RL_results/tensorboard",
        verbose:       int   = 1,
    ):
        if isinstance(env, list):
            env = SubprocVecEnv(env)

        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        ) if use_cnn else {}

        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            clip_range=clip_range,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs or None,
        )

    def train(
        self,
        total_timesteps: int,
        save_path:        str  = "RL_results/training/learned_policy",
        progress_bar:     bool = False,          # True crashes Jupyter without ipywidgets
        tb_log_name:      str  = "AFC_DRL",      # subfolder name inside tensorboard_log/
        print_every:      int  = 50,             # print a heartbeat every N env steps
    ):
        from stable_baselines3.common.callbacks import BaseCallback
        _total = total_timesteps

        class _TrainingMonitor(BaseCallback):
            def __init__(self):
                super().__init__()
                self._ep_count   = 0
                self._ep_rewards = []

            def _on_step(self) -> bool:
                # see training progress at a given few steps (since CFD takes decades to run :( )
                if self.n_calls % print_every == 0:
                    print(f"  step {self.n_calls:>6} / {_total}", flush=True)

                # full episode summary when done flag fires
                for done, info in zip(
                    self.locals.get("dones", []),
                    self.locals.get("infos", []),
                ):
                    if done:
                        ep = info.get("episode", {})
                        if ep:
                            self._ep_count += 1
                            self._ep_rewards.append(ep["r"])
                            print(
                                f"\n>>> Episode {self._ep_count} done | "
                                f"reward = {ep['r']:.4f} | "
                                f"mean so far = {np.mean(self._ep_rewards):.4f}\n",
                                flush=True,
                            )
                return True

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            tb_log_name=tb_log_name,
            callback=_TrainingMonitor(),
        )
        self.model.save(save_path)
        print(f"Policy saved: '{save_path}.zip'")

    def load(self, path: str, env=None):
        self.model = RecurrentPPO.load(path, env=env)
        print(f"Policy loaded: '{path}'")

    def run_inference(self, env, save_dir: str = "RL_results/inference"):
        """
        Run one deterministic episode and save C_D, C_L arrays to save_dir.
        Use those arrays later to plot controlled vs uncontrolled lift-to-drag to replicate work from paper.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        obs, _ = env.reset()
        lstm_states, episode_starts = None, np.ones((1,), dtype=bool)
        C_D_list, C_L_list = [], []

        while True:
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, _, done, _, info = env.step(action)
            episode_starts = np.array([done])
            C_D_list.append(info["C_D"])
            C_L_list.append(info["C_L"])
            if done:
                break

        C_D = np.array(C_D_list)
        C_L = np.array(C_L_list)
        np.save(f"{save_dir}/C_D.npy", C_D)
        np.save(f"{save_dir}/C_L.npy", C_L)
        print(f"Inference data saved → '{save_dir}/'  "
              f"(mean C_L/C_D = {(C_L / np.where(np.abs(C_D) > 1e-8, C_D, 1e-8)).mean():.4f})")
        return C_D, C_L


