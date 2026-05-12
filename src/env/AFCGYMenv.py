import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.env.CFDEnvSolver import CFDEnvSolver
from pathlib import Path

# constants from paper 
Q_MAX         = 0.009   # max jet flow rate 
ALPHA_SMOOTH  = 0.2     # exponential smoothing decay factor α 
N_ACTION      = 25      # CFD timesteps per RL step = T in reward window


class AFCGymEnv(gym.Env):
    """
    Gymnasium environment for Active Flow Control over an airfoil.

    Observation: Box(3, H, W) — channels [u_x, u_y, p] over the probe grid.
    Action:      Box(3,)      — jet flow rates Q_j ∈ [-Q_MAX, Q_MAX]

    Reward:  r_t = <b>_T - r_0
      b       = C_L / C_D computed at each of the T=25 CFD timesteps
      <b>_T   = mean of b over those 25 timesteps
      r_0     = mean(C_L/C_D) of the uncontrolled flow (loaded from checkpoint)

    Episode flow:
      reset(): load checkpoint -> return initial probe obs
      step(a): smooth action -> set jet BCs -> advance N_ACTION CFD steps
               -> compute reward -> sample probes -> return (obs, reward, done, info)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        airfoil_config,          # AirfoilConfig instance
        checkpoint_path: str,    # path to initial_CFD_results/checkpoint/
        mesh_filename:   str,    # path to .msh file
        T_total:         float  = 9.0,
        dt:              int    = 1600,
        mu:              float  = 0.00012,
        rho:             float  = 1.0,
        Re:              float  = 2500,
        U_m:             float  = 0.45,
    ):
        super().__init__()

        self.airfoil_config  = airfoil_config
        self.checkpoint_path = checkpoint_path
        self.jet_normals     = airfoil_config.jet_normals

        self._probe_grid = (12, 16) if airfoil_config.probe_density == "fine" else (6, 8)
        H, W = self._probe_grid

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3, H, W), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-Q_MAX, high=Q_MAX, shape=(3,), dtype=np.float32
        )

        chk = Path(checkpoint_path)
        C_D_base = np.load(chk / "C_D.npy")
        C_L_base = np.load(chk / "C_L.npy")

        half = len(C_D_base) // 2
        b_base = C_L_base[half:] / np.where(np.abs(C_D_base[half:]) > 1e-8,
                                             C_D_base[half:], 1e-8)
        self._r0 = float(np.mean(b_base))
        print(f"Baseline lift-to-drag r_0 = {self._r0:.4f}")

        self._cfd = CFDEnvSolver(
            T=T_total, dt=dt, mu=mu, rho=rho, mesh_filename=mesh_filename
        )
        self._cfd.boundary_conditions_env(Re=Re, U_m=U_m)
        self._cfd.initialize()

        self._Q_smooth   = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self.episode_steps = int(T_total * dt / N_ACTION)  # e.g. 9*1600/25 = 576

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cfd.load_checkpoint(self.checkpoint_path)
        self._Q_smooth   = np.zeros(3, dtype=np.float32)
        self._step_count = 0

        obs = self._cfd.sample_probes(self.airfoil_config.probes, self._probe_grid)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -Q_MAX, Q_MAX).astype(np.float32)

        self._Q_smooth = self._Q_smooth + ALPHA_SMOOTH * (action - self._Q_smooth)

        self._cfd.set_jet_velocities(self._Q_smooth, self.jet_normals)
        C_D_arr, C_L_arr = self._cfd.step_n(N_ACTION)

        reward = self._compute_reward(C_D_arr, C_L_arr)

        obs = self._cfd.sample_probes(self.airfoil_config.probes, self._probe_grid)
        self._step_count += 1
        done = self._step_count >= self.episode_steps

        info = {
            "C_D": float(np.mean(C_D_arr)),
            "C_L": float(np.mean(C_L_arr)),
            "r0":  self._r0,
            "t":   self._cfd.t,
        }
        return obs, reward, done, False, info

    def _compute_reward(self, C_D_arr: np.ndarray, C_L_arr: np.ndarray) -> float:
        b = C_L_arr / np.where(np.abs(C_D_arr) > 1e-8, C_D_arr, 1e-8)
        return float(np.mean(b)) - self._r0

    def close(self):
        pass
