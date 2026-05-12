#!/usr/bin/env python3
"""
Headless training script, runs the AFC-DRL pipeline without Jupyter.
Designed for SLURM/HPC batch jobs. Mirrors the logic in notebooks/main.ipynb.

Usage:
    python scripts/train_headless.py --n_episodes 500
    mpirun -n 8 python scripts/train_headless.py --n_episodes 500

Arguments:
    --airfoil      Path to airfoil .dat file           (default: airfoils/naca001234.dat)
    --alpha        Angle of attack in degrees           (default: 25)
    --checkpoint   Path to uncontrolled CFD checkpoint  (default: checkpoints/naca0012_alpha25)
    --save_path    Where to save the trained policy     (default: results/training/learned_policy)
    --tb_log       TensorBoard log directory            (default: results/tensorboard)
    --n_episodes   Number of training episodes          (default: 500)
    --run_cfd      Re-run uncontrolled CFD even if checkpoint already exists (flag)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration.AirfoilConfig import AirfoilConfig
from src.cfd.CFDParameters import CFDParameters
from src.env.AFCGYMenv import AFCGymEnv
from src.rl.RecPPOAgent import RecPPOAgent


def parse_args():
    p = argparse.ArgumentParser(description="AFC-DRL headless training")
    p.add_argument("--airfoil",    default="airfoils/naca001234.dat")
    p.add_argument("--alpha",      type=int,   default=25)
    p.add_argument("--checkpoint", default="checkpoints/naca0012_alpha25")
    p.add_argument("--save_path",  default="results/training/learned_policy")
    p.add_argument("--tb_log",     default="results/tensorboard")
    p.add_argument("--n_episodes", type=int,   default=500)
    p.add_argument("--run_cfd",    action="store_true",
                   help="Re-run uncontrolled CFD even if checkpoint exists")
    return p.parse_args()


def main():
    args = parse_args()

    # Airfoil and mesh
    print(f"[1/4] Configuring airfoil: {args.airfoil}, alpha={args.alpha}°")
    airfoil = AirfoilConfig(
        alpha=args.alpha,
        NACA_file=args.airfoil,
        jet_locs=[0.2, 0.4, 0.6],
        probe_density='fine',
    )

    # Uncontrolled CFD checkpoint (initial conditions for RL episodes)
    ckpt = Path(args.checkpoint)
    if args.run_cfd or not (ckpt / "u.npy").exists():
        print("[2/4] Running uncontrolled CFD to build checkpoint ...")
        cfd = CFDParameters(
            T=9, dt=1600, mu=0.00012, rho=1.0,
            mesh_filename=airfoil.msh_filename,
        )
        cfd.boundary_conditions(Re=2500, U_m=0.45)
        cfd.IPCS(output_dir='initial_cfd_results', save_interval=20,
                 t_ramp=2.0, name='uncontrolled')
        cfd.save_checkpoint(str(ckpt))
    else:
        print(f"[2/4] Checkpoint found at '{ckpt}' — skipping uncontrolled CFD.")

    # RL environment and agent
    print("[3/4] Initialising Gymnasium environment and RecurrentPPO agent ...")
    env = AFCGymEnv(
        airfoil_config=airfoil,
        checkpoint_path=str(ckpt),
        mesh_filename=airfoil.msh_filename,
        T_total=9.0, dt=1600, mu=0.00012, rho=1.0, Re=2500, U_m=0.45,
    )

    agent = RecPPOAgent(
        env=env,
        use_cnn=False,
        learning_rate=5e-4,
        n_steps=env.episode_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        tensorboard_log=os.path.abspath(args.tb_log),
        verbose=1,
    )

    # Training and save policy 
    total_steps = env.episode_steps * args.n_episodes
    print(f"[4/4] Training for {args.n_episodes} episodes ({total_steps} total timesteps) ...")
    agent.train(
        total_timesteps=total_steps,
        save_path=args.save_path,
        tb_log_name="AFC_DRL",
        print_every=50,
    )
    print(f"Done. Policy saved to '{args.save_path}.zip'")


if __name__ == "__main__":
    main()
