"""
PPO training script for rl-pokerlab.

Trains a PPO agent in PokerEnv against a mix of Random and Heuristic opponents,
then evaluates the result in MultiAgentRunner against all baseline agent types.

Usage:
    python train_ppo.py                    # train from scratch
    python train_ppo.py --eval-only        # skip training, load existing model
    python train_ppo.py --timesteps 1_000_000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from poker_env.env import PokerEnv, MultiAgentRunner
from poker_env.agents.baselines import RandomAgent, HeuristicAgent, CallAgent
from poker_env.agents.ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(seed: int | None = None):
    """
    6-player PokerEnv: hero at seat 0 vs 2 Random + 3 Heuristic opponents.
    Heuristic agents are the stronger baseline so including more of them
    pushes the PPO agent to learn real hand-strength reasoning.
    """
    opponents = [
        RandomAgent(seat=1, seed=None if seed is None else seed + 1),
        RandomAgent(seat=2, seed=None if seed is None else seed + 2),
        HeuristicAgent(seat=3, aggression=0.4, seed=None if seed is None else seed + 3),
        HeuristicAgent(seat=4, aggression=0.6, seed=None if seed is None else seed + 4),
        HeuristicAgent(seat=5, aggression=0.8, seed=None if seed is None else seed + 5),
    ]
    return PokerEnv(
        opponents=opponents,
        hero_seat=0,
        n_players=6,
        starting_stack=1000,
        sb=5,
        bb=10,
        hands_per_episode=200,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(total_timesteps: int, n_envs: int = 4) -> PPO:
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    env = make_vec_env(lambda: make_env(), n_envs=n_envs)
    eval_env = make_env(seed=42)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path="checkpoints/",
        name_prefix="ppo_poker",
        verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # encourage exploration of different bet sizes
        device="cpu",           # MlpPolicy trains faster on CPU than GPU
        verbose=1,
        tensorboard_log="logs/tensorboard/",
    )

    print(f"Training PPO for {total_timesteps:,} timesteps across {n_envs} envs...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    model.save("models/ppo_poker_final")
    print("Saved: models/ppo_poker_final.zip")
    return model


# ---------------------------------------------------------------------------
# Evaluation in MultiAgentRunner
# ---------------------------------------------------------------------------

def evaluate(model_path: str = "models/ppo_poker_final", n_hands: int = 2000):
    """
    Run the trained agent in MultiAgentRunner against all three baselines.
    PPO agent occupies seat 0; baselines fill seats 1-5.
    """
    print(f"\n{'='*55}")
    print(f"Evaluating {model_path} over {n_hands} hands")
    print("="*55)

    for label, opponents in [
        ("vs 5x Random",    [RandomAgent(i) for i in range(1, 6)]),
        ("vs 5x Call",      [CallAgent(i)   for i in range(1, 6)]),
        ("vs 5x Heuristic", [HeuristicAgent(i, aggression=0.5) for i in range(1, 6)]),
        ("vs Mixed",        [
            RandomAgent(1), CallAgent(2),
            HeuristicAgent(3, 0.4), HeuristicAgent(4, 0.6), HeuristicAgent(5, 0.8),
        ]),
    ]:
        agents = [PPOAgent(seat=0, model_path=model_path)] + opponents
        runner = MultiAgentRunner(agents, seed=0)
        result = runner.run(n_hands=n_hands)

        ppo_bb100 = result.bb_per_100()[0]
        print(f"\n{label}")
        print(result.summary())
        print(f"  --> PPO BB/100: {ppo_bb100:+.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model",     type=str, default="models/ppo_poker_final")
    args = parser.parse_args()

    if not args.eval_only:
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)

    if os.path.exists(args.model + ".zip") or os.path.exists(args.model):
        evaluate(model_path=args.model)
    else:
        print(f"No model found at {args.model} — run without --eval-only first.")
