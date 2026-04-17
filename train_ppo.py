"""
PPO training script for rl-pokerlab.

Three training modes, each addressing a known weakness:

  heuristic  (default)  — hero vs Random + Heuristic opponents
                          Good baseline; agent learns basic hand strength.

  threebet              — hero vs Heuristic + ThreeBetAgent opponents
                          ThreeBetAgent 3-bets wide vs BTN opens, punishing
                          the 85% open range leak identified after heuristic training.

  selfplay              — hero vs frozen copies of itself (loaded from ModelPool)
                          Opponents adapt to whatever strategy the hero develops,
                          closing the exploit loop that fixed agents leave open.
                          Requires --base-model to seed the pool.

  league                — generational self-play. Each generation trains against
                          the full pool of all prior generations. Older versions
                          are never evicted, preventing the agent from forgetting
                          how to beat weaker play and avoiding strategy cycling.

# ---------------------------------------------------------------------------
# Recommended training sequence
# ---------------------------------------------------------------------------
#
# Step 1 — baseline (vs heuristic opponents)
#   python train_ppo.py --mode heuristic --timesteps 1_000_000
#   → models/ppo_poker_final.zip
#
# Step 2 — fix BTN over-raising leak (fine-tune from Step 1)
#   python train_ppo.py --mode threebet --timesteps 500_000 \
#       --base-model models/ppo_poker_final
#   → models/ppo_threebet_final.zip
#
# Step 3 — generational self-play (5 gens × 300k steps, fine-tune from Step 2)
#   python train_ppo.py --mode league --generations 5 --timesteps 300_000 \
#       --base-model models/ppo_threebet_final
#   → models/league/gen_1.zip ... gen_5.zip
#   → models/ppo_league_final.zip  (alias for the last generation)
#
#   Pool composition per generation:
#     Gen 1: [threebet_final]
#     Gen 2: [threebet_final, gen_1]
#     Gen 3: [threebet_final, gen_1, gen_2]
#     ...
#
# Step 4 — evaluate the final model
#   python train_ppo.py --eval-only --model models/ppo_league_final
# ---------------------------------------------------------------------------

Usage:
    python train_ppo.py                                        # heuristic, 500k steps
    python train_ppo.py --mode threebet --timesteps 500_000   # fix BTN leak
    python train_ppo.py --mode selfplay --base-model models/ppo_poker_final
    python train_ppo.py --mode league --generations 5 --timesteps 300_000 \
        --base-model models/ppo_threebet_final
    python train_ppo.py --eval-only --model models/ppo_league_final
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from poker_env.env import PokerEnv, MultiAgentRunner
from poker_env.agents.baselines import RandomAgent, HeuristicAgent, CallAgent, ThreeBetAgent
from poker_env.agents.ppo_agent import PPOAgent
from poker_env.agents.selfplay import ModelPool, SelfPlayAgent


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(seed: int | None = None) -> PokerEnv:
    """Hero vs 2 Random + 3 Heuristic."""
    opponents = [
        RandomAgent(seat=1, seed=None if seed is None else seed + 1),
        RandomAgent(seat=2, seed=None if seed is None else seed + 2),
        HeuristicAgent(seat=3, aggression=0.4, seed=None if seed is None else seed + 3),
        HeuristicAgent(seat=4, aggression=0.6, seed=None if seed is None else seed + 4),
        HeuristicAgent(seat=5, aggression=0.8, seed=None if seed is None else seed + 5),
    ]
    return _make_poker_env(opponents, seed)


def make_threebet_env(seed: int | None = None) -> PokerEnv:
    """
    Hero vs 2 Heuristic + 3 ThreeBetAgent.
    ThreeBetAgents 3-bet wide vs BTN opens — forces the PPO agent to learn
    that opening 85%+ of hands is immediately punished.
    """
    opponents = [
        HeuristicAgent(seat=1, aggression=0.5, seed=None if seed is None else seed + 1),
        HeuristicAgent(seat=2, aggression=0.7, seed=None if seed is None else seed + 2),
        ThreeBetAgent(seat=3, bluff_freq=0.4, seed=None if seed is None else seed + 3),
        ThreeBetAgent(seat=4, bluff_freq=0.6, seed=None if seed is None else seed + 4),
        ThreeBetAgent(seat=5, bluff_freq=0.5, seed=None if seed is None else seed + 5),
    ]
    return _make_poker_env(opponents, seed)


def make_selfplay_env(pool: ModelPool, seed: int | None = None) -> PokerEnv:
    """Hero vs 5 SelfPlayAgents that sample from the shared ModelPool."""
    opponents = [SelfPlayAgent(seat=i, pool=pool) for i in range(1, 6)]
    return _make_poker_env(opponents, seed)


def _make_poker_env(opponents, seed):
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
# Self-play callback
# ---------------------------------------------------------------------------

class SelfPlayCallback(BaseCallback):
    """
    Periodically snapshots the current model and adds it to the shared pool.

    Because SelfPlayAgent.reset() samples from the pool at the start of each
    hand, new snapshots propagate to all opponent seats without restarting envs.
    """

    def __init__(self, pool: ModelPool, snapshot_freq: int = 100_000, verbose: int = 1):
        super().__init__(verbose)
        self.pool = pool
        self.snapshot_freq = snapshot_freq
        os.makedirs("checkpoints/selfplay", exist_ok=True)

    def _on_step(self) -> bool:
        # Fire roughly every snapshot_freq steps (n_envs steps per call)
        if self.num_timesteps % self.snapshot_freq < self.training_env.num_envs:
            path = f"checkpoints/selfplay/snap_{self.num_timesteps}"
            self.model.save(path)
            self.pool.add(path + ".zip")
            if self.verbose:
                print(f"[SelfPlay] snapshot → {path}.zip  (pool size: {len(self.pool)})")
        return True


# ---------------------------------------------------------------------------
# Shared model setup helper
# ---------------------------------------------------------------------------

def _build_model(env, base_model_path: str | None = None) -> PPO:
    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
        verbose=1,
        tensorboard_log=None,  # disabled: bytes/str bug on Python 3.14
    )
    if base_model_path and os.path.exists(base_model_path + ".zip"):
        print(f"Fine-tuning from {base_model_path}")
        return PPO.load(base_model_path, env=env, **{k: v for k, v in kwargs.items()
                                                      if k not in ("policy", "env")})
    return PPO(**kwargs)


def _make_callbacks(n_envs: int, save_path: str, eval_env) -> list:
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    freq = max(50_000 // n_envs, 1)
    return [
        CheckpointCallback(save_freq=freq, save_path="checkpoints/",
                           name_prefix=os.path.basename(save_path), verbose=0),
        EvalCallback(eval_env, best_model_save_path="models/", log_path="logs/",
                     eval_freq=freq, n_eval_episodes=200, deterministic=True, verbose=1),
    ]


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train(total_timesteps: int, n_envs: int = 4, base_model: str | None = None) -> PPO:
    env = make_vec_env(lambda: make_env(), n_envs=n_envs)
    model = _build_model(env, base_model)
    callbacks = _make_callbacks(n_envs, "models/ppo_poker_final", make_env(seed=42))
    print(f"[heuristic] Training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save("models/ppo_poker_final")
    print("Saved: models/ppo_poker_final.zip")
    return model


def train_threebet(total_timesteps: int, n_envs: int = 4, base_model: str | None = None) -> PPO:
    env = make_vec_env(lambda: make_threebet_env(), n_envs=n_envs)
    model = _build_model(env, base_model)
    callbacks = _make_callbacks(n_envs, "models/ppo_threebet_final", make_threebet_env(seed=42))
    print(f"[threebet] Training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save("models/ppo_threebet_final")
    print("Saved: models/ppo_threebet_final.zip")
    return model


def train_selfplay(total_timesteps: int, n_envs: int = 4, base_model: str | None = None) -> PPO:
    pool = ModelPool(max_size=5)

    # Seed the pool so opponents don't start with pure random play
    if base_model and os.path.exists(base_model + ".zip"):
        pool.add(base_model + ".zip")
        print(f"[selfplay] Pool seeded with {base_model}")
    else:
        print("[selfplay] WARNING: no base model found — opponents start random until first snapshot")

    env = make_vec_env(lambda: make_selfplay_env(pool), n_envs=n_envs)
    model = _build_model(env, base_model)

    selfplay_cb = SelfPlayCallback(pool, snapshot_freq=100_000, verbose=1)
    callbacks = _make_callbacks(n_envs, "models/ppo_selfplay_final", make_selfplay_env(pool, seed=42))

    print(f"[selfplay] Training for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, callback=[selfplay_cb] + callbacks)
    model.save("models/ppo_selfplay_final")
    print("Saved: models/ppo_selfplay_final.zip")
    return model


def train_league(
    n_generations: int = 5,
    steps_per_gen: int = 300_000,
    n_envs: int = 4,
    base_model: str | None = None,
) -> PPO:
    """
    Generational self-play training loop.

    Each generation trains against the full pool of all prior generations.
    Older versions are never evicted, so the agent can't specialise against
    only the latest opponent and forget how to beat weaker play.

    Pool composition per generation:
      Gen 1:  [base_model]             (5x copies of v1)
      Gen 2:  [v1, v2]                 (random mix)
      Gen 3:  [v1, v2, v3]
      ...

    The pool is frozen during each generation's training (no mid-run snapshots),
    which is the "naive self-play" variant.  Mixing historical versions gives the
    stability benefit of league play without the complexity of dynamic scheduling.
    """
    os.makedirs("models/league", exist_ok=True)

    # Keep every generation — unlimited pool size for the league loop
    pool = ModelPool(max_size=n_generations + 1)

    if base_model and os.path.exists(base_model + ".zip"):
        pool.add(base_model + ".zip")
        print(f"[league] Pool seeded with {base_model}")
    else:
        print("[league] No base model — Gen 1 opponents start random")

    current_path: str | None = base_model
    model: PPO | None = None

    for gen in range(1, n_generations + 1):
        gen_path = f"models/league/gen_{gen}"
        print(f"\n{'='*55}")
        print(f"Generation {gen}/{n_generations}  |  pool size: {len(pool)}")
        print(f"Pool contents: {pool._paths}")
        print("="*55)

        env = make_vec_env(lambda: make_selfplay_env(pool), n_envs=n_envs)

        # Fine-tune from the previous generation's weights, not from scratch
        model = _build_model(env, current_path)

        eval_env = make_selfplay_env(pool, seed=42)
        callbacks = _make_callbacks(n_envs, gen_path, eval_env)

        model.learn(total_timesteps=steps_per_gen, callback=callbacks)
        model.save(gen_path)
        print(f"Gen {gen} saved → {gen_path}.zip")

        pool.add(gen_path + ".zip")
        current_path = gen_path

    # Also save as a convenient "latest" alias
    if model is not None:
        model.save("models/ppo_league_final")
        print("\nSaved final: models/ppo_league_final.zip")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path: str, n_hands: int = 2000):
    print(f"\n{'='*55}")
    print(f"Evaluating {model_path} over {n_hands} hands")
    print("="*55)

    for label, opponents in [
        ("vs 5x Random",    [RandomAgent(i) for i in range(1, 6)]),
        ("vs 5x Call",      [CallAgent(i)   for i in range(1, 6)]),
        ("vs 5x Heuristic", [HeuristicAgent(i, aggression=0.5) for i in range(1, 6)]),
        ("vs 5x ThreeBet",  [ThreeBetAgent(i, bluff_freq=0.5)  for i in range(1, 6)]),
        ("vs Mixed",        [
            RandomAgent(1), CallAgent(2),
            HeuristicAgent(3, 0.4), ThreeBetAgent(4, 0.5), HeuristicAgent(5, 0.8),
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
    parser.add_argument("--mode",        choices=["heuristic", "threebet", "selfplay", "league"],
                        default="heuristic")
    parser.add_argument("--timesteps",   type=int, default=500_000,
                        help="Total steps (heuristic/threebet/selfplay) or steps-per-gen (league)")
    parser.add_argument("--n-envs",      type=int, default=4)
    parser.add_argument("--base-model",  type=str, default=None,
                        help="Path (without .zip) to fine-tune or seed the pool from")
    parser.add_argument("--generations", type=int, default=5,
                        help="Number of league generations (league mode only)")
    parser.add_argument("--eval-only",   action="store_true")
    parser.add_argument("--model",       type=str, default=None,
                        help="Model to evaluate (default: mode-specific output model)")
    args = parser.parse_args()

    output_models = {
        "heuristic": "models/ppo_poker_final",
        "threebet":  "models/ppo_threebet_final",
        "selfplay":  "models/ppo_selfplay_final",
        "league":    "models/ppo_league_final",
    }
    eval_model = args.model or output_models[args.mode]

    if not args.eval_only:
        if args.mode == "heuristic":
            train(args.timesteps, args.n_envs, args.base_model)
        elif args.mode == "threebet":
            train_threebet(args.timesteps, args.n_envs, args.base_model)
        elif args.mode == "selfplay":
            train_selfplay(args.timesteps, args.n_envs, args.base_model)
        elif args.mode == "league":
            train_league(args.generations, args.timesteps, args.n_envs, args.base_model)

    if os.path.exists(eval_model + ".zip") or os.path.exists(eval_model):
        evaluate(model_path=eval_model)
    else:
        print(f"No model at {eval_model} — run without --eval-only first.")
