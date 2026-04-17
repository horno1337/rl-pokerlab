"""
Self-play infrastructure.

  ModelPool      — shared pool of saved model snapshots
  SelfPlayAgent  — opponent that samples a model from the pool each hand

The pool is a plain Python object shared by reference across all env instances
inside a DummyVecEnv, so snapshots added by SelfPlayCallback become visible to
every opponent without restarting the envs.
"""

from __future__ import annotations
import random

from poker_env.game import GameState, Action
from poker_env.agents.base import BaseAgent, build_observation, decode_action


class ModelPool:
    """
    Holds paths to saved PPO model snapshots.

    Thread-safe enough for DummyVecEnv (single-process). For SubprocVecEnv
    you would need a multiprocessing.Manager list instead.
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._paths: list[str] = []
        self._rng = random.Random()

    def add(self, path: str):
        """Add a snapshot path. Evicts oldest if pool is full."""
        self._paths.append(path)
        if len(self._paths) > self.max_size:
            self._paths.pop(0)

    def sample(self) -> str | None:
        """Return a uniformly random path, or None if pool is empty."""
        if not self._paths:
            return None
        return self._rng.choice(self._paths)

    def latest(self) -> str | None:
        return self._paths[-1] if self._paths else None

    def __len__(self) -> int:
        return len(self._paths)


class SelfPlayAgent(BaseAgent):
    """
    Opponent that re-samples its model from a shared ModelPool at the start
    of each hand.  Falls back to random play until the pool has at least one
    snapshot.

    Calling reset() (which PokerEnv does at the start of every hand) is
    enough to trigger a model swap — no VecEnv hacking required.

    Models are cached in a class-level dict so all SelfPlayAgent instances
    across all envs share loaded models — eliminates repeated disk I/O when
    the pool has multiple entries being sampled round-robin.
    """

    _model_cache: dict[str, object] = {}  # shared across all instances

    def __init__(self, seat: int, pool: ModelPool, deterministic: bool = False):
        super().__init__(seat)
        self.pool = pool
        self.deterministic = deterministic
        self._model = None
        self._loaded_path: str | None = None

    def reset(self):
        path = self.pool.sample()
        if path and path != self._loaded_path:
            if path not in SelfPlayAgent._model_cache:
                from stable_baselines3 import PPO
                SelfPlayAgent._model_cache[path] = PPO.load(path)
            self._model = SelfPlayAgent._model_cache[path]
            self._loaded_path = path

    def act(self, state: GameState) -> Action:
        if self._model is None:
            # Pool is empty — fall back to random
            hero = state.players[self.seat]
            all_in_to = hero.bet_street + hero.stack
            choices = [0, 1]
            if all_in_to > state.current_bet:
                choices += [2, 3, 4]
            return decode_action(random.choice(choices), state)

        obs = build_observation(state, self.seat)
        action_idx, _ = self._model.predict(obs, deterministic=self.deterministic)
        return decode_action(int(action_idx), state)
