"""
PPOAgent — wraps a trained stable-baselines3 PPO model as a BaseAgent.

Use this to plug a trained model into MultiAgentRunner for evaluation:

    from poker_env.agents.ppo_agent import PPOAgent
    agent = PPOAgent(seat=0, model_path="models/ppo_poker_final")
    runner = MultiAgentRunner([agent, RandomAgent(1), ...])
    results = runner.run(1000)
"""

from __future__ import annotations
from typing import Dict, Any

from poker_env.game import GameState, Action
from poker_env.agents.base import BaseAgent, build_observation, decode_action


class PPOAgent(BaseAgent):
    """
    Wraps a stable-baselines3 PPO model for use in MultiAgentRunner.

    The model is loaded once at construction and reused for every act() call.
    Observation building and action decoding use the same functions as PokerEnv
    so behaviour matches exactly what the model was trained on.
    """

    def __init__(self, seat: int, model_path: str, deterministic: bool = True):
        super().__init__(seat)
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)
        self.deterministic = deterministic

    def act(self, state: GameState) -> Action:
        obs = build_observation(state, self.seat)
        action_idx, _ = self.model.predict(obs, deterministic=self.deterministic)
        return decode_action(int(action_idx), state)
