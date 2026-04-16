"""
Gymnasium-compatible wrapper around PokerGame.

Supports two modes:
  single_agent  — one learning agent vs N-1 fixed opponents
                  standard gym interface (obs, reward, done, info)
  multi_agent   — all agents are external; env just orchestrates
                  use run_hand() instead of step()

For RL training, use single_agent mode with stable-baselines3.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from poker_env.game import PokerGame, Action, GameState
from poker_env.agents.base import BaseAgent, build_observation, decode_action, OBS_DIM, N_ACTIONS


class PokerEnv(gym.Env):
    """
    Single-agent Gymnasium environment.

    The learning agent occupies one seat; all other seats are
    controlled by opponent agents passed at construction time.

    Observation: 134-dim float32 vector (see agents/base.py)
    Action:      Discrete(5) — fold, check/call, raise-min, raise-pot, all-in
    Reward:      Net chips won/lost this hand, normalized by BB
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        opponents: List[BaseAgent],
        hero_seat: int = 0,
        n_players: int = 6,
        starting_stack: int = 1000,
        sb: int = 5,
        bb: int = 10,
        hands_per_episode: int = 200,
        max_reward_bb: float = 500.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert len(opponents) == n_players - 1, \
            f"Need {n_players - 1} opponents for a {n_players}-player game"

        self.hero_seat = hero_seat
        self.n_players = n_players
        self.bb = bb
        self.hands_per_episode = hands_per_episode
        self.max_reward_bb = max_reward_bb
        self.render_mode = render_mode

        self.game = PokerGame(
            n_players=n_players,
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            seed=seed,
        )

        # Assign opponents to seats (skipping hero_seat)
        self.agents: List[Optional[BaseAgent]] = [None] * n_players
        opp_iter = iter(opponents)
        for seat in range(n_players):
            if seat != hero_seat:
                self.agents[seat] = next(opp_iter)
                self.agents[seat].seat = seat

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._hands_played: int = 0
        self._episode_reward: float = 0.0
        self._pending_reward: Optional[float] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset_session()
        self._hands_played = 0
        self._episode_reward = 0.0
        self._pending_reward = None
        for agent in self.agents:
            if agent is not None:
                agent.reset()
        obs, info = self._start_hand_and_advance()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply hero's action, then run opponents until hero must act again.

        Reward is emitted after every hand completion (not just episode end).
        This gives PPO a tight feedback loop — one reward signal per hand.

        Reward = net chips won/lost this hand, normalised by BB.
        Zero during hands in progress.
        """
        assert not self.game.hand_done, "Call reset() before step()"

        hero_action = decode_action(action, self.game.state)
        last_info = self.game.step(hero_action)

        reward = 0.0
        terminated = False

        # --- Path A: hero's action ended the hand ---
        if last_info["hand_done"]:
            reward = self._hand_reward(last_info)
            terminated = self._check_episode_done()
            obs = self._get_obs() if terminated else self._start_hand_and_advance()[0]
            return obs, reward, terminated, False, self._build_info(last_info)

        # --- Path B: hand still going, run opponents ---
        obs = self._advance_opponents(last_info)

        # _advance_opponents sets self._pending_reward if a hand ended mid-run
        if self._pending_reward is not None:
            reward = self._pending_reward
            self._pending_reward = None
            terminated = self._check_episode_done()
            if not terminated:
                obs = self._start_hand_and_advance()[0]

        return obs, reward, terminated, False, self._build_info(last_info)

    def _hand_reward(self, info: Dict) -> float:
        """
        Extract hero's per-hand reward, normalised by BB.

        Clipped to [-MAX_REWARD_BB, +MAX_REWARD_BB] to keep PPO's value
        function stable. Default cap is 500BB — covers any realistic pot
        at 100BB starting stacks with rebuys, but prevents runaway gradient
        variance from giant deep-stack pots.
        """
        rewards = info["rewards"]
        reward = rewards[self.hero_seat] / self.bb
        reward = float(np.clip(reward, -self.max_reward_bb, self.max_reward_bb))
        self._episode_reward += reward
        self._hands_played += 1
        return reward

    def _check_episode_done(self) -> bool:
        return self._hands_played >= self.hands_per_episode

    def render(self):
        if self.render_mode == "human":
            print(self.game)
        elif self.render_mode == "ansi":
            return str(self.game)

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_hand_and_advance(self) -> Tuple[np.ndarray, Dict]:
        """Start a new hand, run opponents until hero must act."""
        self.game.start_hand()
        obs = self._advance_opponents()
        return obs, {"hand_number": self.game.hand_number}

    def _advance_opponents(self, last_info: Dict | None = None) -> np.ndarray:
        """
        Step through opponent actions until it's the hero's turn or the hand ends.
        If the hand ends during this loop, captures the reward into _pending_reward.
        """
        while (
            not self.game.hand_done
            and self.game.state.acting_seat != self.hero_seat
        ):
            state = self.game.state
            seat = state.acting_seat
            agent = self.agents[seat]
            if agent is None:
                raise RuntimeError(f"No agent for seat {seat}")
            action = agent.act(state)
            info = self.game.step(action)
            agent.observe(state, 0.0, info["hand_done"], info)

            if info["hand_done"]:
                # Hand ended while running opponents — capture reward for step() to return
                self._pending_reward = self._hand_reward(info)
                break

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return build_observation(self.game.state, self.hero_seat)

    def _build_info(self, step_info: Dict) -> Dict:
        return {
            **step_info,
            "hands_played": self._hands_played,
            "episode_reward": self._episode_reward,
            "stacks": [p.stack for p in self.game.players],
        }


# ---------------------------------------------------------------------------
# Multi-agent runner (no Gymnasium, pure orchestration)
# ---------------------------------------------------------------------------

class MultiAgentRunner:
    """
    Run a poker session with all seats controlled by external agents.
    No Gymnasium interface — just a clean game loop.

    Usage:
        agents = [RandomAgent(i) for i in range(6)]
        runner = MultiAgentRunner(agents)
        results = runner.run(n_hands=1000)
        print(results.summary())
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        n_players: int = 6,
        starting_stack: int = 1000,
        sb: int = 5,
        bb: int = 10,
        seed: Optional[int] = None,
    ):
        assert len(agents) == n_players
        self.agents = agents
        for i, a in enumerate(agents):
            a.seat = i
        self.game = PokerGame(
            n_players=n_players,
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            seed=seed,
        )

    def run(self, n_hands: int = 1000, verbose: bool = False) -> "SessionResult":
        self.game.reset_session()
        hand_rewards: List[List[int]] = []

        for _ in range(n_hands):
            self.game.start_hand()
            for agent in self.agents:
                agent.reset()

            while not self.game.hand_done:
                state = self.game.state
                seat = state.acting_seat
                action = self.agents[seat].act(state)

                if verbose:
                    print(f"  Seat {seat}: {action}")

                info = self.game.step(action)

                if info["hand_done"]:
                    rewards = info["rewards"]
                    hand_rewards.append(rewards)
                    for i, agent in enumerate(self.agents):
                        agent.observe(state, rewards[i], True, info)
                    if verbose:
                        print(f"Hand #{self.game.hand_number} done. Rewards: {rewards}")
                        print(f"Stacks: {[p.stack for p in self.game.players]}")
                else:
                    for i, agent in enumerate(self.agents):
                        if i != seat:
                            agent.observe(state, 0.0, False, info)

        return SessionResult(
            n_hands=n_hands,
            n_players=self.game.n_players,
            hand_rewards=hand_rewards,
            final_stacks=[p.stack for p in self.game.players],
            session_rewards=list(self.game.session_rewards),
            bb=self.game.bb_amount,
        )


class SessionResult:
    def __init__(
        self,
        n_hands: int,
        n_players: int,
        hand_rewards: List[List[int]],
        final_stacks: List[int],
        session_rewards: List[int],
        bb: int,
    ):
        self.n_hands = n_hands
        self.n_players = n_players
        self.hand_rewards = hand_rewards
        self.final_stacks = final_stacks
        self.session_rewards = session_rewards
        self.bb = bb

    def bb_per_100(self) -> List[float]:
        """
        Net BB won per 100 hands — standard poker performance metric.
        Computed from per-hand P&L to avoid rebuy capital inflation.
        """
        totals = [0.0] * self.n_players
        for hand in self.hand_rewards:
            for i, r in enumerate(hand):
                totals[i] += r
        return [
            (totals[i] / self.bb) / self.n_hands * 100
            for i in range(self.n_players)
        ]

    def summary(self) -> str:
        lines = [
            f"Session: {self.n_hands} hands, {self.n_players} players",
            f"{'Seat':<6} {'Net chips':>10} {'BB/100':>8} {'Final stack':>12}",
            "-" * 40,
        ]
        for i in range(self.n_players):
            lines.append(
                f"{i:<6} {self.session_rewards[i]:>10} "
                f"{self.bb_per_100()[i]:>8.2f} "
                f"{self.final_stacks[i]:>12}"
            )
        return "\n".join(lines)
