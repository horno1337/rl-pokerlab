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

from poker_env.game import PokerGame, Action, GameState, ActionType, Street
from poker_env.agents.base import BaseAgent, build_observation, decode_action, OBS_DIM, N_ACTIONS

# ---------------------------------------------------------------------------
# GTO BTN RFI raise frequencies (6-max 100BB, condensed to rank pairs).
# Key: (high_rank, low_rank, suited) → raise frequency 0.0–1.0
# Rank encoding: 0=2, 12=A (matches card // 4)
# Hands NOT in this dict have GTO raise freq = 0.0 (pure fold).
# ---------------------------------------------------------------------------
def _build_gto_btn_rfi() -> dict:
    m: dict = {}
    # Pocket pairs
    for r, f in [(12,1.),(11,1.),(10,1.),(9,1.),(8,1.),(7,1.),(6,1.),(5,1.),
                 (4,1.),(3,.9),(2,.8),(1,.6),(0,.5)]:
        m[(r, r, False)] = f
    # Suited Ax
    for r, f in [(11,1.),(10,1.),(9,1.),(8,1.),(7,1.),(6,1.),(5,1.),(4,1.),
                 (3,1.),(2,1.),(1,1.),(0,.8)]:
        m[(12, r, True)] = f
    # Suited Kx
    for r, f in [(10,1.),(9,1.),(8,1.),(7,1.),(6,.9),(5,.6),(4,.6),(3,.5),
                 (2,.5),(1,.4),(0,.4)]:
        m[(11, r, True)] = f
    # Suited Qx
    for r, f in [(9,1.),(8,1.),(7,1.),(6,.7),(5,.5),(2,.4),(1,.4),(0,.4)]:
        m[(10, r, True)] = f
    # Suited Jx
    for r, f in [(8,1.),(7,1.),(6,.8),(5,.5),(1,.4),(0,.4)]:
        m[(9, r, True)] = f
    # Suited connectors / others
    for (h,l), f in [((8,7),1.),((7,6),1.),((6,5),1.),((5,4),1.),
                     ((8,6),1.),((7,5),.9),((6,4),.8),((5,3),.7),
                     ((7,4),.5),((6,3),.5),((8,5),.6)]:
        m[(h, l, True)] = f
    # Offsuit Ax
    for r, f in [(11,1.),(10,1.),(9,1.),(8,1.),(7,.8),(6,.5),(5,.4)]:
        m[(12, r, False)] = f
    # Offsuit Kx
    for r, f in [(10,1.),(9,1.),(8,1.),(7,.7)]:
        m[(11, r, False)] = f
    # Offsuit Qx/Jx/Tx
    for r, f in [(9,1.),(8,.8),(7,.5)]: m[(10, r, False)] = f
    for r, f in [(8,.8),(7,.5)]:        m[(9,  r, False)] = f
    for r, f in [((8,7),.7),((7,6),.5),((6,5),.4)]:
        m[(r[0], r[1], False)] = f
    return m

_GTO_BTN_RFI = _build_gto_btn_rfi()


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
        gto_exploit_bonus: float = 0.5,
        max_ev_scale: float = 1.0,
        kl_coef: float = 0.3,
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
        self.gto_exploit_bonus = gto_exploit_bonus
        self.max_ev_scale = max_ev_scale
        self.kl_coef = kl_coef
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
        self._gto_deviation_this_hand: bool = False   # hero raised a hand GTO says fold
        self._gto_compliant_raise_this_hand: bool = False  # hero raised a hand GTO says raise

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
        # Drain any hands that ended before the hero got a turn
        while self._pending_reward is not None:
            self._pending_reward = None
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

        # Immediate KL-alignment reward at BTN RFI decision points.
        # Fires before the action so it uses the pre-action game state.
        kl_reward = self._compute_kl_reward(action, self.game.state)

        # Check for GTO deviation before applying the action
        self._check_gto_deviation(action, self.game.state)

        hero_action = decode_action(action, self.game.state)
        last_info = self.game.step(hero_action)

        reward = kl_reward
        terminated = False

        # --- Path A: hero's action ended the hand ---
        if last_info["hand_done"]:
            reward = self._hand_reward(last_info)
            terminated = self._check_episode_done()
            obs = self._get_obs() if terminated else self._advance_to_hero()
            terminated = self._check_episode_done()
            return obs, reward, terminated, False, self._build_info(last_info)

        # --- Path B: hand still going, run opponents ---
        obs = self._advance_opponents(last_info)

        # _advance_opponents sets self._pending_reward if a hand ended mid-run
        if self._pending_reward is not None:
            reward = self._pending_reward
            self._pending_reward = None
            terminated = self._check_episode_done()
            if not terminated:
                obs = self._advance_to_hero()
                terminated = self._check_episode_done()

        return obs, reward, terminated, False, self._build_info(last_info)

    def _compute_kl_reward(self, action: int, state: GameState) -> float:
        """
        Immediate per-step reward that nudges the policy toward GTO at BTN RFI spots.

        Behavioral KL approximation (no access to action probabilities needed):
          alignment = gto_freq       if hero raises  (high gto_freq = good raise)
          alignment = 1 - gto_freq   if hero folds   (high gto_freq = bad fold)
          kl_reward = kl_coef × (alignment - 0.5) × 2   → range [-kl_coef, +kl_coef]

        Examples with kl_coef=0.3:
          Raise AA  (gto=1.0)  → +0.30 BB  (aligned)
          Fold  AA  (gto=1.0)  → -0.30 BB  (misaligned)
          Raise 72o (gto=0.0)  → -0.30 BB  (deviation — penalised but GTO-win bonus can offset)
          Fold  72o (gto=0.0)  → +0.30 BB  (aligned)
          Raise 87s (gto=0.5)  →  0.00 BB  (mixed — neutral)
        """
        if self.kl_coef == 0:
            return 0.0
        if state.street != Street.PREFLOP:
            return 0.0
        if state.acting_seat != self.hero_seat:
            return 0.0
        if state.acting_seat != state.button_seat:
            return 0.0
        if state.current_bet != state.bb_amount:
            return 0.0

        hero = state.players[self.hero_seat]
        cards = hero.hole_cards
        r1, r2 = cards[0] // 4, cards[1] // 4
        s1, s2 = cards[0] % 4,  cards[1] % 4
        if r1 < r2:
            r1, r2 = r2, r1
            s1, s2 = s2, s1
        suited = (s1 == s2) and (r1 != r2)

        gto_freq = _GTO_BTN_RFI.get((r1, r2, suited), 0.0)
        raised = action in (2, 3, 4)
        alignment = gto_freq if raised else (1.0 - gto_freq)
        return self.kl_coef * (alignment - 0.5) * 2.0

    def _check_gto_deviation(self, action: int, state: GameState) -> None:
        """
        Flag this hand if the hero makes a profitable-deviation candidate:
        raises/all-ins from the button preflop when GTO says to fold (freq=0).

        Only fires once per hand (the BTN open decision).
        """
        if self._gto_deviation_this_hand:
            return  # already flagged
        if state.street != Street.PREFLOP:
            return
        if state.acting_seat != self.hero_seat:
            return
        if state.acting_seat != state.button_seat:
            return
        if state.current_bet != state.bb_amount:
            return  # not a clean RFI spot (someone already raised)

        # Only care about raises (action 2/3/4); fold/call = not a deviation
        if action not in (2, 3, 4):
            return

        hero = state.players[self.hero_seat]
        cards = hero.hole_cards
        r1, r2 = cards[0] // 4, cards[1] // 4
        s1, s2 = cards[0] % 4,  cards[1] % 4
        if r1 < r2:
            r1, r2 = r2, r1
            s1, s2 = s2, s1
        suited = (s1 == s2) and (r1 != r2)

        gto_freq = _GTO_BTN_RFI.get((r1, r2, suited), 0.0)
        if gto_freq == 0.0:
            self._gto_deviation_this_hand = True
        elif gto_freq >= 0.8:
            # Only flag as compliant for hands GTO raises frequently (≥80%),
            # so mixed-strategy hands don't dilute the signal.
            self._gto_compliant_raise_this_hand = True

    def _hand_reward(self, info: Dict) -> float:
        """
        Extract hero's per-hand reward, normalised by BB.

        Clipped to [-MAX_REWARD_BB, +MAX_REWARD_BB] to keep PPO's value
        function stable. Default cap is 500BB — covers any realistic pot
        at 100BB starting stacks with rebuys, but prevents runaway gradient
        variance from giant deep-stack pots.

        If the hero deviated from GTO (raised a hand GTO says fold) and WON,
        a bonus multiplier is applied to reward successful exploitation.
        Losses on deviations are NOT penalised extra — the chip loss is enough.
        """
        rewards = info["rewards"]
        reward = rewards[self.hero_seat] / self.bb

        if reward > 0:
            # Max-EV pot-size bonus: winning a larger pot gives a superlinear reward.
            # Encourages building big pots with strong hands (AA, sets, etc.).
            # Scale = pot / 100BB so a 100BB pot → 2x, a 200BB pot → 3x, etc.
            # max_ev_scale=0 disables this (default off until mixed league converges).
            if self.max_ev_scale > 0:
                pot_bb = info.get("pot", self.bb * 2) / self.bb
                pot_multiplier = 1.0 + (pot_bb / 100.0) * self.max_ev_scale
                reward *= pot_multiplier

            # GTO-deviation bonus: extra reward for successfully exploiting
            # a spot GTO says fold (e.g. opening 87o BTN and winning).
            if self._gto_deviation_this_hand and self.gto_exploit_bonus > 0:
                reward *= (1.0 + self.gto_exploit_bonus)

        elif reward < 0 and self._gto_compliant_raise_this_hand:
            # GTO-compliant loss cushion: when the hero raised a hand GTO says
            # raise (≥80% freq) but lost due to variance, reduce the penalty.
            # Prevents PPO from learning "don't raise AKs because I lost once" —
            # the process was correct, the outcome was unlucky.
            # Cushion = 20% reduction in loss magnitude (reward × 0.8).
            reward *= 0.8

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

    def _advance_to_hero(self) -> np.ndarray:
        """
        Start new hands until the hero needs to act (or the episode ends).

        Occasionally all opponents fold before the hero gets a turn.
        _hand_reward() is already called inside _advance_opponents, so
        _hands_played and _episode_reward are already updated — we just
        discard _pending_reward and start the next hand.
        """
        obs, _ = self._start_hand_and_advance()
        while self._pending_reward is not None and not self._check_episode_done():
            self._pending_reward = None
            obs, _ = self._start_hand_and_advance()
        if self._pending_reward is not None:
            self._pending_reward = None
        return obs

    def _start_hand_and_advance(self) -> Tuple[np.ndarray, Dict]:
        """Start a new hand, run opponents until hero must act."""
        self.game.start_hand()
        self._gto_deviation_this_hand = False
        self._gto_compliant_raise_this_hand = False
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
