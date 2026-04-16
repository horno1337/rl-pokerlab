"""
Base agent interface and observation builder.

All agents must implement act(). The observe() hook is optional
but needed for learning agents to receive rewards.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

from poker_env.game import GameState, Action, ActionType, Street


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

N_PLAYERS = 6
N_CARD_FEATURES = 52   # one-hot card encoding


def build_observation(state: GameState, hero_seat: int) -> np.ndarray:
    """
    Build a flat numpy observation vector for the agent at hero_seat.

    Layout:
      [0]       street (0-3, normalized)
      [1]       pot (normalized by BB*100)
      [2]       current_bet (normalized)
      [3]       hero stack (normalized)
      [4]       hero bet_street (normalized)
      [5]       position relative to button (0=button, normalized)
      [6:58]    hero hole cards (one-hot, 52 dims)
      [58:110]  community cards (one-hot, 52 dims)
      [110:116] opponent stacks (normalized, 0 if folded/self)
      [116:122] opponent bet_street (normalized)
      [122:128] opponent is_folded flags
      [128:134] opponent is_all_in flags
    Total: 134 dims
    """
    NORM = state.bb_amount * 100  # normalize by 100BB
    hero = state.players[hero_seat]

    obs = np.zeros(134, dtype=np.float32)

    obs[0] = state.street / 3.0
    obs[1] = state.pot / NORM
    obs[2] = state.current_bet / NORM
    obs[3] = hero.stack / NORM
    obs[4] = hero.bet_street / NORM

    # Relative position (0 = button, 1/(n-1) = one left of button, etc.)
    n = len(state.players)
    pos = (hero_seat - state.button_seat) % n
    obs[5] = pos / (n - 1) if n > 1 else 0.0

    # Hole cards (one-hot)
    for card in hero.hole_cards:
        obs[6 + card] = 1.0

    # Community cards (one-hot)
    for card in state.community_cards:
        obs[58 + card] = 1.0

    # Opponents (in seat order, skipping hero)
    opp_idx = 0
    for p in state.players:
        if p.seat == hero_seat:
            continue
        obs[110 + opp_idx] = p.stack / NORM
        obs[116 + opp_idx] = p.bet_street / NORM
        obs[122 + opp_idx] = float(p.is_folded)
        obs[128 + opp_idx] = float(p.is_all_in)
        opp_idx += 1
        if opp_idx >= 6:
            break

    return obs


OBS_DIM = 134


# ---------------------------------------------------------------------------
# Action encoding / decoding
# ---------------------------------------------------------------------------

# Discrete action space:
#   0 = FOLD
#   1 = CHECK/CALL
#   2 = RAISE min
#   3 = RAISE pot
#   4 = ALL_IN
N_ACTIONS = 5


def decode_action(action_idx: int, state: GameState) -> Action:
    """Convert discrete action index to Action object."""
    hero = state.players[state.acting_seat]
    call_amount = state.current_bet - hero.bet_street
    all_in_to = hero.bet_street + hero.stack
    min_raise_to = state.current_bet + state.min_raise
    pot_raise_to = state.current_bet + state.pot

    if action_idx == 0:
        return Action(ActionType.FOLD)
    elif action_idx == 1:
        if call_amount == 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL)
    elif action_idx == 2:
        return Action(ActionType.RAISE, amount=min(min_raise_to, all_in_to))
    elif action_idx == 3:
        return Action(ActionType.RAISE, amount=min(pot_raise_to, all_in_to))
    elif action_idx == 4:
        return Action(ActionType.ALL_IN, amount=all_in_to)
    else:
        raise ValueError(f"Unknown action index: {action_idx}")


def encode_action(action: Action, state: GameState) -> int:
    """Map an Action back to a discrete index (approximate)."""
    if action.action_type == ActionType.FOLD:
        return 0
    if action.action_type in (ActionType.CHECK, ActionType.CALL):
        return 1
    if action.action_type == ActionType.ALL_IN:
        return 4
    # RAISE — map to closest bucket
    hero = state.players[state.acting_seat]
    pot_raise = state.current_bet + state.pot
    if action.amount >= pot_raise:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    All agents must implement act().
    observe() is called after each hand with the final reward.
    """

    def __init__(self, seat: int):
        self.seat = seat

    @abstractmethod
    def act(self, state: GameState) -> Action:
        """Return an action given the current game state."""
        ...

    def observe(self, state: GameState, reward: float, done: bool, info: Dict[str, Any]):
        """Called after each step. Override in learning agents."""
        pass

    def reset(self):
        """Called at the start of each hand. Override if needed."""
        pass
