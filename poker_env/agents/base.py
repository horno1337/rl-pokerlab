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
from poker_env.hand_eval import best_hand


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

N_PLAYERS = 6
N_CARD_FEATURES = 52   # one-hot card encoding


def _postflop_features(hole_cards: list, community_cards: list) -> np.ndarray:
    """
    Compute 6 postflop features appended after the base 86-dim obs.

      [0] made_hand_rank        — best_hand score / 8  (0=high card … 1=SF)
      [1] flush_draw            — 1 if hero has exactly 4-to-flush (not yet made)
      [2] straight_draw_outs    — distinct ranks completing a straight / 8, capped 1
      [3] board_paired          — 1 if any board rank appears twice
      [4] board_flush_possible  — 1 if 3+ same suit on board
      [5] board_straight_possible — 1 if 3 board cards fit in a 5-rank window

    All zeros preflop (no community cards).
    """
    out = np.zeros(6, dtype=np.float32)
    if not community_cards or len(hole_cards) != 2:
        return out

    # 1. Made hand strength
    score, _ = best_hand(hole_cards + community_cards)
    out[0] = score / 8.0

    # 2. Flush draw: exactly 4 cards of hero's suit (hero holds ≥1 of that suit)
    hero_suits  = [c % 4 for c in hole_cards]
    all_suits   = [c % 4 for c in hole_cards + community_cards]
    for suit in range(4):
        if hero_suits.count(suit) >= 1 and all_suits.count(suit) == 4:
            out[1] = 1.0
            break

    # 3. Straight draw outs: ranks not yet held that complete a straight with ≥1 hero card
    hero_ranks = set(c // 4 for c in hole_cards)
    all_ranks  = set(c // 4 for c in hole_cards + community_cards)
    outs = 0
    for r in range(13):
        if r in all_ranks:
            continue
        test = all_ranks | {r}
        for lo in range(9):          # windows 0-4 … 8-12
            window = set(range(lo, lo + 5))
            if window <= test and (hero_ranks & window):
                outs += 1
                break
    out[2] = min(outs / 8.0, 1.0)   # 8 outs (OESD) → 1.0

    # 4. Board paired
    board_ranks = [c // 4 for c in community_cards]
    out[3] = 1.0 if len(set(board_ranks)) < len(board_ranks) else 0.0

    # 5. Board flush possible (3+ of same suit on board)
    board_suits = [c % 4 for c in community_cards]
    if any(board_suits.count(s) >= 3 for s in range(4)):
        out[4] = 1.0

    # 6. Board straight possible (3 board cards within any 5-rank window)
    board_rank_set = set(c // 4 for c in community_cards)
    for lo in range(9):
        if len(set(range(lo, lo + 5)) & board_rank_set) >= 3:
            out[5] = 1.0
            break

    return out


def build_observation(state: GameState, hero_seat: int) -> np.ndarray:
    """
    Build a flat numpy observation vector for the agent at hero_seat.

    Layout:
      [0]     street (0-3, normalized)
      [1]     pot (normalized by BB*100)
      [2]     current_bet (normalized)
      [3]     hero stack (normalized)
      [4]     hero bet_street (normalized)
      [5]     position relative to button (0=button, normalized)
      [6]     hole card high rank (rank/12, 0=2, 1=A)
      [7]     hole card low rank  (rank/12)
      [8]     suited (1.0 if same suit and not a pair)
      [9]     pocket pair (1.0 if same rank)
      [10:62] community cards (one-hot, 52 dims)
      [62:68] opponent stacks (normalized)
      [68:74] opponent bet_street (normalized)
      [74:80] opponent is_folded flags
      [80:86] opponent is_all_in flags
    Total: 86 dims

    A _postflop_features() helper is provided for future experiments that
    want to extend the observation with explicit hand-strength features.
    The shipping model (ppo_poker_final) was trained with 86-dim obs.

    Hole cards are represented ONLY as compact features [6:9].
    Previous versions included a 52-dim one-hot for hole cards, but a shallow
    MLP converged to ignoring it — the model found "raise 65% from BTN" was
    profitable without needing to read individual cards.  Removing the one-hot
    forces the network to use high_rank/low_rank/suited/pocket_pair directly.

    Community cards keep the one-hot [10:62] because postflop hand strength
    (pairs, flushes, straights) genuinely requires knowing which exact cards
    are on the board.
    """
    NORM = state.bb_amount * 100  # normalize by 100BB
    hero = state.players[hero_seat]

    obs = np.zeros(86, dtype=np.float32)

    obs[0] = state.street / 3.0
    obs[1] = state.pot / NORM
    obs[2] = state.current_bet / NORM
    obs[3] = hero.stack / NORM
    obs[4] = hero.bet_street / NORM

    # Relative position (0 = button, 1/(n-1) = one left of button, etc.)
    n = len(state.players)
    pos = (hero_seat - state.button_seat) % n
    obs[5] = pos / (n - 1) if n > 1 else 0.0

    # Compact hole card features — the only hole card information in the obs.
    # These 4 numbers are everything the model knows about its own hand.
    if len(hero.hole_cards) == 2:
        r0, r1 = hero.hole_cards[0] // 4, hero.hole_cards[1] // 4
        s0, s1 = hero.hole_cards[0] % 4,  hero.hole_cards[1] % 4
        high_rank = max(r0, r1)
        low_rank  = min(r0, r1)
        obs[6] = high_rank / 12.0
        obs[7] = low_rank  / 12.0
        obs[8] = 1.0 if (s0 == s1 and r0 != r1) else 0.0  # suited (not for pairs)
        obs[9] = 1.0 if r0 == r1 else 0.0                  # pocket pair

    # Community cards (one-hot, 52 dims at [10:62])
    for card in state.community_cards:
        obs[10 + card] = 1.0

    # Opponents (in seat order, skipping hero)
    opp_idx = 0
    for p in state.players:
        if p.seat == hero_seat:
            continue
        obs[62 + opp_idx] = p.stack / NORM
        obs[68 + opp_idx] = p.bet_street / NORM
        obs[74 + opp_idx] = float(p.is_folded)
        obs[80 + opp_idx] = float(p.is_all_in)
        opp_idx += 1
        if opp_idx >= 6:
            break

    return obs


OBS_DIM = 86


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
