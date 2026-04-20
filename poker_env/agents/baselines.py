"""
Baseline agents for benchmarking.

  RandomAgent     — uniform random over legal actions
  CallAgent       — always call/check, never fold or raise
  HeuristicAgent  — simple rule-based play using hand strength
  ThreeBetAgent   — 3-bets aggressively vs BTN opens; exposes over-raising
  HumanAgent      — reads action from stdin (for manual play)
"""

from __future__ import annotations
import random
from typing import Dict, Any

from poker_env.game import GameState, Action, ActionType, Street
from poker_env.agents.base import BaseAgent, decode_action
from poker_env.hand_eval import best_hand, hand_name


class RandomAgent(BaseAgent):
    """Uniformly samples from legal actions."""

    def __init__(self, seat: int, seed: int | None = None):
        super().__init__(seat)
        self.rng = random.Random(seed)

    def act(self, state: GameState) -> Action:
        # Build legal discrete actions
        hero = state.players[self.seat]
        call_amount = state.current_bet - hero.bet_street
        all_in_to = hero.bet_street + hero.stack
        min_raise_to = state.current_bet + state.min_raise

        choices = [0, 1]  # fold, check/call always available
        if all_in_to > state.current_bet:
            choices += [2, 3, 4]  # min raise, pot raise, all-in

        idx = self.rng.choice(choices)
        return decode_action(idx, state)


class CallAgent(BaseAgent):
    """Always calls or checks. Never folds or raises."""

    def act(self, state: GameState) -> Action:
        hero = state.players[self.seat]
        call_amount = state.current_bet - hero.bet_street
        if call_amount == 0:
            return Action(ActionType.CHECK)
        if call_amount >= hero.stack:
            return Action(ActionType.ALL_IN, amount=hero.bet_street + hero.stack)
        return Action(ActionType.CALL)


class HeuristicAgent(BaseAgent):
    """
    Simple rule-based agent using hand strength as a heuristic.

    Strategy:
      - Strong hand  (rank >= 5): raise pot or call
      - Medium hand  (rank >= 2): call or check
      - Weak hand    (rank < 2):  fold to bets, check if free
      - Preflop: simplified based on hole card ranks
    """

    def __init__(self, seat: int, aggression: float = 0.5, seed: int | None = None):
        super().__init__(seat)
        self.aggression = aggression  # 0=passive, 1=aggressive
        self.rng = random.Random(seed)

    def act(self, state: GameState) -> Action:
        hero = state.players[self.seat]
        call_amount = state.current_bet - hero.bet_street
        all_in_to = hero.bet_street + hero.stack
        pot_raise_to = state.current_bet + state.pot

        strength = self._hand_strength(hero.hole_cards, state.community_cards)

        # Strong hand
        if strength >= 5:
            if self.rng.random() < self.aggression and all_in_to > state.current_bet:
                return Action(ActionType.RAISE, amount=min(pot_raise_to, all_in_to))
            if call_amount >= hero.stack:
                return Action(ActionType.ALL_IN, amount=all_in_to)
            return Action(ActionType.CALL) if call_amount > 0 else Action(ActionType.CHECK)

        # Medium hand
        elif strength >= 2:
            if call_amount == 0:
                return Action(ActionType.CHECK)
            if call_amount <= state.pot * 0.3:  # call up to 30% pot
                return Action(ActionType.CALL)
            return Action(ActionType.FOLD)

        # Weak hand
        else:
            if call_amount == 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.FOLD)

    def _hand_strength(self, hole: list, community: list) -> int:
        if not community:
            # Preflop: use rank of highest card + pair bonus
            ranks = sorted([c // 4 for c in hole], reverse=True)
            strength = ranks[0] / 2  # 0-6
            if ranks[0] == ranks[1]:
                strength += 3  # pocket pair bonus
            return int(strength)
        score, _ = best_hand(hole + community)
        return score


class ThreeBetAgent(BaseAgent):
    """
    Exploits over-raising from the button by 3-betting aggressively.

    Vs a BTN open (preflop raise from button_seat):
      - Value 3-bet  (top ~15%): TT+, AK, AQs, AJs, KQs  → re-raise to 3x
      - Bluff 3-bet  (suited connectors): 65s-T9s with some frequency  → re-raise
      - Call          (medium hands)                        → call
      - Fold          (trash)                               → fold

    In all other spots plays like HeuristicAgent, so it's a valid full-street agent.

    The goal is to make the PPO agent pay for an 85% BTN open range:
    facing a 3-bet with 72o is immediately punishing.
    """

    def __init__(self, seat: int, bluff_freq: float = 0.5, seed: int | None = None):
        super().__init__(seat)
        self.bluff_freq = bluff_freq   # how often suited connectors 3-bet as bluffs
        self.rng = random.Random(seed)

    def act(self, state: GameState) -> Action:
        hero = state.players[self.seat]
        call_amount = state.current_bet - hero.bet_street
        all_in_to = hero.bet_street + hero.stack

        if self._is_btn_open(state):
            return self._vs_btn_open(state, hero, call_amount, all_in_to)

        return self._heuristic(state, hero, call_amount, all_in_to)

    def _is_btn_open(self, state: GameState) -> bool:
        """True if the last aggressor was the button and it's still preflop."""
        return (
            state.street == Street.PREFLOP
            and state.current_bet > state.bb_amount
            and state.last_aggressor == state.button_seat
            and self.seat != state.button_seat
        )

    def _vs_btn_open(self, state, hero, call_amount, all_in_to):
        val = self._preflop_value(hero.hole_cards)
        three_bet_to = min(state.current_bet * 3, all_in_to)

        if val >= 9.0:
            # Value 3-bet: TT+, AK, AQs and equivalents
            return Action(ActionType.RAISE, amount=three_bet_to)

        if val >= 6.5 and self.rng.random() < self.bluff_freq:
            # Bluff 3-bet: suited connectors, small pairs — polarised range
            return Action(ActionType.RAISE, amount=three_bet_to)

        if val >= 5.0:
            # Call with medium equity (call_amount check guards against shove)
            if call_amount < hero.stack:
                return Action(ActionType.CALL)

        return Action(ActionType.FOLD)

    def _preflop_value(self, hole_cards: list) -> float:
        """
        Score a preflop hand:
          Pairs     → pair_rank (22=0, AA=12)
          Suited    → avg_rank + 1.5
          Offsuit   → avg_rank
          Connector → +0.5
        """
        ranks = sorted([c // 4 for c in hole_cards], reverse=True)
        high, low = ranks
        suited = (hole_cards[0] % 4 == hole_cards[1] % 4)

        if high == low:
            return float(high)   # pairs: 0 (22) → 12 (AA)

        base = (high + low) / 2.0
        if suited:
            base += 1.5
        if high - low == 1:      # connector
            base += 0.5
        return base

    def _heuristic(self, state, hero, call_amount, all_in_to):
        pot_raise_to = state.current_bet + state.pot
        strength = self._hand_strength(hero.hole_cards, state.community_cards)

        if strength >= 5:
            if self.rng.random() < 0.5 and all_in_to > state.current_bet:
                return Action(ActionType.RAISE, amount=min(pot_raise_to, all_in_to))
            if call_amount >= hero.stack:
                return Action(ActionType.ALL_IN, amount=all_in_to)
            return Action(ActionType.CALL) if call_amount > 0 else Action(ActionType.CHECK)
        elif strength >= 2:
            if call_amount == 0:
                return Action(ActionType.CHECK)
            if call_amount <= state.pot * 0.3:
                return Action(ActionType.CALL)
            return Action(ActionType.FOLD)
        else:
            if call_amount == 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.FOLD)

    def _hand_strength(self, hole: list, community: list) -> int:
        if not community:
            ranks = sorted([c // 4 for c in hole], reverse=True)
            strength = ranks[0] / 2
            if ranks[0] == ranks[1]:
                strength += 3
            return int(strength)
        score, _ = best_hand(hole + community)
        return score


class FoldToRaiseAgent(BaseAgent):
    """
    Bets medium hands but folds them to any raise this street.

    Creates a training signal that teaches the hero to CALL postflop bets
    with draws instead of always raising or folding.  Against CallAgents,
    raising any two cards is always "wrong" because they never fold, so
    the model learns raise-or-fold.  This opponent teaches the third option:
    call a bet, realize draw equity, then bet/raise on later streets when hit.

    Preflop : value-raise strong (TT+, AK, AQs+), call medium, fold trash.
    Postflop:
      Strong  (strength >= 5) — bet / call / raise normally
      Medium  (strength 2-4)  — bet when checked to; call first bet ≤40% pot;
                                 FOLD to any raise this street
      Weak    (strength < 2)  — check / fold
    """

    def __init__(self, seat: int, aggression: float = 0.5, seed: int | None = None):
        super().__init__(seat)
        self.aggression = aggression
        self.rng = random.Random(seed)

    def act(self, state: GameState) -> Action:
        hero = state.players[self.seat]
        call_amount  = state.current_bet - hero.bet_street
        all_in_to    = hero.bet_street + hero.stack
        pot_raise_to = state.current_bet + state.pot
        half_pot_bet = state.current_bet + max(state.pot // 2, state.bb_amount)

        if state.street == Street.PREFLOP:
            return self._preflop(state, hero, call_amount, all_in_to)
        return self._postflop(state, hero, call_amount, all_in_to,
                              pot_raise_to, half_pot_bet)

    # ------------------------------------------------------------------

    def _preflop(self, state, hero, call_amount, all_in_to):
        val = self._preflop_value(hero.hole_cards)
        three_bet_to = min(state.current_bet * 3, all_in_to)
        open_to      = min(state.bb_amount * 3,   all_in_to)

        if val >= 9.0:            # TT+, AK, AQs — value raise
            if call_amount > 0:
                return Action(ActionType.RAISE, amount=three_bet_to)
            return Action(ActionType.RAISE, amount=open_to)
        if val >= 5.0:            # medium: call or open-limp
            if call_amount == 0:
                return Action(ActionType.CHECK)
            if call_amount < hero.stack:
                return Action(ActionType.CALL)
        if call_amount == 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)

    def _postflop(self, state, hero, call_amount, all_in_to,
                  pot_raise_to, half_pot_bet):
        strength = self._postflop_strength(hero.hole_cards, state.community_cards)

        # Was there already a raise this street (from any player)?
        raised = any(a.action_type in (ActionType.RAISE, ActionType.ALL_IN)
                     for _, a in state.action_history)

        # ── Strong hand ──────────────────────────────────────────────────
        if strength >= 5:
            if call_amount > 0:
                if self.rng.random() < self.aggression and all_in_to > state.current_bet:
                    return Action(ActionType.RAISE, amount=min(pot_raise_to, all_in_to))
                return (Action(ActionType.CALL) if call_amount < hero.stack
                        else Action(ActionType.ALL_IN, amount=all_in_to))
            if self.rng.random() < self.aggression and all_in_to > state.current_bet:
                return Action(ActionType.RAISE, amount=min(half_pot_bet, all_in_to))
            return Action(ActionType.CHECK)

        # ── Medium hand — fold to raises ─────────────────────────────────
        if strength >= 2:
            if raised:
                return Action(ActionType.FOLD)
            if call_amount > 0:
                if call_amount <= state.pot * 0.4:
                    return Action(ActionType.CALL)
                return Action(ActionType.FOLD)
            if self.rng.random() < self.aggression * 0.6 and all_in_to > state.current_bet:
                return Action(ActionType.RAISE, amount=min(half_pot_bet, all_in_to))
            return Action(ActionType.CHECK)

        # ── Weak hand ────────────────────────────────────────────────────
        if call_amount == 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)

    # ------------------------------------------------------------------

    def _preflop_value(self, hole_cards: list) -> float:
        ranks = sorted([c // 4 for c in hole_cards], reverse=True)
        high, low = ranks
        suited = (hole_cards[0] % 4 == hole_cards[1] % 4)
        if high == low:
            return float(high)
        base = (high + low) / 2.0
        if suited:
            base += 1.5
        if high - low == 1:
            base += 0.5
        return base

    def _postflop_strength(self, hole: list, community: list) -> int:
        if not community:
            ranks = sorted([c // 4 for c in hole], reverse=True)
            strength = ranks[0] / 2
            if ranks[0] == ranks[1]:
                strength += 3
            return int(strength)
        score, _ = best_hand(hole + community)
        return score


class HumanAgent(BaseAgent):
    """Reads action from stdin. For manual play and debugging."""

    def act(self, state: GameState) -> Action:
        hero = state.players[self.seat]
        call_amount = state.current_bet - hero.bet_street
        all_in_to = hero.bet_street + hero.stack

        from poker_env.card import cards_to_str
        print(f"\n--- Your turn (seat {self.seat}) ---")
        print(f"  Hole cards : {cards_to_str(hero.hole_cards)}")
        print(f"  Board      : {cards_to_str(state.community_cards)}")
        print(f"  Pot        : {state.pot}  |  To call: {call_amount}  |  Stack: {hero.stack}")
        print("  Actions: [f]old  [c]heck/call  [r]aise <amount>  [a]ll-in")

        while True:
            raw = input("  > ").strip().lower()
            if raw.startswith("f"):
                return Action(ActionType.FOLD)
            if raw.startswith("c"):
                if call_amount == 0:
                    return Action(ActionType.CHECK)
                return Action(ActionType.CALL)
            if raw.startswith("a"):
                return Action(ActionType.ALL_IN, amount=all_in_to)
            if raw.startswith("r"):
                parts = raw.split()
                if len(parts) == 2 and parts[1].isdigit():
                    amount = int(parts[1])
                    return Action(ActionType.RAISE, amount=hero.bet_street + amount)
                print("  Usage: r <total raise amount>")
                continue
            print("  Unknown action. Try: f, c, r <amount>, a")
