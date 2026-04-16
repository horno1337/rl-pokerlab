"""
Baseline agents for benchmarking.

  RandomAgent     — uniform random over legal actions
  CallAgent       — always call/check, never fold or raise
  HeuristicAgent  — simple rule-based play using hand strength
  HumanAgent      — reads action from stdin (for manual play)
"""

from __future__ import annotations
import random
from typing import Dict, Any

from poker_env.game import GameState, Action, ActionType
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
