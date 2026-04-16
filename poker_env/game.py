"""
Core No-Limit Texas Hold'em engine.

Handles: blinds, betting rounds, side pots, showdown, rebuys.
Pure logic
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import IntEnum

from poker_env.card import Deck, cards_to_str
from poker_env.hand_eval import winners, hand_name, best_hand


# ---------------------------------------------------------------------------
# Constants & enums
# ---------------------------------------------------------------------------

class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


class ActionType(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    ALL_IN = 4


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    amount: int = 0  # total bet amount (not raise-by), only for RAISE/ALL_IN

    def __repr__(self):
        if self.action_type == ActionType.RAISE:
            return f"RAISE({self.amount})"
        return self.action_type.name


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    seat: int
    stack: int
    hole_cards: List[int] = field(default_factory=list)
    bet_street: int = 0       # chips committed this street
    bet_total: int = 0        # chips committed this hand
    is_folded: bool = False
    is_all_in: bool = False
    is_sitting_out: bool = False

    def reset_for_hand(self):
        self.hole_cards = []
        self.bet_street = 0
        self.bet_total = 0
        self.is_folded = False
        self.is_all_in = False

    def reset_for_street(self):
        self.bet_street = 0

    def can_act(self) -> bool:
        return not self.is_folded and not self.is_all_in and not self.is_sitting_out

    def put_in(self, amount: int) -> int:
        """Commit chips. Returns actual amount committed (capped by stack)."""
        actual = min(amount, self.stack)
        self.stack -= actual
        self.bet_street += actual
        self.bet_total += actual
        if self.stack == 0:
            self.is_all_in = True
        return actual


# ---------------------------------------------------------------------------
# Side pot calculation
# ---------------------------------------------------------------------------

@dataclass
class Pot:
    amount: int
    eligible_seats: List[int]  # seats that can win this pot


def compute_pots(players: List[PlayerState]) -> List[Pot]:
    """
    Compute side pots from per-player total bets.
    Returns list of Pot objects from main pot to side pots.
    """
    contributions = {
        p.seat: p.bet_total
        for p in players
        if p.bet_total > 0
    }
    active_seats = {
        p.seat for p in players
        if not p.is_folded
    }

    pots: List[Pot] = []
    remaining = dict(contributions)

    while any(v > 0 for v in remaining.values()):
        # Find lowest non-zero contribution
        min_contrib = min(v for v in remaining.values() if v > 0)
        pot_amount = 0
        eligible = []

        for seat, contrib in list(remaining.items()):
            if contrib > 0:
                taken = min(contrib, min_contrib)
                pot_amount += taken
                remaining[seat] -= taken
                if seat in active_seats:
                    eligible.append(seat)

        if pot_amount > 0:
            pots.append(Pot(amount=pot_amount, eligible_seats=sorted(eligible)))

        # Remove players who are fully settled and folded
        remaining = {s: v for s, v in remaining.items() if v > 0}

    return pots


# ---------------------------------------------------------------------------
# Game state snapshot
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Immutable-ish snapshot passed to agents as observation source."""
    hand_number: int
    street: Street
    community_cards: List[int]
    pot: int                       # total pot (all streets)
    current_bet: int               # highest bet on this street
    min_raise: int
    players: List[PlayerState]
    acting_seat: int
    button_seat: int
    sb_seat: int
    bb_seat: int
    sb_amount: int
    bb_amount: int
    last_aggressor: Optional[int]
    action_history: List[Tuple[int, Action]]  # (seat, action) this street


# ---------------------------------------------------------------------------
# Main game engine
# ---------------------------------------------------------------------------

class PokerGame:
    """
    Manages a 6-player No-Limit Texas Hold'em cash game session.

    Usage:
        game = PokerGame(n_players=6, starting_stack=1000, sb=5, bb=10)
        game.reset_session()
        while not session_over:
            game.start_hand()
            while not game.hand_done:
                state = game.state
                action = agent.act(state)
                game.step(action)
            game.end_hand()  # distributes pot, handles rebuys
    """

    REBUY_THRESHOLD = 40   # BB multiples — rebuy if stack < this * bb
    REBUY_TO = 100         # BB multiples — rebuy to this * bb

    def __init__(
        self,
        n_players: int = 6,
        starting_stack: int = 1000,
        sb: int = 5,
        bb: int = 10,
        seed: Optional[int] = None,
    ):
        assert 2 <= n_players <= 9
        self.n_players = n_players
        self.starting_stack = starting_stack
        self.sb_amount = sb
        self.bb_amount = bb
        self.rng = random.Random(seed)
        self.deck = Deck()

        self.players: List[PlayerState] = [
            PlayerState(seat=i, stack=starting_stack)
            for i in range(n_players)
        ]

        self.hand_number: int = 0
        self.button_seat: int = 0
        self.hand_done: bool = True
        self.session_rewards: List[int] = [0] * n_players  # net chips won/lost

        # Set during hand
        self._street: Street = Street.PREFLOP
        self._community: List[int] = []
        self._pot: int = 0
        self._current_bet: int = 0
        self._min_raise: int = bb
        self._acting_seat: int = 0
        self._last_aggressor: Optional[int] = None
        self._action_history: List[Tuple[int, Action]] = []
        self._street_action_count: int = 0

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self):
        for p in self.players:
            p.stack = self.starting_stack
        self.hand_number = 0
        self.button_seat = self.rng.randint(0, self.n_players - 1)
        self.session_rewards = [0] * self.n_players
        self.hand_done = True

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------

    def start_hand(self):
        assert self.hand_done, "Previous hand not finished"
        self.hand_number += 1
        self.hand_done = False

        # Reset player hand state
        for p in self.players:
            p.reset_for_hand()

        # Advance button
        self.button_seat = self._next_active_seat(self.button_seat)
        sb_seat = self._next_active_seat(self.button_seat)
        bb_seat = self._next_active_seat(sb_seat)
        self._sb_seat = sb_seat
        self._bb_seat = bb_seat

        # Reset street state
        self._street = Street.PREFLOP
        self._community = []
        self._pot = 0
        self._current_bet = 0
        self._min_raise = self.bb_amount
        self._last_aggressor = None
        self._action_history = []

        # Deal hole cards
        self.deck.shuffle(self.rng)
        for p in self.players:
            if not p.is_sitting_out:
                p.hole_cards = self.deck.deal(2)

        # Post blinds
        self._post_blind(sb_seat, self.sb_amount)
        self._post_blind(bb_seat, self.bb_amount)
        self._current_bet = self.bb_amount
        self._min_raise = self.bb_amount

        # Action starts left of BB preflop
        self._acting_seat = self._next_active_seat(bb_seat)
        self._last_aggressor = bb_seat  # BB is treated as last aggressor preflop
        self._street_action_count = 0

    def _post_blind(self, seat: int, amount: int):
        p = self.players[seat]
        actual = p.put_in(amount)
        self._pot += actual

    def step(self, action: Action) -> Dict:
        """
        Apply action for current acting player.
        Returns info dict with: seat, action, pot, street, hand_done, rewards.
        """
        assert not self.hand_done
        seat = self._acting_seat
        player = self.players[seat]
        action = self._validate_action(player, action)

        self._action_history.append((seat, action))
        self._street_action_count += 1

        if action.action_type == ActionType.FOLD:
            player.is_folded = True

        elif action.action_type == ActionType.CHECK:
            pass  # nothing to do

        elif action.action_type == ActionType.CALL:
            call_amount = self._current_bet - player.bet_street
            actual = player.put_in(call_amount)
            self._pot += actual

        elif action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
            # action.amount = total bet size for this street
            raise_to = action.amount
            additional = raise_to - player.bet_street
            actual = player.put_in(additional)
            self._pot += actual
            self._current_bet = player.bet_street
            self._min_raise = max(self._min_raise, raise_to - (raise_to - self.bb_amount))
            self._last_aggressor = seat

        info = {
            "seat": seat,
            "action": action,
            "pot": self._pot,
            "street": self._street,
            "hand_done": False,
            "rewards": None,
        }

        # Advance game
        if self._hand_over_early():
            rewards = self._resolve_hand(early=True)
            info["hand_done"] = True
            info["rewards"] = rewards
            self.hand_done = True
        elif self._street_complete():
            if self._street == Street.RIVER:
                rewards = self._resolve_hand(early=False)
                info["hand_done"] = True
                info["rewards"] = rewards
                self.hand_done = True
            else:
                self._advance_street()
        else:
            self._acting_seat = self._next_to_act(seat)

        return info

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def _validate_action(self, player: PlayerState, action: Action) -> Action:
        """Sanitize action to be legal given current state."""
        call_amount = self._current_bet - player.bet_street
        can_check = call_amount == 0

        if action.action_type == ActionType.CHECK and not can_check:
            # Treat check as call when there's a bet
            return Action(ActionType.CALL)

        if action.action_type == ActionType.CALL:
            if call_amount == 0:
                return Action(ActionType.CHECK)
            if call_amount >= player.stack:
                return Action(ActionType.ALL_IN, amount=player.bet_street + player.stack)

        if action.action_type == ActionType.RAISE:
            min_total = self._current_bet + self._min_raise
            max_total = player.bet_street + player.stack
            if action.amount < min_total:
                action = Action(ActionType.RAISE, amount=min_total)
            if action.amount >= max_total:
                return Action(ActionType.ALL_IN, amount=max_total)

        if action.action_type == ActionType.ALL_IN:
            return Action(ActionType.ALL_IN, amount=player.bet_street + player.stack)

        return action

    # ------------------------------------------------------------------
    # Street transitions
    # ------------------------------------------------------------------

    def _street_complete(self) -> bool:
        """True when all active players have acted and bets are equal."""
        active = [p for p in self.players if p.can_act()]
        if not active:
            return True

        # All active players must have matched current bet
        all_called = all(p.bet_street == self._current_bet for p in active)
        if not all_called:
            return False

        # Every active player must have had at least one action this street.
        # This prevents a street completing immediately after _advance_street()
        # resets the counter, and ensures BB gets the option preflop.
        if self._street_action_count < len(active):
            return False

        return True

    def _advance_street(self):
        self._street = Street(self._street + 1)
        for p in self.players:
            p.reset_for_street()
        self._current_bet = 0
        self._min_raise = self.bb_amount
        self._last_aggressor = None
        self._action_history = []
        self._street_action_count = 0

        if self._street == Street.FLOP:
            self._community += self.deck.deal(3)
        elif self._street in (Street.TURN, Street.RIVER):
            self._community += self.deck.deal(1)

        # Post-flop action starts left of button
        self._acting_seat = self._next_active_seat(self.button_seat)

    def _hand_over_early(self) -> bool:
        """True if only one player hasn't folded."""
        active = [p for p in self.players if not p.is_folded]
        return len(active) == 1

    # ------------------------------------------------------------------
    # Showdown & pot resolution
    # ------------------------------------------------------------------

    def _resolve_hand(self, early: bool) -> List[int]:
        """
        Distribute pot(s) to winner(s). Returns per-seat reward list (net chips).
        """
        active = [p for p in self.players if not p.is_folded]

        # Compute side pots
        pots = compute_pots(self.players)

        # Distribute pot(s) to winners
        if early:
            winner = active[0]
            total = sum(pot.amount for pot in pots)
            winner.stack += total
        else:
            for pot in pots:
                eligible_players = [
                    p for p in self.players
                    if p.seat in pot.eligible_seats and not p.is_folded
                ]
                if not eligible_players:
                    eligible_players = [active[0]]

                hole_cards = [p.hole_cards for p in eligible_players]
                winner_indices = winners(hole_cards, self._community)
                split = pot.amount // len(winner_indices)
                remainder = pot.amount % len(winner_indices)

                for i, wi in enumerate(winner_indices):
                    prize = split + (remainder if i == 0 else 0)
                    eligible_players[wi].stack += prize

        # Net reward per player = final stack - stack before this hand started
        # _pre_hand_stacks is set in start_hand()
        rewards = [
            self.players[i].stack - self._pre_hand_stacks[i]
            for i in range(self.n_players)
        ]

        self.session_rewards = [
            self.session_rewards[i] + rewards[i]
            for i in range(self.n_players)
        ]

        # Handle rebuys (after reward is computed)
        self._handle_rebuys()

        return rewards

    def _handle_rebuys(self):
        threshold = self.REBUY_THRESHOLD * self.bb_amount
        rebuy_to = self.REBUY_TO * self.bb_amount
        for p in self.players:
            if p.stack < threshold:
                p.stack = rebuy_to

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _next_active_seat(self, from_seat: int) -> int:
        """Next seat that isn't sitting out, wrapping around."""
        for i in range(1, self.n_players + 1):
            seat = (from_seat + i) % self.n_players
            if not self.players[seat].is_sitting_out:
                return seat
        raise RuntimeError("No active players")

    def _next_to_act(self, from_seat: int) -> int:
        """Next seat that can still act (not folded, not all-in)."""
        for i in range(1, self.n_players + 1):
            seat = (from_seat + i) % self.n_players
            if self.players[seat].can_act():
                return seat
        return from_seat  # fallback (shouldn't happen)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def start_hand(self):
        assert self.hand_done, "Previous hand not finished"
        self.hand_number += 1
        self.hand_done = False

        # Store pre-hand stacks for reward calculation
        self._pre_hand_stacks = {p.seat: p.stack for p in self.players}

        for p in self.players:
            p.reset_for_hand()

        self.button_seat = self._next_active_seat(self.button_seat)
        sb_seat = self._next_active_seat(self.button_seat)
        bb_seat = self._next_active_seat(sb_seat)
        self._sb_seat = sb_seat
        self._bb_seat = bb_seat

        self._street = Street.PREFLOP
        self._community = []
        self._pot = 0
        self._current_bet = 0
        self._min_raise = self.bb_amount
        self._last_aggressor = None
        self._action_history = []
        self._street_action_count = 0

        self.deck.shuffle(self.rng)
        for p in self.players:
            if not p.is_sitting_out:
                p.hole_cards = self.deck.deal(2)

        self._post_blind(sb_seat, self.sb_amount)
        self._post_blind(bb_seat, self.bb_amount)
        self._current_bet = self.bb_amount
        self._min_raise = self.bb_amount

        self._acting_seat = self._next_active_seat(bb_seat)
        self._last_aggressor = bb_seat
        self._street_action_count = 0

    @property
    def state(self) -> GameState:
        return GameState(
            hand_number=self.hand_number,
            street=self._street,
            community_cards=list(self._community),
            pot=self._pot,
            current_bet=self._current_bet,
            min_raise=self._min_raise,
            players=[PlayerState(
                seat=p.seat,
                stack=p.stack,
                hole_cards=list(p.hole_cards),
                bet_street=p.bet_street,
                bet_total=p.bet_total,
                is_folded=p.is_folded,
                is_all_in=p.is_all_in,
            ) for p in self.players],
            acting_seat=self._acting_seat,
            button_seat=self.button_seat,
            sb_seat=self._sb_seat,
            bb_seat=self._bb_seat,
            sb_amount=self.sb_amount,
            bb_amount=self.bb_amount,
            last_aggressor=self._last_aggressor,
            action_history=list(self._action_history),
        )

    def legal_actions(self) -> List[Action]:
        """Return list of legal action types for acting player."""
        player = self.players[self._acting_seat]
        call_amount = self._current_bet - player.bet_street
        actions = [Action(ActionType.FOLD)]

        if call_amount == 0:
            actions.append(Action(ActionType.CHECK))
        else:
            actions.append(Action(ActionType.CALL))

        # Raise options: min raise, pot-size raise, all-in
        min_raise_to = self._current_bet + self._min_raise
        pot_raise_to = self._current_bet + self._pot
        all_in_to = player.bet_street + player.stack

        if all_in_to > self._current_bet:  # can raise at all
            if min_raise_to < all_in_to:
                actions.append(Action(ActionType.RAISE, amount=min_raise_to))
            if pot_raise_to < all_in_to and pot_raise_to > min_raise_to:
                actions.append(Action(ActionType.RAISE, amount=pot_raise_to))
            actions.append(Action(ActionType.ALL_IN, amount=all_in_to))

        return actions

    def __repr__(self):
        p = self.players[self._acting_seat]
        return (
            f"Hand #{self.hand_number} | {self._street.name} | "
            f"Pot: {self._pot} | Bet: {self._current_bet} | "
            f"Acting: seat {self._acting_seat} (stack={p.stack}) | "
            f"Board: {cards_to_str(self._community)}"
        )
