"""Tests for the core game engine."""
import pytest
from poker_env.game import PokerGame, Action, ActionType, Street
from poker_env.hand_eval import hand_name


def fold():  return Action(ActionType.FOLD)
def check(): return Action(ActionType.CHECK)
def call():  return Action(ActionType.CALL)
def raise_(amount): return Action(ActionType.RAISE, amount=amount)
def all_in(): return Action(ActionType.ALL_IN)


class TestBlindsAndSetup:
    def setup_method(self):
        self.game = PokerGame(n_players=6, starting_stack=1000, sb=5, bb=10, seed=0)
        self.game.reset_session()
        self.game.start_hand()

    def test_pot_has_blinds(self):
        assert self.game._pot == 15  # 5 + 10

    def test_stacks_decremented(self):
        sb = self.game.players[self.game._sb_seat]
        bb = self.game.players[self.game._bb_seat]
        assert sb.stack == 995
        assert bb.stack == 990

    def test_acting_seat_is_utg(self):
        # UTG = one left of BB
        expected = self.game._next_active_seat(self.game._bb_seat)
        assert self.game._acting_seat == expected

    def test_hole_cards_dealt(self):
        for p in self.game.players:
            assert len(p.hole_cards) == 2


class TestBettingRound:
    def setup_method(self):
        self.game = PokerGame(n_players=6, starting_stack=1000, sb=5, bb=10, seed=1)
        self.game.reset_session()
        self.game.start_hand()

    def _fold_all_but_one(self):
        """Fold everyone except BB."""
        bb = self.game._bb_seat
        while not self.game.hand_done:
            if self.game._acting_seat == bb:
                self.game.step(check())
            else:
                info = self.game.step(fold())
                if info["hand_done"]:
                    return info
        return None

    def test_early_win_on_folds(self):
        info = self._fold_all_but_one()
        assert info is not None
        assert info["hand_done"] is True
        assert info["rewards"] is not None

    def test_bb_wins_blinds_on_fold(self):
        bb_seat = self.game._bb_seat
        bb_player = self.game.players[bb_seat]
        start_stack = bb_player.stack

        info = self._fold_all_but_one()
        # BB wins the pot (SB blind 5 + BB blind 10 = 15), started at 990
        # so final stack = 990 + 15 = 1005
        final_stack = self.game.players[bb_seat].stack
        assert final_stack == 1005


class TestStreetProgression:
    def setup_method(self):
        self.game = PokerGame(n_players=2, starting_stack=1000, sb=5, bb=10, seed=2)
        self.game.reset_session()
        self.game.start_hand()

    def test_advances_to_flop(self):
        # Heads up preflop: SB acts first, call, BB checks
        assert self.game._street == Street.PREFLOP
        self.game.step(call())   # SB calls
        self.game.step(check())  # BB checks
        assert self.game._street == Street.FLOP
        assert len(self.game._community) == 3

    def test_advances_to_turn(self):
        self.game.step(call())
        self.game.step(check())
        # Flop
        self.game.step(check())
        self.game.step(check())
        assert self.game._street == Street.TURN
        assert len(self.game._community) == 4

    def test_full_hand_to_showdown(self):
        # Just check everything down
        for _ in range(8):  # preflop call + 3 streets * 2 checks
            if self.game.hand_done:
                break
            state = self.game.state
            seat = state.acting_seat
            player = self.game.players[seat]
            call_amt = state.current_bet - player.bet_street
            if call_amt == 0:
                info = self.game.step(check())
            else:
                info = self.game.step(call())
            if info["hand_done"]:
                break
        assert self.game.hand_done


class TestSidePots:
    def test_side_pot_two_players(self):
        from poker_env.game import PlayerState, compute_pots
        players = [
            PlayerState(seat=0, stack=0, bet_total=200, is_all_in=True),
            PlayerState(seat=1, stack=0, bet_total=500),
        ]
        # Give seat 1 a fake active status
        pots = compute_pots(players)
        # Main pot: 200 * 2 = 400, side pot: 300 for seat 1 only
        assert pots[0].amount == 400
        assert 0 in pots[0].eligible_seats

    def test_three_way_side_pots(self):
        from poker_env.game import PlayerState, compute_pots
        players = [
            PlayerState(seat=0, stack=0, bet_total=100, is_all_in=True),
            PlayerState(seat=1, stack=0, bet_total=300, is_all_in=True),
            PlayerState(seat=2, stack=0, bet_total=600),
        ]
        pots = compute_pots(players)
        total = sum(p.amount for p in pots)
        assert total == 1000  # 100 + 300 + 600


class TestLegalActions:
    def setup_method(self):
        self.game = PokerGame(n_players=6, starting_stack=1000, sb=5, bb=10, seed=3)
        self.game.reset_session()
        self.game.start_hand()

    def test_can_fold_call_raise_preflop(self):
        actions = self.game.legal_actions()
        types = {a.action_type for a in actions}
        assert ActionType.FOLD in types
        assert ActionType.CALL in types
        assert ActionType.RAISE in types or ActionType.ALL_IN in types

    def test_check_available_when_no_bet(self):
        # Get to a spot where current_bet == bet_street (e.g. BB after all call)
        # Easiest: 2-player game, SB calls, now BB can check
        game = PokerGame(n_players=2, starting_stack=1000, sb=5, bb=10, seed=3)
        game.reset_session()
        game.start_hand()
        game.step(call())  # SB calls, now BB has option
        actions = game.legal_actions()
        types = {a.action_type for a in actions}
        assert ActionType.CHECK in types
