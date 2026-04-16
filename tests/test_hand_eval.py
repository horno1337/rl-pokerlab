"""Tests for card primitives and hand evaluator."""
import pytest
from poker_env.card import card_from_str, card_to_str, Deck
from poker_env.hand_eval import evaluate_5, best_hand, winners, hand_name


def c(s): return card_from_str(s)


class TestCards:
    def test_round_trip(self):
        for i in range(52):
            assert card_from_str(card_to_str(i)) == i

    def test_specific(self):
        assert card_to_str(c("Ah")) == "Ah"
        assert card_to_str(c("2c")) == "2c"
        assert card_to_str(c("Kd")) == "Kd"

    def test_deck_deals_52(self):
        d = Deck()
        d.shuffle()
        cards = d.deal(52)
        assert len(cards) == 52
        assert len(set(cards)) == 52

    def test_deck_no_overdeal(self):
        d = Deck()
        d.shuffle()
        d.deal(52)
        with pytest.raises(ValueError):
            d.deal(1)


class TestHandEval:
    def test_straight_flush(self):
        hand = [c("Ah"), c("Kh"), c("Qh"), c("Jh"), c("Th")]
        rank, _ = evaluate_5(hand)
        assert rank == 8

    def test_four_of_a_kind(self):
        hand = [c("Ah"), c("Ad"), c("As"), c("Ac"), c("Kh")]
        rank, _ = evaluate_5(hand)
        assert rank == 7

    def test_full_house(self):
        hand = [c("Ah"), c("Ad"), c("As"), c("Kh"), c("Kd")]
        rank, _ = evaluate_5(hand)
        assert rank == 6

    def test_flush(self):
        hand = [c("Ah"), c("Kh"), c("Qh"), c("9h"), c("2h")]
        rank, _ = evaluate_5(hand)
        assert rank == 5

    def test_straight(self):
        hand = [c("9h"), c("8d"), c("7s"), c("6c"), c("5h")]
        rank, _ = evaluate_5(hand)
        assert rank == 4

    def test_wheel_straight(self):
        hand = [c("Ah"), c("2d"), c("3s"), c("4c"), c("5h")]
        rank, _ = evaluate_5(hand)
        assert rank == 4

    def test_three_of_a_kind(self):
        hand = [c("Ah"), c("Ad"), c("As"), c("Kh"), c("Qd")]
        rank, _ = evaluate_5(hand)
        assert rank == 3

    def test_two_pair(self):
        hand = [c("Ah"), c("Ad"), c("Kh"), c("Kd"), c("Qh")]
        rank, _ = evaluate_5(hand)
        assert rank == 2

    def test_one_pair(self):
        hand = [c("Ah"), c("Ad"), c("Kh"), c("Qd"), c("Jh")]
        rank, _ = evaluate_5(hand)
        assert rank == 1

    def test_high_card(self):
        hand = [c("Ah"), c("Kd"), c("Qh"), c("Jd"), c("9h")]
        rank, _ = evaluate_5(hand)
        assert rank == 0

    def test_best_hand_7_cards(self):
        # Royal flush hidden among 7 cards
        cards = [c("Ah"), c("Kh"), c("Qh"), c("Jh"), c("Th"), c("2d"), c("3s")]
        rank, _ = best_hand(cards)
        assert rank == 8

    def test_winners_single(self):
        hole = [[c("Ah"), c("Ad")], [c("2h"), c("2d")]]
        community = [c("As"), c("Ac"), c("Kh"), c("Qd"), c("Jh")]
        w = winners(hole, community)
        assert w == [0]  # four aces beats four 2s

    def test_winners_split(self):
        hole = [[c("Ah"), c("2h")], [c("As"), c("3d")]]
        community = [c("Kh"), c("Kd"), c("Ks"), c("Qh"), c("Qd")]
        # Both have KKKQQ full house from board — should split
        w = winners(hole, community)
        assert set(w) == {0, 1}
