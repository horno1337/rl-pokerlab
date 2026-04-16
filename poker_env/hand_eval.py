"""
Hand evaluator: given up to 7 cards, return the best 5-card hand rank.

Hand ranks (higher = better):
  8 = straight flush
  7 = four of a kind
  6 = full house
  5 = flush
  4 = straight
  3 = three of a kind
  2 = two pair
  1 = one pair
  0 = high card

Returns a tuple (hand_rank, tiebreaker_tuple) for comparison.
"""

from itertools import combinations
from typing import List, Tuple
from collections import Counter


HandScore = Tuple[int, tuple]


def evaluate_5(cards: List[int]) -> HandScore:
    """Evaluate exactly 5 cards. Returns (rank, tiebreakers)."""
    ranks = sorted([c // 4 for c in cards], reverse=True)
    suits = [c % 4 for c in cards]

    is_flush = len(set(suits)) == 1

    # Detect straight (including A-2-3-4-5 wheel)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [12, 3, 2, 1, 0]:  # A-2-3-4-5 wheel
            is_straight = True
            straight_high = 3  # 5-high straight

    counts = Counter(ranks)
    freq = sorted(counts.values(), reverse=True)
    groups = sorted(counts.keys(), key=lambda r: (counts[r], r), reverse=True)

    if is_straight and is_flush:
        return (8, (straight_high,))
    if freq == [4, 1]:
        return (7, tuple(groups))
    if freq == [3, 2]:
        return (6, tuple(groups))
    if is_flush:
        return (5, tuple(ranks))
    if is_straight:
        return (4, (straight_high,))
    if freq == [3, 1, 1]:
        return (3, tuple(groups))
    if freq == [2, 2, 1]:
        return (2, tuple(groups))
    if freq == [2, 1, 1, 1]:
        return (1, tuple(groups))
    return (0, tuple(ranks))


def best_hand(cards: List[int]) -> HandScore:
    """Best 5-card hand from 5-7 cards."""
    if len(cards) <= 5:
        return evaluate_5(cards)
    return max(evaluate_5(list(combo)) for combo in combinations(cards, 5))


def hand_name(score: HandScore) -> str:
    names = [
        "High Card", "One Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
    ]
    return names[score[0]]


def winners(hole_cards: List[List[int]], community: List[int]) -> List[int]:
    """
    Given per-player hole cards and community cards, return list of winner indices.
    Multiple indices means a split pot.
    """
    scores = [best_hand(h + community) for h in hole_cards]
    best = max(scores)
    return [i for i, s in enumerate(scores) if s == best]
