"""
Card and Deck primitives.

Cards are represented as integers 0-51:
  rank = card // 4   (0=2, 1=3, ..., 12=Ace)
  suit = card % 4    (0=clubs, 1=diamonds, 2=hearts, 3=spades)
"""

import random
from typing import List

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_from_str(s: str) -> int:
    """'Ah' -> integer card id."""
    rank = RANKS.index(s[0].upper() if s[0] != 't' else 'T')
    suit = SUITS.index(s[1].lower())
    return rank * 4 + suit


def card_to_str(card: int) -> str:
    return RANKS[card // 4] + SUITS[card % 4]


def cards_to_str(cards: List[int]) -> str:
    return " ".join(card_to_str(c) for c in cards)


class Deck:
    def __init__(self):
        self._cards: List[int] = list(range(52))
        self._dealt: int = 0

    def shuffle(self, rng: random.Random | None = None):
        if rng:
            rng.shuffle(self._cards)
        else:
            random.shuffle(self._cards)
        self._dealt = 0

    def deal(self, n: int = 1) -> List[int]:
        if self._dealt + n > 52:
            raise ValueError("Not enough cards in deck")
        cards = self._cards[self._dealt: self._dealt + n]
        self._dealt += n
        return cards

    def remaining(self) -> int:
        return 52 - self._dealt
