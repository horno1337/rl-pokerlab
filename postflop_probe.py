#!/usr/bin/env python3
"""
postflop_probe.py — Diagnose model postflop decision-making.

Tests two key scenarios across a set of hand/board combinations:
  1. As aggressor (IP, checked to) — should c-bet strong, check back weak
  2. Facing a half-pot bet — should call/raise strong, fold weak

Shows action probabilities (F=fold, X/C=check-call, R-=min-raise,
R+=pot-raise, A=all-in) for each hand on each board to reveal whether
the model discriminates hand strength relative to the board.

Usage:
    python postflop_probe.py                   # uses models/ppo_poker_final.zip
    python postflop_probe.py models/foo.zip
"""

from __future__ import annotations
import sys
import numpy as np
import torch
from stable_baselines3 import PPO

RANKS  = "23456789TJQKA"
SUITS  = "cdhs"
NORM   = 1000.0   # 100BB * 10 = BB*100 with BB=10

RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Observation builder (mirrors base.py exactly)
# ---------------------------------------------------------------------------

def make_obs(
    street: int,          # 0=flop, 1=turn, 2=river  (internally +1 for Street enum)
    pot_bb: float,        # pot in BB
    bet_bb: float,        # current bet in BB (0 = checked to hero)
    hero_stack_bb: float,
    hero_bet_street_bb: float,
    position: float,      # 0.0 = BTN (last to act postflop), 0.2..1.0 = earlier
    hole_cards: tuple,    # (card_int, card_int)
    board: list,          # list of card ints
) -> np.ndarray:
    BB = 10
    NORM = BB * 100
    obs = np.zeros(86, dtype=np.float32)

    obs[0] = (street + 1) / 3.0          # flop=0.333, turn=0.667, river=1.0
    obs[1] = (pot_bb * BB) / NORM
    obs[2] = (bet_bb * BB) / NORM
    obs[3] = (hero_stack_bb * BB) / NORM
    obs[4] = (hero_bet_street_bb * BB) / NORM
    obs[5] = position

    r0, r1 = hole_cards[0] // 4, hole_cards[1] // 4
    s0, s1 = hole_cards[0] % 4,  hole_cards[1] % 4
    obs[6] = max(r0, r1) / 12.0
    obs[7] = min(r0, r1) / 12.0
    obs[8] = 1.0 if (s0 == s1 and r0 != r1) else 0.0
    obs[9] = 1.0 if r0 == r1 else 0.0

    for card in board:
        obs[10 + card] = 1.0

    # 5 opponents: equal stacks, no bets, none folded/all-in
    for i in range(5):
        obs[62 + i] = hero_stack_bb * BB / NORM

    return obs


def card(rank_char: str, suit_char: str) -> int:
    return RANKS.index(rank_char.upper()) * 4 + SUITS.index(suit_char.lower())


# ---------------------------------------------------------------------------
# Hand / board definitions
# ---------------------------------------------------------------------------

BOARDS = {
    "K 7 2 rainbow (dry)": [
        card("K","h"), card("7","d"), card("2","c")
    ],
    "J T 9 two-tone (wet)": [
        card("J","h"), card("T","h"), card("9","d")
    ],
    "A 6 6 paired":  [
        card("A","h"), card("6","d"), card("6","c")
    ],
}

# (display_name, hole_card1, hole_card2, expected_strength)
HANDS = [
    ("AA  (overpair)",        card("A","s"),  card("A","c"),  "strong"),
    ("KK  (top pair/TPTK)",   card("K","s"),  card("K","c"),  "strong"),
    ("K7o (two pair)",        card("K","s"),  card("7","c"),  "strong"),
    ("JTo (open-ended str.)", card("J","s"),  card("T","c"),  "medium"),
    ("A5o (overcards/bdfd)",  card("A","s"),  card("5","c"),  "medium"),
    ("98s (OESD/flush draw)", card("9","h"),  card("8","h"),  "medium"),
    ("55  (underpair)",       card("5","s"),  card("5","c"),  "weak"),
    ("AQo (overcards only)",  card("A","s"),  card("Q","c"),  "weak"),
    ("72o (air)",             card("7","s"),  card("2","c"),  "air"),
    ("32o (pure air)",        card("3","s"),  card("2","c"),  "air"),
]

STRENGTH_COLOR = {
    "strong": GREEN,
    "medium": YELLOW,
    "weak":   YELLOW,
    "air":    RED,
}


# ---------------------------------------------------------------------------
# Model query
# ---------------------------------------------------------------------------

def get_probs(model, obs: np.ndarray) -> np.ndarray:
    device = next(model.policy.parameters()).device
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.squeeze().cpu().numpy()
    return probs   # shape (5,)


def fmt_probs(probs: np.ndarray, strength: str) -> str:
    labels = ["F ", "X/C", "R- ", "R+ ", "AI "]
    col = STRENGTH_COLOR[strength]
    parts = []
    for i, (label, p) in enumerate(zip(labels, probs)):
        bar = "█" * int(p * 20)
        parts.append(f"{label}{col}{p*100:4.0f}%{RESET} {DIM}{bar:<20}{RESET}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model_path: str):
    print(f"\n{BOLD}Loading {model_path}{RESET}")
    model = PPO.load(model_path)

    for board_name, board_cards in BOARDS.items():
        print(f"\n{'='*80}")
        print(f"{BOLD}{CYAN}Board: {board_name}{RESET}")
        print(f"{'='*80}")

        for scenario_name, pot_bb, bet_bb, desc in [
            ("AS AGGRESSOR  (IP, checked to — c-bet spot)",  3.5, 0.0,  "checking or betting"),
            ("FACING HALF-POT BET  (call/raise/fold spot)",  3.5, 1.75, "folding, calling, or raising"),
        ]:
            print(f"\n  {BOLD}{scenario_name}{RESET}")
            print(f"  Pot={pot_bb}BB  Bet={bet_bb}BB  Hero IP (BTN)  Stack=97BB")
            print(f"  {'Hand':<22}  {'F':>5}   {'X/C':>5}   {'R-':>5}   {'R+':>5}   {'AI':>5}")
            print(f"  {'-'*70}")

            for hand_name, c1, c2, strength in HANDS:
                obs = make_obs(
                    street=0,              # flop
                    pot_bb=pot_bb,
                    bet_bb=bet_bb,
                    hero_stack_bb=97.0,
                    hero_bet_street_bb=0.0,
                    position=0.0,          # BTN = in position
                    hole_cards=(c1, c2),
                    board=board_cards,
                )
                probs = get_probs(model, obs)
                col = STRENGTH_COLOR[strength]
                bar = "".join(
                    f"{GREEN if i in (2,3,4) else RED if i==0 else YELLOW}"
                    f"{'█' * max(1, int(probs[i]*20))}{RESET}"
                    for i in range(5)
                )
                print(
                    f"  {col}{hand_name:<22}{RESET}  "
                    f"{probs[0]*100:4.0f}%   "
                    f"{probs[1]*100:4.0f}%   "
                    f"{probs[2]*100:4.0f}%   "
                    f"{probs[3]*100:4.0f}%   "
                    f"{probs[4]*100:4.0f}%   "
                    f"{bar}"
                )

    # Summary: does action vary with hand strength?
    print(f"\n{'='*80}")
    print(f"{BOLD}DISCRIMINATION CHECK — flop, K72r, as aggressor{RESET}")
    print(f"Does betting probability increase with hand strength?")
    print(f"{'='*80}")
    board = BOARDS["K 7 2 rainbow (dry)"]
    results = []
    for hand_name, c1, c2, strength in HANDS:
        obs = make_obs(0, 3.5, 0.0, 97.0, 0.0, 0.0, (c1, c2), board)
        probs = get_probs(model, obs)
        bet_prob = probs[2] + probs[3] + probs[4]
        results.append((hand_name, strength, bet_prob))

    results.sort(key=lambda x: -x[2])
    for hand_name, strength, bet_prob in results:
        col = STRENGTH_COLOR[strength]
        bar = "█" * int(bet_prob * 40)
        print(f"  {col}{hand_name:<22}{RESET}  bet={bet_prob*100:4.0f}%  {col}{bar}{RESET}")

    bet_probs = [r[2] for r in results]
    std = np.std(bet_probs)
    print(f"\n  Std dev of bet frequency across hands: {std:.3f}")
    if std < 0.05:
        print(f"  {RED}UNIFORM — model ignores hand strength postflop{RESET}")
    elif std < 0.15:
        print(f"  {YELLOW}WEAK discrimination — some hand-reading but mostly uniform{RESET}")
    else:
        print(f"  {GREEN}GOOD discrimination — model reads hand strength vs board{RESET}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_poker_final"
    if not path.endswith(".zip"):
        path += ".zip"
    run(path[:-4])
