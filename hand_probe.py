#!/usr/bin/env python3
"""
hand_probe.py — Query the RL poker model for a specific hand.

Usage:
    python hand_probe.py                   # uses models/best_model.zip
    python hand_probe.py models/foo.zip    # use a specific model

Then type hands like:
    AA, AKs, AKo, KQs, T9o, 72o, 98s ...

Positions shown in preflop action order: UTG → MP → CO → BTN → SB → BB
"""

import sys
import os
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Terminal colour helpers
# ────────────────────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ────────────────────────────────────────────────────────────────────────────
# Card constants / parsing
# ────────────────────────────────────────────────────────────────────────────
RANKS    = "23456789TJQKA"
SUITS    = "cdhs"
RANK_MAP = {r: i for i, r in enumerate(RANKS)}

RANK_NAMES = {
    '2': 'Deuce', '3': 'Three', '4': 'Four',  '5': 'Five',
    '6': 'Six',   '7': 'Seven', '8': 'Eight', '9': 'Nine',
    'T': 'Ten',   'J': 'Jack',  'Q': 'Queen', 'K': 'King', 'A': 'Ace',
}


def card_to_str(card: int) -> str:
    return RANKS[card // 4] + SUITS[card % 4]


def parse_hand(hand_str: str):
    """
    Parse hand notation into two card integers.

    Supported formats:
        AA          - pocket pair
        AKs         - suited
        AKo / AK    - offsuit
        T9s, 72o    - any two different ranks

    Returns:
        (card1, card2, display_name, is_suited, is_pair)
    """
    s = hand_str.strip()
    if len(s) < 2:
        raise ValueError(f"Too short: '{hand_str}'")

    r1 = s[0].upper()
    r2 = s[1].upper()

    if r1 not in RANK_MAP:
        raise ValueError(f"Unknown rank '{r1}' in '{hand_str}'")
    if r2 not in RANK_MAP:
        raise ValueError(f"Unknown rank '{r2}' in '{hand_str}'")

    rank1 = RANK_MAP[r1]
    rank2 = RANK_MAP[r2]
    is_pair = (rank1 == rank2)

    if len(s) >= 3:
        suit_tag = s[2].lower()
        if suit_tag == 's':
            is_suited = True
        elif suit_tag == 'o':
            is_suited = False
        else:
            raise ValueError(f"Expected 's' or 'o' after ranks, got '{s[2]}'")
    else:
        if is_pair:
            is_suited = False
        else:
            # default to offsuit 
            is_suited = False

    if is_pair and is_suited:
        raise ValueError("A pair cannot be suited.")

    # Normalise so higher rank is first (for display)
    if rank1 < rank2:
        rank1, rank2 = rank2, rank1
        r1, r2 = r2, r1

    # Pick concrete card ids (hearts=2, spades=3, clubs=0, diamonds=1)
    if is_pair:
        card1 = rank1 * 4 + 2   # Xh
        card2 = rank1 * 4 + 3   # Xs
    elif is_suited:
        card1 = rank1 * 4 + 2   # Xh
        card2 = rank2 * 4 + 2   # Yh  (same suit)
    else:
        card1 = rank1 * 4 + 2   # Xh
        card2 = rank2 * 4 + 3   # Ys  (different suit)

    # Human-readable description
    n1 = RANK_NAMES[r1]
    n2 = RANK_NAMES[r2]
    if is_pair:
        display = f"Pocket {n1}s"
    elif is_suited:
        display = f"{n1}-{n2} suited"
    else:
        display = f"{n1}-{n2} offsuit"

    return card1, card2, display, is_suited, is_pair


# ────────────────────────────────────────────────────────────────────────────
# Synthetic preflop RFI observation
# ────────────────────────────────────────────────────────────────────────────
#
# 6-max positions (relative to button seat 0):
#   BTN=0  SB=1  BB=2  UTG=3  MP=4  CO=5
#
# Preflop action order: UTG(3) → MP(4) → CO(5) → BTN(0) → SB(1) → BB(2)
# For RFI: every player who acts *before* hero has folded.
#
PREFLOP_ORDER = [3, 4, 5, 0, 1, 2]   # seats in preflop action order

# Display order and position names
POSITIONS = [
    ("UTG", 3),
    ("MP",  4),
    ("CO",  5),
    ("BTN", 0),
    ("SB",  1),
    ("BB",  2),
]

STARTING_STACK = 1000
SB_AMOUNT = 5
BB_AMOUNT = 10
NORM = BB_AMOUNT * 100   # 1000


def build_rfi_obs(card1: int, card2: int, pos_rel: int) -> np.ndarray:
    """
    Build a flat 134-dim observation for a preflop raise-first-in scenario.

    pos_rel: hero's seat position relative to the button (BTN=0 … BB=2 … CO=5).
    All players who act before the hero in preflop order are marked as folded.
    The blinds are posted; no voluntary action has happened yet.
    """
    n_players = 6
    button_seat = 0
    hero_seat = pos_rel   # hero sits at seat == pos_rel (button_seat fixed at 0)

    obs = np.zeros(134, dtype=np.float32)

    # ── Scalar game features ──────────────────────────────────────────────
    obs[0] = 0.0                         # street = PREFLOP (0/3)
    obs[1] = (SB_AMOUNT + BB_AMOUNT) / NORM   # pot = SB + BB
    obs[2] = BB_AMOUNT / NORM            # current_bet = BB

    # Hero chip state
    if pos_rel == 1:   # SB: has posted blind
        hero_stack      = STARTING_STACK - SB_AMOUNT
        hero_bet_street = SB_AMOUNT
    elif pos_rel == 2:  # BB: has posted blind
        hero_stack      = STARTING_STACK - BB_AMOUNT
        hero_bet_street = BB_AMOUNT
    else:               # everyone else: hasn't put chips in yet
        hero_stack      = STARTING_STACK
        hero_bet_street = 0

    obs[3] = hero_stack / NORM
    obs[4] = hero_bet_street / NORM

    # Relative position (0 = button)
    obs[5] = pos_rel / (n_players - 1)

    # ── Hero hole cards (one-hot at [6:58]) ──────────────────────────────
    obs[6 + card1] = 1.0
    obs[6 + card2] = 1.0

    # Community cards stay zero (preflop)

    # ── Opponents ([110:134]) ─────────────────────────────────────────────
    # Which seats have folded (acted before hero in preflop order)?
    hero_order_idx = PREFLOP_ORDER.index(pos_rel)
    folded_seats   = set(PREFLOP_ORDER[:hero_order_idx])

    opp_idx = 0
    for seat in range(n_players):
        if seat == hero_seat:
            continue

        if seat == 1:   # SB posted
            opp_stack      = STARTING_STACK - SB_AMOUNT
            opp_bet_street = SB_AMOUNT
        elif seat == 2:  # BB posted
            opp_stack      = STARTING_STACK - BB_AMOUNT
            opp_bet_street = BB_AMOUNT
        else:
            opp_stack      = STARTING_STACK
            opp_bet_street = 0

        is_folded = (seat in folded_seats)

        obs[110 + opp_idx] = opp_stack / NORM
        obs[116 + opp_idx] = opp_bet_street / NORM
        obs[122 + opp_idx] = float(is_folded)
        obs[128 + opp_idx] = 0.0   # nobody is all-in preflop
        opp_idx += 1

    return obs


# ────────────────────────────────────────────────────────────────────────────
# Model inference
# ────────────────────────────────────────────────────────────────────────────
ACTION_NAMES  = ["Fold",      "Call/Check", "Min Raise", "Pot Raise", "All-In"]
ACTION_COLORS = [RED,          YELLOW,       GREEN,       GREEN,       CYAN   ]


def get_action_probs(model, obs: np.ndarray) -> np.ndarray:
    """Return a (5,) probability vector from the PPO policy."""
    import torch as th

    obs_tensor = th.as_tensor(obs[np.newaxis], dtype=th.float32,
                               device=model.policy.device)
    with th.no_grad():
        dist  = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]
    return probs


def decision_label(best_idx: int) -> str:
    if best_idx == 0:
        return f"{RED}{BOLD}FOLD{RESET}"
    elif best_idx == 1:
        return f"{YELLOW}{BOLD}LIMP/CHECK{RESET}"
    elif best_idx in (2, 3):
        return f"{GREEN}{BOLD}RAISE{RESET}"
    else:
        return f"{CYAN}{BOLD}SHOVE{RESET}"


# ────────────────────────────────────────────────────────────────────────────
# Display
# ────────────────────────────────────────────────────────────────────────────
COL_WIDTH = 9   # visible width of each probability cell  " 68.4%  "
SEP       = "│"


def fmt_cell(prob: float, best: bool, color: str) -> str:
    """7-char visible cell, optionally coloured and bolded."""
    text = f"{prob*100:5.1f}%"   # always 6 visible chars
    if best:
        return f"{BOLD}{color}{text}{RESET}"
    return f"{DIM}{text}{RESET}"


def print_results(hand_str: str, display_name: str,
                  card1: int, card2: int, results: list):
    """
    results: list of (pos_name, probs_array)
    """
    c1, c2 = card_to_str(card1), card_to_str(card2)
    width = 68

    print()
    print(f"{BOLD}{'═' * width}{RESET}")
    print(f"{BOLD}  {CYAN}{hand_str.upper()}{RESET}{BOLD}"
          f"  {display_name}  [{c1}  {c2}]{RESET}")
    print(f"{BOLD}{'═' * width}{RESET}")

    # Header
    header = (f"{'Pos':<4} {SEP} {'Fold':>6} {SEP} {'Call':>6} {SEP}"
              f" {'Min↑':>6} {SEP} {'Pot↑':>6} {SEP} {'AllIn':>6} {SEP}"
              f" Decision")
    print(f"{DIM}{header}{RESET}")
    print(f"{DIM}{'─' * width}{RESET}")

    for pos_name, probs in results:
        best = int(np.argmax(probs))
        cells = [fmt_cell(p, i == best, ACTION_COLORS[i])
                 for i, p in enumerate(probs)]
        dec   = decision_label(best)

        # Build row (visible width preserved because cells are pre-padded)
        row = (f"{pos_name:<4} {SEP} {cells[0]} {SEP} {cells[1]} {SEP}"
               f" {cells[2]} {SEP} {cells[3]} {SEP} {cells[4]} {SEP}"
               f" {dec}")
        print(row)

    print(f"{BOLD}{'═' * width}{RESET}")


# ────────────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────────────
def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    default_mdl = os.path.join(script_dir, "models", "best_model.zip")
    model_path  = sys.argv[1] if len(sys.argv) > 1 else default_mdl

    print(f"\n{BOLD}RL Poker — Hand Probe{RESET}")
    print(f"{DIM}Model : {model_path}{RESET}")

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
        print(f"{GREEN}Model loaded.{RESET}")
    except Exception as exc:
        print(f"{RED}Could not load model: {exc}{RESET}")
        sys.exit(1)

    print()
    print("Type a hand to see what the model learned, e.g.:")
    print("  AA   AKs   AKo   KQs   T9o   72o   JTs")
    print("Type 'q' to quit.\n")

    while True:
        try:
            raw = input(f"{BOLD}Hand › {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            print("Bye!")
            break

        try:
            card1, card2, display_name, is_suited, is_pair = parse_hand(raw)
        except ValueError as err:
            print(f"  {RED}⚠  {err}{RESET}")
            continue

        results = []
        for pos_name, pos_rel in POSITIONS:
            obs   = build_rfi_obs(card1, card2, pos_rel)
            probs = get_action_probs(model, obs)
            results.append((pos_name, probs))

        print_results(raw, display_name, card1, card2, results)


if __name__ == "__main__":
    main()
