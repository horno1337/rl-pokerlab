"""
Diagnostic: does the model actually discriminate between hands at BTN RFI?

If the model learned hand-conditional strategy, we'd expect:
  - AA/AK: raise probability ~0.9+
  - 72o/32o: raise probability ~0.1 or less
  - Mid-strength hands: somewhere in between

If raise probability is uniform (~same for all hands), the model
hasn't learned hand selection — it just learned a uniform "always raise"
or "never raise" policy.

Usage:
    python diagnose_model.py models/ppo_poker_final
    python diagnose_model.py models/ppo_mixed_league_final
"""

from __future__ import annotations
import sys
import numpy as np
import torch
from stable_baselines3 import PPO

from poker_env.game import GameState, PlayerState, Street
from poker_env.agents.base import build_observation

# ---------------------------------------------------------------------------
# Synthetic observation builder for a specific BTN RFI spot
# ---------------------------------------------------------------------------

def make_btn_rfi_obs(card1: int, card2: int, n_players: int = 6, hero_seat: int = 0,
                     button_seat: int = 0, bb: int = 10) -> np.ndarray:
    """
    Build an 86-dim observation for the BTN RFI spot with given hole cards.
    Hole cards represented as compact features only (no one-hot).
    """
    obs = np.zeros(86, dtype=np.float32)
    NORM = bb * 100

    obs[0] = 0.0          # street = PREFLOP
    obs[1] = 15 / NORM    # pot = SB+BB
    obs[2] = bb / NORM    # current_bet = BB
    obs[3] = 1000 / NORM  # hero stack
    obs[4] = 0.0          # hero bet_street
    obs[5] = 0.0          # position = button

    # Compact hand features [6:10]
    r0, r1 = card1 // 4, card2 // 4
    s0, s1 = card1 % 4,  card2 % 4
    high_rank = max(r0, r1)
    low_rank  = min(r0, r1)
    obs[6] = high_rank / 12.0
    obs[7] = low_rank  / 12.0
    obs[8] = 1.0 if (s0 == s1 and r0 != r1) else 0.0
    obs[9] = 1.0 if r0 == r1 else 0.0

    # Community cards [10:62] all zero (preflop)

    # Opponents [62:86]
    for i in range(5):
        obs[62 + i] = 1000 / NORM
        obs[68 + i] = (5 if i == 0 else 10 if i == 1 else 0) / NORM
        obs[74 + i] = 0.0
        obs[80 + i] = 0.0

    return obs


def card(rank: int, suit: int) -> int:
    """rank: 0=2 .. 12=A, suit: 0-3 → card index 0-51"""
    return rank * 4 + suit


# Representative hands: (name, card1, card2)
SAMPLE_HANDS = [
    # Premium: GTO says always raise
    ("AA",   card(12, 0), card(12, 1)),
    ("KK",   card(11, 0), card(11, 1)),
    ("QQ",   card(10, 0), card(10, 1)),
    ("AKs",  card(12, 0), card(11, 0)),
    ("AKo",  card(12, 0), card(11, 1)),
    # Strong: GTO says usually raise
    ("JJ",   card(9, 0),  card(9, 1)),
    ("TT",   card(8, 0),  card(8, 1)),
    ("AQs",  card(12, 0), card(10, 0)),
    ("AJs",  card(12, 0), card(9, 0)),
    ("KQs",  card(11, 0), card(10, 0)),
    # Medium: GTO is mixed
    ("55",   card(3, 0),  card(3, 1)),
    ("33",   card(1, 0),  card(1, 1)),
    ("A5s",  card(12, 0), card(3, 0)),
    ("87s",  card(6, 0),  card(5, 0)),
    ("KJo",  card(11, 0), card(9, 1)),
    # Weak: GTO says fold
    ("72o",  card(5, 0),  card(0, 1)),
    ("32o",  card(1, 0),  card(0, 1)),
    ("85o",  card(6, 0),  card(3, 1)),
    ("J3o",  card(9, 0),  card(1, 1)),
    ("92o",  card(7, 0),  card(0, 1)),
]


def diagnose(model_path: str):
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    print(f"\n{'Hand':<8} {'Fold%':>6} {'Call%':>6} {'Raise%':>7} {'AI%':>5}  {'Verdict'}")
    print("-" * 55)

    results = []
    for name, c1, c2 in SAMPLE_HANDS:
        obs = make_btn_rfi_obs(c1, c2)
        device = next(model.policy.parameters()).device
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        # Get action probabilities from policy network
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze().cpu().numpy()

        fold_p  = probs[0]
        call_p  = probs[1]
        raise_p = probs[2] + probs[3]  # min + pot raise
        ai_p    = probs[4]
        total_raise = raise_p + ai_p

        verdict = ""
        if name in ("AA", "KK", "QQ", "AKs", "AKo"):
            verdict = "✓ premium" if total_raise > 0.6 else "✗ SHOULD RAISE"
        elif name in ("72o", "32o", "85o", "J3o", "92o"):
            verdict = "✓ trash fold" if total_raise < 0.3 else "✗ SHOULD FOLD"
        else:
            verdict = "mixed"

        results.append(total_raise)
        print(f"{name:<8} {fold_p*100:>5.1f}% {call_p*100:>5.1f}% {raise_p*100:>6.1f}% {ai_p*100:>4.1f}%  {verdict}")

    std = np.std(results)
    mean = np.mean(results)
    print(f"\n  Mean raise%: {mean*100:.1f}%   Std dev: {std*100:.1f}%")
    if std < 0.05:
        print("  *** DIAGNOSIS: Uniform policy — no hand discrimination ***")
        print("  The model raises the same % regardless of hole cards.")
    elif std < 0.10:
        print("  *** DIAGNOSIS: Weak discrimination — barely hand-conditional ***")
    else:
        print("  *** DIAGNOSIS: Hand-conditional policy — model differentiates hands ***")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        models = [
            "models/ppo_poker_final",
            "models/ppo_threebet_final",
            "models/best_model",
        ]
        for m in models:
            import os
            if os.path.exists(m + ".zip") or os.path.exists(m):
                diagnose(m)
    else:
        diagnose(sys.argv[1])
