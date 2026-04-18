"""
Supervised pre-training of BTN preflop opening ranges.

Teaches the policy network GTO hand selection BEFORE RL training begins,
so PPO starts from a reasonable prior instead of having to discover it
from sparse, noisy chip outcomes.

Pipeline:
  1. Build synthetic BTN RFI observations for all 169 hand types
  2. Use GTO raise frequencies as soft targets
  3. Train the policy head with cross-entropy loss
  4. Save the pre-trained model  → used as --base-model for train_ppo.py

Usage:
    python pretrain_preflop.py                    # saves models/ppo_pretrained.zip
    python pretrain_preflop.py --epochs 2000
    python pretrain_preflop.py --out models/ppo_pretrained
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from poker_env.env import PokerEnv, _GTO_BTN_RFI
from poker_env.agents.baselines import CallAgent, HeuristicAgent
from train_ppo import make_env


# ---------------------------------------------------------------------------
# GTO table (copied from env.py for reference)
# ---------------------------------------------------------------------------

def gto_freq(rank_high: int, rank_low: int, suited: bool) -> float:
    """Return the GTO BTN RFI raise frequency for a given hand."""
    return _GTO_BTN_RFI.get((rank_high, rank_low, suited), 0.0)


# ---------------------------------------------------------------------------
# Synthetic observation builder
# ---------------------------------------------------------------------------

def build_btn_rfi_obs(rank_high: int, rank_low: int, suited: bool,
                      bb: int = 10, stack: int = 1000) -> np.ndarray:
    """
    Build an 86-dim BTN RFI observation for (rank_high, rank_low, suited).
    Ranks: 0=2, 12=A.
    """
    from poker_env.agents.base import OBS_DIM
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    NORM = bb * 100

    obs[0] = 0.0                  # PREFLOP
    obs[1] = (bb * 1.5) / NORM    # pot = SB+BB
    obs[2] = bb / NORM            # current_bet = BB
    obs[3] = stack / NORM         # hero stack
    obs[4] = 0.0                  # hero bet_street
    obs[5] = 0.0                  # position = button

    obs[6] = rank_high / 12.0
    obs[7] = rank_low  / 12.0
    obs[8] = 1.0 if suited else 0.0
    obs[9] = 1.0 if (rank_high == rank_low) else 0.0

    # Community cards [10:62] — all zero (preflop)

    # Opponents [62:86] — 5 active players, standard stacks
    for i in range(5):
        obs[62 + i] = stack / NORM
        obs[68 + i] = (bb // 2 if i == 0 else bb if i == 1 else 0) / NORM
        obs[74 + i] = 0.0   # not folded
        obs[80 + i] = 0.0   # not all-in

    return obs


def build_dataset():
    """
    Build all 169 canonical (hand → GTO_freq) pairs.

    Returns:
        obs_batch:   (169, OBS_DIM) float32
        target_freq: (169,) float32  — GTO raise frequency [0, 1]
    """
    obs_list, freq_list = [], []

    for rh in range(13):
        for rl in range(rh + 1):          # rh >= rl
            if rh == rl:                  # pocket pair
                obs_list.append(build_btn_rfi_obs(rh, rl, suited=False))
                freq_list.append(gto_freq(rh, rl, False))
            else:
                # suited
                obs_list.append(build_btn_rfi_obs(rh, rl, suited=True))
                freq_list.append(gto_freq(rh, rl, True))
                # offsuit
                obs_list.append(build_btn_rfi_obs(rh, rl, suited=False))
                freq_list.append(gto_freq(rh, rl, False))

    obs_batch  = np.array(obs_list,  dtype=np.float32)   # (169, OBS_DIM)
    target_freq = np.array(freq_list, dtype=np.float32)   # (169,)
    return obs_batch, target_freq


# ---------------------------------------------------------------------------
# Supervised training loop
# ---------------------------------------------------------------------------

def pretrain(out_path: str = "models/ppo_pretrained", epochs: int = 3000,
             lr: float = 1e-3, verbose: bool = True):
    """
    Pre-train the PPO policy to output GTO-aligned action probabilities
    for BTN RFI spots.

    Loss: cross-entropy between policy distribution and soft target
          [fold, call, raise_min, raise_pot, all_in] where raise actions
          collectively get weight = gto_freq, and fold/call split the rest.

    Using soft targets (not hard argmax) preserves mixed strategies —
    e.g. 22 has GTO freq 0.5 so the target is 50% raise / 50% fold.
    """
    import os
    os.makedirs("models", exist_ok=True)

    # Build a dummy env just to get the right obs/action space for PPO
    env = make_env(seed=0)
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")

    obs_np, freq_np = build_dataset()
    n = len(obs_np)

    device = next(model.policy.parameters()).device
    obs_t   = torch.FloatTensor(obs_np).to(device)   # (n, obs_dim)
    freq_t  = torch.FloatTensor(freq_np).to(device)  # (n,)

    # Soft target distribution over 5 actions:
    #   raise_min (2) + raise_pot (3) + all_in (4) get (gto_freq / 3) each
    #   fold (0) gets (1 - gto_freq) * 0.4
    #   call (1) gets (1 - gto_freq) * 0.6
    fold_w   = (1.0 - freq_t) * 0.4
    call_w   = (1.0 - freq_t) * 0.6
    raise_w  = freq_t / 3.0

    # target: (n, 5)
    target = torch.stack([fold_w, call_w, raise_w, raise_w, raise_w], dim=1)
    target = torch.clamp(target, min=1e-6)
    target = target / target.sum(dim=1, keepdim=True)   # normalize to sum=1

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        dist = model.policy.get_distribution(obs_t)
        log_probs = dist.distribution.logits   # (n, 5) — unnormalized logits
        # Cross-entropy: -sum(target * log_softmax(logits))
        log_softmax = F.log_softmax(log_probs, dim=1)
        loss = -(target * log_softmax).sum(dim=1).mean()

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if verbose and epoch % 200 == 0:
            # Quick accuracy check: does the argmax action match GTO intent?
            with torch.no_grad():
                probs = dist.distribution.probs  # (n, 5)
                raise_prob = (probs[:, 2] + probs[:, 3] + probs[:, 4]).cpu().numpy()
            correct = np.sum(
                (raise_prob > 0.5) == (freq_np > 0.5)
            )
            print(f"Epoch {epoch:>5}  loss={loss.item():.4f}  "
                  f"directional accuracy={correct}/{n} "
                  f"({correct/n*100:.0f}%)")

    model.save(out_path)
    print(f"\nPre-trained model saved → {out_path}.zip")
    print(f"Best loss: {best_loss:.4f}")

    # Quick verification
    print("\nSample BTN raise probabilities after pre-training:")
    print(f"{'Hand':<8} {'Raise%':>7}  {'GTO%':>6}")
    print("-" * 26)
    check_hands = [
        ("AA",  12, 12, False),
        ("AKs", 12, 11, True),
        ("AKo", 12, 11, False),
        ("55",   3,  3, False),
        ("87s",  6,  5, True),
        ("72o",  5,  0, False),
        ("32o",  1,  0, False),
    ]
    with torch.no_grad():
        for name, rh, rl, s in check_hands:
            obs = torch.FloatTensor(build_btn_rfi_obs(rh, rl, s)).unsqueeze(0).to(device)
            dist = model.policy.get_distribution(obs)
            p = dist.distribution.probs.squeeze().cpu().numpy()
            r = p[2] + p[3] + p[4]
            g = gto_freq(rh, rl, s)
            print(f"{name:<8} {r*100:>6.1f}%  {g*100:>5.0f}%")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--out",    type=str, default="models/ppo_pretrained")
    args = parser.parse_args()

    pretrain(out_path=args.out, epochs=args.epochs, lr=args.lr)
