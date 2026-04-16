"""
Run a 6-player cash game session with different baseline agents and print results.

Usage:
    cd poker-env
    python examples/run_game.py
"""

import sys
sys.path.insert(0, ".")

from poker_env import (
    MultiAgentRunner,
    RandomAgent, CallAgent, HeuristicAgent,
)

AGENT_NAMES = [
    "HeuristicAgg",
    "HeuristicPass",
    "CallAgent-A",
    "Random-42",
    "Random-43",
    "CallAgent-B",
]


def main():
    agents = [
        HeuristicAgent(seat=0, aggression=0.7, seed=0),
        HeuristicAgent(seat=1, aggression=0.3, seed=1),
        CallAgent(seat=2),
        RandomAgent(seat=3, seed=42),
        RandomAgent(seat=4, seed=43),
        CallAgent(seat=5),
    ]

    runner = MultiAgentRunner(
        agents=agents,
        n_players=6,
        starting_stack=1000,
        sb=5,
        bb=10,
        seed=0,
    )

    n_hands = 2000
    print(f"Running {n_hands}-hand session...\n")
    result = runner.run(n_hands=n_hands, verbose=False)
    print(result.summary())

    print("\nBB/100 by agent:")
    bb100s = result.bb_per_100()
    max_abs = max(abs(x) for x in bb100s) or 1
    bar_width = 40

    for name, bb100 in zip(AGENT_NAMES, bb100s):
        bar_len = int(abs(bb100) / max_abs * bar_width)
        bar = "█" * bar_len
        sign = "+" if bb100 >= 0 else "-"
        print(f"  {name:<15} {sign}{abs(bb100):6.1f}  {bar}")


if __name__ == "__main__":
    main()
