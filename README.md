# rl-pokerlab

No-Limit Texas Hold'em environment for training and benchmarking RL agents. 6-player cash game with persistent stacks, auto-rebuys, and a Gymnasium-compatible interface.

## Setup

```bash
uv add gymnasium numpy pytest
uv add stable-baselines3 torch tensorboard
```

## Usage

Baseline tournament:

```bash
uv run python examples/run_game.py
```

Train a PPO agent:

```bash
uv run python train_ppo.py --timesteps 1_000_000
```

Tests:

```bash
uv run pytest tests/
```

## Interface

```python
from poker_env import PokerEnv
from poker_env.agents.baselines import RandomAgent
from stable_baselines3 import PPO

env = PokerEnv(
    opponents=[RandomAgent(i) for i in range(1, 6)],
    hero_seat=0,
    n_players=6,
    starting_stack=1000,
    sb=5, bb=10,
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

- **Observation:** `Box(134,)` — street, pot, stacks, hole/board cards, opponent status
- **Action:** `Discrete(5)` — fold, check/call, min-raise, pot-raise, all-in
- **Reward:** net chips per hand, normalised by BB

## Custom agents

```python
from poker_env.agents.base import BaseAgent
from poker_env.game import Action, ActionType

class MyAgent(BaseAgent):
    def act(self, state):
        return Action(ActionType.CALL)
```

## Structure

```
poker_env/
├── card.py        # Card and Deck primitives
├── hand_eval.py   # 7-card evaluator, side-pot-aware winner detection
├── game.py        # engine: blinds, betting, streets, side pots, rebuys
├── env.py         # PokerEnv (Gymnasium) + MultiAgentRunner
└── agents/        # BaseAgent + baselines + PPO wrapper
notebooks/         # GTO comparison
examples/          # baseline tournament
tests/             # pytest suite
train_ppo.py       # training entry point
hand_probe.py      # evaluator debugging utility
```

## curent limitations

5-action discrete space, no action history in observations, PPO trained against fixed baselines (no self-play yet).
