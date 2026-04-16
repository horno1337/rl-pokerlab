# rl-poker

No-Limit Texas Hold'em environment for training and benchmarking reinforcement learning agents.

6-player cash game with persistent stacks, auto-rebuys, and a Gymnasium-compatible interface for plugging in PPO or any other RL algorithm.

---

## Structure

```
poker_env/
├── card.py          # Card and Deck primitives
├── hand_eval.py     # 7-card hand evaluator, side-pot-aware winner detection
├── game.py          # Core game engine — blinds, betting, streets, side pots, rebuys
├── env.py           # PokerEnv (Gymnasium) + MultiAgentRunner (tournament)
└── agents/
    ├── base.py      # BaseAgent ABC + 134-dim observation builder
    ├── baselines.py # RandomAgent, CallAgent, HeuristicAgent, HumanAgent
    └── ppo_agent.py # PPOAgent — wraps a trained SB3 model for evaluation
notebooks/
└── btn_rfi_analysis.ipynb  # Compares PPO raise frequency to GTO BTN RFI range
examples/
└── run_game.py      # 6-player baseline tournament with BB/100 reporting
train_ppo.py         # PPO training script with checkpointing and evaluation
```

---

## Setup

```bash
uv add gymnasium numpy pytest
uv add stable-baselines3 torch tensorboard  # for RL training
```

Run the baseline tournament:

```bash
uv run python examples/run_game.py
```

Run tests:

```bash
uv run python -m pytest tests/
```

---

## Training a PPO agent

```bash
uv run python train_ppo.py --timesteps 1_000_000
```

Options:

```
--timesteps   Total environment steps (default: 500_000)
--n-envs      Parallel environments (default: 4)
--eval-only   Skip training, evaluate existing model
--model       Path to model (default: models/ppo_poker_final)
```

Monitor training:

```bash
uv run tensorboard --logdir logs/tensorboard
```

---

## Plugging in your own agent

Implement `BaseAgent` and override `act()`:

```python
from poker_env.agents.base import BaseAgent
from poker_env.game import GameState, Action, ActionType

class MyAgent(BaseAgent):
    def act(self, state: GameState) -> Action:
        # state gives you: hole cards, community cards, pot,
        # stacks, current bet, position, action history
        return Action(ActionType.CALL)
```

Run it in a tournament:

```python
from poker_env import MultiAgentRunner
from poker_env.agents.baselines import RandomAgent

agents = [MyAgent(seat=0)] + [RandomAgent(i) for i in range(1, 6)]
runner = MultiAgentRunner(agents, n_players=6)
result = runner.run(n_hands=1000)
print(result.summary())
```

---

## Gymnasium interface (for RL training)

```python
from poker_env import PokerEnv
from poker_env.agents.baselines import RandomAgent
from stable_baselines3 import PPO

opponents = [RandomAgent(i) for i in range(1, 6)]
env = PokerEnv(
    opponents=opponents,
    hero_seat=0,
    n_players=6,
    starting_stack=1000,   # chips (100BB at bb=10)
    sb=5, bb=10,
    hands_per_episode=200, # hands before episode resets
    max_reward_bb=500.0,   # reward clipping
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("models/my_agent")
```

Observation space: 134-dim float32 vector covering street, pot, stacks, hole cards (one-hot), community cards (one-hot), and opponent status.

Action space: `Discrete(5)` — fold, check/call, raise min, raise pot, all-in.

Reward: net chips won/lost per hand, normalised by BB, clipped to `±max_reward_bb`.

---

## GTO comparison notebook


---

## Game rules

- 6-player No-Limit Texas Hold'em cash game
- Fixed blinds: SB=5, BB=10 (configurable)
- Starting stack: 1000 chips (100BB)
- Auto-rebuy: players below 40BB are topped up to 100BB between hands
- Side pots handled correctly for all-in situations
- Stacks persist across hands within a session

---
