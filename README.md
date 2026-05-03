# Blackjack Reasoning Tutor

A terminal Blackjack game where four AG2 Swarm agents debate the rationality of your move before you act.

## Agents

| Agent | Role | Tool |
|---|---|---|
| **Dealer_Agent** | Game master. Manages all game state transitions. | `place_bet`, `deal_cards`, `resolve_action`, `settle_round` |
| **Math_Oracle** | Pure logic. Calculates Bust% and EV via Monte Carlo (200 trials). | `run_math_analysis` |
| **Safe_Player** | Risk-averse veteran. Cites Bust% to argue for caution. | `record_safe_advice` |
| **The_Shark** | Aggressive EV-chaser. Reads Safe_Player's advice and rebuts it. | `record_shark_advice` |

## Swarm Flow

```
Dealer (betting) ──► REVERT_TO_USER        ← player types bet
Dealer calls place_bet → deal_cards
OnContextCondition ──► Math_Oracle          ← phase=advisory, math_done=False
Math_Oracle prints JSON analysis
Math_Oracle ──► Safe_Player ──► The_Shark ──► Dealer
Dealer summarises debate ──► REVERT_TO_USER ← player types HIT / STAND / DOUBLE
  HIT (no bust) → reset flags, re-run advisory cycle for new hand state
  bust / STAND / DOUBLE → Dealer calls settle_round
Dealer ──► REVERT_TO_USER                   ← next round or quit
```

`context_variables` is the single source of truth. All agents read live state via `UpdateSystemMessage` callables injected before each reply.

## Setup

**Prerequisites:** Python 3.11+, an [OpenRouter](https://openrouter.ai) API key.

```bash
cd agent-py
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `agent-py/.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
# Optional — override the default model:
# MODEL=google/gemini-2.5-flash
```

## Run

```bash
cd agent-py
source .venv/bin/activate
python blackjack_tutor.py
```

## Gameplay

| Input | Action |
|---|---|
| A number (e.g. `200`) | Place that bet to start a round |
| `HIT` | Draw one card; advisory panel re-runs on the new hand |
| `STAND` | Dealer plays out; round settles |
| `DOUBLE` | Double bet, draw one card, dealer plays out |
| `quit` / `exit` | Print session summary and stop |

## Files

```
agent-py/
├── blackjack_tutor.py   # AG2 Swarm entry point
├── engine.py            # Stateless BlackjackEngine (no AG2 imports)
├── requirements.txt
└── .env                 # OPENROUTER_API_KEY (not committed)
```
