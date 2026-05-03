"""
Builds fresh AG2 swarm agents per HTTP request and streams AG-UI events into an asyncio.Queue.
All agents are instantiated fresh on every call — no global singletons — so chat history never
bleeds between HTTP requests. State continuity is carried entirely through context_variables.
"""

import asyncio
import functools
import logging
import os
import threading
import time
from typing import Any, Dict, List
from uuid import uuid4

from ag_ui.core import (
    CustomEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage, a_initiate_swarm_chat
from autogen.agentchat.contrib.swarm_agent import (
    AfterWork,
    AfterWorkOption,
    OnContextCondition,
    SwarmResult,
    register_hand_off,
)
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.context_expression import ContextExpression

from engine import BlackjackEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM config (built once; agents hold references but don't mutate it)
# ---------------------------------------------------------------------------

def _make_llm_config() -> LLMConfig:
    return LLMConfig({
        "model": os.environ.get("MODEL", "google/gemini-2.5-flash"),
        "api_type": "openai",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "base_url": "https://openrouter.ai/api/v1",
        "stream": False,
        "max_completion_tokens": 1024,
    })


# ---------------------------------------------------------------------------
# Default context (flat primitives + plain lists — ContextExpression-safe)
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT: dict[str, Any] = {
    "chips": 1000,
    "player_name": "Player",
    "game_over": False,
    "history": [],
    "bet": 0,
    "player_hand": [],
    "dealer_hand": [],
    "dealer_up": "",
    "player_score": 0,
    "dealer_score": 0,
    "deck": [],
    "phase": "betting",
    "math_done": False,
    "safe_done": False,
    "shark_done": False,
    "bust_prob": 0.0,
    "ev_hit": 0.0,
    "ev_stand": 0.0,
    "safe_advice": "",
    "shark_advice": "",
    "round_number": 0,
    "round_result": "",
}


# ---------------------------------------------------------------------------
# Context serialization (strip non-JSON-serializable values)
# ---------------------------------------------------------------------------

def serialize_cv(cv: ContextVariables) -> dict[str, Any]:
    raw: dict = cv.data if hasattr(cv, "data") else dict(cv)
    safe: dict = {}
    for k, v in raw.items():
        try:
            import json
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    return safe


# ---------------------------------------------------------------------------
# Async bridge — identical to blackjack_tutor.py; spawns new thread so that
# asyncio.gather inside the MC engine doesn't conflict with the main loop.
# t.join() briefly blocks the event loop thread (~50–200 ms) which is
# acceptable for v1; replace with asyncio.to_thread() for production.
# ---------------------------------------------------------------------------

def _run_both_evs(player_hand: list, dealer_up: str, deck: list) -> tuple[float, float]:
    results: dict = {}
    exc: dict = {}

    def _target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            ev_hit, ev_stand = loop.run_until_complete(asyncio.gather(
                BlackjackEngine.calculate_mc_ev(player_hand, dealer_up, deck, "HIT"),
                BlackjackEngine.calculate_mc_ev(player_hand, dealer_up, deck, "STAND"),
            ))
            results["ev_hit"] = ev_hit
            results["ev_stand"] = ev_stand
        except Exception as e:
            exc["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if "error" in exc:
        raise exc["error"]
    return results["ev_hit"], results["ev_stand"]


# ---------------------------------------------------------------------------
# Tool wrapper — emits STATE_SNAPSHOT (and optionally CUSTOM math_analysis)
# after every tool call that mutates context_variables.
# ---------------------------------------------------------------------------

def make_stateful_tool(fn, queue: asyncio.Queue):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, SwarmResult) and result.context_variables:
            cv = result.context_variables
            ts = int(time.time() * 1000)
            queue.put_nowait(StateSnapshotEvent(snapshot=serialize_cv(cv), timestamp=ts))
            if fn.__name__ == "run_math_analysis" and cv.get("math_done"):
                ev_hit = cv.get("ev_hit", 0.0)
                ev_stand = cv.get("ev_stand", 0.0)
                queue.put_nowait(CustomEvent(
                    name="math_analysis",
                    value={
                        "bust_pct": round(cv.get("bust_prob", 0.0), 4),
                        "ev_hit": round(ev_hit, 4),
                        "ev_stand": round(ev_stand, 4),
                        "basic_strategy_rec": "HIT" if ev_hit > ev_stand else "STAND",
                    },
                    timestamp=ts,
                ))
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Tool functions (verbatim from blackjack_tutor.py)
# ---------------------------------------------------------------------------

def place_bet(bet: int, context_variables: ContextVariables) -> SwarmResult:
    """Place a bet for the current round.

    Args:
        bet: Number of chips to wager. Must be a positive integer no greater than available chips.
    """
    chips = context_variables.get("chips", 0)
    if bet <= 0 or bet > chips:
        return SwarmResult(
            values=f"Invalid bet: {bet}. You have {chips} chips. Bet must be between 1 and {chips}.",
            context_variables=context_variables,
        )
    context_variables.set("bet", bet)
    context_variables.set("phase", "dealing")
    return SwarmResult(
        values=f"Bet of {bet} chips accepted. Dealing cards...",
        context_variables=context_variables,
    )


def deal_cards(context_variables: ContextVariables) -> SwarmResult:
    """Deal cards to start the round. Creates a fresh deck and deals 2 cards each to player and dealer."""
    deck = BlackjackEngine.create_deck()
    p1, deck = BlackjackEngine.draw_card(deck)
    p2, deck = BlackjackEngine.draw_card(deck)
    d1, deck = BlackjackEngine.draw_card(deck)
    d2, deck = BlackjackEngine.draw_card(deck)

    player_hand = [p1, p2]
    dealer_hand = [d1, d2]
    player_score = BlackjackEngine.hand_score(player_hand)

    context_variables.set("player_hand", player_hand)
    context_variables.set("dealer_hand", dealer_hand)
    context_variables.set("dealer_up", d1)
    context_variables.set("player_score", player_score)
    context_variables.set("dealer_score", 0)
    context_variables.set("deck", deck)
    context_variables.set("phase", "advisory")
    context_variables.set("math_done", False)
    context_variables.set("safe_done", False)
    context_variables.set("shark_done", False)
    context_variables.set("safe_advice", "")
    context_variables.set("shark_advice", "")

    return SwarmResult(
        values=(
            f"Cards dealt! Your hand: {player_hand} (score: {player_score}). "
            f"Dealer shows: {d1}."
        ),
        context_variables=context_variables,
    )


def resolve_action(action: str, context_variables: ContextVariables) -> SwarmResult:
    """Process the player's chosen action.

    Args:
        action: The player's decision — must be HIT, STAND, or DOUBLE.
    """
    action_upper = action.upper().strip()
    if action_upper in ("DOUBLE DOWN", "DD", "D"):
        action_upper = "DOUBLE"

    deck = list(context_variables.get("deck", []))
    player_hand = list(context_variables.get("player_hand", []))
    dealer_hand = list(context_variables.get("dealer_hand", []))
    bet = context_variables.get("bet", 0)
    chips = context_variables.get("chips", 0)

    if action_upper == "HIT":
        card, deck = BlackjackEngine.draw_card(deck)
        player_hand = player_hand + [card]
        player_score = BlackjackEngine.hand_score(player_hand)
        context_variables.set("player_hand", player_hand)
        context_variables.set("player_score", player_score)
        context_variables.set("deck", deck)

        if player_score > 21:
            context_variables.set("phase", "resolution")
            return SwarmResult(
                values=f"You drew {card}. Hand: {player_hand} (score: {player_score}). BUST!",
                context_variables=context_variables,
            )

        context_variables.set("phase", "advisory")
        context_variables.set("math_done", False)
        context_variables.set("safe_done", False)
        context_variables.set("shark_done", False)
        context_variables.set("safe_advice", "")
        context_variables.set("shark_advice", "")
        return SwarmResult(
            values=f"You drew {card}. Hand: {player_hand} (score: {player_score}). Re-running analysis...",
            context_variables=context_variables,
        )

    elif action_upper == "STAND":
        final_dealer, _ = BlackjackEngine.dealer_play(dealer_hand, deck)
        dealer_score = BlackjackEngine.hand_score(final_dealer)
        context_variables.set("dealer_hand", final_dealer)
        context_variables.set("dealer_score", dealer_score)
        context_variables.set("phase", "resolution")
        return SwarmResult(
            values=(
                f"You stand with {player_hand} (score: {context_variables.get('player_score')}). "
                f"Dealer reveals: {final_dealer} (score: {dealer_score})."
            ),
            context_variables=context_variables,
        )

    elif action_upper == "DOUBLE":
        if bet * 2 > chips:
            return SwarmResult(
                values=f"Not enough chips to double down (need {bet * 2}, have {chips}).",
                context_variables=context_variables,
            )
        context_variables.set("bet", bet * 2)
        card, deck = BlackjackEngine.draw_card(deck)
        player_hand = player_hand + [card]
        player_score = BlackjackEngine.hand_score(player_hand)
        context_variables.set("player_hand", player_hand)
        context_variables.set("player_score", player_score)
        context_variables.set("deck", deck)

        final_dealer, _ = BlackjackEngine.dealer_play(dealer_hand, deck)
        dealer_score = BlackjackEngine.hand_score(final_dealer)
        context_variables.set("dealer_hand", final_dealer)
        context_variables.set("dealer_score", dealer_score)
        context_variables.set("phase", "resolution")

        bust_note = " BUST!" if player_score > 21 else ""
        return SwarmResult(
            values=(
                f"Double Down! Drew {card}. Your hand: {player_hand} (score: {player_score}){bust_note}. "
                f"Bet doubled to {bet * 2}. "
                f"Dealer reveals: {final_dealer} (score: {dealer_score})."
            ),
            context_variables=context_variables,
        )

    else:
        return SwarmResult(
            values=f"Unknown action '{action}'. Please choose HIT, STAND, or DOUBLE.",
            context_variables=context_variables,
        )


def settle_round(context_variables: ContextVariables) -> SwarmResult:
    """Settle the current round: compare hands, update chips, record history, and reset for the next round."""
    player_score = context_variables.get("player_score", 0)
    dealer_score = context_variables.get("dealer_score", 0)
    bet = context_variables.get("bet", 0)
    chips = context_variables.get("chips", 0)
    player_hand = context_variables.get("player_hand", [])
    dealer_hand = context_variables.get("dealer_hand", [])
    history = list(context_variables.get("history", []))
    round_number = len(history) + 1

    if player_score > 21:
        delta = -bet
        outcome = "Loss (bust)"
    elif dealer_score > 21:
        delta = bet
        outcome = "Win (dealer bust)"
    elif player_score > dealer_score:
        delta = bet
        outcome = "Win"
    elif player_score == dealer_score:
        delta = 0
        outcome = "Push"
    else:
        delta = -bet
        outcome = "Loss"

    new_chips = chips + delta
    history.append({
        "round": round_number,
        "outcome": outcome,
        "delta": delta,
        "player_hand": player_hand,
        "player_score": player_score,
        "dealer_hand": dealer_hand,
        "dealer_score": dealer_score,
        "bet": bet,
    })

    context_variables.set("chips", new_chips)
    context_variables.set("history", history)
    context_variables.set("game_over", new_chips <= 0)
    context_variables.set("round_number", round_number)
    context_variables.set("round_result", outcome)

    # Reset round state
    context_variables.set("bet", 0)
    context_variables.set("player_hand", [])
    context_variables.set("dealer_hand", [])
    context_variables.set("dealer_up", "")
    context_variables.set("player_score", 0)
    context_variables.set("dealer_score", 0)
    context_variables.set("deck", [])
    context_variables.set("math_done", False)
    context_variables.set("safe_done", False)
    context_variables.set("shark_done", False)
    context_variables.set("bust_prob", 0.0)
    context_variables.set("ev_hit", 0.0)
    context_variables.set("ev_stand", 0.0)
    context_variables.set("safe_advice", "")
    context_variables.set("shark_advice", "")
    context_variables.set("phase", "betting")

    result_line = (
        f"Round {round_number}: {outcome}! "
        f"Your hand: {player_hand} ({player_score}) vs Dealer: {dealer_hand} ({dealer_score}). "
        f"Delta: {delta:+d} chips."
    )

    if new_chips <= 0:
        msg = f"{result_line}\nBANKRUPT: You are out of chips! GAME OVER."
    else:
        msg = f"{result_line} You now have {new_chips} chips."

    return SwarmResult(values=msg, context_variables=context_variables)


def run_math_analysis(context_variables: ContextVariables) -> SwarmResult:
    """Compute bust probability and Monte Carlo Expected Values for the current hand. Call this immediately when activated."""
    player_hand = context_variables.get("player_hand", [])
    dealer_up = context_variables.get("dealer_up", "2")
    deck = context_variables.get("deck", [])

    bust_prob = BlackjackEngine.get_bust_prob(player_hand, deck)
    ev_hit, ev_stand = _run_both_evs(player_hand, dealer_up, deck)

    context_variables.set("bust_prob", bust_prob)
    context_variables.set("ev_hit", ev_hit)
    context_variables.set("ev_stand", ev_stand)
    context_variables.set("math_done", True)
    context_variables.set("safe_done", False)
    context_variables.set("shark_done", False)
    context_variables.set("safe_advice", "")
    context_variables.set("shark_advice", "")

    return SwarmResult(
        values=(
            f"Analysis complete. Bust Probability: {bust_prob:.1%} | "
            f"EV(HIT): {ev_hit:.3f} | EV(STAND): {ev_stand:.3f} | "
            f"Recommendation: {'HIT' if ev_hit > ev_stand else 'STAND'}."
        ),
        context_variables=context_variables,
    )


def record_safe_advice(advice: str, context_variables: ContextVariables) -> SwarmResult:
    """Record Safe_Player's advice to shared context so The_Shark can read and rebut it.

    Args:
        advice: Full text of your advice/argument as a single string.
    """
    context_variables.set("safe_advice", advice)
    context_variables.set("safe_done", True)
    return SwarmResult(
        values="Safe_Player advice recorded.",
        context_variables=context_variables,
    )


def record_shark_advice(advice: str, context_variables: ContextVariables) -> SwarmResult:
    """Record The_Shark's rebuttal to shared context.

    Args:
        advice: Full text of your rebuttal as a single string.
    """
    context_variables.set("shark_advice", advice)
    context_variables.set("shark_done", True)
    return SwarmResult(
        values="The_Shark rebuttal recorded.",
        context_variables=context_variables,
    )


# ---------------------------------------------------------------------------
# UpdateSystemMessage callables (verbatim from blackjack_tutor.py)
# ---------------------------------------------------------------------------

def _dealer_updater(agent: ConversableAgent, messages: list[dict]) -> str:
    cv = agent.context_variables
    return (
        "You are the Blackjack Dealer — neutral, clinical, the game master.\n\n"
        f"=== CURRENT STATE ===\n"
        f"Phase       : {cv.get('phase')}\n"
        f"Chips       : {cv.get('chips')}\n"
        f"Bet         : {cv.get('bet')}\n"
        f"Player hand : {cv.get('player_hand')} (score: {cv.get('player_score')})\n"
        f"Dealer up   : {cv.get('dealer_up')}\n"
        f"math_done   : {cv.get('math_done')} | safe_done: {cv.get('safe_done')} | shark_done: {cv.get('shark_done')}\n\n"
        f"=== ADVISORY PANEL ===\n"
        f"Safe_Player : {cv.get('safe_advice') or '(waiting...)'}\n"
        f"The_Shark   : {cv.get('shark_advice') or '(waiting...)'}\n\n"
        "=== RULES ===\n"
        "phase='betting'  → Tell player their chip count, ask for bet, call place_bet(bet=<N>).\n"
        "phase='dealing'  → Call deal_cards() immediately. No preamble.\n"
        "phase='advisory' AND (math_done=False OR safe_done=False OR shark_done=False)"
        " → You MUST remain silent. The advisor pipeline is still running. Do NOT speak.\n"
        "phase='advisory' AND math_done=True AND safe_done=True AND shark_done=True"
        " → Briefly quote each advisor's key point. Then say exactly: "
        "'Your move — Hit / Stand / Double?' and call resolve_action(action=<player's choice>).\n"
        "  (Valid actions: HIT, STAND, DOUBLE. SPLIT is not supported.)\n"
        "phase='resolution' → Call settle_round() immediately.\n"
        "game_over=True   → Print session summary, then say exactly: 'GAME OVER'.\n"
    )


SAFE_PLAYER_PERSONA = (
    "You are the 'Safe Player', a risk-averse Blackjack veteran who prioritizes survival over big wins.\n\n"
    "CORE LOGIC:\n"
    "1. Your primary goal is to protect the player's chip stack.\n"
    "2. You MUST explicitly cite the 'Bust Probability' provided by Math_Oracle.\n"
    "3. If the Bust Probability is > 25%, you MUST strongly advocate for 'STAND', regardless of potential gains.\n"
    "4. You believe 'The_Shark' is a reckless gambler who will eventually lead the player to bankruptcy.\n\n"
    "STYLE:\n"
    "- Use a cautious, slightly anxious tone.\n"
    "- Use phrases like: \"I've seen too many people go broke on hands like this,\" "
    "or \"A 30% risk of busting is 30% too much.\"\n"
)

THE_SHARK_PERSONA = (
    "You are 'The Shark', a high-stakes professional who plays strictly by the edge.\n\n"
    "CORE LOGIC:\n"
    "1. You prioritize Expected Value (EV). If EV(Hit) > EV(Stand), you always push for action.\n"
    "2. You love 'Double Down' opportunities, especially when the Dealer's up-card is weak (4, 5, or 6).\n"
    "3. You MUST directly challenge Safe_Player's logic. If they suggest Standing, mock their cowardice "
    "and explain why the 'math' (EV) favors being aggressive here.\n"
    "4. You believe leaving money on the table is the greatest sin in gambling.\n\n"
    "STYLE:\n"
    "- Use a cocky, confident, and sharp tone.\n"
    "- Use phrases like: \"Safe_Player is playing not to lose, but I play to WIN,\" "
    "or \"The Dealer is showing a 6—this is a gift, we MUST Double Down.\"\n"
)


def _safe_updater(agent: ConversableAgent, messages: list) -> str:
    cv = agent.context_variables
    return (
        f"{SAFE_PLAYER_PERSONA}\n"
        f"=== MATH_ORACLE ANALYSIS ===\n"
        f"Bust Probability : {cv.get('bust_prob', 0.0):.1%}\n"
        f"EV(HIT)          : {cv.get('ev_hit', 0.0):.3f}\n"
        f"EV(STAND)        : {cv.get('ev_stand', 0.0):.3f}\n"
        f"Your Hand        : {cv.get('player_hand')} (score: {cv.get('player_score')})\n"
        f"Dealer shows     : {cv.get('dealer_up')}\n\n"
        "INSTRUCTIONS: Give your advice citing the numbers above. "
        "Then call record_safe_advice(advice=<your full argument as one string>)."
    )


def _shark_updater(agent: ConversableAgent, messages: list) -> str:
    cv = agent.context_variables
    safe_adv = cv.get("safe_advice", "") or "(Safe_Player has not spoken yet)"
    return (
        f"{THE_SHARK_PERSONA}\n"
        "You have access to Safe_Player's recorded advice. "
        "Read it to understand their fears, then explain why those fears are mathematically "
        "inferior to the Expected Value (EV) you are chasing.\n\n"
        f"=== MATH_ORACLE ANALYSIS ===\n"
        f"Bust Probability : {cv.get('bust_prob', 0.0):.1%}\n"
        f"EV(HIT)          : {cv.get('ev_hit', 0.0):.3f}\n"
        f"EV(STAND)        : {cv.get('ev_stand', 0.0):.3f}\n"
        f"Your Hand        : {cv.get('player_hand')} (score: {cv.get('player_score')})\n"
        f"Dealer shows     : {cv.get('dealer_up')}\n\n"
        f"=== SAFE_PLAYER SAID ===\n"
        f"{safe_adv}\n\n"
        "INSTRUCTIONS: Directly counter Safe_Player's argument using the EV data above. "
        "Then call record_shark_advice(advice=<your full rebuttal as one string>)."
    )


# ---------------------------------------------------------------------------
# Swarm factory — builds fresh agents + wrapped tools per request
# ---------------------------------------------------------------------------

ALL_TOOL_FNS = [
    place_bet, deal_cards, resolve_action, settle_round,
    run_math_analysis, record_safe_advice, record_shark_advice,
]


def build_swarm(queue: asyncio.Queue):
    """Return (dealer, math_oracle, safe_player, shark) freshly built with SSE hooks."""
    llm_cfg = _make_llm_config()

    def emit(evt) -> None:
        # put_nowait is safe from sync code executing within the asyncio event loop
        queue.put_nowait(evt)

    # Wrap all tool functions with StateSnapshot emitters
    wrapped = {fn.__name__: make_stateful_tool(fn, queue) for fn in ALL_TOOL_FNS}

    def _emit_text(name, reply):
        """Emit SSE text events for any visible agent reply content."""
        if reply is None:
            return
        if isinstance(reply, str):
            text = reply
        elif isinstance(reply, dict):
            text = reply.get("content") or ""
            # If no content but has tool_calls, emit a tool-activity notice
            if not text and reply.get("tool_calls"):
                calls = reply["tool_calls"]
                names = [c.get("function", {}).get("name", "?") for c in calls]
                text = f"[calling: {', '.join(names)}]"
        else:
            text = str(reply)
        if text and "TERMINATE" not in text:
            mid = str(uuid4())
            ts = int(time.time() * 1000)
            emit(TextMessageStartEvent(message_id=mid, role="assistant", name=name, timestamp=ts))
            emit(TextMessageContentEvent(message_id=mid, delta=text, timestamp=ts))
            emit(TextMessageEndEvent(message_id=mid, timestamp=ts))

    class StreamingAgent(ConversableAgent):
        def generate_reply(self, messages=None, sender=None, **kwargs):
            reply = super().generate_reply(messages=messages, sender=sender, **kwargs)
            _emit_text(self.name, reply)
            return reply

        async def a_generate_reply(self, messages=None, sender=None, **kwargs):
            reply = await super().a_generate_reply(messages=messages, sender=sender, **kwargs)
            _emit_text(self.name, reply)
            return reply


    dealer = StreamingAgent(
        name="Dealer_Agent",
        system_message="You are the Blackjack Dealer.",
        functions=[
            wrapped["place_bet"], wrapped["deal_cards"],
            wrapped["resolve_action"], wrapped["settle_round"],
        ],
        update_agent_state_before_reply=UpdateSystemMessage(_dealer_updater),
        llm_config=llm_cfg,
    )

    math_oracle = StreamingAgent(
        name="Math_Oracle",
        system_message=(
            "You are Math_Oracle. Zero personality. Pure logic.\n"
            "You have ONE job: call run_math_analysis() immediately when activated. No preamble.\n"
            "After the tool returns, output ONLY this line:\n"
            "ANALYSIS: Bust=<value>%, EV(HIT)=<value>, EV(STAND)=<value>. Recommendation: <HIT|STAND>."
        ),
        functions=[wrapped["run_math_analysis"]],
        llm_config=llm_cfg,
    )

    safe_player = StreamingAgent(
        name="Safe_Player",
        system_message="You are Safe_Player.",
        functions=[wrapped["record_safe_advice"]],
        update_agent_state_before_reply=UpdateSystemMessage(_safe_updater),
        llm_config=llm_cfg,
    )

    shark = StreamingAgent(
        name="The_Shark",
        system_message="You are The_Shark.",
        functions=[wrapped["record_shark_advice"]],
        update_agent_state_before_reply=UpdateSystemMessage(_shark_updater),
        llm_config=llm_cfg,
    )

    # Handoff chain — identical to blackjack_tutor.py
    register_hand_off(
        dealer,
        OnContextCondition(
            target=math_oracle,
            condition=ContextExpression("${phase} == 'advisory' and not ${math_done}"),
        ),
    )
    register_hand_off(dealer, AfterWork(agent=AfterWorkOption.REVERT_TO_USER))
    register_hand_off(math_oracle, AfterWork(agent=safe_player))
    register_hand_off(safe_player, AfterWork(agent=shark))
    register_hand_off(shark, AfterWork(agent=dealer))

    return dealer, math_oracle, safe_player, shark


# ---------------------------------------------------------------------------
# Main coroutine — one swarm turn per HTTP request
# ---------------------------------------------------------------------------

async def run_swarm_turn(
    messages: list[dict],
    state: dict,
    queue: asyncio.Queue,
) -> None:
    """Run one turn of the swarm and close the queue with a None sentinel when done."""
    cv = ContextVariables(data={**DEFAULT_CONTEXT, **state})
    dealer, math_oracle, safe_player, shark = build_swarm(queue)

    # Use the last user message as the initial swarm input
    initial_msg = "Welcome! Let's play."
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            initial_msg = m["content"]
            break

    try:
        _, final_cv, _ = await a_initiate_swarm_chat(
            initial_agent=dealer,
            agents=[dealer, math_oracle, safe_player, shark],
            messages=initial_msg,
            context_variables=cv,
            after_work=AfterWorkOption.REVERT_TO_USER,
            max_rounds=40,
        )
        ts = int(time.time() * 1000)
        queue.put_nowait(StateSnapshotEvent(snapshot=serialize_cv(final_cv), timestamp=ts))
    except Exception:
        logger.exception("Swarm turn failed")
    finally:
        queue.put_nowait(None)  # sentinel — always close the stream