import asyncio
import json
import os
import threading

from dotenv import load_dotenv

from autogen import (
    ConversableAgent,
    LLMConfig,
    UpdateSystemMessage,
    UserProxyAgent,
    a_initiate_swarm_chat,
)
from autogen.agentchat.contrib.swarm_agent import (
    AfterWork,
    AfterWorkOption,
    OnContextCondition,
    SwarmResult,
    register_hand_off,
)
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.context_expression import ContextExpression
from autogen.agents.experimental import ReasoningAgent

from engine import BlackjackEngine

load_dotenv()

llm_config=LLMConfig({
    "model": "google/gemini-2.5-flash",
    "api_type": "openai",
    "api_key": "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",
    "base_url": "https://openrouter.ai/api/v1",
    "stream": True,
    "max_completion_tokens": 1024,
}),


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def make_initial_context() -> ContextVariables:
    return ContextVariables(data={
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
    })


# ---------------------------------------------------------------------------
# Async bridge — runs MC coroutines in a new thread's event loop so they
# can be called from sync tool functions that execute inside the swarm's
# running event loop (asyncio.run() cannot be nested).
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
# Tool functions
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

        # Not bust — re-trigger the full advisory cycle for the new hand state
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
        "round": len(history) + 1,
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

    # Reset all round + flag state
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
        f"Round {len(history)}: {outcome}! "
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

    analysis = {
        "player_hand": player_hand,
        "player_score": context_variables.get("player_score", 0),
        "dealer_up_card": dealer_up,
        "bust_probability": round(bust_prob, 4),
        "ev_hit": round(ev_hit, 4),
        "ev_stand": round(ev_stand, 4),
        "recommendation": "HIT" if ev_hit > ev_stand else "STAND",
    }
    print(f"\n{'=' * 52}")
    print("  [MATH_ORACLE]")
    print(json.dumps(analysis, indent=4))
    print(f"{'=' * 52}\n")

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
        values=f"Safe_Player advice recorded.",
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
        values=f"The_Shark rebuttal recorded.",
        context_variables=context_variables,
    )


# ---------------------------------------------------------------------------
# UpdateSystemMessage callables — callable form avoids .format() injection
# risk from LLM-generated advice strings containing { } characters.
# ---------------------------------------------------------------------------

def _dealer_updater(agent: ConversableAgent, messages: list) -> str:
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
# Agent creation
# ---------------------------------------------------------------------------

dealer_agent = ConversableAgent(
    name="Dealer_Agent",
    system_message="You are the Blackjack Dealer.",
    functions=[place_bet, deal_cards, resolve_action, settle_round],
    update_agent_state_before_reply=UpdateSystemMessage(_dealer_updater),
    llm_config=llm_config,
)

math_oracle = ReasoningAgent(
    name="Math_Oracle",
    system_message=(
        "You are Math_Oracle. Zero personality. Pure logic.\n"
        "You have ONE job: call run_math_analysis() immediately when activated. No preamble.\n"
        "After the tool returns, output ONLY this line:\n"
        "ANALYSIS: Bust=<value>%, EV(HIT)=<value>, EV(STAND)=<value>. Recommendation: <HIT|STAND>."
    ),
    functions=[run_math_analysis],
    llm_config=llm_config,
    reason_config={"method": "beam_search", "max_depth": 2, "beam_size": 2},
)

safe_player = ConversableAgent(
    name="Safe_Player",
    system_message="You are Safe_Player.",
    functions=[record_safe_advice],
    update_agent_state_before_reply=UpdateSystemMessage(_safe_updater),
    llm_config=llm_config,
)

the_shark = ConversableAgent(
    name="The_Shark",
    system_message="You are The_Shark.",
    functions=[record_shark_advice],
    update_agent_state_before_reply=UpdateSystemMessage(_shark_updater),
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="PlayerProxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda m: any(
        w in (m.get("content") or "").lower() for w in ["quit", "exit", "game over"]
    ),
    code_execution_config=False,
)

# ---------------------------------------------------------------------------
# Handoff registration
# ---------------------------------------------------------------------------

# Dealer → Math_Oracle: intercepts Dealer BEFORE its LLM fires whenever a
# new advisory cycle begins (phase=advisory, math not yet done).
register_hand_off(
    dealer_agent,
    OnContextCondition(
        target=math_oracle,
        condition=ContextExpression("${phase} == 'advisory' and not ${math_done}"),
    ),
)

# Dealer → REVERT_TO_USER: after Dealer speaks (to collect bet or player action).
register_hand_off(dealer_agent, AfterWork(agent=AfterWorkOption.REVERT_TO_USER))

# Math_Oracle → Safe_Player → The_Shark → Dealer (linear debate chain).
register_hand_off(math_oracle, AfterWork(agent=safe_player))
register_hand_off(safe_player, AfterWork(agent=the_shark))
register_hand_off(the_shark, AfterWork(agent=dealer_agent))


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def print_banner() -> None:
    print("\n" + "=" * 60)
    print("   BLACKJACK REASONING TUTOR")
    print("   Multi-Agent Advisory System (AG2 Swarm)")
    print("=" * 60)
    print("  Advisors: Math_Oracle  |  Safe_Player  |  The_Shark")
    print("  Actions : HIT  |  STAND  |  DOUBLE")
    print("  Quit    : type 'quit' or 'exit'")
    print("=" * 60 + "\n")


def print_session_summary(cv: ContextVariables) -> None:
    history = cv.get("history", [])
    chips = cv.get("chips", 0)
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Final chips  : {chips}")
    print(f"  Rounds played: {len(history)}")
    if history:
        wins = sum(1 for r in history if r.get("delta", 0) > 0)
        losses = sum(1 for r in history if r.get("delta", 0) < 0)
        pushes = len(history) - wins - losses
        net = sum(r.get("delta", 0) for r in history)
        print(f"  W / L / P    : {wins} / {losses} / {pushes}")
        print(f"  Net chips    : {net:+d}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    context_variables = make_initial_context()
    print_banner()

    chat_result, final_cv, _ = await a_initiate_swarm_chat(
        initial_agent=dealer_agent,
        agents=[dealer_agent, math_oracle, safe_player, the_shark],
        user_agent=user_proxy,
        messages="Welcome to Blackjack Reasoning Tutor! Let's play.",
        context_variables=context_variables,
        after_work=AfterWorkOption.REVERT_TO_USER,
        max_rounds=200,
    )

    print_session_summary(final_cv)


if __name__ == "__main__":
    asyncio.run(main())
