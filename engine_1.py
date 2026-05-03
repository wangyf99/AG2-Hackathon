import asyncio
import random


class BlackjackEngine:
    @staticmethod
    def card_value(card: str) -> int:
        if card in ("J", "Q", "K"):
            return 10
        if card == "A":
            return 1
        return int(card)

    @staticmethod
    def hand_score(hand: list[str]) -> int:
        total = sum(BlackjackEngine.card_value(c) for c in hand)
        aces = hand.count("A")
        for _ in range(aces):
            if total + 10 <= 21:
                total += 10
        return total

    @staticmethod
    def create_deck() -> list[str]:
        values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return values * 4

    @staticmethod
    def draw_card(deck: list[str]) -> tuple[str, list[str]]:
        deck_copy = list(deck)
        idx = random.randrange(len(deck_copy))
        card = deck_copy.pop(idx)
        return card, deck_copy

    @staticmethod
    def dealer_play(dealer_hand: list[str], deck: list[str]) -> tuple[list[str], list[str]]:
        hand = list(dealer_hand)
        remaining = list(deck)
        while BlackjackEngine.hand_score(hand) < 17:
            card, remaining = BlackjackEngine.draw_card(remaining)
            hand.append(card)
        return hand, remaining

    @staticmethod
    def get_bust_prob(hand: list[str], deck: list[str]) -> float:
        if not deck:
            return 0.0
        bust_count = sum(
            1 for card in deck
            if BlackjackEngine.hand_score(hand + [card]) > 21
        )
        return bust_count / len(deck)

    @staticmethod
    def _simulate_single_trial(
        player_hand: list[str],
        dealer_up: str,
        deck: list[str],
        action: str,
    ) -> float:
        ph = list(player_hand)
        dk = list(deck)

        if action == "HIT":
            card, dk = BlackjackEngine.draw_card(dk)
            ph.append(card)
            if BlackjackEngine.hand_score(ph) > 21:
                return -1.0

        # Dealer gets their hole card from the remaining deck
        if not dk:
            return 0.0
        hole, dk = BlackjackEngine.draw_card(dk)
        dh = [dealer_up, hole]
        dh, _ = BlackjackEngine.dealer_play(dh, dk)

        ps = BlackjackEngine.hand_score(ph)
        ds = BlackjackEngine.hand_score(dh)

        if ds > 21 or ps > ds:
            return 1.0
        if ps == ds:
            return 0.0
        return -1.0

    @staticmethod
    async def calculate_mc_ev(
        player_hand: list[str],
        dealer_up: str,
        deck: list[str],
        action: str,
        trials: int = 200,
    ) -> float:
        results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    BlackjackEngine._simulate_single_trial,
                    player_hand, dealer_up, deck, action,
                )
                for _ in range(trials)
            ]
        )
        return sum(results) / trials
