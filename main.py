import random
from itertools import combinations

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.rank} of {self.suit}"


class Deck:
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

    def __init__(self):
        self.reset()

    def __repr__(self):
        return f"Deck of {len(self.cards)} cards: " + ", ".join(map(str, self.cards))

    def reset(self):
        """Rebuild to a fresh 52-card deck (no jokers)."""
        self.cards = [Card(suit, rank) for suit in self.suits for rank in self.ranks]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if not self.cards:
            raise RuntimeError("Cannot draw from an empty deck")
        return self.cards.pop()

    def draw_hand(self, hand_size):
        if hand_size > len(self.cards):
            raise RuntimeError(f"Not enough cards to draw {hand_size}; only {len(self.cards)} left")
        return [self.draw_card() for _ in range(hand_size)]


class Player:
    def __init__(self, name, chips=1000):
        self.name = name
        self.hand = []
        self.chips = chips
        self.current_bet = 0
        self.folded = False

        self.total_commit = 0 
        self.all_in = False

    def draw_card(self, deck):
        self.hand.append(deck.draw_card())
        return self.hand
    

    def bet(self, amount):  
        """Put 'amount' more chips into the pot for this action; return actual amount put in."""
        if self.folded:
            print(f"{self.name} has folded and cannot bet")
            return 0

        if amount <= 0:
            print("Bet must be positive.")
            return 0

        actual_bet = min(amount, self.chips)   # handles all-in if requested amount > stack
        self.chips -= actual_bet
        self.current_bet += actual_bet
        self.total_commit += actual_bet

        if self.chips == 0:
            self.all_in = True
            print(f"{self.name} goes ALL-IN with {actual_bet} chips!")
        else:
            print(f"{self.name} bets {actual_bet} chips (remaining: {self.chips})")

        return actual_bet
        
        

    def call(self, amount_to_call):
        """Call a bet - match the current bet"""
        if self.folded:
            print(f"{self.name} has folded and cannot call")
            return 0
        
        needed = amount_to_call - self.current_bet
        if needed <= 0:
            print(f"{self.name} is already caught up")
            return 0
        
        return self.bet(needed)

    def fold(self):
        """Fold hand"""
        self.folded = True
        print(f"{self.name} folds")

    def reset_for_new_hand(self):
        """Reset player for new hand"""
        self.hand = []
        self.current_bet = 0
        self.folded = False
        self.total_commit = 0
        self.all_in = False

    def __str__(self):
        status = " (FOLDED)" if self.folded else ""
        return f"{self.name}: {self.hand} | Chips: {self.chips} | Bet: {self.current_bet}{status}"


class Game:
    def __init__(self, small_blind=1, big_blind=2):
        self.deck = Deck()
        self.players = []  
        self.community_cards = []
        self.stage = "preflop"
        self.pot = 0 
        self.current_bet = 0
        self.pots = []

        self.small_blind = small_blind
        self.big_blind = big_blind
        self.button_index = -1  # will be set at start of the first hand
        self.last_raise_size = None  # min raise amount for this street; set each street

        self.decide_action = None  #function(player, obs, legal) -> ("fold"/"check"/"call"/"bet"/"raise", amount_or_None)

    @staticmethod
    def _rank_number(rank_str):
        order = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,
                'Jack':11,'Queen':12,'King':13,'Ace':14}
        return order[rank_str]

    @staticmethod
    def _number_to_label(n):
        inv = {11:'Jack',12:'Queen',13:'King',14:'Ace'}
        return inv.get(n, str(n))

    @staticmethod
    def _straight_high(values):
        """Return high card of a 5-card straight within 'values' (ints), or None."""
        vals = set(values)
        if 14 in vals:            # Ace present → allow Ace-as-1
            vals.add(1)
        uniq = sorted(vals)       # ensure sorted after adding 1

        run = 1
        best = None
        for i in range(1, len(uniq)):
            if uniq[i] == uniq[i-1] + 1:
                run += 1
            else:
                run = 1
            if run >= 5:
                best = uniq[i]    # end of the 5+ run (e.g., 5 for A-2-3-4-5)
        return best


    @classmethod
    def _hand_rank_5(cls, cards5):
        """
        Rank a 5-card hand. Returns a tuple that compares lexicographically:
        (category, tiebreakers...)
        with category:
        8=Straight Flush, 7=Four, 6=Full House, 5=Flush, 4=Straight,
        3=Trips, 2=Two Pair, 1=One Pair, 0=High Card
        """
        vals = sorted([cls._rank_number(c.rank) for c in cards5], reverse=True)
        suits = [c.suit for c in cards5]
        # counts by value
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        # sort by (count desc, value desc)
        groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        counts_sorted = [cnt for _, cnt in groups]

        is_flush = len(set(suits)) == 1
        # Straight: use unique ranks; wheel support
        straight_high = None
        if len(set(vals)) >= 5:
            straight_high = cls._straight_high(vals)

        # Straight Flush
        if is_flush:
            flush_vals = vals[:]  # all 5 are same suit
            sf_high = cls._straight_high(flush_vals)
            if sf_high is not None:
                return (8, sf_high)

        # Four of a Kind
        if counts_sorted == [4,1]:
            four_val = groups[0][0]
            kicker = max([v for v in vals if v != four_val])
            return (7, four_val, kicker)

        # Full House
        if counts_sorted == [3,2]:
            three_val = groups[0][0]
            pair_val = groups[1][0]
            return (6, three_val, pair_val)

        # Flush
        if is_flush:
            return (5, *vals)

        # Straight
        if straight_high is not None:
            return (4, straight_high)

        # Trips
        if counts_sorted == [3,1,1]:
            three_val = groups[0][0]
            kickers = [v for v in vals if v != three_val][:2]
            return (3, three_val, *kickers)

        # Two Pair
        if counts_sorted == [2,2,1]:
            pair_high = max(groups[0][0], groups[1][0])
            pair_low  = min(groups[0][0], groups[1][0])
            kicker = max([v for v in vals if v != pair_high and v != pair_low])
            return (2, pair_high, pair_low, kicker)

        # One Pair
        if counts_sorted == [2,1,1,1]:
            pair_val = groups[0][0]
            kickers = [v for v in vals if v != pair_val][:3]
            return (1, pair_val, *kickers)

        # High Card
        return (0, *vals)

    def _best_5_from_7(self, seven_cards):
        """Return (rank_tuple, best5_list). Compares by rank_tuple."""
        best = None
        best5 = None
        for combo in combinations(seven_cards, 5):
            score = self._hand_rank_5(combo)
            if (best is None) or (score > best):
                best = score
                best5 = list(combo)
        return best, best5

    @classmethod
    def _hand_name_from_score(cls, score):
        cat = score[0]
        if cat == 8:
            return f"Straight Flush ({cls._number_to_label(score[1])}-high)"
        if cat == 7:
            return f"Four of a Kind ({cls._number_to_label(score[1])}s), kicker {cls._number_to_label(score[2])}"
        if cat == 6:
            return f"Full House ({cls._number_to_label(score[1])}s over {cls._number_to_label(score[2])}s)"
        if cat == 5:
            cards = ', '.join(cls._number_to_label(x) for x in score[1:6])
            return f"Flush ({cards})"
        if cat == 4:
            return f"Straight ({cls._number_to_label(score[1])}-high)"
        if cat == 3:
            kick = ', '.join(cls._number_to_label(x) for x in score[2:4])
            return f"Three of a Kind ({cls._number_to_label(score[1])}s), kickers {kick}"
        if cat == 2:
            return f"Two Pair ({cls._number_to_label(score[1])}s and {cls._number_to_label(score[2])}s), kicker {cls._number_to_label(score[3])}"
        if cat == 1:
            kick = ', '.join(cls._number_to_label(x) for x in score[2:5])
            return f"One Pair ({cls._number_to_label(score[1])}s), kickers {kick}"
        # High card
        cards = ', '.join(cls._number_to_label(x) for x in score[1:6])
        return f"High Card ({cards})"

    def add_player(self, player):
        self.players.append(player)

    def start(self, players):
        # Fresh deck every hand
        self.deck = Deck()
        self.deck.shuffle()

        self.players = players
        self.community_cards = []
        self.stage = "preflop"
        self.pot = 0
        self.current_bet = 0
        self.pots = []

        # Rotate the button (first hand sets to 0)
        if self.button_index == -1:
            self.button_index = 0
        else:
            self.advance_button()

        # Reset all players for new hand
        for player in self.players:
            player.reset_for_new_hand()

        # Deal hole cards
        for player in self.players:
            player.draw_card(self.deck)
            player.draw_card(self.deck)

        print("=== HOLE CARDS DEALT ===")
        for player in self.players:
            # Don’t reveal others’ hands. You can uncomment the next line while debugging:
            # print(f"{player.name}: {player.hand}")
            print(f"{player.name}: [2 cards]")
        print(f"Pot: {self.pot}\n")

        # Post SB/BB now that hands are dealt
        self.post_blinds()

    def betting_round(self):
        """Loop until betting is closed or only one player remains.
        Returns True if the hand ended early (all but one folded), else False.
        """
        print(f"=== BETTING ROUND ({self.stage.upper()}) ===")
        print(f"Current pot: {self.pot}")
        print(f"Current bet to match: {self.current_bet}\n")

        # If only one player already, nothing to do.
        if len(self._active_players()) <= 1:
            return self._award_if_single_player_left()

        # Correct order for this street
        order = self.get_betting_order()

        # Build the initial "players who still need to act" queue.
        # Players with zero chips are skipped (they're effectively all-in).
        def can_act(p): return (not p.folded) and (p.chips > 0)

        players_to_act = [p for p in order if can_act(p)]

        # Helper to rotate the queue to "everyone except raiser, starting after raiser"
        def reset_after_raiser(raiser):
            idx = order.index(raiser)
            rotated = order[idx+1:] + order[:idx]
            return [p for p in rotated if can_act(p)]

        def amount_to_call(p):
            return max(0, self.current_bet - p.current_bet)

        # Closure check: all active players are either folded, all-in,
        # or have matched the current bet.
        def betting_closed():
            for p in self.players:
                if p.folded or p.chips == 0:
                    continue
                if amount_to_call(p) > 0:
                    return False
            return True

        # Main loop
        while players_to_act and len(self._active_players()) > 1:
            player = players_to_act.pop(0)

            # Skip if player can no longer act
            if player.folded or player.chips == 0:
                continue

            atc = amount_to_call(player)  # amount to call
            print(f"\n{player.name}'s turn:")
            print(f"Your hand: {player.hand}")
            print(f"Your chips: {player.chips}")
            print(f"Your current bet: {player.current_bet}")
            print(f"Amount to call: {atc}")
            
            legal = self.legal_actions(player)
            action, param = self._decide(player, legal)

            if action == "fold":
                player.fold()
                if self._award_if_single_player_left():
                    print("-" * 40)
                    return True
                print(f"Pot is now: {self.pot}")
                print("-" * 40)
                continue

            elif action == "check":
                if not legal["can_check"]:
                    print("Cannot check: there is a bet to call.")
                    players_to_act.append(player)
                    continue
                print(f"{player.name} checks.")
                print(f"Pot is now: {self.pot}")
                print("-" * 40)

            elif action == "call":
                if not legal["can_call"]:
                    print(f"Nothing to call.")
                    # treat as check
                else:
                    actual = player.call(self.current_bet)
                    self.pot += actual
                    print(f"{player.name} puts in {actual}.")
                    print(f"Pot is now: {self.pot}")
                    print("-" * 40)

            elif action == "bet":
                if not legal["can_bet"]:
                    print("There is already a bet. Use 'raise' instead.")
                    players_to_act.append(player)
                    continue
                try:
                    amt = int(param)
                except Exception:
                    amt = 0
                min_bet = legal["bet_min"]
                if amt <= 0:
                    print("Bet must be positive.")
                    players_to_act.append(player); continue
                # Enforce min bet unless all-in short
                if amt < min_bet and player.chips > amt:
                    print(f"Bet must be at least {min_bet} unless all-in.")
                    players_to_act.append(player); continue

                actual = player.bet(amt)
                self.pot += actual
                self.current_bet = player.current_bet

                if actual >= min_bet:
                    self.last_raise_size = actual
                    players_to_act = reset_after_raiser(player)
                print(f"{player.name} bets {actual}.")
                print(f"New bet to match: {self.current_bet}")
                print(f"Pot is now: {self.pot}")
                print("-" * 40)

            elif action == "raise":
                if not legal["can_raise"]:
                    print("No bet to raise (or not enough chips).")
                    players_to_act.append(player); continue
                try:
                    inc = int(param)
                except Exception:
                    inc = 0
                atc = legal["to_call"]
                # Enforce min raise unless all-in short
                if inc < legal["raise_min"] and player.chips > (atc + inc):
                    print(f"Min raise is {self.last_raise_size}. Enter {self.last_raise_size} or more, or 'call'.")
                    players_to_act.append(player); continue

                target = atc + inc
                actual = player.bet(target)
                self.pot += actual

                if actual < atc:
                    print(f"{player.name} goes all-in for less than a call: {actual}.")
                elif actual == atc:
                    print(f"{player.name} calls {actual}.")
                else:
                    effective_raise = actual - atc
                    self.current_bet = player.current_bet
                    if effective_raise >= self.last_raise_size:
                        self.last_raise_size = effective_raise
                        print(f"{player.name} raises by {effective_raise} to total {self.current_bet}.")
                        players_to_act = reset_after_raiser(player)
                    else:
                        print(f"{player.name} all-in raise of {effective_raise} (below min raise). Betting not re-opened.")
                print(f"New bet to match: {self.current_bet}")
                print(f"Pot is now: {self.pot}")
                print("-" * 40)

            else:
                print("Invalid action from policy; folding to be safe.")
                player.fold()
                if self._award_if_single_player_left():
                    print("-" * 40)
                    return True

        print(f"Betting round complete. Pot: {self.pot}\n")
        return False  # hand continues
    
    def advance_button(self):
        if not self.players:
            raise RuntimeError("No players to advance button")
        self.button_index = (self.button_index + 1) % len(self.players)

    def position_indexes(self):
        """Return (button, small blind, big blind) indexes given current button."""
        n = len(self.players)
        btn = self.button_index % n
        if n == 2:
            # Heads-up: button is SB; other player is BB
            sb = btn
            bb = (btn + 1) % n
        else:
            sb = (btn + 1) % n
            bb = (btn + 2) % n
        return btn, sb, bb
    
    def post_blinds(self):
        """Post small/big blinds and set the opening bet size (BB)."""
        btn, sb_idx, bb_idx = self.position_indexes()
        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]

        print(f"Button: {self.players[btn].name} | SB: {sb_player.name} | BB: {bb_player.name}")

        sb_posted = sb_player.bet(self.small_blind)
        bb_posted = bb_player.bet(self.big_blind)

        self.pot += sb_posted + bb_posted
        # Set the current bet to the largest posted (normally the BB)
        self.current_bet = max(sb_player.current_bet, bb_player.current_bet)

        self.last_raise_size = self.big_blind
        print(f"Blinds posted. Pot: {self.pot}. Current bet to match: {self.current_bet}\n")

    def get_betting_order(self):
        """
        Return the correct turn order for the current street:
        - Preflop:
            * 2 players (HU): Button (SB) acts first.
            * 3+ players: first to act is left of the BB.
        - Postflop/Turn/River: first to act is left of the button.
        Folded players are skipped here; all-ins are handled in betting_round.
        """
        n = len(self.players)
        if n == 0:
            return []

        if self.stage == "preflop":
            if n == 2:
                start_idx = self.button_index  # HU: button/SB acts first preflop
            else:
                _, _, bb_idx = self.position_indexes()
                start_idx = (bb_idx + 1) % n  # first left of BB
        else:
            start_idx = (self.button_index + 1) % n  # left of button

        order = []
        for i in range(n):
            p = self.players[(start_idx + i) % n]
            if not p.folded:
                order.append(p)
        return order

    def flop(self):
        """Reveal the first 3 community cards"""
        if self.stage != "preflop":
            print(f"Cannot deal flop - game is already at {self.stage} stage")
            return self.community_cards
        
        # Burn one card (discard)
        self.deck.draw_card()
        
        # Deal 3 cards for the flop
        for _ in range(3):
            self.community_cards.append(self.deck.draw_card())
        
        self.stage = "flop"  
        self.current_bet = 0
        self.last_raise_size = self.big_blind  # NEW
        for player in self.players:
            player.current_bet = 0
            
        print("=== FLOP ===")
        print(f"Community cards: {self.community_cards}")
        print()
        return self.community_cards

    def turn(self):
        """Reveal the 4th community card"""
        if self.stage != "flop":
            print(f"Cannot deal turn - game is at {self.stage} stage (need flop first)")
            return self.community_cards
        
        # Burn one card (discard)
        self.deck.draw_card()
        
        # Deal 1 card for the turn
        self.community_cards.append(self.deck.draw_card())
        
        self.stage = "turn" 
        self.current_bet = 0
        self.last_raise_size = self.big_blind  # NEW
        for player in self.players:
            player.current_bet = 0
            
        print("=== TURN ===")
        print(f"Community cards: {self.community_cards}")
        print()
        return self.community_cards

    def river(self):
        """Reveal the 5th and final community card"""
        if self.stage != "turn":
            print(f"Cannot deal river - game is at {self.stage} stage (need turn first)")
            return self.community_cards
        
        # Burn one card (discard)
        self.deck.draw_card()
        
        # Deal 1 card for the river
        self.community_cards.append(self.deck.draw_card())
        
        self.stage = "river" 
        self.current_bet = 0
        self.last_raise_size = self.big_blind  # NEW
        for player in self.players:
            player.current_bet = 0
            
        print("=== RIVER ===")
        print(f"Community cards: {self.community_cards}")
        print()
        return self.community_cards

    def get_community_cards(self):
        """Get all currently revealed community cards"""
        return self.community_cards

    def get_stage(self):
        """Get current game stage"""
        return self.stage

    def play(self):
        print("=== STARTING POKER GAME ===")

        # Pre-flop betting
        if self.betting_round():
            print("=== GAME COMPLETE ===")
            return self.community_cards

        # Flop
        self.flop()
        if self.betting_round():
            print("=== GAME COMPLETE ===")
            return self.community_cards

        # Turn
        self.turn()
        if self.betting_round():
            print("=== GAME COMPLETE ===")
            return self.community_cards

        # River
        self.river()
        if self.betting_round():
            print("=== GAME COMPLETE ===")
            return self.community_cards

        print("=== GAME COMPLETE ===")
        print(f"Final community cards: {self.community_cards}")
        print(f"Final pot: {self.pot}")

        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.chips += self.pot
            print(f"{winner.name} wins the pot of {self.pot} chips!")
        else:
            # Build pots and award by best hands among each pot's eligibles
            self.build_side_pots()
            # (Optional) print pot structure
            for i, pot in enumerate(self.pots):
                names = [p.name for p in pot["eligibles"]]
                print(f"Pot {i} - amount: {pot['amount']}, eligibles: {names}")
            self._award_pots()

        return self.community_cards
    
    def _active_players(self):
        return [p for p in self.players if not p.folded]

    def _award_if_single_player_left(self):
        active = self._active_players()
        if len(active) == 1:
            winner = active[0]
            winner.chips += self.pot
            print(f"{winner.name} wins the pot of {self.pot} chips (everyone else folded).")
            return True
        return False
        
    def build_side_pots(self):
        """
        Partition total contributions into main + side pots.
        Each pot: {"amount": int, "eligibles": set(Player)} where eligibles excludes folded players.
        Uses each player's per-hand total_commit (includes blinds).
        """
        # Map players to how much they've committed this hand
        contrib = {p: p.total_commit for p in self.players}
        folded = {p for p in self.players if p.folded}

        pots = []
        # Work on a mutable copy
        remaining = contrib.copy()

        def positive_players():
            return [p for p, amt in remaining.items() if amt > 0]

        while True:
            participants = positive_players()
            if not participants:
                break

            # The next "layer" is limited by the smallest remaining stack contribution
            layer = min(remaining[p] for p in participants)
            # Amount in this pot layer = layer taken from each participant who still has >0
            amount = sum(min(remaining[p], layer) for p in participants)

            # Eligible winners are non-folded participants for this layer
            eligibles = {p for p in participants if p not in folded}

            if amount > 0:
                pots.append({"amount": amount, "eligibles": eligibles})

            # Subtract this layer from all current participants
            for p in participants:
                remaining[p] -= layer

        # Optional sanity: total of pots should equal printed pot
        total_from_layers = sum(pot["amount"] for pot in pots)
        if total_from_layers != self.pot:
            print(f"[WARN] Side-pot sum {total_from_layers} != table pot {self.pot}. "
                f"(This can happen if pot accounting is out of sync.)")

        self.pots = pots
        return pots
        
    def _seat_order_left_of_button(self):
        """Return players in clockwise order starting left of the button."""
        n = len(self.players)
        start = (self.button_index + 1) % n
        return [self.players[(start + i) % n] for i in range(n)]
        
    def _evaluate_player(self, player):
        """Return (score_tuple, best5, name_str) for a player's 7-card hand."""
        seven = player.hand + self.community_cards
        score, best5 = self._best_5_from_7(seven)
        return score, best5, self._hand_name_from_score(score)

    def _award_pots(self):
        """
        Assumes self.pots built; evaluate eligibles, pick winners per pot,
        split (with odd chips going left of the button).
        """
        if not self.pots:
            return

        order_from_button = self._seat_order_left_of_button()
        seat_index = {p: i for i, p in enumerate(order_from_button)}
        total_awarded = 0

        print("\n=== SHOWDOWN ===")
        # Show each active player's hand strength (optional)
        active = [p for p in self.players if not p.folded]
        for p in active:
            score, _, name = self._evaluate_player(p)
            print(f"{p.name}: {name}")

        for i, pot in enumerate(self.pots):
            eligibles = [p for p in pot["eligibles"] if not p.folded]
            if not eligibles:
                continue

            # Evaluate all eligibles
            scores = {}
            for p in eligibles:
                scores[p], _, _ = self._evaluate_player(p)

            # Find best score and winners
            best_score = max(scores.values())
            winners = [p for p, sc in scores.items() if sc == best_score]

            amount = pot["amount"]
            base = amount // len(winners)
            remainder = amount % len(winners)

            # Split base equally
            for w in winners:
                w.chips += base
                total_awarded += base

            # Odd chips go left of the button among the winners
            winners_in_seat_order = sorted(winners, key=lambda p: seat_index[p])
            for w in winners_in_seat_order:
                if remainder <= 0:
                    break
                w.chips += 1
                remainder -= 1
                total_awarded += 1

            # Print pot result
            winners_names = ", ".join(w.name for w in winners)
            print(f"Pot {i}: {amount} chips -> {winners_names}")

        if total_awarded != self.pot:
            print(f"[WARN] Awarded {total_awarded} != table pot {self.pot}.")

    def set_decide_action(self, fn):
        """Register a decision function: fn(player, obs, legal) -> (action, amount_or_None)."""
        self.decide_action = fn

    def _decide(self, player, legal):
        """Call the registered decide hook; fallback to a console prompt."""
        if self.decide_action is not None:
            return self.decide_action(player, self.get_observation_for(player), legal)
        return self._default_decide_action(player, legal)

    def get_observation_for(self, player):
        """Imperfect-info view for 'player'."""
        seats = []
        for p in self.players:
            seats.append({
                "name": p.name,
                "chips": p.chips,
                "current_bet": p.current_bet,
                "folded": p.folded,
                "is_self": (p is player)
            })
        obs = {
            "stage": self.stage,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "last_raise_size": self.last_raise_size,
            "community": list(self.community_cards),
            "self": {
                "name": player.name,
                "chips": player.chips,
                "current_bet": player.current_bet,
                "hand": list(player.hand),  # only the actor sees their own cards
                "folded": player.folded
            },
            "seats": seats,
            "button_index": self.button_index,
            "blinds": {"sb": self.small_blind, "bb": self.big_blind}
        }
        return obs

    def legal_actions(self, player):
        """Return legal actions and numeric bounds for this player right now."""
        atc = max(0, self.current_bet - player.current_bet)
        can_check = (atc == 0)
        can_call  = (atc > 0 and player.chips > 0)
        can_bet   = (self.current_bet == 0 and player.chips > 0)
        can_raise = (self.current_bet > 0 and player.chips > atc)

        legal = {
            "to_call": atc,
            "can_check": can_check,
            "can_call": can_call,
            "can_bet": can_bet,
            "bet_min": self.big_blind if can_bet else None,
            "bet_max": player.chips if can_bet else None,
            "can_raise": can_raise,
            "raise_min": self.last_raise_size if can_raise else None,  # increment on top of call
            "raise_max": (player.chips - atc) if can_raise else None,  # max increment (all-in)
        }
        return legal

    def _default_decide_action(self, player, legal):
        """Simple console fallback for manual play; AI can override via set_decide_action."""
        options = []
        if legal["can_check"]:
            options.append("check")
        if legal["can_call"]:
            options.append(f"call({legal['to_call']})")
        if legal["can_bet"]:
            options.append(f"bet[{legal['bet_min']}..{legal['bet_max']}]")
        if legal["can_raise"]:
            options.append(f"raise+[{legal['raise_min']}..{legal['raise_max']}]")
        options.append("fold")

        print(f"\n{player.name}'s turn:")
        print(f"Legal: {', '.join(options)}")
        act = input("Action: ").strip().lower()

        if act == "fold":
            return ("fold", None)
        if act == "check":
            return ("check", None)
        if act == "call":
            return ("call", None)
        if act.startswith("bet"):
            try:
                amt = int(input("Bet amount: ").strip())
            except Exception:
                amt = 0
            return ("bet", amt)
        if act.startswith("raise"):
            try:
                inc = int(input("Raise increment (on top of call): ").strip())
            except Exception:
                inc = 0
            return ("raise", inc)
        # default
        return ("fold", None)



if __name__ == "__main__":
    game = Game()
    players = [Player("Player 1"), Player("Player 2", 500)]
    
    # Start and play the complete game
    game.start(players)
    game.play()