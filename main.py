import random

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.rank} of {self.suit}"


class Deck:
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
    jokers = ['Joker']

    def __init__(self):
        self.cards = [Card(suit, rank) for suit in self.suits for rank in self.ranks]
        self.cards.extend([Card(None, joker) for joker in self.jokers])

    def __repr__(self):
        return f"Deck of {len(self.cards)} cards: " + ", ".join(map(str, self.cards))

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        return self.cards.pop()

    def draw_hand(self, hand_size):
        return [self.draw_card() for _ in range(hand_size)]


class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def draw_card(self, deck):
        self.hand.append(deck.draw_card())
        return self.hand

    def __str__(self):
        return f"{self.name}: {self.hand}"


class Game: 
    def __init__(self):
        self.deck = Deck()
        self.players = []

    def add_player(self, player):
        self.players.append(player)

    def start(self, players):
        self.deck.shuffle()
        self.players = players

        for player in self.players:
            player.draw_card(self.deck)
            player.draw_card(self.deck)

        for player in self.players:
            print(player)


if __name__ == "__main__":
    game = Game()
    players = [Player("Player 1"), Player("Player 2")]
    game.start(players)

