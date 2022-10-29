import numpy as np
import random

class Card:
    def __init__ (self, suit, value):
        self.suit = suit
        self.value = value
    
    #just for testing purposes
    def printCard(self):
        print(str(self.value)+str(self.suit))

class Deck:
    def __init__ (self):
        self.Cards = []
    
    def buildDeck(self):
        #adds the 52 cards in the deck to Cards list
        for suits in ['D','H','C','S']:
            self.Cards.append(Card(suits,'A'))
            for values in range(2,11):
                self.Cards.append(Card(suits,str(values)))
            self.Cards.append(Card(suits,'J'))
            self.Cards.append(Card(suits,'Q'))
            self.Cards.append(Card(suits,'K'))

    def shuffleDeck(self):
        random.shuffle(self.Cards)

    def drawCards(self,numCards):
        drawDeck = []
        for i in range(numCards):
            drawDeck.append(self.Cards[0])
            del self.Cards[0]
        return drawDeck 

    #below 2 functions just for testing purposes
    def printCard(self):
        for card in self.Cards:
            card.printCard()

    def printDrawnCards(self):
        testDeck = Deck.drawCards(self,30)
        for card in testDeck:
            card.printCard()

class Player:
    def __init__(self, numCards, cardList):
        self.numCards = numCards
        self.cardList = cardList

#below are the check functions for seeing if the cards are there

#Checks if high card exists in the card pool, True if it does, False if it does not
def CheckHigh(cardPool, highCard):
    cardList = [card[1:] for card in cardPool] 
    if cardList.count(highCard) >= 1:
        return True
    return False

#Checks if 2 of pairCard exist in the card pool, True if it does, False if it does not
def CheckPair(cardPool, pairCard):
    cardList = [card[1:] for card in cardPool] 
    if cardList.count(pairCard) >= 2:
        return True
    return False

#Checks if the two of pairCard1 and two of pairCard2 exist in the card pool, True if it does, False if it does not
def CheckTwoPair(cardPool, pairCard1, pairCard2):
    if CheckPair(cardPool,pairCard1) and CheckPair(cardPool,pairCard2):
        return True
    return False

#Checks if the three of tripleCard exist in the card pool, True if it does, False if it does not
def CheckThreeKind(cardPool, tripleCard):
    cardList = [card[1:] for card in cardPool] 
    if cardList.count(tripleCard) >= 3:
        return True
    return False

def CheckStraight(cardPool, highCard):
    noDuplicateArray = ['placeholder','2','3','4','5','6','7','8','9','10','J','Q','K','A']
    orderList = ['A','2','3','4','5','6','7','8','9','10','J','Q','K','A']
    topCard = noDuplicateArray.index(highCard)
    valueList = []
    for card in cardPool:
        valueList.append(card.value)
    for i in range(5):
        if orderList[topCard-i] not in valueList:
            return False
    return True

#How to make sure cards lower than highCard are not considered?
def CheckFlush(cardPool, highCard, suit):
    suitList = [card[:1] for card in cardPool]
    if suitList.count(suit) >= 5 and CheckHigh(cardPool,highCard):
        return True
    return False


def CheckFullHouse(cardPool, tripleCard, pairCard):
    if CheckThreeKind(cardPool,tripleCard) and CheckPair(cardPool,pairCard):
        return True
    return False

#Checks if the four of quadCard exist in the card pool, True if it does, False if it does not
def CheckFourKind(cardPool, quadCard):
    cardList = [card[1:] for card in cardPool] 
    if cardList.count(quadCard) == 4:
        return True
    return False

def CheckStraightFlush(cardPool, highCard, suit):
    if CheckStraight(cardPool,highCard) and CheckFlush(cardPool,highCard,suit):
        return True
    return False

def CheckRoyalFlush(cardPool, suit):
    return CheckStraightFlush(cardPool,'A',suit)
#end of check cards

#This function is for making sure that the last hand is smaller than the hand user wants to play
def isHandLarger(previousHand,proposedHand):   
    
    return

#Currently supports only 2 players
def gameRound(p1,p2):
    #creates a fresh shuffled deck
    deck = Deck()
    deck.buildDeck()
    deck.shuffleDeck()
    #store_move[0] is 
    store_move = []

    #draws cards for each player
    p1Cards = deck.drawCards(p1.numCards)
    p1.cardList = p1Cards
    p2Cards = deck.drawCards(p2.numCards)
    p2.cardList = p2Cards

    #2 ways to end a round - call bluff or declare royal flush
    while True:
        for p in [p1,p2]:

            print('Your move, choose:')
            print('1 - Call Bluff')
            print('2 - High card')
            print('3 - Pair')
            print('4 - 2 Pair')
            print('5 - 3 of a kind')
            print('6 - Straight')
            print('7 - Flush')
            print('8 - Full House')
            print('9 - Four of a kind')
            print('10 - Straight Flush')
            print('11 - Royal Flush')
            
            #makes sure user enters a valid response
            while True:
                choice = input('Enter your choice: ')
                if store_move != [] or choice != '1':
                    if choice in ['1','2','3','4','5','6','7','8','9','10','11']:
                        break
                    else:
                        print('Please enter a number from 1 to 11')
                else:
                    print('You cannot call bluff on the first move!')
            
            #handles the next move based on user's response
            if choice =='1':
                if store_move[1] == '2':
                    if CheckHigh(p1Cards+p2Cards,store_move[2]):
                        return p #If not bluff, p loses
                    else:
                        return store_move[0] #If bluff, the player that made the bluff loses
                if store_move[1] == '3':
                    if CheckPair(p1Cards+p2Cards,store_move[2]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '4':
                    if CheckTwoPair(p1Cards+p2Cards,store_move[2],store_move[3]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '5':
                    if CheckThreeKind(p1Cards+p2Cards,store_move[2]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '6':
                    if CheckStraight(p1Cards+p2Cards,store_move[2]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '7':
                    if CheckFlush(p1Cards+p2Cards,store_move[3],store_move[2]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '8':
                    if CheckFullHouse(p1Cards+p2Cards,store_move[2],store_move[3]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '9':
                    if CheckFourKind(p1Cards+p2Cards,store_move[2]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '10':
                    if CheckStraightFlush(p1Cards+p2Cards,store_move[2],store_move[3]):
                        return p
                    else:
                        return store_move[0]
                if store_move[1] == '11':
                    if CheckRoyalFlush(p1Cards+p2Cards,store_move[2]):
                        return p
                    else:
                        return store_move[0]

                #use store_move to call one of the check functions, then update player cardnum
                break
            elif choice == '2':
                choice2 = input('Enter High card: ')
                store_move = [p,choice,choice2]
                print("You played: High card of " + choice2)

            elif choice == '3':
                choice2 = input('Enter Pair card: ')
                store_move = [p,choice,choice2]
                print("You played: Pair of " + choice2 + "'s")

            elif choice == '4':
                choice2 = input('Enter 1st pair card: ')
                choice3 = input('Enter 2nd pair card: ')
                store_move = [p,choice,choice2,choice3]
                print("You played: Two " + choice2 + "'s and two " + choice3 + "'s")

            elif choice == '5':
                choice2 = input('Enter Triple card: ')
                store_move = [p,choice,choice2]

            elif choice == '6':
                choice2 = input('Enter highest card in straight: ')
                #this one may need to prevent entering 2, 3, and 4
                store_move = [p,choice,choice2]

            elif choice == '7':
                choice2 = input('Enter flush suite: ')
                choice3 = input('Enter high card in flush: ')
                store_move = [p,choice,choice2,choice3]

            elif choice == '8':

                choice2 = input('Enter triple card: ')
                choice3 = input('Enter pair card: ')
                store_move = [p,choice,choice2,choice3]

            elif choice == '9':
                choice2 = input('Enter quadruple card: ')
                store_move = [p,choice,choice2]

            elif choice == '10':
                choice2 = input('Enter high card in straight: ')
                choice3 = input('Enter suite of straight: ')
                store_move = [p,choice,choice2,choice3]

            elif choice == '11':
                choice2 = input('Enter suite of royal flush: ')
                store_move = [p,choice,choice2]
                break

p1 = Player(2,[])
p2 = Player(2,[])
loser = gameRound(p1,p2)
print(loser)