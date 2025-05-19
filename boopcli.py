from boop import Boop
from bcolors import bcolors

class BoopCLI:
    def __init__(self, game, hasBot = True, aiMove = None):
        self.game = game
        self.hasBot = hasBot
        self.aiMove = aiMove

    def get_user_move(self):
        """Prompts the user for a move based on the current game state and returns it."""
        if self.game.playerState == Boop.PlayerState.PLAY_CAT:
            return self._get_play_cat_move()
        elif self.game.playerState == Boop.PlayerState.PROMOTE_CAT:
            return self._get_promote_move()
        else:
            print("Game is finished!")
            return None

    def _get_play_cat_move(self):
        """Handles input for placing a cat on the board."""
        player = self.game.players[self.game.currentPlayer]
        
        while True:
            try:
                size = input("Place a kitten (0) or cat (1)? ").strip()
                if size not in ("0", "1") or player.catCounts[int(size)] <= 0:
                    print("Invalid size or no pieces available!")
                    continue
                
                x = int(input(f"Enter row (0-{Boop.boardSize-1}): "))
                y = int(input(f"Enter column (0-{Boop.boardSize-1}): "))
                posI = Boop.posToIndex(x, y)
                if self.game.isOutOfBoard(x, y) or self.game.board[posI] != Boop.stateEmpty:
                    print("Invalid position or space occupied!")
                    continue
                
                return ("playCat", (posI, int(size)))
            except ValueError:
                print("Please enter valid numbers!")

    def _get_promote_move(self):
        """Handles input for promoting cats."""
        options = self.game.promotionOptions
        print("Promotion options:")
        for i, opt in enumerate(options):
            print(f"{i}: {opt}")
        
        while True:
            try:
                choice = int(input(f"Choose an option (0-{len(options)-1}): "))
                if 0 <= choice < len(options):
                    return ("promoteCat", choice)
                print("Invalid option!")
            except ValueError:
                print("Please enter a valid number!")
    
    def displayPiece(self, boardIndex, lastMove, lastBoard):
        str = self.game.stateDisplays[self.game.board[boardIndex]]
        isCurrent = lastMove is not None and lastMove[0] == 'playCat' and boardIndex == lastMove[1][0]
        isDifferent = lastBoard is not None and self.game.board[boardIndex] != lastBoard[boardIndex]
        return f"{bcolors.OKGREEN}{str}{bcolors.ENDC}" if isCurrent else f"{bcolors.OKBLUE}{str}{bcolors.ENDC}" if isDifferent else str
    
    def displayGameState(self, lastMove=None, lastBoard=None):
        # TODO: compare new game state with previous one
        displayString = " _ _ _ _ _ _ \n"
        for i in range(0, self.game.boardSize * self.game.boardSize, self.game.boardSize):
            textArr = [self.displayPiece(i+j, lastMove, lastBoard) for j in range(self.game.boardSize)]
            line = "|".join(textArr)
            displayString += f"|{line}|\n"
        displayString += " T T T T T T "
        displayString += "\n" + "\n".join([f"P{i}: K{p.catCounts[0]} C{p.catCounts[1]}" for i, p in enumerate(self.game.players)])
        print(displayString)

    def play(self):
        """Runs a simple game loop for user interaction."""
        if self.hasBot:
            # TODO: random first player
            playerControls = ( # (isHuman)
                (True),
                (False)
            )
        else:
            playerControls = ( # (isHuman)
                (True),
                (True)
            )
        print("\n")
        # print(self.game.displayGameState())
        self.displayGameState()
        while not self.game.isCompleted():
            print(f"Current player: {self.game.currentPlayer + 1}")
            (isHuman) = playerControls[self.game.currentPlayer]
            if isHuman:
                move = self.get_user_move()
                if not move: continue
            else:
                move = self.aiMove(self.game)
                print(f"AI move: {move}")
            # save previous board state and compare with new one
            lastBoard = self.game.board.copy()
            self.game.makeMove(move)
            # print(self.game.displayGameState())
            self.displayGameState(move, lastBoard)
        print(f"Player {self.game.currentPlayer + 1} wins!")
