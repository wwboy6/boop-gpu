import numpy as np
import cupy as cp
from enum import Enum
import re

# Precompute connections once
def boop_setup(cls):
    connections = []
    directions = [
        [(0, 0), (1, 0), (2, 0)],  # right
        [(0, 0), (1, 1), (2, 2)],  # right bottom
        [(0, 0), (0, 1), (0, 2)],  # bottom
        [(0, 0), (-1, 1), (-2, 2)],  # left bottom
    ]
    for x in range(cls.boardSize):
        for y in range(cls.boardSize):
            for dir in directions:
                connection = [(x + dx, y + dy) for dx, dy in dir]
                if all(0 <= pos[0] < cls.boardSize and 0 <= pos[1] < cls.boardSize for pos in connection):
                    connections.append(tuple(connection))
    connections = np.array(connections, dtype="int")
    cls.allPositions = [i for i in range(cls.boardSize * cls.boardSize)]
    # for checking if player can immediately win
    # data structure [[2d_pos0, 2d_pos1, ...], [2d_empty_pos0, 2d_empty_pos1, ...], 2d_win_pos]
    # the immediate winning loggic would check if all 2d_pos has current player's cats,
    # and all 2d_empty_pos and 2d_win_pos are empty
    # 2d_win_pos would be the placement of cat to be immediately win
    # win by 2 nearby cats with an available spot
    wpp = [[[c[0], c[1]], [], c[2]] for c in connections]
    if __name__ == "__main__": print(f"wpp {wpp[0]}")
    winningPatterns = wpp
    wpp = [[[c[1], c[2]], [], c[0]] for c in connections]
    if __name__ == "__main__": print(f"wpp {wpp[0]}")
    winningPatterns.extend(wpp)
    # win by 2 cats in a single connection (not nearby), with a cat next to the middle spot, and an available spot to push that cat into it
    # direction from middle to third cat is perpendicular with the connection
    # mirror the direction of the connection along x=y => flip(c1-c0)
    # mirror the it again along y=0, x=0 => *[-1,1], *[1,-1]
    extractMiddle = [[[c[0], c[2]], c[1], np.flip(c[1] - c[0]) * [-1, 1]] for c in connections]
    extractMiddle.extend([[[c[0], c[2]], c[1], np.flip(c[1] - c[0]) * [1, -1]] for c in connections])
    wpp = [[[*em[0], em[1]+em[2]], [em[1]], em[1]+em[2]+em[2]] for em in extractMiddle]
    wpp = [p for p in wpp if not cls.isOutOfBoard(*p[2])]
    if __name__ == "__main__": print(f"wpp {wpp[0]}, len {len(wpp)}")
    winningPatterns.extend(wpp)
    if __name__ == "__main__": print(f"wp count {len(winningPatterns)}")
    # convert those data into 1d array index system
    cls.connections = [cls.posAToIndexA(conn) for conn in connections]
    cls.winningPatterns = [[cls.posAToIndexA(wp[0]), cls.posAToIndexA(wp[1]), cls.posToIndex(*wp[2])] for wp in winningPatterns]
    # winningPatterns would be indexed by first element of pos (2d_pos0) as winningPatternMap
    cls.winningPatternMap = dict()
    for p in cls.winningPatterns:
        pos0 = p[0][0]
        if not pos0 in cls.winningPatternMap:
            cls.winningPatternMap[pos0] = [p]
        else:
            cls.winningPatternMap[pos0].append(p)
    return cls

@boop_setup
class Boop:
    boardSize = 6
    # stateKitten0 = 0  # Player 0, kitten
    # stateCat0 = 1     # Player 0, cat
    # stateKitten1 = 2  # Player 1, kitten
    # stateCat1 = 3     # Player 1, cat
    stateEmpty = 4
    stateDisplays = np.array(["o", "O", "x", "X", "_"])
    connectionLength = 3
    initPieceCount = 8
    # Adjacent offsets for booping (excluding 0,0), corresponding to [-1,-1] to [1,1]
    adjacentOffsets = [-7, -6, -5, -1, 1, 5, 6, 7]

    class PlayerState(Enum):
        PLAY_CAT = 0
        PROMOTE_CAT = 1
        FINISHED = 2
    @classmethod
    def isOutOfBoard(cls, x, y):
        return x < 0 or x >= cls.boardSize or y < 0 or y >= cls.boardSize
    
    @classmethod
    def posToIndex(cls, x, y):
        return x * cls.boardSize + y
    
    @classmethod
    def posAToIndexA(cls, posA):
        return [cls.posToIndex(*pos) for pos in posA]
    
    @classmethod
    def indexToPos(cls, index):
        return (index // cls.boardSize, index % cls.boardSize)

    class Player:
        def __init__(self, player=None):
            if player == None:
                self.catCounts = [Boop.initPieceCount, 0]  # [kittens, cats]
            else:
                self.catCounts = player.catCounts.copy()
        def __repr__(self):
            return f"[bp c:{self.catCounts}]"

    def __init__(self, saveStr=None, game=None):
        # copy game
        if game != None:
            self.board = game.board.copy()
            self.players = [Boop.Player(game.players[i]) for i in range(2)]
            self.currentPlayer = game.currentPlayer
            self.playerState = game.playerState
            self.promotionOptions = game.promotionOptions.copy()
            self.winningPieces = game.winningPieces.copy()
            # TODO: i doubt the performance improvement of using empty_spaces, as it would be copied for every new step
            self.empty_spaces = game.empty_spaces.copy()
            # no need to copy caches like immediatelyWinningMove
            self.immediatelyWinningMove = None
            return
        #
        self.players = [Boop.Player(), Boop.Player()]
        self.promotionOptions = []
        self.winningPieces = np.array([])
        self.playerState = Boop.PlayerState.PLAY_CAT
        self.immediatelyWinningMove = None
        self.lastMove = None
        if saveStr == None:
            self.board = np.full(self.boardSize * self.boardSize, self.stateEmpty, dtype="int8")
            self.currentPlayer = 0
            self.empty_spaces = list(range(self.boardSize * self.boardSize))
        else:
            # load save
            sdl = list(self.stateDisplays)
            lines = saveStr.split('\n')
            # parse map
            mapStr = [line.split('|')[1:7] for line in lines[1:7]]
            board = []
            for row in mapStr:
                for v in row:
                    board.append(sdl.index(v))
            self.board = np.array(board, dtype="int8")
            # parse tokens
            self.players[0].catCounts = [int(str) for str in re.findall(r"\d", lines[8])[1:3]]
            self.players[1].catCounts = [int(str) for str in re.findall(r"\d", lines[9])[1:3]]
            # parse current player
            self.currentPlayer = int(re.findall(r"\d", lines[10])[0]) - 1
            # parse state
            match re.search(r"\w+", lines[11])[0]:
                case "Place": self.playerState = Boop.PlayerState.PLAY_CAT
                case "Promotion":
                    self.playerState = Boop.PlayerState.PROMOTE_CAT
                    # compute promotionOptions
                    self.promotionOptions = self.getPromotionOptions()
            # compute empty_space
            self.empty_spaces = [i for i in range(self.boardSize * self.boardSize) if self.board[i] == self.stateEmpty]

    def getPromotionOptions(self):
        allOptions = []
        player = self.players[self.currentPlayer]
        # if all pieces are placed on board, 1 kitten can be promoted 
        playerKitten = self.currentPlayer * 2
        if player.catCounts == [0, 0]:
            allOptions.extend([(i,) for i in self.allPositions if self.board[i] == playerKitten])
        # check if any pieces form line
        for conn in self.connections:
            if all(self.board[pos] // 2 == self.currentPlayer for pos in conn):
                allOptions.append(tuple(conn))
        return allOptions

    def getWinningConnections(self):
        playerCat = self.currentPlayer * 2 + 1
        bcPositions = [i for i in self.allPositions if self.board[i] == playerCat]
        if len(bcPositions) == self.initPieceCount:
            self.winningPieces = np.array(bcPositions)
        for conn in self.connections:
            if all(self.board[pos] == playerCat for pos in conn):
                self.winningPieces = np.unique(conn, axis=0)
        return self.winningPieces

    def playCat(self, idx, isBig):
        # TODO: prohibit move that produce same game state as before ??
        player = self.players[self.currentPlayer]
        piece = self.currentPlayer * 2 + isBig
        player.catCounts[isBig] -= 1
        self.empty_spaces.remove(idx)
        self.board[idx] = piece

        x, y = self.indexToPos(idx)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                tx, ty = x + dx, y + dy
                if self.isOutOfBoard(tx, ty):
                    continue
                tidx = self.posToIndex(tx, ty)
                target = self.board[tidx]
                if target == self.stateEmpty or target % 2 > isBig:
                    continue
                nx, ny = tx + dx, ty + dy
                if self.isOutOfBoard(nx, ny):
                    self.players[target // 2].catCounts[target % 2] += 1
                    self.board[tidx] = self.stateEmpty
                    self.empty_spaces.append(tidx)
                elif self.board[self.posToIndex(nx, ny)] == self.stateEmpty:
                    nidx = self.posToIndex(nx, ny)
                    self.board[nidx] = target
                    self.board[tidx] = self.stateEmpty
                    self.empty_spaces.remove(nidx)
                    self.empty_spaces.append(tidx)
        self.checkNextState()

    def checkNextState(self):
        if len(self.getWinningConnections()) > 0:
            self.playerState = Boop.PlayerState.FINISHED
        else:
            self.promotionOptions = self.getPromotionOptions()
            if self.promotionOptions:
                self.playerState = Boop.PlayerState.PROMOTE_CAT
            else:
                self.currentPlayer = 1 - self.currentPlayer
                self.playerState = Boop.PlayerState.PLAY_CAT

    def promote(self, index):
        for pos in self.promotionOptions[index]:
            self.board[pos] = self.stateEmpty
            self.empty_spaces.append(pos)
        self.players[self.currentPlayer].catCounts[1] += len(self.promotionOptions[index])
        self.currentPlayer = 1 - self.currentPlayer
        self.playerState = Boop.PlayerState.PLAY_CAT

    def displayBoardState(self):
        result = " _ _ _ _ _ _ \n"
        for i in range(0, self.boardSize * self.boardSize, self.boardSize):
            line = "|".join(self.stateDisplays[self.board[i + j]] for j in range(self.boardSize))
            result += f"|{line}|\n"
        result += " T T T T T T "
        return result

    def displayGameState(self):
        board = self.displayBoardState()
        players = "\n".join([f"P{i}: K{p.catCounts[0]} C{p.catCounts[1]}" for i, p in enumerate(self.players)])
        return f"{board}\n{players}"
    
    def __repr__(self):
        return f"[Boop\n{self.displayBoardState()}\nps:{self.players} cp:{self.currentPlayer} s:{self.playerState}\nprom:{self.promotionOptions}\nwp:{self.winningPieces}]"

    def getPlayerCount(self) -> int:
        return 2

    def getCurrentPlayer(self) -> int:
        return self.currentPlayer
    
    # check if current player can win by actions without any other factor
    # return the next action to win
    def checkImmediatelyWin(self) -> any:
        # return None
        # current player need to place cat to win on next move
        if self.playerState is not self.PlayerState.PLAY_CAT or self.players[self.currentPlayer].catCounts[1] == 0: return None
        # return cache if any
        if self.immediatelyWinningMove != None: return self.immediatelyWinningMove
        # filter pattern
        playerCat = self.currentPlayer * 2 + 1
        # find pattern with matching cat
        pattern = None
        # search for cat and find potential patterns in winningPatternMap
        # TODO: loop with self.winningPatternMap key only
        for pos in [i for i in self.allPositions if self.board[i] == playerCat]:
            patterns = self.winningPatternMap.get(pos, [])
            for p in patterns:
                nextPattern = False
                for pos in p[0]:
                    if self.board[pos] != playerCat:
                        nextPattern = True
                        break
                if nextPattern: continue
                for pos in p[1]:
                    if self.board[pos] != self.stateEmpty:
                        nextPattern = True
                        break
                if nextPattern: continue
                if self.board[p[2]] != self.stateEmpty: continue
                pattern = p
                break
            if pattern: break
        if pattern:
            self.immediatelyWinningMove = ("playCat", [pattern[2], 1])
        return self.immediatelyWinningMove

    def getPossibleMoves(self) -> list:
        # TODO: prioritize moves
        if self.playerState == Boop.PlayerState.PLAY_CAT:
            player = self.players[self.currentPlayer]
            moves = []
            if player.catCounts[0]:
                moves.extend(("playCat", (i, 0)) for i in self.empty_spaces)
            if player.catCounts[1]:
                moves.extend(("playCat", (i, 1)) for i in self.empty_spaces)
            return moves
        elif self.playerState == Boop.PlayerState.PROMOTE_CAT:
            return [("promoteCat", i) for i in range(len(self.promotionOptions))]
        # TODO: prohibit move that produce same game state as before ??
        return []

    @classmethod
    def describeMove(cls, move):
        if move is None: return None
        match(move[0]):
            case 'playCat':
                return [move[0], [*cls.indexToPos(move[1][0]), move[1][1]]]
            case _:
                return move
    
    # this is faster than copy.deepcopy
    def copy(self) -> 'Boop':
        return Boop(game=self)

    def makeMove(self, move) -> 'Boop':
        action, value = move
        if action == "playCat":
            self.playCat(*value)
        elif action == "promoteCat":
            self.promote(value)
        return self

    def isCompleted(self) -> bool:
        return self.playerState == Boop.PlayerState.FINISHED

    evalScoreReserveKitten = 0
    evalScoreBoardKitten = 100
    evalScoreReserveCat = 400 # upgrading 3 kitten in the center of board = (400-100)*3-4-4-3 = 889 >>> removing 1 kitten / cat on edge
    evalScoreBoardCat = 495 # score loss of removing kitten on edge is more than removing cat
    # FIXME: promoting 1 kitten with 2 cats = 300 - 101*2 = 98
    evalScoreBoardPieceBonuses = np.array([
        [0, 1, 1, 1, 1, 0],
        [1, 2, 3, 3, 2, 1],
        [1, 3, 4, 4, 3, 1],
        [1, 3, 4, 4, 3, 1],
        [1, 2, 3, 3, 2, 1],
        [0, 1, 1, 1, 1, 0],
    ], dtype="int").flatten()
    evalScoreBoardPieceBonusesMul = [1, 2] # score gain of placing kitten in the center is more than placing cat
    # set a large value for winning
    evalScoreWin = (evalScoreBoardCat * (initPieceCount - 1) + 2*16 + evalScoreReserveCat) * 2
    evalScoreLose = 0

    currentPlayerScore = min(evalScoreBoardKitten - evalScoreReserveKitten,
                             evalScoreBoardCat - evalScoreReserveCat) + 2

    def prepareDataForEvaluate(self) -> list:
        return [*self.board, self.currentPlayer, self.playerState.value]
    
    def evaluateGames(self, dataArr):
        # Extract boards and current players
        num_games = len(dataArr)
        boards = dataArr[:, 0:36]
        current_players = dataArr[:, 36:37]  # [num_games]
        playerStates = dataArr[:, 37:38] # FIXME: TODO: FINISHED
        # Initialize scores
        scores = cp.zeros((num_games, 2), dtype=cp.float32)  # [num_games, num_players]

        isCompleteds = playerStates == self.PlayerState.FINISHED

        # TODO: check immediately win
        # win_mask = self.checkImmediatelyWin(boards, stream)  # [num_games, 2]

        isNotCompleted = not isCompleteds

        for player_idx in range(2):
            # # Game end scores
            is_current_player = current_players == player_idx
            
            # scores[isCompleteds, player_idx] = self.evalScoreWin if is_current_player else self.evalScoreLose
            scores[isCompleteds and is_current_player, player_idx] = self.evalScoreWin
            scores[isCompleteds and not is_current_player, player_idx] = self.evalScoreLose
            
            # scores[:, player_idx] = cp.where(win_mask[:, player_idx],
            #                                 cp.where(is_current_player, self.evalScoreWin, self.evalScoreLose),
            #                                 scores[:, player_idx])
            
            # # Non-finished games
            # non_finished = ~win_mask[:, player_idx]
            # scores[:, player_idx] = cp.where(non_finished,
            #                                 scores[:, player_idx] + cp.where(is_current_player,
            #                                                                 min(self.evalScoreBoardKitten - self.evalScoreReserveKitten,
            #                                                                     self.evalScoreBoardCat - self.evalScoreReserveCat) + 2,
            #                                                                 0),
            #                                 scores[:, player_idx])
            # TODO: check immediately win
            
            # FIXME: this works in np but not in cp
            # dummy = cp.where(is_current_player, float(self.currentPlayerScore), 0)
            # print(dummy.shape, scores[:, player_idx].shape)
            # scores[:, player_idx] += dummy

            # playerZero = cp.zeros(2, dtype=cp.float32)
            playerScoreVector = cp.zeros(2, dtype=cp.float32)
            playerScoreVector[player_idx] = self.currentPlayerScore
            # scores += cp.where(isNotCompleted and is_current_player, playerScoreVector, playerZero)
            scores[isNotCompleted and is_current_player, player_idx] += playerScoreVector
            
            # Piece scores
            player_mask = (boards // 2 == player_idx) & (boards != self.stateEmpty)
            piece_types = boards % 2
            base_scores = cp.where(piece_types == 0, self.evalScoreBoardKitten, self.evalScoreBoardCat)
            bonuses = cp.array(self.evalScoreBoardPieceBonuses, dtype=cp.float32)[None, :] * \
                      cp.array(self.evalScoreBoardPieceBonusesMul, dtype=cp.float32)[piece_types]
            piece_scores = cp.sum(cp.where(player_mask, base_scores + bonuses, 0), axis=1)
            # scores[:, player_idx] = cp.where(non_finished, scores[:, player_idx] + piece_scores, scores[:, player_idx])
            scores[isNotCompleted, player_idx] += piece_scores
            
            # TODO:
            # # Promotion scoring
            # promote_mask = (current_players == player_idx) & (playerStates == self.PlayerState.PROMOTE_CAT)
            # if cp.any(promote_mask):
            #     maxCount = 3  # Simplified
            #     score_increasment = max(0, maxCount * (self.evalScoreReserveCat - self.evalScoreBoardKitten) - 
            #                           (3 - maxCount) * (self.evalScoreBoardCat))
            #     scores[:, player_idx] = cp.where(promote_mask & non_finished,
            #                                     scores[:, player_idx] + score_increasment,
            #                                     scores[:, player_idx])
        
        return scores
