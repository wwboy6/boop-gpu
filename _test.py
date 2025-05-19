import asyncio
import cupy as cp
import numpy as np
from enum import Enum
import re
import time

class BoopG:
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
        
    class Player:
        def __init__(self, player=None):
            if player == None:
                self.catCounts = [BoopG.initPieceCount, 0]  # [kittens, cats]
            else:
                self.catCounts = player.catCounts.copy()
        def __repr__(self):
            return f"[bp c:{self.catCounts}]"

    def __init__(self, saveStr=None, game=None):
        # copy game
        if game != None:
            self.board = game.board.copy()
            self.players = [self.Player(game.players[i]) for i in range(2)]
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
        self.players = [self.Player(), self.Player()]
        self.promotionOptions = []
        self.winningPieces = np.array([])
        self.playerState = self.PlayerState.PLAY_CAT
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
                case "Place": self.playerState = self.PlayerState.PLAY_CAT
                case "Promotion":
                    self.playerState = self.PlayerState.PROMOTE_CAT
                    # compute promotionOptions
                    self.promotionOptions = self.getPromotionOptions()
            # compute empty_space
            self.empty_spaces = [i for i in range(self.boardSize * self.boardSize) if self.board[i] == self.stateEmpty]

    def prepareDataForEvaluate(self) -> list:
        # TODO: TODO: expand catCounts
        return [*self.board, self.currentPlayer, *self.players[0].catCounts, *self.players[1].catCounts, self.playerState.value]
    
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
    def evaluateGames(self, dataArr):
        # Extract boards and current players
        num_games = len(dataArr)
        boards = dataArr[:, 0:36]
        current_players = dataArr[:, 36]  # [num_games]
        # TODO: TODO: extract catCounts
        catCounts0 = dataArr[:, 37:39]
        catCounts1 = dataArr[:, 39:41]
        catCountsArr = [catCounts0, catCounts1]
        playerStates = dataArr[:, 41]
        # Initialize scores
        scores = cp.zeros((num_games, 2), dtype=cp.float32)  # [num_games, num_players]

        isCompleteds = playerStates == self.PlayerState.FINISHED.value

        # TODO: check immediately win
        # win_mask = self.checkImmediatelyWin(boards, stream)  # [num_games, 2]

        isNotCompleted = cp.logical_not(isCompleteds)

        for player_idx in range(2):
            # reserve
            catCounts = catCountsArr[player_idx]
            scores[:, player_idx] += catCounts[:, 0] * self.evalScoreReserveKitten + catCounts[:, 1] * self.evalScoreReserveCat

            is_current_player = current_players == player_idx
            
            # scores[isCompleteds, player_idx] = self.evalScoreWin if is_current_player else self.evalScoreLose
            mask = cp.logical_and(isCompleteds, is_current_player)
            scores[mask, player_idx] = self.evalScoreWin
            mask = cp.logical_and(isCompleteds, cp.logical_not(is_current_player))
            scores[mask, player_idx] = self.evalScoreLose
            
            # playerZero = cp.zeros(2, dtype=cp.float32)
            # playerScoreVector = cp.zeros(2, dtype=cp.float32)
            # playerScoreVector[player_idx] = self.currentPlayerScore
            # scores += cp.where(isNotCompleted and is_current_player, playerScoreVector, playerZero)
            mask = cp.logical_and(isNotCompleted, is_current_player)
            scores[mask, player_idx] += self.currentPlayerScore
            
            # Piece scores
            player_mask = (boards // 2 == player_idx) & (boards != self.stateEmpty)
            piece_types = boards % 2
            base_scores = cp.where(piece_types == 0, self.evalScoreBoardKitten, self.evalScoreBoardCat)
            bonuses = cp.array(self.evalScoreBoardPieceBonuses, dtype=cp.float32)[None, :] * \
                      cp.array(self.evalScoreBoardPieceBonusesMul, dtype=cp.float32)[piece_types]
            piece_scores = cp.sum(cp.where(player_mask, base_scores + bonuses, 0), axis=1)
            # scores[:, player_idx] = cp.where(non_finished, scores[:, player_idx] + piece_scores, scores[:, player_idx])
            scores[isNotCompleted, player_idx] += piece_scores

        return scores

class BoopC:
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
    
    class Player:
        def __init__(self, player=None):
            if player == None:
                self.catCounts = [BoopC.initPieceCount, 0]  # [kittens, cats]
            else:
                self.catCounts = player.catCounts.copy()
        def __repr__(self):
            return f"[bp c:{self.catCounts}]"

    def __init__(self, saveStr=None, game=None):
        # copy game
        if game != None:
            self.board = game.board.copy()
            self.players = [self.Player(game.players[i]) for i in range(2)]
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
        self.players = [self.Player(), self.Player()]
        self.promotionOptions = []
        self.winningPieces = np.array([])
        self.playerState = self.PlayerState.PLAY_CAT
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
                case "Place": self.playerState = self.PlayerState.PLAY_CAT
                case "Promotion":
                    self.playerState = self.PlayerState.PROMOTE_CAT
                    # compute promotionOptions
                    self.promotionOptions = self.getPromotionOptions()
            # compute empty_space
            self.empty_spaces = [i for i in range(self.boardSize * self.boardSize) if self.board[i] == self.stateEmpty]

    def evaluate(self) -> list[float]:
        # TODO: cache in case it would be called more than once
        return [self.evaluatePlayer(i) for i in range(2)]

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

    def evaluatePlayer(self, playerIndex: int) -> float:
        # game end score
        isCurrentPlayer = playerIndex == self.currentPlayer
        
        # this happen only if opponent push player's cat to right position
        if self.playerState == self.PlayerState.FINISHED:
            return self.evalScoreWin if isCurrentPlayer else self.evalScoreLose

        score = 0
        # score for being current player
        # this score is to balance the comparison of game state with different current player
        # normally current player would have less score, and he can have at least some score to gain for a move
        if isCurrentPlayer: score += min(self.evalScoreBoardKitten-self.evalScoreReserveKitten, self.evalScoreBoardCat-self.evalScoreReserveCat) + 2
        # score of pieces in reserve
        player = self.players[playerIndex]
        rk, rc = player.catCounts
        score += rk * self.evalScoreReserveKitten + rc * self.evalScoreReserveCat
        # score of pieces on board
        for i in range(self.boardSize * self.boardSize):
            piece = self.board[i]
            if piece // 2 == playerIndex:
                base = self.evalScoreBoardKitten if piece % 2 == 0 else self.evalScoreBoardCat
                score += base + self.evalScoreBoardPieceBonuses[i] * self.evalScoreBoardPieceBonusesMul[piece % 2]

        # # if the next move is to promote, increase score by promotion
        # if self.currentPlayer == playerIndex:
        #     match self.playerState:
        #         case self.PlayerState.PROMOTE_CAT:
        #             # find all possible promotion and add the highest score that can be increased
        #             playerKitten = playerIndex * 2
        #             kittenCountInOption = [sum(1 for p in opt if self.board[p] == playerKitten) for opt in self.promotionOptions]
        #             maxCount = np.max(kittenCountInOption) if kittenCountInOption else 0
        #             scoreIncreasment = max(0, maxCount*(self.evalScoreReserveCat-self.evalScoreBoardKitten)-(3-maxCount)*(self.evalScoreBoardCat))
        #             score += scoreIncreasment

        return score

def main():
    gameCount = 1_000_000

    boardDisplay = """ _ _ _ _ _ _
    |O|x|_|_|_|X|
    |_|_|_|_|O|_|
    |_|_|X|_|_|_|
    |X|_|_|_|_|_|
    |_|_|O|_|_|_|
    |_|_|_|_|_|_|
    T T T T T T
    P0: K1 C4
    P1: K0 C4
    Current player: 1
    Place"""

    # warm up gpu
    cp.sum(cp.array([0], dtype=cp.int16))
    cp.cuda.Stream.null.synchronize()

    print("start")
    
    ### cpu
    # TODO: multi threading
    gameC = BoopC(boardDisplay)
    ts = time.perf_counter()
    for _ in range(gameCount):
        # gameC.playerState = BoopC.PlayerState.FINISHED
        scoreC = gameC.evaluate()
    print("score(C)", scoreC)
    print("time", time.perf_counter() - ts)

    ### gpu
    gameG = BoopG(boardDisplay)
    ts = time.perf_counter()
    # gameG.playerState = gameG.PlayerState.FINISHED
    data = gameG.prepareDataForEvaluate()
    # datas = cp.repeat(datas, repeats=[gameCount], axis=0)
    datas = []
    for _ in range(gameCount):
        datas.append(data)
    datas = cp.array(datas, dtype=cp.int16)
    cp.cuda.Stream.null.synchronize()
    print("time to prepare data", time.perf_counter() - ts)
    ts = time.perf_counter()
    scoreG = gameG.evaluateGames(datas)
    cp.cuda.Stream.null.synchronize()
    print("score(G)", scoreG[gameCount - 1])
    print("time", time.perf_counter() - ts)

main()
