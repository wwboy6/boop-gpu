import concurrent
import multiprocessing
import asyncio
from math import nan
import cupy as cp
from gamegpu import GameGpu
from asyncutil import wrapAsCoroutine

class BiasedMaxSearcherGpu:

    class Evaluation:
        # TODO: add game
        def __init__(self, currentMove, evaluation, bestMove = None, children = None, childIndex = nan):
            self.currentMove = currentMove
            self.evaluation = evaluation
            self.bestMove = bestMove
            self.children = children
            self.childIndex = childIndex
    
    class EvaluationRegistration:
        def __init__(self, future, startIndex, endIndex):
            self.future = future
            self.startIndex = startIndex
            self.endIndex = endIndex
    
    def __init__(self, game: GameGpu, max_workers = 8):
        self.game = game
        self.playerCount = game.getPlayerCount()
        self.max_workers = max_workers
        # FIXME: make sure it is seperated for each thread
        self.evaluationDatas = []
        self.evaluationDatasIndex = 0
        self.evaluationRegistrations = []
        self.eventLoop = asyncio.get_event_loop()
        self.buildBiasedEvaluateKernel(self.playerCount)
    
    # @staticmethod
    # def biasedEvaluate(playerCount: int, scores: list[float]) -> list[float]:
    #     return [scores[playerIndex] - cp.sum(v for i, v in enumerate(scores) if i != playerIndex) / (playerCount - 1) for playerIndex in range(playerCount)]
    
    # @staticmethod
    # def biasedEvaluate(playerCount: int, scoresArr: cp.ndarray[cp.ndarray[float]]) -> cp.ndarray[cp.ndarray[float]]:
    #     # Compute total sum for each score set (shape: [num_sets])
    #     total_sums = cp.sum(scoresArr, axis=1)
    #     # Broadcast total_sums to shape [num_sets, playerCount]
    #     # Subtract scoresArr to exclude each player's score
    #     sum_excluding_self = total_sums[:, cp.newaxis] - scoresArr
    #     # Compute mean of other players' scores
    #     mean_excluding_self = sum_excluding_self / (playerCount - 1)
    #     # Compute final result: scores - mean_excluding_self
    #     result = scoresArr - mean_excluding_self
    #     return result

    # biasedEvaluateKernal
    def buildBiasedEvaluateKernel(self, playerCount):
        # FIXME: support playerCount > 2
        assert(playerCount == 2)
        in_params = ", ".join([f"float32 s{i}" for i in range(playerCount)]) # float32 s0, float32 s1
        out_params = ", ".join([f"float32 r{i}" for i in range(playerCount)]) # float32 r0, float32 r1
        if playerCount == 2:
            operation='''
                r0 = s0 - s1;
                r1 = s1 - s0;
            '''
        else:
            # FIXME:
            pass
        self.biasedEvaluateKernal = cp.ElementwiseKernel(
            in_params = in_params,
            out_params = out_params,
            operation = operation,
            name = f"biased_evaluation_kernal_p{playerCount}"
        )
    
    def biasedEvaluate(self, scoresArr):
        inParams = [scoresArr[:, i] for i in range(self.playerCount)]
        result = cp.empty_like(scoresArr)
        outParams = [result[:, i] for i in range(self.playerCount)]
        self.biasedEvaluateKernal(*inParams, *outParams)
        return result

    # return: (scores, nextMove, children, choosenIndex)
    def evaluate(self, depth=3, traceChildren=False, game=None, parallel=True) -> Evaluation:
        if game is None:
            game = self.game
        with multiprocessing.Manager() as manager:
            self.cache = manager.dict()
            resultFuture = self._evaluate(depth, traceChildren, game, None, parallel, True)
            result = asyncio.get_event_loop().run_until_complete(resultFuture)
        self.cache = None
        return result
    
    # FIXME: this return future
    async def _evaluate(self, depth, traceChildren, game, currentMove, parallel=False, isTopLevel=False) -> asyncio.Future[Evaluation] :
        # Cache key based on game state
        state_key = (depth, game.currentPlayer, game.playerState.value, tuple(map(tuple, game.board.reshape(-1, 2))))

        if state_key in self.cache:
            return self.cache[state_key]
        
        # TODO: check cache in progress
        # if the cache is in progress, suspend the evaluation and check it again later

        if game.isCompleted():
            result = self.Evaluation(currentMove, game.evaluate())
            if not state_key in self.cache: self.cache[state_key] = result
            return result
        
        moves = None
        # shortcut searching for winning
        winningMove = game.checkImmediatelyWin()
        if winningMove != None:
            moves = [winningMove]
        else:
            moves = game.getPossibleMoves()

        if not moves:
            # game.getPossibleMoves should return at least one move. Even pass turn should be a move
            raise Exception("no move is available")
        
        # shortcut for only 1 move
        if isTopLevel and len(moves) == 1:
            return self.Evaluation(None, None, moves[0])
        
        # TODO: check some of moves instead of every moves to reduce workload, but how to choose move?

        evaluationsTask = None

        if depth == 1:
            # At depth 1, evaluate moves using evaluator
            evaluationsTask = self._evaluateMovesGpu(game, moves)
            # trigger _performEvaluateMovesGpu for last child of the whole searching
        else:
            # TODO: dynamic thread allocation: check available thread (cpu core) can be obtained and create new executor w.r.t. that number
            # with a max thread limit per tree layer, say 1 for first layer and 3 other layer
            # when a sub-search using executor is completed, release the available thread count
            if parallel and len(moves) > 1:
                # Use multi-threading for deeper evaluations
                evaluations = self._evaluateMovesParallel(moves, depth - 1, traceChildren, game)
            else:
                evaluationFutures = [self._evaluate(depth - 1, traceChildren, game.copy().makeMove(m), m) for m in moves]
                evaluationsTask = wrapAsCoroutine(asyncio.gather(*evaluationFutures))

        if evaluationsTask:
            if isTopLevel:
                # run task until it return to the loop
                evaluationsTask = asyncio.create_task(evaluationsTask)
                await asyncio.sleep(0)
                # flush evaluationRegistrations after all states are prepared
                self._performEvaluateMovesGpu()
            evaluations = await evaluationsTask

        evaluations = sorted(evaluations, key=lambda x: x.evaluation[game.currentPlayer], reverse=True)
        # TODO: handle more than 1 best choice
        bestIndex = 0
        bestEval = evaluations[bestIndex]
        
        # trace children
        if traceChildren:
            evaluation = self.Evaluation(currentMove, bestEval.evaluation, bestEval.currentMove, evaluations, bestIndex)
        else:
            evaluation = self.Evaluation(currentMove, bestEval.evaluation, bestEval.currentMove, [evaluations[bestIndex]], 0)
        
        if not state_key in self.cache: self.cache[state_key] = evaluation
        return evaluation
    
    # multi-threading using concurrent

    def _evaluateMovesParallel(self, moves, depth, traceChildren, game):
        """Evaluates all moves in parallel using ProcessPoolExecutor."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all move evaluations to the thread pool
            moveCount = len(moves)
            futureToMove = {executor.submit(self._evaluateMove, move, depth, traceChildren, game): move for i, move in enumerate(moves)}
            # Collect results as they complete
            evaluations = [future.result() for future in concurrent.futures.as_completed(futureToMove)]
        return evaluations

    def _evaluateMove(self, move, depth, traceChildren, game):
        """Helper function to evaluate a single move (used in threading)."""
        next_game = game.copy().makeMove(move)
        self.eventLoop = asyncio.get_event_loop()
        result = self.eventLoop.run_until_complete(self._evaluate(depth, traceChildren, next_game, move))
        print(f"_evaluateMove {result}")
        return result

    # evaluationDataCountThreshold = 10000000
    evaluationDataCountThreshold = 100
    # async GPU process using asyncio
    async def _evaluateMovesGpu(self, game, moves):
        moveCount = len(moves)
        datas = [game.copy().makeMove(m).prepareDataForEvaluate() for m in moves]
        future = self.eventLoop.create_future()
        registration = self.EvaluationRegistration(future, self.evaluationDatasIndex, self.evaluationDatasIndex+moveCount)
        self.evaluationDatasIndex += moveCount
        self.evaluationDatas.extend(datas)
        self.evaluationRegistrations.append(registration)
        if self.evaluationDatasIndex >= self.evaluationDataCountThreshold:
            self._performEvaluateMovesGpu()
        bes = await future
        print(f"_evaluateMovesGpu {bes}")
        return [self.Evaluation(moves[i], bes[i]) for i in range(moveCount)]
    
    def _performEvaluateMovesGpu(self):
        if self.evaluationDatasIndex == 0: return
        evaluationDatas = self.evaluationDatas
        evaluationRegistrations = self.evaluationRegistrations
        self.evaluationDatas = []
        self.evaluationDatasIndex = 0
        self.evaluationRegistrations = []
        # TODO: it supports only int8 for now
        scoresArr = self.game.evaluateGames(cp.array(evaluationDatas, dtype=cp.int8))
        bes = self.biasedEvaluate(scoresArr)
        cp.cuda.Stream.null.synchronize()
        bes = cp.asarray(bes)
        print(f"_performEvaluateMovesGpu {bes}")
        for er in evaluationRegistrations:
            er.future.set_result(bes[er.startIndex:er.endIndex])
