from typing import Protocol, Self

class GameGpu(Protocol):

    def getPlayerCount(self) -> int: pass

    def getCurrentPlayer(self) -> int: pass

    def getPossibleMoves(self) -> list: pass

    # TODO: consider different results can be obtained by chance
    def makeMove(self, move) -> Self: pass

    def isCompleted(self) -> bool: pass

    # evaluate using CPU
    
    # def evaluate(self) -> list[float]: pass

    # evaluate using GPU
    
    # TODO: report memory usage for game state grouping -> let user / searcher descide how many group should be made
    
    def prepareDataForEvaluate() -> list: pass

    # dataArr should be using CuPy-like library
    @staticmethod
    def evaluateGames(dataArr): pass
