def elaborateState(game, evaluation):
    game = game.copy()
    while evaluation != None:
        if evaluation.bestMove == None: break
        print(f"p{game.currentPlayer} {evaluation.bestMove}")
        game.makeMove(evaluation.bestMove)
        print(game)
        if evaluation.children == None:
            break
        else:
            evaluation = evaluation.children[evaluation.childIndex]
    print(f"eval: {evaluation.evaluation}")

def countEndChild(evaluation):
    if not evaluation.children:
        return 1
    return sum([countEndChild(child) for child in evaluation.children])
