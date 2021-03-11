import chess

import FeatureExtraction


def get_next_move(fen_string, model):
    start_board = chess.Board(fen_string)

    best_move = None
    best_evaluation = 0

    for i, move in enumerate(start_board.legal_moves):
        start_board.push(move)
        test_features = FeatureExtraction.features_from_board(start_board)

        evaluation = model(test_features).item()
        if i == 0 or evaluation > best_evaluation:
            best_move = move
            best_evaluation = evaluation

        # Remove the move from the stack so we're always evaluating from the same start state
        start_board.pop()

    return best_move, best_evaluation
