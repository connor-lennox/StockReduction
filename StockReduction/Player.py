import threading

import chess
from chessboard import display

import ModelPersistence
import Predictor


_queued_move: chess.Move = None


def get_player_move():
    global _queued_move
    while True:
        move_string = input()
        try:
            input_move = chess.Move.from_uci(move_string)
            if board.is_legal(input_move):
                _queued_move = input_move
                return
            else:
                print("Invalid move!")
        except ValueError as e:
            print(e)


def get_model_move(maximize=True):
    return Predictor.get_next_move(board.fen(), model, maximize)


def play_model_move(maximize=True):
    model_move, evaluation = get_model_move(maximize)
    print(f"Playing {model_move} ({evaluation})")
    board.push(model_move)


if __name__ == '__main__':

    # Load a model to play against the player
    MODEL_TO_LOAD = "test_real"
    model = ModelPersistence.unpickle_model(MODEL_TO_LOAD)

    # Board setup
    board = chess.Board(chess.Board.starting_fen)
    display.start(board.fen())

    # When the model plays as white, it should be maximizing the evaluation function. When it plays
    # as black, it needs to minimize the evaluation function as large negative values are desirable
    # for the black player.
    MODEL_PLAYS_WHITE = False
    MODEL_MAXIMIZES = MODEL_PLAYS_WHITE

    # If the model is playing as white, it should make a move before the player can input anything
    # Since player color is just determined by what moves are legal according to FEN string, this
    # effectively swaps the default of the player being white.
    if MODEL_PLAYS_WHITE:
        play_model_move(MODEL_MAXIMIZES)

    # Start thread to get user input
    threading.Thread(target=get_player_move).start()

    while True:
        # Must draw the board every frame to avoid hanging and keep up to date with board state
        display.update(board.fen())

        # Listen for the player input to occur from the other thread
        if _queued_move is not None:
            # Push player move and clear queued move so that new input can be taken
            board.push(_queued_move)
            _queued_move = None

            # Check for checkmate/stalemate
            if board.is_checkmate():
                print("Checkmate by player.")
                break

            if board.is_stalemate():
                print("Stalemate")
                break

            # Model's turn
            play_model_move(MODEL_MAXIMIZES)

            # Check for checkmate/stalemate
            if board.is_checkmate():
                print("Checkmate by computer.")
                break

            if board.is_stalemate():
                print("Stalemate")
                break

            # Start thread so that input can be retrieved while still drawing board
            threading.Thread(target=get_player_move).start()
