import chess
import numpy as np
import torch


embeddings = torch.eye(13)

piece_dict = {
    chess.PAWN: 1,
    chess.ROOK: 2,
    chess.KNIGHT: 3,
    chess.BISHOP: 4,
    chess.QUEEN: 5,
    chess.KING: 6
}


def features_from_fen(fen_string):
    # Convert fen string to a board
    board = chess.Board(fen_string)

    # Return board-based features
    return features_from_board(board)


def features_from_board(board):
    # Construct one-hot features for piece types on a given square
    layout_pieces = [[board.piece_at(chess.square(file, rank)) for file in range(8)] for rank in range(8)]
    layout_nums = torch.flatten(
        torch.tensor([[piece_dict[layout_pieces[rank][file].piece_type] + 6 * (layout_pieces[rank][file].color == chess.BLACK)
                       if layout_pieces[rank][file] is not None else 0 for file in range(8)] for rank in range(8)])
    )

    layout_features = embeddings[layout_nums]

    # Count features: counts of each piece type (numeric)

    # Get counts for each piece type
    counts = np.zeros(13)
    us, cs = torch.unique(layout_nums, return_counts=True)
    for u, c in zip(us, cs):
        counts[u] = c

    # Convert counts to a tensor for concatenation purposes
    counts = torch.tensor(counts)

    # Indicator variables for current turn
    turn = torch.tensor([board.turn == chess.WHITE, board.turn == chess.BLACK])

    # Castling availability per player
    castling = torch.tensor([
        board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)
    ])

    # En passant features - not needed: rare scenario and difficult to encode
    # Turn/Ply features - not needed: only care about board position in this instant

    # Concatenate all extra features together
    extra_features = torch.cat([counts, turn, castling])

    # Flatten layout features into a 832 length vector and concatenate with extra features
    flat_features = torch.cat([torch.flatten(layout_features), extra_features])

    return flat_features.float()


if __name__ == '__main__':
    from timeit import timeit

    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    test_board = chess.Board(test_fen)

    # print(timeit(lambda: chess.Board(test_fen), number=5000))
    # print(timeit(lambda: features_from_board(test_board), number=10000))

    print(features_from_fen(test_fen))
