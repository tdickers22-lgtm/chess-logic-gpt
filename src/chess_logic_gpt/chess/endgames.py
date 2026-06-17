from __future__ import annotations

import random

import chess

from chess_logic_gpt.records import stable_id


ENDGAME_TEMPLATES: list[list[tuple[chess.Color, chess.PieceType]]] = [
    [(chess.WHITE, chess.QUEEN)],
    [(chess.WHITE, chess.ROOK)],
    [(chess.WHITE, chess.BISHOP), (chess.WHITE, chess.KNIGHT)],
    [(chess.WHITE, chess.PAWN)],
    [(chess.WHITE, chess.ROOK), (chess.BLACK, chess.PAWN)],
    [(chess.WHITE, chess.QUEEN), (chess.BLACK, chess.ROOK)],
    [(chess.WHITE, chess.ROOK), (chess.BLACK, chess.BISHOP)],
    [(chess.WHITE, chess.ROOK), (chess.BLACK, chess.KNIGHT)],
    [(chess.WHITE, chess.BISHOP), (chess.WHITE, chess.PAWN), (chess.BLACK, chess.PAWN)],
    [(chess.WHITE, chess.KNIGHT), (chess.WHITE, chess.PAWN), (chess.BLACK, chess.PAWN)],
    [(chess.WHITE, chess.PAWN), (chess.WHITE, chess.PAWN), (chess.BLACK, chess.PAWN)],
]


def generate_endgame_candidates(n: int, seed: int = 0, max_attempts: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    max_attempts = max_attempts or n * 200
    records: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(records) < n and attempts < max_attempts:
        attempts += 1
        template = rng.choice(ENDGAME_TEMPLATES)
        if rng.random() < 0.5:
            template = invert_template(template)
        board = random_board_from_template(rng, template)
        if board is None:
            continue
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        records.append(to_record(fen, template, len(records)))
    if len(records) < n:
        raise RuntimeError(f"generated only {len(records)} valid endgame candidates after {attempts} attempts")
    return records


def invert_template(template: list[tuple[chess.Color, chess.PieceType]]) -> list[tuple[chess.Color, chess.PieceType]]:
    return [(not color, piece_type) for color, piece_type in template]


def random_board_from_template(
    rng: random.Random,
    template: list[tuple[chess.Color, chess.PieceType]],
) -> chess.Board | None:
    board = chess.Board.empty()
    occupied: set[chess.Square] = set()

    white_king, black_king = random_kings(rng)
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    occupied.update([white_king, black_king])

    for color, piece_type in template:
        square = random_square_for_piece(rng, piece_type, occupied)
        if square is None:
            return None
        board.set_piece_at(square, chess.Piece(piece_type, color))
        occupied.add(square)

    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.clear_stack()

    if board.status() != chess.STATUS_VALID:
        return None
    return board


def random_kings(rng: random.Random) -> tuple[chess.Square, chess.Square]:
    while True:
        white_king = rng.randrange(64)
        black_king = rng.randrange(64)
        if white_king == black_king:
            continue
        if chess.square_distance(white_king, black_king) > 1:
            return white_king, black_king


def random_square_for_piece(
    rng: random.Random,
    piece_type: chess.PieceType,
    occupied: set[chess.Square],
) -> chess.Square | None:
    candidates = list(range(64))
    rng.shuffle(candidates)
    for square in candidates:
        if square in occupied:
            continue
        if piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
            continue
        return square
    return None


def to_record(
    fen: str,
    template: list[tuple[chess.Color, chess.PieceType]],
    index: int,
) -> dict:
    board = chess.Board(fen)
    pieces = [
        f"{'white' if color == chess.WHITE else 'black'} {chess.piece_name(piece_type)}"
        for color, piece_type in template
    ]
    prompt = (
        "This is a legal low-piece chess endgame candidate for exact Syzygy labeling.\n\n"
        f"FEN: {fen}\n"
        f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}\n"
        f"Extra material beyond kings: {', '.join(pieces) if pieces else 'none'}\n\n"
        "After Syzygy labeling, explain the exact WDL/DTZ result from the side-to-move perspective."
    )
    return {
        "id": stable_id("endgame-candidate", index, fen),
        "domain": "chess",
        "task": "syzygy_endgame_candidate",
        "source": {
            "name": "generated_endgame_positions",
            "url": "generated",
            "license": "generated-by-project",
            "provenance": "generated",
        },
        "messages": [
            {"role": "system", "content": "You are a precise chess endgame model. Exact tablebase labels outrank intuition."},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "Pending Syzygy verification. Do not include this unlabeled candidate in training.",
            },
        ],
        "verification": {
            "status": "unverified",
            "method": "legal-position-generator",
        },
        "metadata": {
            "fen": fen,
            "material_template": pieces,
        },
    }
