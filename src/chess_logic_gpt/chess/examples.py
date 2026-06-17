from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.pgn

from chess_logic_gpt.records import stable_id


PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


@dataclass(frozen=True)
class PositionSample:
    game_id: str
    ply: int
    fen: str
    san_history: str
    next_move_san: str
    result: str


def sample_positions_from_game(
    game: chess.pgn.Game,
    game_index: int,
    every_n_plies: int = 8,
    min_ply: int = 10,
    max_positions: int = 12,
) -> list[PositionSample]:
    board = game.board()
    sans: list[str] = []
    out: list[PositionSample] = []
    moves = list(game.mainline_moves())
    result = game.headers.get("Result", "*")
    game_id = game.headers.get("LichessId") or game.headers.get("Site") or f"game-{game_index}"
    for ply, move in enumerate(moves):
        san = board.san(move)
        if ply >= min_ply and ply % every_n_plies == 0 and len(out) < max_positions:
            out.append(
                PositionSample(
                    game_id=str(game_id),
                    ply=ply,
                    fen=board.fen(),
                    san_history=" ".join(sans[-24:]),
                    next_move_san=san,
                    result=result,
                )
            )
        board.push(move)
        sans.append(san)
    return out


def board_facts(board: chess.Board) -> list[str]:
    facts: list[str] = []
    facts.append(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}.")
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    facts.append(f"Castling rights: {board.castling_xfen() or '-'}; en passant: {ep}.")
    facts.append(f"Legal moves: {', '.join(board.san(m) for m in list(board.legal_moves)[:80])}.")
    if board.is_check():
        facts.append("The side to move is in check.")
    for color, color_name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
        pieces = []
        for piece_type in [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            squares = board.pieces(piece_type, color)
            if squares:
                sq = ", ".join(chess.square_name(s) for s in sorted(squares))
                pieces.append(f"{PIECE_NAMES[piece_type]}: {sq}")
        facts.append(f"{color_name} pieces: {'; '.join(pieces)}.")
    return facts


def pgn_sample_to_record(sample: PositionSample) -> dict:
    board = chess.Board(sample.fen)
    facts = board_facts(board)
    prompt = (
        "Study this chess position and infer the next human move from the game. "
        "List concrete board facts first, then candidate ideas, then give the move.\n\n"
        f"FEN: {sample.fen}\n"
        f"Recent SAN history: {sample.san_history or '(none)'}\n"
        f"Game result: {sample.result}"
    )
    answer = (
        "Board facts:\n"
        + "\n".join(f"- {fact}" for fact in facts)
        + "\n\nCandidate reasoning:\n"
        "- Use legal moves only.\n"
        "- Prefer forcing moves, king safety, material balance, and known strategic plans.\n\n"
        f"Move played in the source game: {sample.next_move_san}"
    )
    return {
        "id": stable_id("pgn-sample", sample.game_id, sample.ply, sample.fen),
        "domain": "chess",
        "task": "pgn_human_move_reasoning",
        "source": {
            "name": "lichess_standard_games",
            "url": "https://database.lichess.org/",
            "license": "database-public-export-see-lichess-terms",
            "provenance": "derived",
        },
        "messages": [
            {"role": "system", "content": "You are a rigorous chess reasoning model. Never invent illegal moves."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "verification": {
            "status": "tool-labeled",
            "method": "python-chess-legal-move-board-state",
        },
        "metadata": {
            "fen": sample.fen,
            "game_id": sample.game_id,
            "ply": sample.ply,
            "next_move_san": sample.next_move_san,
        },
    }
