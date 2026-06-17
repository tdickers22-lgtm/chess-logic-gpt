from __future__ import annotations

from chess_logic_gpt.chess.curriculum import (
    build_motif_curriculum,
    motif_counts,
    primary_motif,
)


def rec(idx: int, themes: list[str], rating: int = 1500, fen: str | None = None) -> dict:
    return {
        "domain": "chess",
        "task": "chess_tactic_solve",
        "metadata": {
            "fen": fen or f"fen-{idx}",
            "line_uci": ["e2e4"],
            "themes": themes,
            "rating": rating,
        },
    }


def _motif_of(record: dict) -> str | None:
    return primary_motif(record["metadata"]["themes"])


def test_primary_motif_uses_priority_order():
    assert primary_motif(["short", "fork"]) == "fork"
    assert primary_motif(["fork", "mateIn1"]) == "mateIn1"  # mate outranks fork
    assert primary_motif(["endgame"]) is None


def test_motif_counts_dedupes_positions():
    records = [
        rec(0, ["fork"], fen="A"),
        rec(1, ["fork"], fen="A"),  # duplicate position
        rec(2, ["fork"], fen="B"),
        rec(3, ["pin"], fen="C"),
    ]
    assert motif_counts(records) == {"fork": 2, "pin": 1}


def test_blocked_order_groups_by_motif_with_repetition():
    records = [rec(i, ["fork"], rating=1000 + i, fen=f"f{i}") for i in range(6)]
    records += [rec(100 + i, ["pin"], rating=1000 + i, fen=f"p{i}") for i in range(6)]

    seq = build_motif_curriculum(
        records, motifs=["fork", "pin"], per_motif=3, repeat=2, order="blocked"
    )
    # 3 distinct x 2 passes per motif = 6 each, 12 total.
    assert len(seq) == 12
    motifs = [_motif_of(r) for r in seq]
    assert motifs[:6] == ["fork"] * 6
    assert motifs[6:] == ["pin"] * 6


def test_interleaved_order_alternates_motifs():
    records = [rec(i, ["fork"], fen=f"f{i}") for i in range(4)]
    records += [rec(100 + i, ["pin"], fen=f"p{i}") for i in range(4)]
    seq = build_motif_curriculum(
        records, motifs=["fork", "pin"], per_motif=4, repeat=1, order="interleaved"
    )
    motifs = [_motif_of(r) for r in seq]
    assert motifs == ["fork", "pin", "fork", "pin", "fork", "pin", "fork", "pin"]


def test_oversamples_scarce_motif_to_target():
    # Only 2 distinct fork positions, but we want 5 per motif -> cycled to 5.
    records = [rec(0, ["fork"], fen="A"), rec(1, ["fork"], fen="B")]
    seq = build_motif_curriculum(
        records, motifs=["fork"], per_motif=5, repeat=1, order="blocked"
    )
    assert len(seq) == 5
    assert all(_motif_of(r) == "fork" for r in seq)


def test_empty_motif_contributes_nothing():
    records = [rec(i, ["fork"], fen=f"f{i}") for i in range(3)]
    seq = build_motif_curriculum(
        records, motifs=["fork", "skewer"], per_motif=3, repeat=1, order="blocked"
    )
    assert len(seq) == 3  # skewer bucket empty
    assert all(_motif_of(r) == "fork" for r in seq)
