"""Motif-weighted curriculum for tactics drilling.

Tactical pattern recognition is built by *massed repetition of a motif across
many different positions* (the "Woodpecker Method"). This module turns a pool of
labelled puzzle records into an ordered training sequence that:

- attributes each puzzle to ONE primary motif (Lichess puzzles carry several
  themes; we pick the highest-priority/most-instructive one present),
- heavily over-samples a chosen set of core motifs to a per-motif target,
- repeats the selection across passes (re-exposure) with later passes shuffled,
  so repetition is *spaced*, not adjacent (adjacent identical FEN->move pressures
  the model to memorise the position instead of the pattern),
- orders either ``blocked`` (drill one motif at a time -- fast acquisition),
  ``interleaved`` (round-robin across motifs -- better discrimination/retention),
  or ``blocked_then_interleaved`` (acquire, then discriminate).

Exact-FEN duplicates are removed before sampling so "lots of the same motif"
means many distinct positions, with the unseen-position eval proving the model
learned the pattern rather than a lookup table.
"""

from __future__ import annotations

import random

# Most specific / most instructive motifs first. A puzzle is attributed to the
# highest-priority theme it carries so each drilled item has one clear pattern.
DEFAULT_MOTIF_PRIORITY = [
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "backRankMate",
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "doubleCheck",
    "deflection",
    "decoy",
    "removalOfDefender",
    "hangingPiece",
    "trappedPiece",
]

DEFAULT_CORE_MOTIFS = [
    "mateIn1",
    "mateIn2",
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "hangingPiece",
    "backRankMate",
]


def primary_motif(themes: list[str], priority: list[str] = DEFAULT_MOTIF_PRIORITY) -> str | None:
    themeset = set(themes)
    for motif in priority:
        if motif in themeset:
            return motif
    return None


def motif_counts(
    records: list[dict],
    priority: list[str] = DEFAULT_MOTIF_PRIORITY,
) -> dict[str, int]:
    """Distinct-position counts per primary motif (diagnostics for coverage)."""
    counts: dict[str, int] = {}
    seen_fen: set[str] = set()
    for record in records:
        md = record.get("metadata", {})
        fen = md.get("fen")
        if fen in seen_fen:
            continue
        if fen is not None:
            seen_fen.add(fen)
        motif = primary_motif(md.get("themes", []), priority)
        if motif:
            counts[motif] = counts.get(motif, 0) + 1
    return counts


def _stratified_take(pool: list[dict], n: int) -> list[dict]:
    """Take n items from a rating-sorted pool, spread easy->hard; cycle if scarce."""
    if not pool:
        return []
    if len(pool) <= n:
        return [pool[i % len(pool)] for i in range(n)]  # oversample by cycling
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def _round_robin(sequences: list[list[dict]]) -> list[dict]:
    out: list[dict] = []
    longest = max((len(s) for s in sequences), default=0)
    for i in range(longest):
        for seq in sequences:
            if i < len(seq):
                out.append(seq[i])
    return out


def build_motif_curriculum(
    records: list[dict],
    *,
    motifs: list[str] | None = None,
    per_motif: int = 200,
    repeat: int = 1,
    order: str = "blocked",
    dedupe_fen: bool = True,
    priority: list[str] = DEFAULT_MOTIF_PRIORITY,
    seed: int = 0,
) -> list[dict]:
    """Build a drilling sequence: per_motif distinct puzzles x repeat passes.

    order:
      "blocked"                 motif A (all), motif B (all), ...
      "interleaved"             round-robin across motifs
      "blocked_then_interleaved" first pass blocked, remaining passes interleaved
    """
    motifs = motifs or DEFAULT_CORE_MOTIFS
    rng = random.Random(seed)

    buckets: dict[str, list[dict]] = {m: [] for m in motifs}
    seen_fen: set[str] = set()
    for record in records:
        md = record.get("metadata", {})
        motif = primary_motif(md.get("themes", []), priority)
        if motif not in buckets:
            continue
        fen = md.get("fen")
        if dedupe_fen and fen is not None:
            if fen in seen_fen:
                continue
            seen_fen.add(fen)
        buckets[motif].append(record)

    # One pass per motif: distinct puzzles, ordered easy -> hard for acquisition.
    base: dict[str, list[dict]] = {}
    for motif in motifs:
        pool = sorted(buckets[motif], key=lambda r: r.get("metadata", {}).get("rating", 0))
        base[motif] = _stratified_take(pool, per_motif)

    # Re-exposure passes (spaced: later passes shuffled to avoid adjacent repeats).
    passes: list[dict[str, list[dict]]] = []
    for p in range(repeat):
        this_pass: dict[str, list[dict]] = {}
        for motif in motifs:
            block = list(base[motif])
            if p > 0:
                rng.shuffle(block)
            this_pass[motif] = block
        passes.append(this_pass)

    if order == "blocked":
        # All of a motif's passes stay together (massed drilling), then next motif.
        return [r for m in motifs for p in passes for r in p[m]]
    if order == "interleaved":
        return [r for p in passes for r in _round_robin([p[m] for m in motifs])]
    if order == "blocked_then_interleaved":
        out: list[dict] = []
        for i, p in enumerate(passes):
            if i == 0:
                out.extend(r for m in motifs for r in p[m])
            else:
                out.extend(_round_robin([p[m] for m in motifs]))
        return out
    raise ValueError(f"unknown order: {order!r}")
