# Dataset Plan

This project is designed around clean, high-signal supervision rather than scraped text volume.

## Default Mix

- 50% chess:
  - Lichess PGN positions for human move pattern recognition.
  - Lichess puzzles for tactical motifs.
  - Generated low-piece positions labeled by Syzygy for exact endgame truth.
  - Stockfish-labeled positions for candidate moves, PVs, and blunder checks.
- 25% logic:
  - Generated Fitch-style propositional and predicate proofs.
  - Generated applied reasoning across philosophy, social domains, causality, fallacy audits, hidden assumptions, and counterexamples.
  - Generated ethics calibration for over-refusal reduction, precise boundaries, and direct answers to benign sensitive analysis.
  - Lean/mathlib traces after license review and conversion.
- 25% memory and puzzles:
  - Generated working-memory recall.
  - Ordered-list transformations.
  - Small constraint puzzles with known solutions.

## Use In Weights

Train into the adapter:

- tactical motifs and candidate-move discipline,
- exact endgame result patterns from tablebase labels,
- proof step habits,
- hidden assumption detection,
- counterexample generation,
- causal skepticism,
- ethical boundary precision without blanket refusal,
- state tracking and short-term recall.

## Keep In Retrieval Or Tools

Do not try to memorize everything in the model weights:

- complete PGN archives,
- exact tablebase files,
- raw engine caches,
- copyrighted book notes,
- large theorem libraries.

Use retrieval, Stockfish, and Syzygy at inference time for exact facts. Use training for the reasoning habits that decide when and how to consult those facts.

## Excluded

Do not ingest raw Silman books, paid chess course text, proprietary IQ tests, leaked interviews, private chats, jailbreak/evasion datasets, or random web scrapes. If you own notes from a copyrighted source, use short original summaries written by you, not copied text.
