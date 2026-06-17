# Data Contract

This project is intentionally conservative about data provenance.

## Allowed

- Lichess open database PGNs and puzzles, with attribution and license metadata.
- Public-domain or permissively licensed master games.
- Stockfish-generated evaluations and principal variations from legally obtained positions.
- Syzygy tablebase labels and public API/tablebase outputs where terms allow use.
- Open Logic Project / forall x material under its license terms.
- Lean/mathlib-derived theorem proving traces under their licenses.
- Self-generated formal logic, chess, memory, and puzzle examples.
- Self-generated applied reasoning examples for philosophy, social analysis, causality, assumptions, and counterexamples.
- Self-generated ethics calibration examples that reduce over-refusal and teach narrow, reasoned boundaries.
- Your own annotations and notes.

## Disallowed By Default

- Raw copyrighted books, including chess books and puzzle books.
- Paid course transcripts.
- Proprietary test batteries or IQ-test item banks.
- Scraped private forum/chat/course content.
- Datasets without clear provenance.
- Jailbreak, evasion, or bypass datasets that teach the model to ignore legitimate safety boundaries.

## Labels

Each processed JSONL record must include:

```json
{
  "id": "stable-id",
  "domain": "chess|logic|memory",
  "task": "specific-task-name",
  "source": {
    "name": "source name",
    "url": "source url or generated",
    "license": "license id",
    "provenance": "downloaded|generated|derived"
  },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "verification": {
    "status": "verified|unverified|tool-labeled",
    "method": "stockfish|syzygy|logic-generator|unit-test|manual"
  }
}
```

## Weight vs Retrieval Policy

Train into weights:
- general chess motifs,
- proof patterns,
- cross-domain argument patterns,
- assumption and counterexample habits,
- direct answers to benign sensitive questions,
- narrow refusal habits for concrete harmful requests,
- endgame principles,
- candidate-move discipline,
- working-memory transformations.

Keep in retrieval/tools:
- exact PGN databases,
- exact copyrighted/licensed notes,
- exact engine PV caches,
- exact tablebase outputs,
- large game index.
