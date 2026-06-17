"""Tests for completion-only SFT label masking (torch-free)."""

from __future__ import annotations

from chess_logic_gpt.training.formatting import build_supervised_example, mask_prompt_labels


def test_mask_prompt_labels_masks_only_the_prefix():
    assert mask_prompt_labels([10, 11, 12, 13], 2) == [-100, -100, 12, 13]


def test_mask_prompt_labels_handles_prompt_longer_than_sequence():
    # After truncation the prompt can cover the whole window; nothing is supervised.
    assert mask_prompt_labels([10, 11], 5) == [-100, -100]


class _CharTokenizer:
    """A 1-char-per-token stand-in so the prompt render is an exact token prefix."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_build_supervised_example_supervises_only_the_assistant_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "A"},
    ]
    tok = _CharTokenizer()
    ex = build_supervised_example(messages, tok, max_len=1000)

    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])

    prompt_len = ex["labels"].index(next(x for x in ex["labels"] if x != -100))
    # Everything before the assistant content is masked...
    assert all(label == -100 for label in ex["labels"][:prompt_len])
    # ...and the assistant content + its closing tokens carry loss unchanged.
    assert ex["labels"][prompt_len:] == ex["input_ids"][prompt_len:]
    # The assistant's actual content char ('A') must be among the supervised tokens.
    assert ord("A") in ex["labels"][prompt_len:]


def test_build_supervised_example_respects_max_len_truncation():
    messages = [
        {"role": "user", "content": "x" * 50},
        {"role": "assistant", "content": "y" * 50},
    ]
    ex = build_supervised_example(messages, _CharTokenizer(), max_len=20)
    assert len(ex["input_ids"]) == 20
    assert len(ex["labels"]) == 20
