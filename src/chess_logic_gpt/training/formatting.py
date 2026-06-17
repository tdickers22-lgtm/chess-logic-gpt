from __future__ import annotations


def messages_to_chat_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    return "\n".join(parts)


def row_to_text(row: dict, tokenizer=None) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return messages_to_chat_text(messages)
    if "text" in row:
        return str(row["text"])
    raise ValueError("row has neither messages nor text")


def mask_prompt_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """Labels for completion-only loss: the first ``prompt_len`` tokens are -100.

    -100 is the HF cross-entropy ignore index, so loss is computed only on the
    assistant turn (the <reasoning>/<answer> trace), not on the prompt the model
    is merely conditioned on (system text, the FEN, the puzzle statement).
    """
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    return labels


def render_chat(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    """Apply the chat template with Qwen3 native thinking disabled.

    We canonicalize on our own ``<reasoning>/<answer>`` trace, so we suppress
    Qwen3's built-in ``<think>`` mode. Tokenizers whose template doesn't accept
    ``enable_thinking`` (e.g. Qwen2.5 used for the smoke test) simply fall back.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def build_supervised_example(messages: list[dict], tokenizer, max_len: int) -> dict:
    """Tokenize a chat example for completion-only SFT.

    Renders the prompt (everything but the final assistant turn, plus the
    generation prefix) and the full conversation through the tokenizer's chat
    template, then masks the prompt span so only the assistant completion carries
    loss. The prompt render is a token prefix of the full render for ChatML-style
    templates (Qwen included), which makes the mask boundary exact.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("completion-only SFT needs a tokenizer with apply_chat_template")
    prompt_text = render_chat(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_len)
    input_ids = full["input_ids"]
    attention_mask = full.get("attention_mask", [1] * len(input_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": mask_prompt_labels(input_ids, prompt_len),
    }
