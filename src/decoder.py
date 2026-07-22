"""
Greedy CTC decoding and standard ASR evaluation metrics (WER, CER).
"""

from __future__ import annotations

import torch

from src.dataset import BLANK_IDX, IDX2CHAR


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """
    log_probs: (time, batch, vocab)
    lengths: (batch,) valid output frames per sample
    Returns list of decoded strings, one per batch item.
    """
    # (time, batch, vocab) -> (batch, time)
    best_paths = torch.argmax(log_probs, dim=-1).transpose(0, 1)  # (batch, time)

    results = []
    for path, length in zip(best_paths, lengths):
        path = path[: int(length.item())].tolist()
        decoded_chars = []
        prev = None
        for idx in path:
            if idx != prev and idx != BLANK_IDX:
                decoded_chars.append(IDX2CHAR[idx])
            prev = idx
        results.append("".join(decoded_chars))
    return results


def _levenshtein(a: list, b: list) -> int:
    """Standard edit-distance DP, used for both WER (words) and CER (chars)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


def word_error_rate(hypothesis: str, reference: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(hyp_words, ref_words) / len(ref_words)


def char_error_rate(hypothesis: str, reference: str) -> float:
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return _levenshtein(hyp_chars, ref_chars) / len(ref_chars)


def batch_wer_cer(hypotheses: list[str], references: list[str]) -> tuple[float, float]:
    wers = [word_error_rate(h, r) for h, r in zip(hypotheses, references)]
    cers = [char_error_rate(h, r) for h, r in zip(hypotheses, references)]
    return sum(wers) / len(wers), sum(cers) / len(cers)
