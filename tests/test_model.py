import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DeepSpeechLite
from src.decoder import greedy_decode, word_error_rate, char_error_rate, batch_wer_cer
from src.dataset import text_to_indices, indices_to_text, VOCAB_SIZE, BLANK_IDX


def test_model_forward_shape():
    model = DeepSpeechLite(n_mels=80, vocab_size=VOCAB_SIZE)
    x = torch.randn(2, 1, 80, 200)
    lengths = torch.tensor([200, 150])
    log_probs, out_lengths = model(x, lengths)
    assert log_probs.shape[1] == 2          # batch dim preserved
    assert log_probs.shape[2] == VOCAB_SIZE  # vocab dim correct
    assert out_lengths[0] >= out_lengths[1]  # shorter input -> shorter (or equal) output


def test_tokenizer_roundtrip():
    text = "hello world"
    indices = text_to_indices(text)
    assert indices_to_text(indices) == text


def test_greedy_decode_removes_blanks_and_repeats():
    # construct log_probs that spell "ab" with blanks and repeats: a a _ b b _
    vocab = VOCAB_SIZE
    seq = [3, 3, BLANK_IDX, 4, 4, BLANK_IDX]  # 'a'=idx3, 'b'=idx4 given CHARS ordering
    log_probs = torch.full((len(seq), 1, vocab), -10.0)
    for t, idx in enumerate(seq):
        log_probs[t, 0, idx] = 0.0
    lengths = torch.tensor([len(seq)])
    decoded = greedy_decode(log_probs, lengths)
    assert decoded[0] == "ab"


def test_wer_cer_perfect_match():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert char_error_rate("abc", "abc") == 0.0


def test_wer_cer_mismatch():
    wer = word_error_rate("hello there", "hello world")
    assert wer == 0.5  # 1 substitution / 2 words

    batch_wer, batch_cer = batch_wer_cer(["hello there"], ["hello world"])
    assert batch_wer == 0.5
    assert batch_cer > 0.0
