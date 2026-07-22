"""
DeepSpeech-Lite model definition.

Architecture:
    log-mel spectrogram -> 2D CNN feature extractor -> BiGRU stack -> FC -> CTC log-softmax

This mirrors the DeepSpeech2 / early wav2letter family: a convolutional front-end
that downsamples the frequency axis and extracts local acoustic patterns, feeding
a stack of bidirectional recurrent layers that model temporal context, followed by
a linear classifier over the character vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskConv(nn.Module):
    """Applies a sequence of conv/BN/ReLU/Hardtanh layers while respecting padding masks."""

    def __init__(self, seq_module: nn.Sequential):
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        for module in self.seq_module:
            x = module(x)
        return x, lengths


class BatchRNN(nn.Module):
    """A single bidirectional GRU layer with batch-norm applied between layers."""

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        if self.rnn.bidirectional:
            # sum the forward/backward directions instead of concatenating,
            # keeps hidden size constant across stacked layers
            b, t, h2 = x.shape
            x = x.view(b, t, 2, h2 // 2).sum(dim=2)
        return x


class DeepSpeechLite(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        vocab_size: int = 29,
        cnn_channels: list[int] | None = None,
        rnn_hidden_size: int = 512,
        rnn_layers: int = 5,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [32, 32]

        self.conv = MaskConv(
            nn.Sequential(
                nn.Conv2d(1, cnn_channels[0], kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(cnn_channels[1]),
                nn.Hardtanh(0, 20, inplace=True),
            )
        )

        # frequency dimension after two stride-2 convs along freq axis
        reduced_freq = n_mels
        for _ in range(2):
            reduced_freq = (reduced_freq - 1) // 2 + 1
        rnn_input_size = cnn_channels[1] * reduced_freq

        rnn_layers_list = []
        for i in range(rnn_layers):
            in_size = rnn_input_size if i == 0 else rnn_hidden_size
            rnn_layers_list.append(BatchRNN(in_size, rnn_hidden_size, bidirectional))
        self.rnns = nn.ModuleList(rnn_layers_list)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, vocab_size, bias=False),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (batch, 1, n_mels, time)
        lengths: (batch,) number of valid time frames per sample (pre-conv)
        Returns: log_probs (time, batch, vocab), output_lengths (batch,)
        """
        x, _ = self.conv(x, lengths)
        # x: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.view(b, c * f, t).transpose(1, 2)  # (batch, time, features)

        for rnn in self.rnns:
            x = rnn(x)
            x = self.dropout(x)

        x = x.transpose(1, 2)  # (batch, hidden, time) for BatchNorm1d
        x = self.fc[0](x)
        x = x.transpose(1, 2)  # (batch, time, hidden)
        x = self.fc[1](x)      # (batch, time, vocab)

        log_probs = torch.log_softmax(x, dim=-1).transpose(0, 1)  # (time, batch, vocab)

        # output lengths shrink by the same conv strides applied to the time axis
        output_lengths = lengths
        for stride in (2, 1):
            output_lengths = torch.div(output_lengths - 1, stride, rounding_mode="floor") + 1

        return log_probs, output_lengths


if __name__ == "__main__":
    # quick smoke test
    model = DeepSpeechLite(n_mels=80, vocab_size=29)
    dummy = torch.randn(4, 1, 80, 300)
    lengths = torch.tensor([300, 280, 250, 300])
    out, out_lengths = model(dummy, lengths)
    print("log_probs shape:", out.shape)       # (time, batch, vocab)
    print("output lengths:", out_lengths)
