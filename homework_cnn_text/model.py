"""
Свёрточная нейросеть для классификации текстов на уровне слов.

Архитектура: Embedding → Conv1D × 2 → GlobalMaxPooling → Dropout → Linear
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Word-level CNN для многоклассовой классификации текстов."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        # Несколько свёрток с разным размером ядра — типичный приём для текстовых CNN
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)              # (batch, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)     # (batch, embed_dim, seq_len)

        conv_outputs = []
        for conv in self.convs:
            feature_map = F.relu(conv(embedded))
            pooled = F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        return self.fc(dropped)
