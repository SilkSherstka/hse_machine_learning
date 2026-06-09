"""
Обучение word-level CNN на собранном корпусе новостей Lenta.ru.

Запуск:
    python collect_corpus.py   # один раз — сбор данных
    python train_cnn.py        # предобработка, обучение, метрики
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import TextCNN

# --- Воспроизводимость ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CORPUS_CSV = DATA_DIR / "corpus.csv"
MODEL_PATH = OUTPUT_DIR / "text_cnn_weights.pt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

CLASS_NAMES = ["sport", "economics", "science", "culture"]

# --- Гиперпараметры (обоснование — в README.md) ---
MAX_SEQ_LEN = 120          # достаточно для заголовка + 2–3 абзаца
MIN_WORD_FREQ = 2          # отсекаем редкие опечатки/шум
EMBED_DIM = 128            # баланс ёмкости и переобучения на ~700 примерах
NUM_FILTERS = 96           # умеренное число фильтров для небольшого корпуса
KERNEL_SIZES = (3, 4, 5)   # n-граммы разной длины (слово/биграмма/триграмма)
DROPOUT = 0.5              # снижает переобучение на малом датасете
BATCH_SIZE = 32
LEARNING_RATE = 1e-3       # Adam стабильно сходится для текстовых CNN
WEIGHT_DECAY = 1e-4        # L2-регуляризация
NUM_EPOCHS = 15
PATIENCE = 4               # early stopping по val loss


@dataclass
class Vocab:
    word2idx: dict[str, int]
    idx2word: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.word2idx)


class TextDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int]) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def normalize_text(text: str) -> str:
    """Приведение текста к нижнему регистру и удаление лишних символов."""
    text = text.lower()
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def build_vocab(texts: list[str], min_freq: int) -> Vocab:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # 0 — padding, 1 — unknown
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}
    return Vocab(word2idx=word2idx, idx2word=idx2word)


def encode_text(text: str, vocab: Vocab, max_len: int) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab.word2idx.get(t, 1) for t in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
    return ids


def load_corpus(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {path} не найден. Сначала выполните: python collect_corpus.py"
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделение 70% / 15% / 15% с сохранением баланса классов."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_loader(
    df: pd.DataFrame, vocab: Vocab, batch_size: int, shuffle: bool
) -> DataLoader:
    sequences = [encode_text(t, vocab, MAX_SEQ_LEN) for t in df["text"]]
    labels = df["label"].astype(int).tolist()
    dataset = TextDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: TextCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        ),
    }


def predict_all(
    model: TextCNN, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_y.tolist())
    return all_labels, all_preds


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # 1. Загрузка и разделение данных
    df = load_corpus(CORPUS_CSV)
    print(f"Загружено примеров: {len(df)}")
    train_df, val_df, test_df = split_data(df)
    print(
        f"Разделение: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # 2. Построение словаря только по train (без утечки)
    vocab = build_vocab(train_df["text"].tolist(), MIN_WORD_FREQ)
    print(f"Размер словаря: {vocab.size}")

    train_loader = make_loader(train_df, vocab, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_df, vocab, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_df, vocab, BATCH_SIZE, shuffle=False)

    # 3. Модель
    num_classes = df["label"].nunique()
    model = TextCNN(
        vocab_size=vocab.size,
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT,
        padding_idx=0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Обучение с early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, float]] = []

    print("\n=== Обучение ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab.word2idx,
                    "hyperparameters": {
                        "max_seq_len": MAX_SEQ_LEN,
                        "embed_dim": EMBED_DIM,
                        "num_filters": NUM_FILTERS,
                        "kernel_sizes": KERNEL_SIZES,
                        "dropout": DROPOUT,
                    },
                    "class_names": CLASS_NAMES,
                },
                MODEL_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping на эпохе {epoch}")
                break

    # 5. Загрузка лучших весов и оценка на test
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict_all(model, test_loader, device)
    metrics = evaluate_metrics(y_true, y_pred)

    print("\n=== Метрики на тестовой выборке ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro):  {metrics['f1_macro']:.4f}")
    print("\nПодробный отчёт по классам:")
    print(metrics["classification_report"])

    # 6. Сохранение метрик и истории
    result = {
        "test_metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "classification_report": metrics["classification_report"],
        "training_history": history,
        "dataset_sizes": {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "vocab_size": vocab.size,
        "model_path": str(MODEL_PATH),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nВеса модели: {MODEL_PATH}")
    print(f"Метрики:       {METRICS_PATH}")


if __name__ == "__main__":
    main()
