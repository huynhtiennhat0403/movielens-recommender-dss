"""PyTorch dataset helpers for recommendation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DataSource = Union[str, Path, pd.DataFrame]


class InteractionDataset(Dataset):
    """
    Dataset containing encoded (user, item, label) interactions.

    Required columns
    ----------------
    user_idx:
        Contiguous encoded user index.
    movie_idx:
        Contiguous encoded movie index.
    label:
        Binary target: 1 for positive, 0 for sampled negative.
    """

    REQUIRED_COLUMNS = ("user_idx", "movie_idx", "label")

    def __init__(self, data: DataSource) -> None:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            frame = pd.read_csv(data)

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in frame.columns
        ]
        if missing:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing)
            )

        if frame.empty:
            raise ValueError("InteractionDataset cannot be empty.")

        if frame[list(self.REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(
                "user_idx, movie_idx and label must not contain NaN."
            )

        labels = set(frame["label"].astype(int).unique().tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(
                f"label must be binary (0/1). Found values: {sorted(labels)}"
            )

        self.user_idx = torch.tensor(
            frame["user_idx"].to_numpy(copy=True),
            dtype=torch.long,
        )

        self.movie_idx = torch.tensor(
            frame["movie_idx"].to_numpy(copy=True),
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            frame["label"].to_numpy(copy=True),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.user_idx[index],
            self.movie_idx[index],
            self.labels[index],
        )


def create_dataloader(
    data: DataSource,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a DataLoader from a DataFrame or CSV path.

    ``num_workers=0`` is a safe default for Windows/Jupyter.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    dataset = InteractionDataset(data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
