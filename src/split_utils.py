from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def lesion_level_split_indices(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices by lesion_id (group-aware) to avoid leakage.
    The same lesion_id never appears across train/val/test.
    """
    if "lesion_id" not in df.columns:
        raise ValueError("DataFrame must include 'lesion_id' for lesion-level split.")
    if "label" not in df.columns:
        raise ValueError("DataFrame must include 'label' column.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    # One row per lesion_id with lesion label
    lesion_df = (
        df[["lesion_id", "label"]]
        .drop_duplicates(subset=["lesion_id"])
        .reset_index(drop=True)
    )

    train_lesions, temp_lesions = train_test_split(
        lesion_df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=lesion_df["label"],
    )

    val_share_of_temp = val_ratio / (val_ratio + test_ratio)
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=temp_lesions["label"],
    )

    train_ids = set(train_lesions["lesion_id"].tolist())
    val_ids = set(val_lesions["lesion_id"].tolist())
    test_ids = set(test_lesions["lesion_id"].tolist())

    train_idx = df.index[df["lesion_id"].isin(train_ids)].tolist()
    val_idx = df.index[df["lesion_id"].isin(val_ids)].tolist()
    test_idx = df.index[df["lesion_id"].isin(test_ids)].tolist()

    return train_idx, val_idx, test_idx

