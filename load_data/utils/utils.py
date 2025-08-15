import os
from typing import Any

from datasets import Dataset
import numpy as np


def calculate_length(batch: Any) -> Any:
    lengths = [len(text) for text in batch['text']]
    batch['length'] = lengths
    return batch

def show_length_distribution(data: Dataset, num_bins: int = 20) -> None:
    """Display the distribution of text lengths in the dataset."""
    lengths = np.array(data)

    min_length, max_length = lengths.min(), lengths.max()
    bins: Any = np.linspace(min_length, max_length, num_bins + 1)
    bins = bins.astype(np.int64)
    hist, _ = np.histogram(lengths, bins=bins)

    total_count = len(lengths)

    for i in range(num_bins):
        count = hist[i]
        percentage = (count / total_count) * 100
        print(f"{bins[i]:,} - {bins[i+1]:,}: ({count:,}, {percentage:.6f}%)")

def split_by_paragraph(batch, length_range=(64, 512)):
    min_length, max_length = length_range
    splited_text = []
    for text in batch['text']:
        if len(text) < min_length:
            continue
        elif len(text) <= max_length:
            splited_text.append(text)
        else:
            parts = text.split('\n')
            parts = [p for p in parts if min_length <= len(p) <= max_length]
            splited_text.extend(parts)
    return {'text': splited_text}

def save_as_parquets(data: Dataset, batch_size: int, local_name: str, save_dir: str) -> None:
    """Save the dataset as Parquet files."""
    num_shards: int = (len(data) + batch_size - 1) // batch_size

    for shard_id in range(num_shards):
        shard: Any = data.shard(num_shards=num_shards, index=shard_id)
        shard_path: str = os.path.join(save_dir, f"{local_name}_shard_{shard_id}.parquet")
        shard.to_parquet(shard_path)
        print(f"Saved shard {shard_id+1}/{num_shards} → {shard_path}")

