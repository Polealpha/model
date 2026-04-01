from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .io import read_jsonl, write_jsonl
from .schema import EpisodeWindowV1, episode_from_dict


def load_episodes(path: str | Path) -> List[EpisodeWindowV1]:
    rows = read_jsonl(path)
    return [episode_from_dict(row) for row in rows]


def save_episodes(path: str | Path, episodes: Sequence[EpisodeWindowV1]) -> None:
    write_jsonl(path, [episode.to_dict() for episode in episodes])


def split_episodes(episodes: Sequence[EpisodeWindowV1], train_ratio: float, dev_ratio: float) -> Tuple[List[EpisodeWindowV1], List[EpisodeWindowV1], List[EpisodeWindowV1]]:
    items = list(episodes)
    n = len(items)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    return items[:train_end], items[train_end:dev_end], items[dev_end:]

