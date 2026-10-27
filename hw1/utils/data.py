"""Загрузка и подготовка KuaiRec."""

from pathlib import Path
import polars as pl
import numpy as np


def load_csv(data_dir: Path, filename: str) -> pl.DataFrame:
    paths = list(data_dir.rglob(filename))

    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one {filename}, found: {paths}"
        )

    return pl.read_csv(paths[0])


def prepare_warm_sample(
    df: pl.DataFrame,
    selected_users: list[int],
    n_items: int = 1200,
    threshold: float = 0.8,
):
    # Оставляем выбранных пользователей
    pool = df.filter(
        pl.col("user_id").is_in(selected_users)
        & pl.col("timestamp").is_not_null()
        & pl.col("watch_ratio").is_finite()
        & (pl.col("watch_ratio") >= 0)
        & (pl.col("video_duration") > 0)
        & (pl.col("play_duration") >= 0)
    )

    # Первое взаимодействие каждой пары
    pool = (
        pool.sort("timestamp")
        .unique(
            subset=["user_id", "video_id"],
            keep="first",
            maintain_order=True
        )
    )

    # Фиксируем временные границы ДО отбора видео
    train_end, val_end = pool.select(
        pl.col("timestamp")
        .quantile(0.8).alias("train_end"),
        pl.col("timestamp")
        .quantile(0.9).alias("val_end")
    ).row(0)

    # Считаем популярность исключительно на train
    train_pool = pool.filter(
        pl.col("timestamp") <= train_end
    )

    selected_items = (
        train_pool
        .group_by("video_id")
        .len(name="popularity")
        .sort(
            ["popularity", "video_id"],
            descending=[True, False]
        )
        .head(n_items)["video_id"]
        .to_list()
    )

    sample = (
        pool
        .filter(pl.col("video_id").is_in(selected_items))
        .with_columns(
            (pl.col("watch_ratio") >= threshold)
            .cast(pl.Int8)
            .alias("label")
        )
    )

    return sample, train_end, val_end



def prepare_sample(
    df: pl.DataFrame,
    seed: int = 42,
    n_users: int = 2000,
    n_items: int = 1500,
    threshold: float = 0.8,
) -> pl.DataFrame:

    df = df.filter(
        pl.col("user_id").is_not_null()
        & pl.col("video_id").is_not_null()
        & pl.col("timestamp").is_finite()
        & pl.col("watch_ratio").is_finite()
        & (pl.col("video_duration") > 0)
        & (pl.col("play_duration") >= 0)
        & (pl.col("watch_ratio") >= 0)
    )

    rng = np.random.default_rng(seed)

    users = np.sort(df["user_id"].unique().to_numpy())
    items = np.sort(df["video_id"].unique().to_numpy())

    selected_users = rng.choice(
        users, size=n_users, replace=False
    )
    selected_items = rng.choice(
        items, size=n_items, replace=False
    )


    sample = df.filter(
        pl.col("user_id").is_in(selected_users.tolist())
        & pl.col("video_id").is_in(selected_items.tolist())
    )

    # Оставляем первое наблюдение каждой пары, т.к. одно и то же видео не является двумя разными объектами
    # для рекомендации
    sample = (
        sample
        .sort("timestamp")
        .unique(
            subset=["user_id", "video_id"],
            keep="first",
            maintain_order=True
        )
    )

    # Общий критерий релевантности
    sample = sample.with_columns(
        (pl.col("watch_ratio") >= threshold)
        .cast(pl.Int8)
        .alias("label")
    )

    return sample



def temporal_split(
    df: pl.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
):
    """Глобальное временное разбиение."""

    assert 0 < train_fraction < 1
    assert 0 < val_fraction < 1 - train_fraction

    train_end, val_end = df.select(
        pl.col("timestamp").quantile(train_fraction).alias("train_end"),
        pl.col("timestamp").quantile(
            train_fraction + val_fraction
        ).alias("val_end"),
    ).row(0)

    assert train_end < val_end

    train = df.filter(
        pl.col("timestamp") <= train_end
    )

    val = df.filter(
        (pl.col("timestamp") > train_end)
        & (pl.col("timestamp") <= val_end)
    )

    test = df.filter(
        pl.col("timestamp") > val_end
    )

    return train, val, test


def encode_splits(train, val, test):
    """Кодируем ID на основе обучающей выборки."""

    users = np.sort(
        train["user_id"].unique().to_numpy()
    )
    items = np.sort(
        train["video_id"].unique().to_numpy()
    )

    user_map = pl.DataFrame({
        "user_id": users,
        "user_idx": np.arange(len(users)),
    })

    item_map = pl.DataFrame({
        "video_id": items,
        "item_idx": np.arange(len(items)),
    })

    def encode(df):
        return (
            df
            .join(user_map, on="user_id", how="inner")
            .join(item_map, on="video_id", how="inner")
        )

    return (
        encode(train),
        encode(val),
        encode(test),
        user_map,
        item_map,
    )
