#!/usr/bin/env python3
"""Compute semantic-ID migration stability metrics for continual recommendation.

The script compares candidate code files against the previous-period code and
reports item-level, support-weighted, head-item, and sequence-level churn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-code", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--valid-parquet", default=None)
    parser.add_argument("--test-parquet", default=None)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--head-fracs", default="0.01,0.05,0.10")
    return parser.parse_args()


def load_sequences(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    return pd.read_parquet(path)


def iter_items(df: pd.DataFrame):
    for row in df.itertuples(index=False):
        if hasattr(row, "history"):
            for item in list(row.history):
                yield int(item) - 1
        if hasattr(row, "target"):
            yield int(row.target) - 1


def item_support(train_df: pd.DataFrame, n_items: int) -> np.ndarray:
    support = np.zeros(n_items, dtype=np.int64)
    for idx in iter_items(train_df):
        if 0 <= idx < n_items:
            support[idx] += 1
    return support


def sequence_perturbation(df: pd.DataFrame | None, changed: np.ndarray) -> dict[str, float | int | None]:
    if df is None:
        return {
            "seq_mean_changed_frac": None,
            "seq_p50_changed_frac": None,
            "seq_p90_changed_frac": None,
            "seq_any_changed_frac": None,
            "seq_target_changed_frac": None,
            "n_sequences": None,
        }

    seq_fracs: list[float] = []
    target_changed = []
    for row in df.itertuples(index=False):
        indices: list[int] = []
        if hasattr(row, "history"):
            indices.extend(int(item) - 1 for item in list(row.history))
        if hasattr(row, "target"):
            indices.append(int(row.target) - 1)
            target_idx = int(row.target) - 1
            target_changed.append(bool(0 <= target_idx < len(changed) and changed[target_idx]))
        valid = [idx for idx in indices if 0 <= idx < len(changed)]
        if not valid:
            seq_fracs.append(0.0)
        else:
            seq_fracs.append(float(np.mean(changed[valid])))

    seq = np.asarray(seq_fracs, dtype=float)
    return {
        "seq_mean_changed_frac": float(seq.mean()) if len(seq) else 0.0,
        "seq_p50_changed_frac": float(np.quantile(seq, 0.5)) if len(seq) else 0.0,
        "seq_p90_changed_frac": float(np.quantile(seq, 0.9)) if len(seq) else 0.0,
        "seq_any_changed_frac": float(np.mean(seq > 0)) if len(seq) else 0.0,
        "seq_target_changed_frac": float(np.mean(target_changed)) if target_changed else 0.0,
        "n_sequences": int(len(seq)),
    }


def candidate_metrics(
    name: str,
    path: str,
    old: np.ndarray,
    support: np.ndarray,
    frames: dict[str, pd.DataFrame | None],
    head_fracs: list[float],
) -> dict[str, float | int | str | None]:
    cand = np.load(path)
    n_old = len(old)
    cand_old = cand[:n_old]

    changed = (old != cand_old).any(axis=1)
    token_changed = old != cand_old
    hamming = token_changed.mean(axis=1)

    support_sum = support.sum()
    support_weighted_churn = float(support[changed].sum() / support_sum) if support_sum else 0.0
    changed_support = support[changed]
    unchanged_support = support[~changed]

    out: dict[str, float | int | str | None] = {
        "name": name,
        "path": path,
        "n_old_items": int(n_old),
        "n_candidate_items": int(len(cand)),
        "old_item_churn": float(changed.mean()),
        "changed_old_items": int(changed.sum()),
        "token_churn": float(token_changed.mean()),
        "mean_hamming_all_items": float(hamming.mean()),
        "mean_hamming_changed_items": float(hamming[changed].mean()) if changed.any() else 0.0,
        "support_weighted_churn": support_weighted_churn,
        "changed_support_mean": float(changed_support.mean()) if len(changed_support) else 0.0,
        "unchanged_support_mean": float(unchanged_support.mean()) if len(unchanged_support) else 0.0,
        "changed_support_p90": float(np.quantile(changed_support, 0.9)) if len(changed_support) else 0.0,
        "new_item_count": int(max(0, len(cand) - n_old)),
        "new_item_ratio_vs_old": float(max(0, len(cand) - n_old) / n_old),
    }

    order = np.argsort(-support)
    for frac in head_fracs:
        k = max(1, int(round(frac * n_old)))
        head = order[:k]
        label = f"head{int(frac * 100)}"
        out[f"{label}_churn"] = float(changed[head].mean())
        out[f"{label}_support_weighted_churn"] = (
            float(support[head][changed[head]].sum() / support[head].sum()) if support[head].sum() else 0.0
        )

    for split, frame in frames.items():
        seq_metrics = sequence_perturbation(frame, changed)
        for key, value in seq_metrics.items():
            out[f"{split}_{key}"] = value

    return out


def main() -> None:
    args = parse_args()
    old = np.load(args.old_code)
    train_df = load_sequences(args.train_parquet)
    assert train_df is not None
    frames = {
        "train": train_df,
        "valid": load_sequences(args.valid_parquet),
        "test": load_sequences(args.test_parquet),
    }
    support = item_support(train_df, len(old))
    head_fracs = [float(x) for x in args.head_fracs.split(",") if x.strip()]

    rows = [
        candidate_metrics(name, path, old, support, frames, head_fracs)
        for name, path in args.candidate
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2))

    print(f"wrote {out_csv}")
    if args.out_json:
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
