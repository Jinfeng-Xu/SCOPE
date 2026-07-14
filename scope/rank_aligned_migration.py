#!/usr/bin/env python3
"""Ranking-aligned constrained semantic-ID migration selection.

The selector uses a full-migration GRM teacher to estimate item-level
validation rank utility. It runs constrained beam search under the updated
codebook, records the rank/NDCG contribution of each validation target, and
selects migrations under support and head-item constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import T5Config, T5ForConditionalGeneration


class Trie:
    def __init__(self, sequences: List[List[int]] | None = None):
        self.trie_dict: Dict[int, dict] = {}
        if sequences:
            for sequence in sequences:
                self.add(sequence)

    def add(self, sequence: List[int]) -> None:
        node = self.trie_dict
        for token in sequence:
            node = node.setdefault(int(token), {})

    def get(self, prefix: List[int]) -> List[int]:
        node = self.trie_dict
        for token in prefix:
            if int(token) not in node:
                return []
            node = node[int(token)]
        return list(node.keys())


def prefix_allowed_tokens_fn(trie: Trie):
    def prefix_allowed_tokens(batch_id, sentence):
        return trie.get(sentence.tolist())

    return prefix_allowed_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-code", required=True)
    parser.add_argument("--new-code", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--valid-parquet", required=True)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="Tools_0.7_scopev3")
    parser.add_argument("--budgets", default="0.15,0.20")
    parser.add_argument(
        "--selection",
        choices=["rank_s1p0_head0", "rank_s2p0_head0", "rank_drift_s1p0_head0"],
        default="rank_s1p0_head0",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=1505)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--d-kv", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_t5(args: argparse.Namespace) -> T5ForConditionalGeneration:
    config = T5Config(
        num_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        d_kv=args.d_kv,
        dropout_rate=0.1,
        vocab_size=args.vocab_size,
        pad_token_id=0,
        eos_token_id=0,
        decoder_start_token_id=0,
        feed_forward_proj="relu",
    )
    return T5ForConditionalGeneration(config)


def load_teacher(args: argparse.Namespace, device: torch.device) -> T5ForConditionalGeneration:
    model = build_t5(args)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    if all(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def offset_codes(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    offsets = np.arange(codes.shape[1], dtype=np.int64).reshape(1, -1) * codebook_size + 1
    return codes.astype(np.int64) + offsets


def pad_or_truncate(history: list[int], max_len: int) -> list[int]:
    if len(history) > max_len:
        return history[-max_len:]
    return [0] * (max_len - len(history)) + history


def encode_histories(df: pd.DataFrame, item_to_code: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories: list[list[int]] = []
    targets: list[int] = []
    for row in df.itertuples(index=False):
        flat: list[int] = []
        for item in pad_or_truncate(list(row.history), max_len):
            idx = int(item) - 1
            if 0 <= idx < len(item_to_code):
                flat.extend(int(x) for x in item_to_code[idx])
            else:
                flat.extend([0, 0, 0, 0])
        histories.append(flat)
        targets.append(int(row.target) - 1)
    input_ids = np.asarray(histories, dtype=np.int64)
    attention = (input_ids != 0).astype(np.int64)
    return input_ids, attention, np.asarray(targets, dtype=np.int64)


def item_support(path: str, n_items: int) -> np.ndarray:
    df = pd.read_parquet(path)
    support = np.zeros(n_items, dtype=np.int64)
    for row in df.itertuples(index=False):
        for item in list(row.history):
            idx = int(item) - 1
            if 0 <= idx < n_items:
                support[idx] += 1
        idx = int(row.target) - 1
        if 0 <= idx < n_items:
            support[idx] += 1
    return support


def minmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    if not mask.any():
        return out
    v = values[mask].astype(float)
    lo = float(v.min())
    hi = float(v.max())
    out[mask] = 0.0 if hi <= lo else (values[mask] - lo) / (hi - lo)
    return out


def resolve_collisions(codes: np.ndarray, codebook_size: int) -> tuple[np.ndarray, int]:
    result = codes.copy()
    used: dict[tuple[int, ...], int] = {}
    fixes = 0
    for idx in range(result.shape[0]):
        key = tuple(int(x) for x in result[idx])
        if key not in used:
            used[key] = idx
            continue
        base = result[idx].copy()
        for suffix in range(codebook_size):
            cand = base.copy()
            cand[-1] = suffix
            cand_key = tuple(int(x) for x in cand)
            if cand_key not in used:
                result[idx] = cand
                used[cand_key] = idx
                fixes += 1
                break
        else:
            raise RuntimeError(f"Unable to resolve duplicate code for row {idx}: {key}")
    return result, fixes


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old = np.load(args.old_code)
    new = np.load(args.new_code)
    n_old = len(old)
    changed_mask = (old != new[:n_old]).any(axis=1)
    new_offsets = offset_codes(new, args.codebook_size)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(args, device)

    valid_df = pd.read_parquet(args.valid_parquet)
    input_ids_np, attention_np, targets = encode_histories(valid_df, new_offsets, args.max_len)
    valid_old_mask = (targets >= 0) & (targets < n_old)
    input_ids_np = input_ids_np[valid_old_mask]
    attention_np = attention_np[valid_old_mask]
    targets = targets[valid_old_mask]

    trie = Trie([[0] + [int(x) for x in code] + [0] for code in new_offsets])
    constraint_fn = prefix_allowed_tokens_fn(trie)

    rank_sum = np.zeros(n_old, dtype=float)
    ndcg_sum = np.zeros(n_old, dtype=float)
    hit_sum = np.zeros(n_old, dtype=float)
    count = np.zeros(n_old, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(targets), args.batch_size):
            end = start + args.batch_size
            input_ids = torch.as_tensor(input_ids_np[start:end], dtype=torch.long, device=device)
            attention = torch.as_tensor(attention_np[start:end], dtype=torch.long, device=device)
            batch_targets = targets[start:end]
            output = teacher.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_length=5,
                num_beams=args.beam_size,
                num_return_sequences=args.beam_size,
                prefix_allowed_tokens_fn=constraint_fn,
            )
            preds = output[:, 1:].reshape(len(batch_targets), args.beam_size, -1).detach().cpu().numpy()
            labels = new_offsets[batch_targets]
            for item, beams, label in zip(batch_targets, preds, labels):
                count[item] += 1
                rank = 0
                for pos, beam in enumerate(beams, start=1):
                    if np.array_equal(beam, label):
                        rank = pos
                        break
                if rank:
                    hit_sum[item] += 1.0
                    ndcg_sum[item] += float(1.0 / np.log2(rank + 1))
                    rank_sum[item] += float(rank)

    observed = count > 0
    mean_ndcg = np.zeros(n_old, dtype=float)
    hit_rate = np.zeros(n_old, dtype=float)
    mean_rank_utility = np.zeros(n_old, dtype=float)
    mean_ndcg[observed] = ndcg_sum[observed] / count[observed]
    hit_rate[observed] = hit_sum[observed] / count[observed]
    mean_rank_utility[observed] = mean_ndcg[observed] + 0.25 * hit_rate[observed]

    support = item_support(args.train_parquet, n_old)
    support_norm = np.log1p(support).astype(float)
    if support_norm.max() > 0:
        support_norm /= support_norm.max()
    drift = np.linalg.norm(old.astype(float) - new[:n_old].astype(float), axis=1)
    drift_norm = minmax(drift, changed_mask)
    benefit_norm = minmax(mean_rank_utility, changed_mask & observed)

    configs = {
        "rank_s1p0_head0": {"support_lambda": 1.0, "drift_weight": 0.0},
        "rank_s2p0_head0": {"support_lambda": 2.0, "drift_weight": 0.0},
        "rank_drift_s1p0_head0": {"support_lambda": 1.0, "drift_weight": 0.5},
    }
    cfg = configs[args.selection]
    utility = benefit_norm + cfg["drift_weight"] * drift_norm - cfg["support_lambda"] * support_norm
    utility[~changed_mask] = -np.inf

    head_k = max(1, int(round(0.05 * n_old)))
    head = set(int(i) for i in np.argsort(-support)[:head_k])
    ranked = [int(i) for i in np.argsort(-utility) if changed_mask[i] and i not in head]

    diagnostics = pd.DataFrame(
        {
            "item_index": np.arange(n_old),
            "changed": changed_mask,
            "support": support,
            "valid_target_count": count,
            "teacher_hit_rate": hit_rate,
            "teacher_mean_ndcg": mean_ndcg,
            "rank_benefit_norm": benefit_norm,
            "drift_norm": drift_norm,
            "support_norm": support_norm,
            "utility": utility,
        }
    )
    diagnostics.to_csv(out_dir / f"{args.prefix}_{args.selection}_diagnostics.csv", index=False)

    summary = {
        "selection": args.selection,
        "valid_old_rows": int(len(targets)),
        "observed_changed_items": int(np.sum(observed & changed_mask)),
        "mean_hit_observed_changed": float(hit_rate[observed & changed_mask].mean())
        if np.any(observed & changed_mask)
        else 0.0,
        "mean_ndcg_observed_changed": float(mean_ndcg[observed & changed_mask].mean())
        if np.any(observed & changed_mask)
        else 0.0,
        "outputs": [],
    }

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    for budget in budgets:
        k = int(round(budget * n_old))
        accepted = np.asarray(ranked[:k], dtype=np.int64)
        result = np.concatenate([old.copy(), new[n_old:]], axis=0)
        result[accepted] = new[accepted]
        result, fixes = resolve_collisions(result, args.codebook_size)
        actual_changed = (old != result[:n_old]).any(axis=1)
        budget_tag = f"b{budget:.4f}".replace(".", "p")
        out_path = out_dir / f"{args.prefix}_{args.selection}_{budget_tag}.npy"
        np.save(out_path, result)
        row = {
            "budget": budget,
            "path": str(out_path),
            "accepted_changes": int(len(accepted)),
            "actual_changed_old_items": int(actual_changed.sum()),
            "actual_changed_old_ratio": float(actual_changed.mean()),
            "collision_fixes": int(fixes),
            "mean_teacher_hit_accepted": float(hit_rate[accepted].mean()) if len(accepted) else 0.0,
            "mean_teacher_ndcg_accepted": float(mean_ndcg[accepted].mean()) if len(accepted) else 0.0,
            "mean_support_accepted": float(support[accepted].mean()) if len(accepted) else 0.0,
        }
        summary["outputs"].append(row)
        print(json.dumps(row, indent=2))

    summary_path = out_dir / f"{args.prefix}_{args.selection}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
