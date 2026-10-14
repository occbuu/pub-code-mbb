"""Aggregate the Claude frontier archive. Does not call the API."""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_13b_upgrade import COLS, profile

ARCHIVE = ROOT / "derived" / "claude_frontier" / "generations.jsonl"
OUT = ROOT / "derived" / "claude_frontier"


def load_scores() -> pd.DataFrame:
    rows = []
    for line in ARCHIVE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error"):
            continue
        row = {column: record.get(column) for column in ["key", "doc_id", "corpus", "model", "condition", "schema_valid", "truncated", "elapsed_s", "response_model"]}
        row.update({column: record.get(column) for column in COLS})
        rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates(["model", "condition", "doc_id"], keep="last")
    return frame


def matched(frame: pd.DataFrame, reps: int = 1000) -> pd.DataFrame:
    paired = []
    for model, group in frame.groupby("model"):
        wide = {}
        for condition, part in group.groupby("condition"):
            wide[condition] = part.set_index("doc_id")
        if "joint" not in wide or "contrastive" not in wide:
            continue
        left, right = wide["joint"], wide["contrastive"]
        for a, b in combinations(COLS, 2):
            joined = left[list((a, b))].join(right[list((a, b))], lsuffix="_base", rsuffix="_alt").dropna()
            row = {"model": model, "pair": f"{a}/{b}", "n_common": len(joined)}
            if len(joined) >= 30 and (joined.std() > 0).all():
                arr = joined.to_numpy(float)

                def difference(z):
                    if (z.std(0) == 0).any():
                        return np.nan
                    return np.corrcoef(z[:, 2:4], rowvar=False)[0, 1] - np.corrcoef(z[:, :2], rowvar=False)[0, 1]

                row["halo_delta_on_common_support"] = difference(arr)
                rng = np.random.default_rng(20261011)
                boot = np.array([difference(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(reps)])
                finite = boot[np.isfinite(boot)]
                if len(finite):
                    row["delta_ci_low"], row["delta_ci_high"] = np.quantile(finite, [0.025, 0.975])
            paired.append(row)
    return pd.DataFrame(paired)


def main():
    scores = load_scores()
    scores.to_csv(OUT / "claude_scores.csv", index=False)
    parse = (scores.groupby(["model", "condition"])
             .agg(n=("doc_id", "size"), schema_valid_rate=("schema_valid", "mean"),
                  truncated_rate=("truncated", "mean"))
             .reset_index())
    parse.to_csv(OUT / "claude_parse_audit.csv", index=False)
    rows = []
    for (model, condition), part in scores.groupby(["model", "condition"]):
        part = part.set_index("doc_id")
        rows.append(profile(part, f"{model}/{condition}", "pooled", 1000))
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "claude_structure_audit.csv", index=False)
    pairs = matched(scores)
    pairs.to_csv(OUT / "claude_matched_support.csv", index=False)
    print(parse.to_string(index=False))
    keep = [c for c in ["method", "n_docs", "n_complete", "coverage", "n_estimable_pairs", "halo_pairwise", "pc1_total_variance_complete", "pc1_ci_low", "pc1_ci_high"] if c in audit.columns]
    print(audit[keep].to_string(index=False))
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
