"""Frontier Claude audit on the fixed 300-document local sample.

Reads ANTHROPIC_API_KEY from the environment. The key is never written to the
archive, the log, or the manuscript. Interrupted runs resume from generations.jsonl.

python scripts/run_13b_claude.py --models claude-sonnet-4-5 claude-opus-4-1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cft.codebook import CONSTRUCTS
from cft.llm import SYSTEM
from run_13b_local import prompt, strict_scores

API = "https://api.anthropic.com/v1/messages"


def load_sample() -> pd.DataFrame:
    ids = set()
    archive = ROOT / "derived" / "local_upgrade" / "generations.jsonl"
    for line in archive.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["doc_id"])
    frames = []
    for path in sorted((ROOT / "derived").glob("corpus_*.parquet")):
        frame = pd.read_parquet(path, columns=["doc_id", "corpus", "text"])
        frames.append(frame[frame.doc_id.isin(ids)])
    sample = pd.concat(frames, ignore_index=True)
    if sample.doc_id.nunique() != 300 or len(sample) != 300:
        raise RuntimeError(f"Expected the fixed 300-document sample, found {len(sample)}")
    return sample.sort_values(["corpus", "doc_id"]).reset_index(drop=True)


def done_keys(archive: Path) -> set[str]:
    found = set()
    if not archive.exists():
        return found
    for line in archive.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("error"):
            found.add(record["key"])
    return found


def call_claude(session, model, system, user, max_tokens):
    # These checkpoints reject a client temperature. Sampling is the provider default.
    response = session.post(
        API,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=180,
    )
    if response.status_code == 429 or response.status_code >= 500:
        raise RuntimeError(f"retryable {response.status_code}: {response.text[:300]}")
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text[:500]}")
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["joint", "contrastive"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="claude_frontier")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0, help="If positive, score only the first N documents")
    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in the environment. Do not pass the key as an argument.")

    dest = ROOT / "derived" / args.output
    dest.mkdir(parents=True, exist_ok=True)
    sample = load_sample()
    if args.limit > 0:
        sample = sample.head(args.limit).copy()
    archive = dest / "generations.jsonl"
    log_path = ROOT / "logs" / "13b_claude_frontier.log"
    log_path.parent.mkdir(exist_ok=True)
    finished = done_keys(archive)
    manifest = {
        "models": args.models,
        "conditions": args.conditions,
        "n_docs": int(len(sample)),
        "doc_ids": sample.doc_id.tolist(),
        "temperature": None,
        "max_tokens": args.max_tokens,
        "seed": None,
        "workers": args.workers,
        "protocol": (
            "Same 300 documents, codebook and joint/contrastive prompts as the local extension. "
            "Anthropic Messages API rejects a client temperature for these checkpoints, so sampling "
            "uses the provider default. No client seed is accepted. Invalid construct fields stay missing."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    jobs = []
    total = len(sample) * len(args.models) * len(args.conditions)
    seen = 0
    for model in args.models:
        for _, doc in sample.iterrows():
            for condition in args.conditions:
                user = prompt(doc.text, condition)
                spec = {
                    "model": model,
                    "condition": condition,
                    "doc_id": doc.doc_id,
                    "max_tokens": args.max_tokens,
                    "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
                }
                key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
                seen += 1
                if key in finished:
                    continue
                jobs.append((seen, key, model, condition, doc.doc_id, doc.corpus, user))

    lock = threading.Lock()
    failures = []

    def run_job(job):
        index, key, model, condition, doc_id, corpus, user = job
        record = {
            "key": key,
            "doc_id": doc_id,
            "corpus": corpus,
            "model": model,
            "condition": condition,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        started = time.monotonic()
        payload = None
        error = None
        for attempt in range(6):
            try:
                payload = call_claude(requests.Session(), model, SYSTEM, user, args.max_tokens)
                error = None
                break
            except Exception as exc:
                error = str(exc)
                if "credit balance is too low" in error.lower():
                    break
                time.sleep(min(30, 2 ** attempt))
        record["elapsed_s"] = time.monotonic() - started
        if payload is None:
            record["error"] = error
        else:
            blocks = payload.get("content") or []
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
            record["response_text"] = text
            record["stop_reason"] = payload.get("stop_reason")
            record["usage"] = {
                "input_tokens": (payload.get("usage") or {}).get("input_tokens"),
                "output_tokens": (payload.get("usage") or {}).get("output_tokens"),
            }
            record["response_model"] = payload.get("model")
            record.update(strict_scores(text))
            record["truncated"] = payload.get("stop_reason") == "max_tokens"
        line = (
            f"{index}/{total} {model} {condition} {doc_id} "
            f"{record['elapsed_s']:.1f}s valid={record.get('schema_valid')} error={record.get('error')}"
        )
        with lock:
            with archive.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)
        return record

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            if record.get("error"):
                failures.append(record["error"])
    if failures:
        raise RuntimeError(f"{len(failures)} calls failed. Resume the same command. First error: {failures[0]}")
    print("COMPLETE", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMPLETE\n")


if __name__ == "__main__":
    main()
