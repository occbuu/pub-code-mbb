"""Resumable local model-size/contrastive audit; no downloads or paid APIs.

python Paper13/scripts/run_13b_local.py --models qwen2.5:14b-instruct --per-corpus 75
All attempts are archived; interrupted runs resume using content-addressed keys.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cft.codebook import CONSTRUCTS
from cft.llm import SYSTEM, build_prompt, parse_generation

CONTRAST = """
DISAMBIGUATION RULES (apply independently to each construct):
ATT is an explicit overall evaluation; a functional benefit alone is PE, not ATT.
PE requires an intended function/task/outcome; liking alone is not usefulness.
EE requires ease or difficulty; effectiveness does not imply ease.
SI requires a third-party opinion or normative pressure; the writer's own
recommendation or future plan is BI, not SI.
BI requires expressed future engagement/recommendation; praise is not intention.
Mixed signs are legitimate: useful but hard to use is positive PE and negative EE;
others recommend it but I will not return is positive SI and negative BI.
Do not copy valence between constructs. Do not infer unmentioned constructs.
When evidence is absent use sal=0, lvl=null, even if the document is very positive.
"""

def prompt(text, condition):
    base = build_prompt(text, "zeroshot")
    # Insert before the document: avoid an instruction appended inside source text.
    return base if condition == "joint" else base.replace("DOCUMENT:", CONTRAST + "\nDOCUMENT:", 1)

def strict_scores(raw):
    obj = parse_generation(raw)
    out = {"parsed": obj is not None, "schema_valid": True}
    for c in CONSTRUCTS:
        v = (obj or {}).get(c)
        valid = isinstance(v, dict)
        sal, lvl = (v.get("sal"), v.get("lvl")) if valid else (None, None)
        valid = valid and type(sal) is int and sal in range(4)
        valid = valid and ((sal == 0 and lvl is None) or
                          (sal is not None and sal > 0 and type(lvl) is int and -3 <= lvl <= 3))
        out[c + "_valid"] = bool(valid)
        out[c + "_sal"] = sal / 3 if valid else None
        out[c + "_pol"] = lvl / 3 if valid and sal > 0 else None
        out["schema_valid"] &= bool(valid)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5:14b-instruct", "gemma2:27b", "qwen2.5:32b-instruct-q4_K_M"])
    ap.add_argument("--conditions", nargs="+", choices=["joint", "contrastive"], default=["joint", "contrastive"])
    ap.add_argument("--per-corpus", type=int, default=75)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--output", default="local_upgrade")
    args = ap.parse_args()
    if args.per_corpus < 1: ap.error("per-corpus must be positive")
    dest = ROOT / "derived" / args.output
    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    url = "http://127.0.0.1:11434"
    tags = session.get(url + "/api/tags", timeout=20).json()
    available = {m["name"]: m for m in tags["models"]}
    for model in args.models:
        if model not in available: raise ValueError(f"Model not installed: {model}")
    docs = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "derived").glob("corpus_*.parquet"))])
    ids = pd.read_parquet(ROOT / "derived/mitigation_scores.parquet").doc_id.unique()
    docs = docs[docs.doc_id.isin(ids)].sort_values("doc_id")
    sample = pd.concat([g.sample(n=min(len(g), args.per_corpus), random_state=args.seed)
                        for _, g in docs.groupby("corpus")]).sort_values(["corpus", "doc_id"])
    if (dest / "sample.parquet").exists():
        previous = pd.read_parquet(dest / "sample.parquet")
        if previous[["doc_id", "text"]].to_dict("records") != sample[["doc_id", "text"]].to_dict("records"):
            raise ValueError("Existing output has a different sample; choose a new --output directory")
    if (dest / "manifest.json").exists():
        previous = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["args"]
        for field in ["seed", "temperature", "max_tokens", "context", "conditions", "models"]:
            if previous[field] != vars(args)[field]:
                raise ValueError(f"Existing output has different {field}; choose a new --output directory")
    sample.to_parquet(dest / "sample.parquet", index=False)
    # Human coders see source text and codebook only, never model predictions or ratings.
    human = sample[["doc_id", "corpus", "text"]].copy()
    for c in CONSTRUCTS:
        for field in ["sal", "lvl", "quote"]: human[c + "_" + field] = ""
    human.to_csv(dest / "human_coding_blinded.csv", index=False, encoding="utf-8-sig")
    manifest = {"args": vars(args), "created_utc": datetime.now(timezone.utc).isoformat(),
                "models": {m: available[m] for m in args.models},
                "ollama_version": session.get(url + "/api/version", timeout=20).json(),
                "n_docs": len(sample), "protocol": "unconstrained JSON; same sample across all cells; one seed; quantized local inference"}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = dest / "generations.jsonl"
    done = set()
    if archive.exists():
        for line in archive.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if not r.get("error"): done.add(r["key"])
            except json.JSONDecodeError: raise RuntimeError("Damaged JSONL: repair final partial line before resuming")
    total = len(sample) * len(args.models) * len(args.conditions)
    count = 0
    for model in args.models:
        for _, doc in sample.iterrows():
            for condition in args.conditions:
                p = prompt(doc.text, condition)
                spec = {"model_digest": available[model]["digest"], "model": model,
                        "prompt": p, "seed": args.seed, "temperature": args.temperature,
                        "max_tokens": args.max_tokens, "context": args.context, "system": SYSTEM}
                key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
                count += 1
                if key in done: continue
                rec = {"key": key, "doc_id": doc.doc_id, "corpus": doc.corpus,
                       "model": model, "condition": condition, "spec": spec,
                       "utc": datetime.now(timezone.utc).isoformat()}
                started = time.monotonic()
                try:
                    response = session.post(url + "/api/chat", json={"model": model,
                        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                        "stream": False, "keep_alive": "10m",
                        "options": {"temperature": args.temperature, "top_p": 0.9,
                                    "seed": args.seed, "num_ctx": args.context, "num_predict": args.max_tokens}}, timeout=1800)
                    response.raise_for_status()
                    result = response.json()
                    rec["response"] = result
                    rec.update(strict_scores(result.get("message", {}).get("content", "")))
                    rec["truncated"] = result.get("done_reason") == "length"
                except Exception as exc:
                    rec["error"] = str(exc)
                rec["elapsed_s"] = time.monotonic() - started
                with archive.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{count}/{total} {model} {condition} {doc.doc_id} {rec['elapsed_s']:.1f}s valid={rec.get('schema_valid')} error={rec.get('error')}", flush=True)
                if rec.get("error"): raise RuntimeError(rec["error"])
        session.post(url + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30).raise_for_status()
    print("COMPLETE", flush=True)

if __name__ == "__main__": main()
