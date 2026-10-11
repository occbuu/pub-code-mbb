"""Measurement arm 3 -- fine-tuned transformer encoder.

The encoder is trained by *distillation from the LLM consensus* on the benchmark
pool, with a human-verified subsample used to check that the teacher labels are
themselves defensible.  This is what applied papers actually do, and it has a
consequence the benchmark is designed to expose: the encoder inherits the
teacher's method variance, so encoder-LLM agreement is not independent evidence
of validity.  The manuscript reports it as such.

Ten regression heads (5 constructs x {salience, polarity}) sit on one shared
DeBERTa-v3 encoder; polarity targets are masked where the teacher marked the
construct absent, so the polarity head is trained only on documents where the
construct is present.

Requires ``torch`` + ``transformers``; import lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .codebook import CONSTRUCTS

TARGETS = [f"{c}_{k}" for c in CONSTRUCTS for k in ("sal", "pol")]


@dataclass
class EncoderConfig:
    model_id: str = "microsoft/deberta-v3-base"
    max_length: int = 320
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 64
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    seed: int = 20260909
    grad_accum: int = 1
    fp16: bool = True
    dataloader_workers: int = 0        # 0: safe after CUDA has been initialised

    def provenance(self):
        d = asdict(self)
        d["run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return d


def _build_dataset(df, tok, cfg):
    import torch
    enc = tok(list(df["text"]), truncation=True, max_length=cfg.max_length,
              padding="max_length", return_tensors="pt")
    y = np.stack([df[t].values.astype("float32") for t in TARGETS], axis=1)
    mask = np.isfinite(y).astype("float32")
    y = np.nan_to_num(y, nan=0.0)
    return torch.utils.data.TensorDataset(
        enc["input_ids"], enc["attention_mask"],
        torch.tensor(y), torch.tensor(mask))


class EncoderScorer:
    """Multi-head regression on a shared transformer encoder."""

    def __init__(self, cfg: EncoderConfig = None):
        self.cfg = cfg or EncoderConfig()
        self.model = None
        self.tok = None

    def _build(self):
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        cfg = self.cfg
        # DeBERTa-v3's SPM tokenizer is unreliable as a fast tokenizer.
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=False)

        class Net(nn.Module):
            def __init__(self, model_id, n_out):
                super().__init__()
                # Keep master weights in FP32. transformers 5 may otherwise load
                # DeBERTa in FP16 on CUDA, which breaks GradScaler.
                self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
                hid = self.enc.config.hidden_size
                self.drop = nn.Dropout(0.1)
                self.head = nn.Linear(hid, n_out)

            def forward(self, ids, mask):
                h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).to(dtype=h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                return self.head(self.drop(pooled))

        self.model = Net(cfg.model_id, len(TARGETS))
        return self

    def fit(self, train: pd.DataFrame, val: pd.DataFrame = None, log=print):
        import torch
        from contextlib import nullcontext
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if self.model is None:
            self._build()
        # One visible GPU only (CUDA_VISIBLE_DEVICES is set by the launcher).
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(dev)
        self.model.float()

        ds = _build_dataset(train, self.tok, cfg)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.dataloader_workers, pin_memory=dev.type == "cuda")
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        steps = max(len(dl) * cfg.epochs // cfg.grad_accum, 1)
        sch = get_linear_schedule_with_warmup(opt, int(steps * cfg.warmup_ratio), steps)

        use_cuda = dev.type == "cuda"
        use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
        use_fp16_scaler = bool(cfg.fp16 and use_cuda and not use_bf16)
        amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16_scaler else None)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
        log(f"encoder device={dev} amp={amp_dtype} scaler={use_fp16_scaler}")

        def _amp():
            if amp_dtype is None:
                return nullcontext()
            try:
                return torch.amp.autocast("cuda", dtype=amp_dtype)
            except Exception:
                return torch.cuda.amp.autocast(dtype=amp_dtype)

        for ep in range(cfg.epochs):
            self.model.train()
            tot, nb = 0.0, 0
            for ids, mask, y, w in dl:
                ids, mask, y, w = ids.to(dev), mask.to(dev), y.to(dev), w.to(dev)
                with _amp():
                    pred = self.model(ids, mask)
                    loss = (((pred - y) ** 2) * w).sum() / w.sum().clamp(min=1.0)
                opt.zero_grad(set_to_none=True)
                if use_fp16_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                sch.step()
                tot += float(loss); nb += 1
            msg = f"epoch {ep+1}/{cfg.epochs}  train_loss={tot/max(nb,1):.4f}"
            if val is not None and len(val):
                msg += f"  val_r={self.validate(val):.3f}"
            log(msg)
        return self

    def predict(self, df: pd.DataFrame, id_col="doc_id") -> pd.DataFrame:
        import torch
        dev = next(self.model.parameters()).device
        self.model.eval()
        out = []
        bs = max(int(self.cfg.eval_batch_size), 1)
        texts = list(df["text"])
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok(texts[i:i + bs], truncation=True,
                               max_length=self.cfg.max_length, padding=True,
                               return_tensors="pt")
                pred = self.model(enc["input_ids"].to(dev),
                                  enc["attention_mask"].to(dev)).float().cpu().numpy()
                out.append(pred)
                del enc, pred
        P = np.vstack(out) if out else np.zeros((0, len(TARGETS)), dtype=np.float32)
        res = pd.DataFrame(P, columns=TARGETS)
        for c in CONSTRUCTS:
            res[f"{c}_sal"] = res[f"{c}_sal"].clip(0, 1)
            res[f"{c}_pol"] = res[f"{c}_pol"].clip(-1, 1)
            res.loc[res[f"{c}_sal"] < 0.05, f"{c}_pol"] = np.nan
            res[f"{c}_n"] = (res[f"{c}_sal"] >= 0.05).astype(int)
        res.insert(0, "doc_id", list(df[id_col]))
        return res

    def validate(self, val: pd.DataFrame) -> float:
        pred = self.predict(val)
        rs = []
        for t in TARGETS:
            a = pd.to_numeric(val[t], errors="coerce").values
            b = pred[t].values
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 20 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
        return float(np.mean(rs)) if rs else float("nan")


def llm_consensus_targets(llm_scores: pd.DataFrame, min_runs=2) -> pd.DataFrame:
    """Median across LLM runs -> distillation targets for the encoder."""
    cols = [c for c in llm_scores.columns
            if c.endswith("_sal") or c.endswith("_pol")]
    g = llm_scores.groupby("doc_id")
    med = g[cols].median()
    cnt = g[cols].count()
    med = med.where(cnt >= min_runs)
    return med.reset_index()
