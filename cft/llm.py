"""Measurement arm 4 -- LLM as a measurement instrument.

Runs an instruction-tuned model as a *rater* against the same codebook the human
coders and the dictionary use.  Three prompt variants (zero-shot, few-shot,
chain-of-thought) x K independent runs give the two reliability quantities that
matter for a stochastic instrument: run-to-run stability (test-retest) and
prompt sensitivity.

Everything needed for reproducibility is recorded with the scores: model id and
revision, quantisation, dtype, temperature, top_p, seed, prompt variant, prompt
hash, and the run date.  Raw generations are always kept.

The heavy dependencies (torch / transformers / vllm) are imported lazily, so this
module can be imported on a machine that has none of them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .codebook import CONSTRUCTS, CONSTRUCT_META

# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def _definition_block() -> str:
    lines = []
    for c in CONSTRUCTS:
        m = CONSTRUCT_META[c]
        lines.append(f"- {c} ({m['short']}): {m['definition']} Coding rule: {m['coding_rule']}")
    return "\n".join(lines)


SYSTEM = (
    "You are a trained research assistant coding text for latent behavioural "
    "constructs. You follow the codebook literally and you never speculate about "
    "information the writer did not express."
)

SCALE = (
    "For every construct give two numbers:\n"
    '  "sal" = 0, 1, 2 or 3: how much the document is ABOUT that construct '
    "(0 = not discussed at all, 3 = a central concern of the document).\n"
    '  "lvl" = an integer from -3 to +3: the LEVEL of the construct expressed '
    "(-3 = strongly low/negative, 0 = neutral or mixed, +3 = strongly high/positive). "
    'If "sal" is 0, set "lvl" to null.\n'
)

OUTPUT = (
    'Answer with a single JSON object and nothing else, in exactly this form:\n'
    '{"ATT": {"sal": <0-3>, "lvl": <-3..3 or null>}, '
    '"PE": {...}, "EE": {...}, "SI": {...}, "BI": {...}}'
)

FEWSHOT = [
    ("Ordered the stand for my laptop. Setup took two minutes and it holds the "
     "weight fine. My colleague had one and told me to get it. I'll buy the "
     "bigger version next.",
     {"ATT": {"sal": 1, "lvl": 1}, "PE": {"sal": 2, "lvl": 2},
      "EE": {"sal": 2, "lvl": 3}, "SI": {"sal": 2, "lvl": 2},
      "BI": {"sal": 2, "lvl": 3}}),
    ("Waited 45 minutes for a table we had booked, and when the food finally "
     "came it was cold. Everyone raves about this place, I don't see it. "
     "Not going back.",
     {"ATT": {"sal": 3, "lvl": -3}, "PE": {"sal": 2, "lvl": -2},
      "EE": {"sal": 3, "lvl": -3}, "SI": {"sal": 2, "lvl": 2},
      "BI": {"sal": 3, "lvl": -3}}),
    ("The proposed rule would require small pharmacies to file monthly reports. "
     "The paperwork alone would take a day a month. I ask the agency to "
     "withdraw it.",
     {"ATT": {"sal": 2, "lvl": -2}, "PE": {"sal": 1, "lvl": -1},
      "EE": {"sal": 3, "lvl": -3}, "SI": {"sal": 0, "lvl": None},
      "BI": {"sal": 3, "lvl": -3}}),
]


def build_prompt(text: str, variant: str = "zeroshot") -> str:
    head = ("Code the DOCUMENT below for five constructs.\n\n"
            "CODEBOOK\n" + _definition_block() + "\n\n" + SCALE + "\n")
    if variant == "zeroshot":
        body = head + OUTPUT + f"\n\nDOCUMENT:\n\"\"\"{text}\"\"\"\n\nJSON:"
    elif variant == "fewshot":
        ex = "\n\n".join(
            f"DOCUMENT:\n\"\"\"{t}\"\"\"\nJSON: {json.dumps(j)}" for t, j in FEWSHOT)
        body = head + OUTPUT + "\n\nEXAMPLES\n" + ex + \
            f"\n\nNow code this one.\nDOCUMENT:\n\"\"\"{text}\"\"\"\n\nJSON:"
    elif variant == "cot":
        body = (head +
                "First think step by step in at most three short sentences: which "
                "constructs the document actually addresses, and how strongly. "
                "Then give the JSON.\n\n" + OUTPUT +
                f"\n\nDOCUMENT:\n\"\"\"{text}\"\"\"\n\nReasoning:")
    else:
        raise ValueError(variant)
    return body


def prompt_hash(variant: str) -> str:
    return hashlib.blake2b(build_prompt("X", variant).encode(), digest_size=6).hexdigest()


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.S)


def parse_generation(txt: str) -> dict | None:
    """Extract the last well-formed JSON object with the expected keys."""
    for m in reversed(_JSON_RE.findall(txt or "")):
        try:
            obj = json.loads(m)
        except Exception:
            continue
        if isinstance(obj, dict) and any(c in obj for c in CONSTRUCTS):
            return obj
    return None


def to_scores(obj: dict) -> dict:
    """Codebook JSON -> the project's ``_sal`` / ``_pol`` convention."""
    out = {}
    for c in CONSTRUCTS:
        d = (obj or {}).get(c) or {}
        try:
            sal = float(d.get("sal", 0) or 0)
        except Exception:
            sal = 0.0
        lvl = d.get("lvl", None)
        try:
            lvl = float(lvl) if lvl is not None else np.nan
        except Exception:
            lvl = np.nan
        out[f"{c}_sal"] = np.clip(sal / 3.0, 0, 1)
        out[f"{c}_pol"] = np.nan if (sal <= 0 or not np.isfinite(lvl)) \
            else float(np.clip(lvl / 3.0, -1, 1))
        out[f"{c}_n"] = int(sal > 0)
    return out


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
@dataclass
class LLMRunConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    max_new_tokens: int = 220
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 1
    variant: str = "zeroshot"
    batch_size: int = 8
    backend: str = "transformers"      # {"transformers", "vllm", "openai"}
    api_model: str | None = None
    gpu_memory_utilization: float = 0.88
    max_model_len: int = 2048
    tensor_parallel_size: int = 1      # always one GPU for this project

    def provenance(self) -> dict:
        d = asdict(self)
        d["prompt_hash"] = prompt_hash(self.variant)
        d["run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return d


class LLMScorer:
    """Thin wrapper that keeps every raw generation next to its parsed score."""

    def __init__(self, cfg: LLMRunConfig):
        self.cfg = cfg
        self._pipe = None
        self._llm = None

    # -- backends ---------------------------------------------------------
    def _load_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # transformers 5 pre-allocates the whole model as a CUDA warmup and
        # OOMs on a shared workstation even when the 7B weights would fit.
        try:
            import transformers.modeling_utils as _mu
            _mu.caching_allocator_warmup = lambda *a, **k: None
        except Exception:
            pass
        kw = {}
        if self.cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": 0}
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_id,
                                                 revision=self.cfg.revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = (getattr(torch, self.cfg.dtype)
                 if torch.cuda.is_available() else torch.float32)
        load_kw = dict(revision=self.cfg.revision, dtype=dtype,
                       low_cpu_mem_usage=True, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, **load_kw)
        if "device_map" not in kw and torch.cuda.is_available():
            self.model.to("cuda:0")
        self.model.eval()

    def _format_chat(self, prompt: str) -> str:
        """Apply the model's chat template.  Some instruct checkpoints (Mistral
        v0.2, older Llama) reject a dedicated system turn; fold it into user."""
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tok.apply_chat_template(messages, **kw)
        except Exception:
            merged = SYSTEM + "\n\n" + prompt
            return self.tok.apply_chat_template(
                [{"role": "user", "content": merged}], **kw)

    def _load_vllm(self):
        from vllm import LLM
        self._llm = LLM(model=self.cfg.model_id, dtype=self.cfg.dtype,
                        seed=self.cfg.seed,
                        tensor_parallel_size=self.cfg.tensor_parallel_size,
                        gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                        max_model_len=self.cfg.max_model_len)

    # -- generation -------------------------------------------------------
    def generate(self, prompts: list[str]) -> list[str]:
        cfg = self.cfg
        if cfg.backend == "vllm":
            from vllm import SamplingParams
            if self._llm is None:
                self._load_vllm()
            sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p,
                                max_tokens=cfg.max_new_tokens, seed=cfg.seed)
            chats = [[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": p}] for p in prompts]
            outs = self._llm.chat(chats, sp)
            return [o.outputs[0].text for o in outs]

        if cfg.backend == "openai":                      # optional API branch
            from openai import OpenAI
            client = OpenAI()
            res = []
            for p in prompts:
                r = client.chat.completions.create(
                    model=cfg.api_model or cfg.model_id,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": p}],
                    temperature=cfg.temperature, top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens, seed=cfg.seed)
                res.append(r.choices[0].message.content)
            return res

        import torch
        if self._pipe is None:
            self._load_transformers()
            self._pipe = True
        torch.manual_seed(cfg.seed)
        device = next(self.model.parameters()).device
        outs = []
        bs = max(int(cfg.batch_size), 1)
        i = 0
        n = len(prompts)
        last_log = 0
        oom_err = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

        def _is_oom(exc: BaseException) -> bool:
            if isinstance(exc, oom_err):
                return True
            return "out of memory" in str(exc).lower()

        while i < n:
            chunk = prompts[i:i + bs]
            enc = gen = None
            try:
                texts = [self._format_chat(p) for p in chunk]
                enc = self.tok(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=cfg.max_model_len).to(device)
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc, max_new_tokens=cfg.max_new_tokens, do_sample=True,
                        temperature=cfg.temperature, top_p=cfg.top_p,
                        pad_token_id=self.tok.pad_token_id, use_cache=True)
                plen = enc["input_ids"].shape[1]
                for j in range(len(chunk)):
                    outs.append(self.tok.decode(gen[j][plen:],
                                                skip_special_tokens=True))
                i += len(chunk)
                if i - last_log >= 40 or i >= n:
                    from .resources import log
                    log(f"llm generate {i}/{n} docs (batch={bs})")
                    last_log = i
            except Exception as e:
                if not _is_oom(e):
                    raise
                if bs <= 1:
                    raise
                from .resources import log
                bs = max(1, bs // 2)
                cfg.batch_size = bs
                log(f"CUDA OOM in generate; retry from doc {i} with batch_size={bs}")
            finally:
                del gen, enc
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return outs

    # -- public -----------------------------------------------------------
    def score(self, docs: pd.DataFrame, text_col="text", id_col="doc_id",
              keep_raw=True) -> pd.DataFrame:
        prompts = [build_prompt(t, self.cfg.variant) for t in docs[text_col]]
        raw = self.generate(prompts)
        rows = []
        for did, r in zip(docs[id_col], raw):
            obj = parse_generation(r)
            rec = dict(doc_id=did, parsed=obj is not None)
            rec.update(to_scores(obj or {}))
            if keep_raw:
                rec["raw"] = r
            rows.append(rec)
        out = pd.DataFrame(rows)
        prov = self.cfg.provenance()
        for k, v in prov.items():
            out[f"prov_{k}"] = v
        out["run"] = f"{self.cfg.variant}_s{self.cfg.seed}"
        return out


# ---------------------------------------------------------------------------
# bias-audit perturbations
# ---------------------------------------------------------------------------
NAME_SETS = {
    # Perturbation sets for the name audit.  These are *stimuli*, used only to
    # test whether the instrument's output moves when an irrelevant identity cue
    # is inserted; no inference about any real person is made or implied.
    "set_a": ["Jake", "Connor", "Emily", "Katie"],
    "set_b": ["Jamal", "DeShawn", "Latoya", "Keisha"],
    "set_c": ["Nguyen", "Minh", "Wei", "Ling"],
}

DESIRABILITY_CUE = ("This review will be shown to the manager of the business "
                    "together with my name. ")


def insert_name(text: str, name: str) -> str:
    return f"My name is {name}. " + text


def make_perturbations(docs: pd.DataFrame, text_col="text",
                       seed=20260909) -> pd.DataFrame:
    """Long frame of perturbed documents: name insertion (three sets) and a
    social-desirability cue, each keyed back to the original ``doc_id``."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in docs.iterrows():
        rows.append(dict(doc_id=r["doc_id"], perturbation="none", text=r[text_col]))
        for k, names in NAME_SETS.items():
            nm = names[int(rng.integers(0, len(names)))]
            rows.append(dict(doc_id=r["doc_id"], perturbation=k,
                             text=insert_name(r[text_col], nm)))
        rows.append(dict(doc_id=r["doc_id"], perturbation="desirability_cue",
                         text=DESIRABILITY_CUE + r[text_col]))
    return pd.DataFrame(rows)
