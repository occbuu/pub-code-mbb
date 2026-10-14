# Paper 13B — reproducibility package

Code, run logs, and aggregate result tables for the structural audit of
LLM construct scores (five TPB/UTAUT constructs; dictionary, topic model,
distilled encoder, Qwen2.5-7B, and Mistral-7B, a local 7B–32B extension, and
Claude Sonnet 5.5 and Claude Opus 5.5 on the same 300 documents).

The manuscript numbers were checked against these logs and tables. See
`VERIFICATION.md`. Heavy model inference is not repeated from this package.

## What is included

| Path | Contents |
| --- | --- |
| `cft/` | Scoring library: codebook, dictionary, topic model, encoder, LLM prompts, reliability and validity helpers |
| `codebook/CODEBOOK.md` | Construct definitions and coding rules |
| `scripts/` | Batch runners for preparation, the four baseline arms, the Mistral replication, the elicitation study, the local Ollama extension, and the Claude frontier arm |
| `logs/` | Inference and analysis logs. No new model calls are required to read them |
| `tables/` | Study 1–3 aggregate tables cited by the manuscript |
| `upgrade/` | Complete-case audit, matched-support prompt comparisons, held-out rating check, local scores, and model digests |
| `frontier/` | Claude Sonnet 5.5 and Claude Opus 5.5 scores, parse audit, structure audit, and matched-support comparison on the same 300 documents |
| `notebooks/Paper13B_Reproducibility.ipynb` | Reads the archived tables |

## What is not included

Source documents are not redistributed. Amazon Reviews, the Yelp Open Dataset,
Regulations.gov comments, and Bluesky posts remain under their own licenses
and access conditions.

Also omitted, because they embed those source texts:

- normalized corpus files
- `derived/local_upgrade/generations.jsonl` (prompts and responses)
- the Claude `generations.jsonl` (response text can quote the source)
- blinded human-coding sheets
- the Anthropic API key (`ANTHROPIC_API_KEY` is read from the environment and is never written here)

`upgrade/local_scores.csv` and `frontier/claude_scores.csv` keep document identifiers and polarity scores only.
`upgrade/model_manifest.json` records Ollama digests, parameter size,
quantization, and server version for the local extension.

## Layout of the evidence

- Study 1 checkpoint comparison, agreement, and Campbell–Fiske summary: `logs/llm2.log`, `tables/t29`–`t32`
- Run stability, prompt wording, perturbations, and language coverage: `logs/gpu.log`, `tables/t24`–`t27b`
- Study 2 and Study 3 elicitation profile: `logs/mitigation.log`, `tables/t33`–`t35`
- Complete-case and local 7B–32B extension: `logs/13b_local_upgrade.log` (ends with `COMPLETE`), `logs/13b_local_analysis.log`, `upgrade/`
- Claude Sonnet 5.5 and Claude Opus 5.5, joint and contrastive, 300 documents: `logs/13b_claude_frontier.log` (ends with `COMPLETE`), `frontier/`

## Rerun notes

The original Qwen, Mistral, encoder, and translation calls used local GPUs and
are recorded in `logs/gpu.log` and `logs/llm2.log`. The local extension used
installed Ollama models (`qwen2.5:7b-instruct`, `qwen2.5:14b-instruct`,
`gemma2:27b`, `qwen2.5:32b-instruct-q4_K_M`), temperature 0.7, top-p 0.9,
context 8192, 512 output tokens, and sample seed `20261011`.

`scripts/analyze_13b_upgrade.py` rebuilds `upgrade/` only when the normalized
corpora and score archives are already on disk. It does not download models.

The Claude arm (`scripts/run_13b_claude.py`) calls the Anthropic Messages API
for `claude-sonnet-5-5` and `claude-opus-5-5`. The client does not send a
temperature or a seed: this API rejects a client temperature, so sampling is
the provider default. Output is capped at 512 tokens. Invalid fields stay
missing. `scripts/analyze_13b_claude.py` rebuilds `frontier/` from a local
generations archive and does not call the API. That archive is not in this
package.

Python dependencies are listed in `pyproject.toml`.
