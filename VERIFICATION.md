# Manuscript check against archived logs

Checked on 11 October 2026 by reading the run logs and the tables those runs
wrote. The language-model calls were not repeated.

## Study 1

`logs/gpu.log` records `train=3363`, encoder `train=(3027, 14) val=(336, 14)`,
and an overall Qwen parse rate of `0.9933` (99.3%).
`logs/llm2.log` records Mistral-7B-Instruct-v0.2 on 1,600 documents × 3 seeds,
parse rate `0.7904` (79.0%), overlap 1,600, and the halo, agreement, and
Campbell–Fiske figures in Tables 2–4.

| Manuscript | Source | Logged value |
| --- | --- | --- |
| PC1 37.0 / 44.1 / 95.0 / 85.1 / 75.8% | `tables/t29_general_factor_share.csv` | 0.370, 0.441, 0.950, 0.851, 0.758 |
| Halo .19 / .94 / .77 / .68 / .28 | `logs/llm2.log` | 0.195, 0.935, 0.774, 0.685, 0.280 |
| Smallest pair n, Mistral 92 vs Qwen 920 | `tables/t29_general_factor_share.csv` | 92 and 920 |
| Qwen–Mistral mean r .72; SI .49, n = 187 | `logs/llm2.log` | mean r 0.716; SI r 0.491, n 187 |
| Campbell–Fiske .46 vs .58; criterion 3 = 0%; N = 1,597 | `logs/llm2.log` | 0.460, 0.581, 0.0, n_docs 1597 |
| ICC .81–.94 | `tables/t24_llm_run_stability.csv` | 0.809–0.936 |
| Prompt r .85–.94; eta-squared at most 0.008 | `tables/t25_llm_prompt_sensitivity.csv` | 0.846–0.942; max eta2 0.0077 |
| Pools 28,478 / 29,999 / 1,563 / 23,325; medians 25 / 75 / 152 / 13 | `tables/t03_corpus_descriptives.csv` | same |
| English 96.6 / 99.9 / 99.4 / 66.0%; p90 81 / 219 / 568.8 / 40 | `tables/t03_corpus_descriptives.csv` | 96.60, 99.91, 99.36, 65.96; p90 as stated |

Table 1 reports mean absolute correlation (.81 for Qwen). Table 2 reports the
signed halo on the 1,600-document overlap (.77). Both match their own files.
The prose uses .77 and cites Table 2.

## Study 2 and Study 3

`logs/mitigation.log` prints the profile after the Qwen rerun
(1,000 documents, seed 1, temperature 0.7):

| Condition | Coverage | Halo | PC1 | Convergent r | Margin | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| joint | 46.12 | 0.784 | 0.829 | 0.820 | 0.036 | 0.573 |
| isolated | 37.42 | 0.766 | 0.814 | 0.817 | 0.051 | 0.557 |
| evidence | 33.04 | 0.647 | 0.722 | 0.763 | 0.116 | 0.587 |
| gate | 11.72 | 0.854 | 0.371 | 0.843 | −0.012 | 0.522 |
| joint + PC1 removed | 46.12 | −0.096 | 0.341 | 0.285 | 0.381 | 0.522 |

Unrounded halo drop, joint minus isolated: 0.783640 − 0.766337 = 0.0173,
reported as .017. Coverage drop 46.12 − 33.04 = 13.08 points, reported as 13.1.
Quote audit in the same log matches Table 12, including verbatim rates
71.7% (ATT) and 8.1% (EE). The five-construct mean of those verbatim rates is
32.94%, reported as 32.9%. Construct coverage matches Table 11, including
EE 32.2% → 8.4%, SI 30.9% → 14.1%, and gate EE 1.1% / SI 1.6%.

Study 2 word counts in `upgrade/study2_corpus_summary.csv`: medians 24, 82.5,
145.5, 11 and totals 10,470, 28,897, 55,981, 4,087 in Amazon, Yelp,
Regulations.gov, Bluesky order.

## Complete-case and local extension

`logs/13b_local_upgrade.status` is `COMPLETE`. The inference log ends at
`2400/2400` and `COMPLETE`. `logs/13b_local_analysis.log` and
`upgrade/structure_audit.csv` agree on:

- Qwen joint complete cases: 146 (Amazon 28, Yelp 102, Regulations.gov 15, Bluesky 1)
- complete-case PC1 0.851; bootstrap interval [0.821, 0.880]
- Gaussian one-factor fit: chi-square 35.00 on 5 df, CFI 0.965, SRMR 0.020, RMSEA 0.203
- complete-case counts: Mistral 27, dictionary 4, isolated 36, evidence 27, gate 0. The manuscript withholds a complete-case PC1 when n is below 30, which covers Mistral, the dictionary, evidence, and the gate
- Yelp held-out rating R²: PC1 0.858, five scores 0.867, increment 0.009 (`upgrade/heldout_global_criterion.csv`)

Local joint halo runs from 0.689 (Gemma 27B) to 0.864 (Qwen 7B). Contrastive
halo is lower for local 7B and Gemma and slightly higher for 14B and 32B.
Schema-valid rates and complete-case counts match the local audit table.
Only 32B joint (PC1 0.789, n = 32, interval [0.691, 0.876]) and 7B contrastive
(PC1 0.880, n = 38, interval [0.820, 0.933]) meet the n ≥ 30 rule.
On matched supports, Qwen 14B PE–EE changes by −0.154 (n = 51, interval
[−0.272, −0.063]). Every reported pooled interval for local 7B and 32B
includes zero (`upgrade/matched_support_prompt_comparison.csv`).

`upgrade/model_manifest.json` records Q4_K_M for the three Qwen checkpoints,
Q4_0 for Gemma2 27B, and Ollama 0.40.1.

## Rounding note, not a table error

The native-language sentence says the four non-attitude constructs are detected
in at most 6% of posts. Table 9 prints 6.1 for Portuguese performance
expectancy (`tables/t27b_native_nonenglish.csv`, 6.0606). The table is the
precise figure.
