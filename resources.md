# Resources Catalog

## Summary
Resources for the project "Detect different lies": separating hallucinations (model does not know)
from lies (model knows but misreports under an incentive) for one model on matched questions, and
testing whether white-box lie detectors and hallucination detectors tell them apart.

- Papers: 25 PDFs + 1 blog post (10 user-specified, all obtained and deep-read)
- Datasets: 6 downloaded (≈260 MB), both user-specified ones included
- Code repositories: 13 cloned (one is a placeholder with no code)
- Environment: `uv` venv at `.venv/` with `pypdf requests arxiv datasets huggingface_hub httpx`;
  `pyproject.toml` + `uv.lock` at root. torch/transformers are **not** installed yet.
- Hardware seen: 1× NVIDIA RTX A6000 (48 GB), idle. Disk: 425 GB free.

## Papers

| Title | Authors | Year | File | Key info |
|---|---|---|---|---|
| The MASK Benchmark | Ren et al. | 2025 | papers/2503.03750_mask_benchmark.pdf | Belief elicitation vs pressure; lie ≠ inaccuracy |
| Liars' Bench | Kretschmar et al. | 2025 | papers/2511.16035_liars_bench.pdf | 7 lie datasets; standard probes near chance on harm-pressure sets |
| Detecting Strategic Deception Using Linear Probes | Goldowsky-Dill et al. | 2025 | papers/2502.03407_strategic_deception_linear_probes.pdf | Standard deception probe recipe |
| One Probe Won't Catch Them All | Natarajan et al. | 2026 | papers/2602.01425_one_probe_wont_catch_them_all.pdf | Prompt pair drives probe performance |
| Probing the Limits of the Lie Detector Approach | Thormann | 2026 | papers/2603.10003_limits_of_lie_detector_approach.pdf | Truth probes track falsity |
| Asymmetries in Spontaneous and Instructed Deception | Luikham | 2026 | papers/2609.00180_spontaneous_vs_instructed_deception.pdf | Instructed vs incentive-only lies |
| Beyond Liars' Bench | Moustafa, Feser, Mai | 2026 | papers/2607.20479_beyond_liars_bench.pdf | Probe transfer below chance on harm pressure |
| Do LLMs Really Know What They Don't Know? | Cheang et al. | 2025 | papers/2510.09033_do_llms_know_what_they_dont_know.pdf | Hallucination detectors track recall |
| "Did you lie?" | Cooney, Africa, Irving | 2026 | papers/2606.12618_did_you_lie.pdf | Belief-verified TriviaQA lies; self-report probe |
| Fine-Tuned Lie Detectors Failed to Generalize | Hopkins et al. (Anthropic) | 2026 | papers/anthropic_lie_detectors.html | On-policy lie data; label noise |
| 16 further papers (truth probes, hallucination detectors, probe robustness) | — | 2022–2026 | papers/*.pdf | Abstract-level review; see papers/README.md |

See `papers/README.md` for the full list and `papers/notes/*.md` for detailed notes on the first ten.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| MASK | HF `cais/MASK` | 1,000 rows, 6 configs | Pressure prompts + belief elicitation | datasets/mask/ | User-specified. Canary string. |
| TriviaQA (no context) | HF `mandarjoshi/trivia_qa` `rc.nocontext` | 138,384 / 17,944 / 17,210 | Open-domain QA | datasets/trivia_qa_nocontext/ | User-specified. Heavy columns dropped; test split has no answers. |
| PopQA | HF `akariasai/PopQA` | 14,267 | Long-tail entity QA with popularity | datasets/popqa/ | Source of unknown items |
| Liars' Bench | HF `Cadenza-Labs/liars-bench` | 8 configs, ≈79k transcripts | Lie/honest transcripts | datasets/liars_bench/ | Gated (auto-approved). 24B–72B models only. |
| MASK generations | HF `Cadenza-Labs/mask-generations` | 1,045 | MASK transcripts with labels | datasets/mask_generations/ | |
| Anthropic lies | HF `Noddybear/lies` | 106 parquet files, 88 MB | On-policy lie data, Gemma-3 etc. | datasets/anthropic_lies_noddybear/ | Raw parquet snapshot |

See `datasets/README.md` for schemas and download instructions.

## Code Repositories

| Name | URL | Purpose | Location |
|---|---|---|---|
| deception-detection | github.com/ApolloResearch/deception-detection | Apollo probes | code/deception-detection/ |
| cadenza-deception-detection | github.com/Cadenza-Labs/deception-detection | Liars' Bench probe fork | code/cadenza-deception-detection/ |
| liars-bench | github.com/Cadenza-Labs/liars-bench | Dataset construction, belief verification | code/liars-bench/ |
| mask | github.com/centerforaisafety/mask | MASK pipeline and judge prompts | code/mask/ |
| geometry-of-truth | github.com/saprmarks/geometry-of-truth | Truth probes, true/false datasets | code/geometry-of-truth/ |
| Truth_is_Universal | github.com/sciai-lab/Truth_is_Universal | TTPD lie detector | code/Truth_is_Universal/ |
| LLMsKnow | github.com/technion-cs-nlp/LLMsKnow | Hallucination probes | code/LLMsKnow/ |
| semantic-entropy-probes | github.com/OATML/semantic-entropy-probes | Semantic entropy, SEPs | code/semantic-entropy-probes/ |
| LLM-LieDetector | github.com/LoryPack/LLM-LieDetector | Black-box detector, lie prompts | code/LLM-LieDetector/ |
| Beyond-Liars-Bench | github.com/amrgaber249/Beyond-Liars-Bench | Compact probe training code | code/Beyond-Liars-Bench/ |
| Spont-Instructed-Deception | github.com/JosiahL98/Spont-Instructed-Deception | Incentive-only pressure pipeline | code/Spont-Instructed-Deception/ |
| Probing_Limits_Lie_Detectors | github.com/Tom-Felix-Thormann/Probing_Limits_Lie_Detectors | Attention-head truth probe | code/Probing_Limits_Lie_Detectors/ |
| knowledge-recall-vs-truthfulness | github.com/AndyCheang/knowledge-recall-vs-truthfulness | Placeholder only | code/knowledge-recall-vs-truthfulness/ |

See `code/README.md` for key files. No repository was executed in this phase.

## Resource Gathering Notes

### Search strategy
1. Downloaded all ten user-specified sources first (nine arXiv PDFs, one blog post as HTML).
2. Paper-finder was tried and was unavailable (first run: missing `httpx`; after install: HTTP 500
   from the local service). Fell back to 14 Semantic Scholar API queries plus citation chasing
   from the specified papers.
3. Dataset and code links were taken from the papers' text and the blog's HTML.

### Selection criteria
Papers that define the lie/hallucination distinction, supply a standard detector for either, or
document confounds in such detectors. Datasets that give matched factual questions, pressure
prompts, or existing labelled lies. Code that implements a detector we need to reproduce.

### Challenges encountered
- Paper-finder down (above).
- `Cadenza-Labs/liars-bench` is gated; access was granted automatically after an access request
  with the workspace HF token.
- `liars-bench/src/probes` is an unfetched submodule; the fork was cloned separately.
- `ai-safety-institute/lie-detection` (paper 2606.12618) could not be located; the paper's
  64+64 system prompts are not printed.
- Cheang et al.'s repository contains no code or data yet.
- Text extraction garbled some figures (2602.01425 Fig. 3, 2607.20479 Fig. 2); specific values
  from those figures should be checked against the PDFs before being cited.

### Gaps and workarounds
- No existing dataset labels the *mechanism* of a false output for a small open model. The labels
  must be generated on-policy (neutral elicitation, then pressure), following MASK and
  "Did you lie?".
- Lie-inducing prompts must be written or adapted from MASK, `LLM-LieDetector/data`, and
  `Spont-Instructed-Deception`.
- Hallucination-detector code for Cheang et al. is unavailable; `LLMsKnow` and
  `semantic-entropy-probes` cover the same detector families.
- Sixteen search-found papers were reviewed from abstracts only; in particular Rift
  (2606.17229) and PIR (2609.21996) claim results adjacent to the hypothesis and should be read
  before novelty claims are made.

## Recommendations for Experiment Design

1. **Primary datasets**: TriviaQA `rc.nocontext` validation (matched question pool), PopQA for
   extra unknown items, MASK for an external-validity replication; Azaria–Mitchell facts and
   Alpaca for probe training and threshold calibration.
2. **Baseline methods**: Apollo Instructed-Pairs probe (default and pressure-style prompt pair),
   follow-up/self-report probe, truth probe (difference of means / logistic regression on
   true/false statements), correctness probe at the answer token, answer-token probability,
   semantic entropy or sample consistency, and an in-distribution lie-vs-hallucination probe as a
   ceiling.
3. **Evaluation metrics**: shares of false outputs by mechanism with confidence intervals; AUROC
   for each detector on lie-vs-honest, hallucination-vs-correct and lie-vs-hallucination;
   balanced accuracy at a per-sample 1% FPR Alpaca threshold.
4. **Code to adapt**: `deception-detection` or `Beyond-Liars-Bench/probes.py` for probes;
   `LLMsKnow/src` and `semantic-entropy-probes/semantic_uncertainty` for hallucination detectors;
   `mask/mask/evaluate.py` for belief and judge prompts.
5. **Model**: `meta-llama/Llama-3.1-8B-Instruct` primary, `google/gemma-3-12b-it` fallback (HF
   access to both verified with the workspace token). `uv add torch transformers accelerate
   scikit-learn` is needed first.

The three retained research directions and the pruned ones are in `literature_review.md`
("Direction ranking") and summarised in `STATE.md`.

## Experiment-runner addendum (2026-10-04)

Used: `datasets/trivia_qa_nocontext` (3,000 + 300 pilot validation questions), `datasets/mask`
(known_facts, disinformation, continuations), `datasets/liars_bench/alpaca` (500 prompts for the
control threshold), `code/deception-detection/data/repe/true_false_facts.csv` (instructed pairs),
`code/geometry-of-truth/datasets/*.csv` (truth probe). Not used: PopQA, Liars' Bench transcripts,
`anthropic_lies_noddybear`, `mask_generations`. No cloned repository was executed; the probe
recipes were re-implemented in `src/analysis.py`. Model: `meta-llama/Llama-3.1-8B-Instruct`
(local). Judge: `openai/gpt-5.6-luna` via OpenRouter.
