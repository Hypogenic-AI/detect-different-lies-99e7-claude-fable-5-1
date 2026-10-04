# Cloned Repositories

All cloned with `--depth 1`. None were executed in this phase (no user-specified code_references);
entry points below are from README/code inspection.

| Repo | URL | Purpose | Key files |
|---|---|---|---|
| `deception-detection/` | github.com/ApolloResearch/deception-detection | Apollo linear deception probes (paper 2502.03407). Reference implementation of the standard white-box lie detector. | `deception_detection/detectors.py` (LR / mean-diff probes), `activations.py`, `experiment.py`, `scripts/configs/`, `data/repe/` + `data/internal_state/` (Azaria–Mitchell facts for Instructed-Pairs), `example_results/` (trained probe weights for Llama-3.3-70B). Needs TOGETHER/OPENAI keys only for rollout generation/grading. |
| `liars-bench/` (913 MB, mostly `results/`) | github.com/Cadenza-Labs/liars-bench | Liars' Bench dataset construction + detectors. | `src/harm-refusal/harm-refusal.ipynb` (harm-pressure belief verification), `src/instruct-dishonesty/instruct_dishonesty.ipynb`, `src/mask/`, `src/blackbox/`, `scripts/full_pipeline.sh`, `data/raw/azaria_mitchell`. **`src/probes/` is an empty submodule** (points to Cadenza-Labs/deception-detection; cloned separately below). |
| `cadenza-deception-detection/` | github.com/Cadenza-Labs/deception-detection | Cadenza fork of the Apollo code used for the Liars' Bench probes (mean probe, follow-up probe, upper-bound probe). | same layout as `deception-detection/`. |
| `mask/` | github.com/centerforaisafety/mask | MASK evaluation pipeline. | `mask/generate_responses.py`, `mask/evaluate.py` (LLM-judge proposition resolution), `mask/metric.py`, `mask/prompts/evaluation_prompts.py`. Written for API models; swap in a local-model generate function. Judge needs OPENAI_API_KEY (available in env). |
| `geometry-of-truth/` | github.com/saprmarks/geometry-of-truth | True/false statement datasets and truth probes. | `datasets/*.csv` (cities, sp_en_trans, larger_than, common_claim, …), `probes.py` (LR, mass-mean, CCS), `generate_acts.py`. |
| `Truth_is_Universal/` | github.com/sciai-lab/Truth_is_Universal | TTPD lie detector, 2-D truth subspace. | `probes.py`, `generate_acts.py`, `lie_detection.ipynb`, `datasets/`. |
| `LLMsKnow/` | github.com/technion-cs-nlp/LLMsKnow | Hallucination/error probes on exact-answer tokens (Orgad et al.). | `src/generate_model_answers.py`, `src/extract_exact_answer.py`, `src/probe.py`, `src/logprob_detection.py`, `src/p_true_detection.py`; has TriviaQA loaders. |
| `semantic-entropy-probes/` | github.com/OATML/semantic-entropy-probes | Semantic entropy + SEPs. | `semantic_uncertainty/` (sampling, NLI clustering, entropy), `semantic_entropy_probes/train-latent-probe.ipynb`. |
| `LLM-LieDetector/` | github.com/LoryPack/LLM-LieDetector | Black-box follow-up-question lie detector (Pacchiardi et al.). | `lllm/`, `data/` (lie-instruction prompts, elicitation questions), `tutorial.ipynb`. Useful source of lie-instruction prompts. |
| `Beyond-Liars-Bench/` | github.com/amrgaber249/Beyond-Liars-Bench | Probe variants + SAE features on Liars' Bench. | `probes.py`, `activations.py`, `main.py`, `train_data_azaria_mitchell/`. Compact, readable probe-training code. |
| `Spont-Instructed-Deception/` (271 MB) | github.com/JosiahL98/Spont-Instructed-Deception | Instructed vs incentive-only deception pipeline. | `pipeline/`, `src/`, `paper_runs_manifest.md`. Incentive-only pressure prompts. |
| `Probing_Limits_Lie_Detectors/` | github.com/Tom-Felix-Thormann/Probing_Limits_Lie_Detectors | Truth probe on attention heads; misleading-but-true stimuli. | small repo. |
| `knowledge-recall-vs-truthfulness/` | github.com/AndyCheang/knowledge-recall-vs-truthfulness | Cheang et al. — **README only ("Coming soon"), no code or data yet.** | — |

Not obtainable: `ai-safety-institute/lie-detection` (printed in 2606.12618 without a host; GitHub
404 under `ai-safety-institute` and `UKGovernmentBEIS`). Its 64+64 honest/lie system prompts are
not printed in the paper, so the experiment runner must write its own.

## Suggested reuse
- Probe training/eval: adapt `deception-detection/deception_detection/detectors.py` or the simpler
  `Beyond-Liars-Bench/probes.py`; Instructed-Pairs facts are in `deception-detection/data/repe/`.
- Hallucination detectors: `LLMsKnow/src/probe.py` (correctness probe, exact answer token),
  `semantic-entropy-probes/semantic_uncertainty/` (semantic entropy), token log-prob baselines.
- Belief elicitation: `mask/mask/evaluate.py` prompts and `liars-bench/src/harm-refusal/`.
