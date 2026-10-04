# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-04T07:25:28.006880Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/detect-different-lies-99e7-claude-fable-5-1`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase**: resource_finder — complete. Artifacts: `papers/` (25 PDFs + 1 HTML, `README.md`, `notes/`, `text/`), `datasets/` (6 datasets, `README.md`, `.gitignore`), `code/` (13 repos, `README.md`), `literature_review.md`, `resources.md`, `.resource_finder_complete`.

**Key findings**
- All 10 user-specified sources obtained and read in full; per-paper notes with page refs in `papers/notes/`. 16 search-found papers reviewed at abstract level only.
- In the 10 deep-read sources, no experiment scores a lie probe and a hallucination detector on both lies and hallucinations for one model on matched questions. "Did you lie?" filters unknown questions out; Cheang et al. has no lie condition; MASK has no detectors.
- Adjacent evidence: Liars' Bench HP-KR (genuine vs feigned "I don't know") — Apollo-style probe AUROC 0.36, in-distribution probe 0.96. Cheang: hallucination detectors 0.86–0.93 on no-knowledge errors, 0.48–0.69 on confident-wrong ones. Thormann: truth probes track falsity. Apollo probe fires on honest answers under pressure prompts.
- Lie labels from one neutral-vs-pressure contradiction are noisy (13–34% disputed in 2606.12618; 25% overturned in the Anthropic post).

**Retained directions (budget 3)** — full scoring table in `literature_review.md`
- A. Matched 2x2 on TriviaQA (+PopQA): known/unknown via repeated neutral elicitation x neutral/incentive prompt; share of false outputs by mechanism.
- B. Cross-evaluate standard detectors on those cells (Apollo probe default + pressure pair, follow-up probe, truth probe, correctness probe, token probability, semantic entropy), with honest-under-pressure controls.
- C. MASK replication on the same model + in-distribution lie-vs-hallucination probe as separability ceiling.

**Pruned**: fine-tuned model organisms (cost, belief verification); black-box detectors as main object (hypothesis is white-box); SAE / residual-rank analyses (no gain reported in 2607.20479; Rift unverified); multi-model scale sweep (one model specified); re-probing Liars' Bench 24–72B transcripts (quantisation, disputed labels).

**Decisions**
- Model: `meta-llama/Llama-3.1-8B-Instruct` (MASK Table 3: P(lie) 23.5, accuracy 62.0); fallback `google/gemma-3-12b-it`. HF access to both verified. GPU: 1x RTX A6000 48 GB.
- Split the "does not know" bucket into confident-wrong vs inconsistent; count evasions separately; force short answers under pressure.

**Next phase (experiment_runner)**
1. `source .venv/bin/activate`; `uv add torch transformers accelerate scikit-learn`.
2. Build neutral elicitation on TriviaQA validation (`datasets/trivia_qa_nocontext`) to label known / confident-wrong / no-knowledge.
3. Write incentive prompts (adapt MASK, `code/LLM-LieDetector/data`, `code/Spont-Instructed-Deception`); generate pressured answers; label cells.
4. Train/apply detectors (see `code/README.md` "Suggested reuse"); sweep layers; Alpaca per-sample 1% FPR threshold.

**Failures / uncertainty**
- Paper-finder unavailable (HTTP 500); used Semantic Scholar API instead.
- `ai-safety-institute/lie-detection` repo not found; Cheang et al. repo is an empty placeholder.
- Rift (2606.17229) and PIR (2609.21996) claim adjacent results but were not read beyond abstracts — read before making novelty claims.
- Figure values in 2602.01425 (Fig. 3) and 2607.20479 (Fig. 2) came from garbled extraction; verify against PDFs before citing.
- Alpaca is not saved standalone; use `datasets/liars_bench/alpaca` prompts or `tatsu-lab/alpaca`, generating responses on-policy.
- No code repo was executed in this phase.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase**: experiment_runner — complete (planning, collection, analysis, documentation, validation). Artifacts: `planning.md`, `src/`, `results/*.json`, `figures/fig1-4*.png`, `REPORT.md`, `README.md`, `requirements.txt`.

**What was run**
- Llama-3.1-8B-Instruct, 3,000 TriviaQA questions, 4 system prompts (neutral, truth-pays control, falsehood-pays incentive, instructed lie); knowledge labels from 3 greedy paraphrases + 10 samples; judge `openai/gpt-5.6-luna`.
- Detectors: Apollo instructed-pairs probe (LR, mean-diff, prompt-token-only), self-report follow-up, truth probe, correctness probe, answer log-prob, sample inconsistency, in-distribution ceilings. MASK subset (510 items) as external check.

**Key findings** (evidence: `results/behaviour.json`, `detectors.json`, `extra_stats.json`, `mask_results.json`)
- Incentive-only lie yield on known questions: 2.9% vs 2.0% control; net 0.9 pp [0.3, 1.5], McNemar p = 0.004. Instructed: 34.5%; knowing lies = 40% of false answers.
- Deception probe across prompts = prompt detector (lie vs neutral correct 1.00; honest-under-lie-prompt vs neutral 1.00; prompt-token-only 1.00; flags 96.8% of honest answers under the lie prompt).
- Within prompt: lie vs honest 0.87, lie vs hallucination 0.80, hallucination vs honest 0.62 (layer 12; other layers 0.52-0.82).
- Hallucination detectors: hallucination vs correct 0.91-0.94; also lie vs honest (same prompt) 0.76-0.88; correctness probe lie vs hallucination 0.38.
- In-distribution lie-vs-hallucination probe 0.92; prompt-token-only 0.90.
- MASK: deception probe lie vs honest 0.75 [0.69, 0.81]; prompt-token-only 0.70.

**Decisions / deviations from plan**
- Pool is TriviaQA only (PopQA dropped); added a truth-pays control prompt after the pilot; follow-up probe not trained (activations saved); per-example mean activations for probe training.
- Logistic regression solved with L-BFGS on GPU (`TorchLR` in `src/analysis.py`) because the host CPU was heavily loaded.

**Failures / uncertainty**
- `TORCH_DISABLE_NATIVE_JIT=1` is required (no C compiler for torch's Triton kernels).
- Judge returned unusable output for ~0.1% of calls; those fall back to alias match.
- Incentive-only "lie" cell (n = 52) is mostly noise; no claim about spontaneous deception is supported.
- Deception-probe layer was chosen on a saturated contrast; within-prompt numbers are near best case.
- Rift (2606.17229) and PIR (2609.21996) still unread beyond abstracts; novelty claims are hedged.
- Generation was run once; analysis was re-run and reproduced identical numbers.

**Next phase**: paper writing can start from `REPORT.md`; follow-ups listed in REPORT.md section 7.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
