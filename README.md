# Detect different lies

One model (Llama-3.1-8B-Instruct), 3,000 matched TriviaQA questions, knowledge crossed with
incentive: which false answers are hallucinations (the model does not know) and which are lies
(it knows and misreports), and do white-box lie detectors and hallucination detectors tell them
apart? Full write-up: [REPORT.md](REPORT.md).

## Key findings

- **Incentive alone produces almost no knowing lies.** False answers on known questions: 2.9% when
  falsehood pays vs 2.0% under a matched truth-pays control (net 0.9 points, 95% CI 0.3 to 1.5).
  With an explicit lie instruction: 34.5%, and knowing lies are 40% of all false answers.
- **Across prompts the deception probe detects the prompt.** Lie vs neutral correct answers:
  AUROC 1.00, but honest correct answers under the lie instruction vs neutral: also 1.00, and the
  probe read before any answer token: 1.00. At the Alpaca 1% FPR threshold it flags 96.8% of honest
  answers under the lie instruction.
- **Within a prompt it is lie-specific to a degree, not a general falsehood detector.** Lie vs
  honest-correct 0.87, lie vs hallucination 0.80, hallucination vs honest-correct 0.62 (best-case
  layer; 0.52 to 0.82 at other layers).
- **Hallucination detectors fire on both kinds of false answer.** Hallucination vs correct 0.91 to
  0.94; lie vs honest under the same prompt 0.76 to 0.88; the correctness probe ranks
  hallucinations slightly above lies (0.38).
- **A probe trained in-distribution separates lie from hallucination at 0.92**, but 0.90 is
  available from the prompt token alone, i.e. from whether the model knows the question.
- On a 510-item MASK subset the deception probe separates lies from honest answers at 0.75
  (0.70 from the prompt token alone).

Limits: one model, one run, lies are instructed lies, one incentive prompt, short answers.

## Reproduce

```bash
source .venv/bin/activate            # or: uv venv && uv pip install -r requirements.txt
export HF_TOKEN=... OPENROUTER_KEY=...
export TORCH_DISABLE_NATIVE_JIT=1    # needed on hosts without a C compiler
cd src
python pilot.py                      # pressure-prompt pilot (300 questions)
./run_all.sh                         # main collection, probe data, MASK (about 2 h on one A6000)
python analysis.py                   # results/behaviour.json, detectors.json, scores.npz
STRICT=1 python analysis.py          # strict-label sensitivity
python extra_stats.py && python mask_analysis.py && python plots.py
```

Needs a GPU with about 24 GB free, 7 GB of disk for activations, and under $2 of judge calls.

## Layout

| Path | Content |
|---|---|
| `planning.md` | Pre-experiment plan |
| `src/` | Collection, analysis and plotting scripts (prompts are in `src/common.py`) |
| `results/` | Responses, grades, result tables (activations `*.npz` are not in git) |
| `figures/` | Figures used in the report |
| `literature_review.md`, `resources.md` | Background gathered before the experiments |
