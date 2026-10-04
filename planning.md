# Planning: Detect different lies

## Motivation & Novelty Assessment

### Why This Research Matters
White-box deception probes are proposed as monitors for deployed models. A monitor is only useful
if it responds to a *mismatch between belief and statement*. If it instead responds to falsehood,
to low-knowledge states, or to the wording of the prompt, it will flag honest mistakes and can
miss confident knowing lies. Knowing which of these a probe reads is a precondition for using it.

### Gap in Existing Work
(From `literature_review.md`.) MASK separates honesty from accuracy behaviourally but evaluates no
detector. Liars' Bench, Goldowsky-Dill et al., Natarajan et al. and Luikham evaluate lie detectors
with *honest (mostly correct)* negatives, or compare lie types with each other. Cheang et al. study
hallucination detectors with no lie condition. "Did you lie?" filters unknown questions out. In
the ten deep-read sources no experiment scores a lie probe and a hallucination detector on false
outputs of both mechanisms for the same model and the same questions. (Rift 2606.17229 and PIR
2609.21996 are adjacent and were read only at abstract level; novelty claims are hedged accordingly.)

### Our Novel Contribution
A matched design on one model (Llama-3.1-8B-Instruct): every question is answered under a neutral
prompt, an incentive-only pressure prompt and an explicit lie instruction; knowledge is labelled
independently by repeated neutral elicitation. This gives a knowledge x incentive table of false
outputs, and lets standard lie detectors and hallucination detectors be scored on
lie-vs-hallucination with the prompt held fixed.

### Experiment Justification
- **Exp 1 (behavioural 2x2)**: needed to answer "what share of false outputs is epistemic vs
  incentive-driven", and to report the yield of genuine belief-contradicting answers without an
  explicit lie instruction.
- **Exp 2 (detector cross-evaluation)**: needed to see whether each detector separates lies from
  hallucinations, or only separates prompts / false from true / known from unknown.
- **Exp 3 (ceiling + external check)**: an in-distribution lie-vs-hallucination probe tells us
  whether the information is linearly present at all (so a failure in Exp 2 is a failure of
  transfer, not of representation); a prompt-token-only probe measures the prompt confound; MASK
  `known_facts` checks whether conclusions survive on realistic pressure scenarios.

## Research Question
For one model and matched questions: (1) what share of false outputs comes from not knowing
(hallucination) versus knowing and misreporting under an incentive (lie)? (2) Do standard
white-box lie detectors and hallucination detectors tell the two apart, or do they respond to
falsehood, uncertainty, or the prompt?

## Direction ranking (budget 3)
Retained (inherits the resource_finder ranking, which the evidence does not contradict):
A. matched 2x2 on TriviaQA; B. detector cross-evaluation with honest-under-pressure controls;
C. in-distribution ceiling probe + prompt-only control + small MASK replication.
Pruned: fine-tuned model organisms (cost; belief verification); black-box detectors as main
object (hypothesis is white-box); SAE analyses (no reported gain); multi-model sweep (one model
specified; time); re-probing Liars' Bench 24-72B transcripts (different models, disputed labels);
PopQA as a second pool (TriviaQA alone already yields enough unknown items at ~8B scale).

## Hypothesis Decomposition
- H1: under a neutral prompt nearly all false outputs are epistemic; under pressure the share of
  knowing misreports rises, much more with an explicit instruction than with incentive only.
- H2: an Apollo-style instructed-pairs deception probe separates lies from neutral honest answers
  well, but much of that separation is the prompt: it drops for lie vs honest-under-same-pressure
  and for lie vs false-answer-without-knowledge under the same prompt.
- H3: hallucination detectors (correctness probe, answer probability, sample consistency) track
  knowledge/uncertainty; they separate hallucinations from correct answers but do not flag
  knowing lies as "false" in the same way (or flag them for a different reason).
- H4: a probe trained in-distribution can separate lie from hallucination (information is present).

Independent variables: knowledge state (known / confident-wrong / no-knowledge / ambiguous);
prompt condition (neutral / incentive-only / instructed lie). Dependent: answer correctness,
evasion; detector scores.

## Proposed Methodology
### Model
`meta-llama/Llama-3.1-8B-Instruct`, local, fp16, one RTX A6000. Judge:
`openai/gpt-5.6-luna` via OpenRouter (live catalog checked 2026-10-04) to grade answers as
correct / incorrect / evasive against TriviaQA aliases; alias string match as a second grader.

### Steps
1. **Pool**: 2,000 random TriviaQA `rc.nocontext` validation questions (seed 0).
2. **Belief elicitation (neutral)**: 3 paraphrased neutral templates, greedy answer each, plus 10
   samples at T=1 with template 1. Labels (pre-specified):
   - *known*: all 3 greedy answers correct and >=8/10 samples correct;
   - *confident-wrong*: all 3 greedy wrong, and >=8/10 samples agree with the greedy wrong answer;
   - *no-knowledge*: all 3 greedy wrong (or "don't know"), <=1/10 samples correct, not confident-wrong;
   - *ambiguous*: everything else (excluded from probe contrasts, reported in shares).
3. **Statement conditions** (same user turn for all; only the system prompt changes):
   neutral; incentive-only pressure (pilot 4-5 prompts on 300 questions, keep the one with the
   highest lie yield on known questions, report all yields); instructed lie. Greedy, short answers.
   Under each condition also 10 samples for a same-prompt consistency score.
4. **Cells**: known x false = lie (after removing evasions); not-known x false = hallucination
   (neutral) or "false without knowledge" (pressure); known x correct under pressure =
   honest-under-pressure control.
5. **Detectors**
   - Lie: Apollo instructed-pairs logistic-regression probe (and difference-of-means variant),
     trained on Azaria-Mitchell facts under honest/dishonest persona prompts; scored as mean over
     response tokens. Follow-up probe ("Was your last response deceptive?" -> "No").
   - Truth: LR probe on Geometry-of-Truth true/false statements (cities, sp_en_trans, etc.).
   - Hallucination: correctness probe trained on neutral answers (train questions only);
     mean answer log-probability; sample consistency (same-prompt) as a semantic-entropy proxy.
   - Layer: chosen on training data / a held-out dev split of questions, never on test cells;
     full layer sweep shown as a figure.
6. **Controls**: prompt-token-only scores (activation at the last prompt token, before any answer
   is produced); honest-under-pressure vs honest-neutral contrast; Alpaca on-policy responses for
   a 1% FPR threshold.
7. **Ceiling**: in-distribution LR probe, lie vs hallucination, question-level 5-fold CV.
8. **MASK**: `known_facts` (209) + `disinformation`/`continuations` if time: belief by elicit_1
   x3, pressure statement, judge mapping to proposition side; apply the same probes.

### Contrasts (AUROC, bootstrap 95% CI, question-level split 50/50 train/test)
a. lie vs honest-correct neutral (standard, prompt-confounded)
b. lie vs honest-under-same-pressure
c. lie vs hallucination (neutral prompt) — both false, prompt differs
d. lie vs false-without-knowledge under the same prompt — both false, prompt fixed (key)
e. hallucination vs correct, neutral (hallucination detectors' home contrast)
f. honest-under-pressure vs honest-neutral (pure prompt effect)

### Statistics
Shares with Wilson 95% CIs; AUROC with 1,000-sample bootstrap CIs; paired comparisons across
contrasts by bootstrap; recall at the Alpaca 1% FPR threshold. Train/test split by question so no
question appears on both sides.

## Expected Outcomes
Support: probe AUROC high on (a), clearly lower on (b)/(d); hallucination detectors high on (e)
and near chance or inverted on lies. Refute: deception probe keeps high AUROC on (d) and stays
quiet on hallucinations, i.e. it reads belief-statement mismatch.

## Timeline
Setup 20 min; generation + activations 60-90 min; probes/analysis 60 min; MASK 30 min; report 40 min.

## Potential Challenges
- Incentive-only lying may be rare on an 8B model -> report yield; analyses that need lies fall
  back to the instructed condition, flagged as such.
- Under a lie instruction the model also "lies" on unknown questions; those are false statements
  with deceptive intent but without knowledge — reported as a separate cell, not as hallucination.
- Short answers give few response tokens; report both mean-over-response and last-token scores.
- Belief labels are noisy; report a sensitivity analysis with stricter thresholds.

## Success Criteria
A populated knowledge x incentive table with CIs, the honest yield of incentive-only lies, and a
detector x contrast AUROC table with prompt-confound controls, whatever the direction of the result.
