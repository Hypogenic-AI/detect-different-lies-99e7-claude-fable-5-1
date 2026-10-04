# Downloaded Datasets

Data files are NOT committed to git (see `.gitignore`); only this README and `*/samples/` are.
All datasets below are downloaded and loadable locally (total ≈ 260 MB).

| Dir | Source (HF) | Size | Role |
|---|---|---|---|
| `mask/<config>` | `cais/MASK` | 1,000 rows over 6 configs (test only) | Incentive-pressure prompts + belief-elicitation prompts |
| `trivia_qa_nocontext/` | `mandarjoshi/trivia_qa`, config `rc.nocontext` | train 138,384 / val 17,944 / test 17,210 | Matched factual questions; known vs unknown split |
| `popqa/` | `akariasai/PopQA` | 14,267 (test) | Long-tail entity questions with popularity → supply of "does not know" items |
| `liars_bench/<config>` | `Cadenza-Labs/liars-bench` (gated, auto-approve) | 8 configs, ≈79k transcripts | Existing labelled lie/honest transcripts for 70B/24B/27B models |
| `mask_generations/mask` | `Cadenza-Labs/mask-generations` | 1,045 | MASK transcripts with deceptive labels (Liars' Bench format) |
| `anthropic_lies_noddybear/` | `Noddybear/lies` | 106 parquet files, 88 MB | Anthropic on-policy lie datasets (Gemma-3 4b/12b/27b, gpt-oss-120b, gpt-4o) |

## 1. MASK (`cais/MASK`) — user-specified
- Configs (rows): `known_facts` 209, `provided_facts` 274, `continuations` 176, `disinformation` 125, `doubling_down_known_facts` 120, `statistics` 96. Split: `test` only.
- Fields: `system_prompt` + `user_prompt` (pressure), `proposition` (with `<pivotal>a/b</pivotal>`), `ground_truth`, `belief_elicit_1..3` (neutral belief questions; absent in `provided_facts`, only `_1` in `statistics`), `type` (binary/numerical), `formatted_ground_truth`.
- Contains a canary string — do not publish rows in training corpora.
- `provided_facts` supplies the truth in the prompt, so it has no parametric-knowledge check; prefer `known_facts`, `continuations`, `disinformation`, `doubling_down_known_facts`.
```python
from datasets import load_dataset
for c in ["known_facts","provided_facts","continuations","disinformation","doubling_down_known_facts","statistics"]:
    load_dataset("cais/MASK", c).save_to_disk(f"datasets/mask/{c}")
# load: from datasets import load_from_disk; load_from_disk("datasets/mask/known_facts")["test"]
```

## 2. TriviaQA (`mandarjoshi/trivia_qa`, `rc.nocontext`) — user-specified
- Fields: `question`, `question_id`, `question_source`, `answer` (dict with `value`, `aliases`, `normalized_aliases`, …). The heavy `entity_pages`/`search_results` columns were dropped before saving. The full dataset is ~35 GB; only the no-context config (≈64 MB parquet) is needed.
- `test` has no answers; use `validation`/`train`.
```python
d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
d = d.remove_columns(["entity_pages", "search_results"]); d.save_to_disk("datasets/trivia_qa_nocontext")
```

## 3. PopQA (`akariasai/PopQA`)
- Fields: `question`, `possible_answers` (JSON list string), `subj`, `prop`, `obj`, `s_pop`, `o_pop` (Wikipedia page views). Low `s_pop` items are a controllable source of questions the model does not know.
```python
load_dataset("akariasai/PopQA").save_to_disk("datasets/popqa")
```

## 4. Liars' Bench (`Cadenza-Labs/liars-bench`)
- Gated with automatic approval: accept terms on the dataset page (or POST `/datasets/Cadenza-Labs/liars-bench/ask-access` with the HF token), then load with `HF_TOKEN` set.
- Configs (rows): `alpaca` 8,000 test (control, all honest) + train split, `instructed-deception` 26,426, `harm-pressure-choice` 5,400, `harm-pressure-knowledge-report` 7,068, `convincing-game` 888, `gender-secret` 762, `insider-trading` 6,873, `soft-trigger` 24,000.
- Fields: `model`, `messages`, `deceptive` (bool), `meta`, `temperature`, `dataset_index`.
- Models: llama-v3.3-70b-instruct, qwen-2.5-72b-instruct, mistral-small-3.1-24b-instruct, gemma-3-27b-it (+ LoRA variants, kimi-k2.6). Transcripts are on-policy for those models: probing them requires running the same model (Mistral-Small-24B and Gemma-3-27B fit a 48 GB GPU in bf16/4-bit; the 70B models need 4-bit).
```python
from datasets import load_dataset, get_dataset_config_names
for c in get_dataset_config_names("Cadenza-Labs/liars-bench"):
    load_dataset("Cadenza-Labs/liars-bench", c).save_to_disk(f"datasets/liars_bench/{c}")
```

## 5. MASK generations (`Cadenza-Labs/mask-generations`)
- 1,045 rows; same schema as Liars' Bench.
```python
load_dataset("Cadenza-Labs/mask-generations", "mask").save_to_disk("datasets/mask_generations/mask")
```

## 6. Anthropic lie-detector data (`Noddybear/lies`)
- Raw parquet snapshot; one sub-directory per `<model>_<aggregation>_<fold>` with `train`/`validation` files.
- Fields: `trace` (prior messages), `lie_detection_prompt`, `did_lie` (bool), `task`, `model`, `fold`.
- Task groups include `mask-factual`, `mask-roleplay`, `sycophancy`, `self-sycophancy`, `sandbagging`, `unanswerable`, `games`, `offpolicy`, `cot-unfaithfulness`, `ascii`. The dataset card on the Hub lists only the gpt-oss-120b megafolds; the other configs exist as files.
```python
from huggingface_hub import snapshot_download
snapshot_download("Noddybear/lies", repo_type="dataset", local_dir="datasets/anthropic_lies_noddybear")
# load: pd.read_parquet("datasets/anthropic_lies_noddybear/google-gemma-3-12b-it_task-group_mask-factual/train-00000-of-00001.parquet")
```

## Other data shipped inside cloned repos
- True/false statement sets (cities, sp_en_trans, larger_than, common_claim, companies …): `code/geometry-of-truth/datasets/`, `code/Truth_is_Universal/datasets/`.
- Azaria–Mitchell facts for Instructed-Pairs probe training: `code/deception-detection/data/repe/`, `code/deception-detection/data/internal_state/`, `code/liars-bench/data/raw/azaria_mitchell`, `code/Beyond-Liars-Bench/train_data_azaria_mitchell/`.

## Notes
- None of these datasets labels *why* an output is false for a given small model; the
  known/unknown × honest/pressured labels must be generated on-policy by the experiment runner.
- Samples (2–3 rows per config, long fields truncated) are in each dataset's `samples/` directory.
