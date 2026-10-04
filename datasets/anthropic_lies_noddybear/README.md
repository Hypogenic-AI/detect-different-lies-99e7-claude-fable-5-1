
---
language:
- en
license: mit
task_categories:
- text-classification
pretty_name: Lie Detection Dataset
tags:
- lie-detection
- truthfulness
- fine-tuning
- safety
---

# Lie Detection Dataset

This dataset contains lie detection samples for fine-tuning language models to detect when they are lying.

## Dataset Summary

Each sample contains a conversation between a user and an AI model, followed by a lie detection prompt and the ground truth label indicating whether the model lied.

## Available Configurations

| Configuration | Model | Aggregation | Fold | Samples | Description |
|--------------|-------|-------------|------|---------|-------------|
| `openai-gpt-oss-120b_megafolds_fold_1` | openai/gpt/oss/120b | megafolds_fold | 1 | 7860 | Fold containing only '1' category samples |
| `openai-gpt-oss-120b_megafolds_fold_2` | openai/gpt/oss/120b | megafolds_fold | 2 | 10919 | Fold containing only '2' category samples |


## Usage

```python
from datasets import load_dataset

# Load a specific configuration
dataset = load_dataset("Noddybear/lies", "openai-gpt-oss-120b_megafolds_fold_2")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
```

## Dataset Structure
Each sample contains:

- sample_id: Unique identifier
- task: Task name
- category: Task category from aggregation
- did_lie: Whether the model lied (boolean)
- trace: Full conversation history
- model: Model that generated responses

And more fields...

@dataset{lie_detection_2025,
  title={Lie Detection Dataset},
  year={2025},
  publisher={Hugging Face}
}