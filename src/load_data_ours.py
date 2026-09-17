"""Load WikiText-103 with an optional training-sample size limit.

Adapted from src/load_data.py (kept untouched — see FORK_NOTES.md and
README_OURS.md for why this is a separate file rather than an edit to the
original). Two differences from the original loader:

1. Uses the namespaced dataset id "Salesforce/wikitext" instead of the bare
   "wikitext" id, which is required under current `datasets`/
   `huggingface_hub` versions (see FORK_NOTES.md, 2026-08-05 entry).
2. Loads the "wikitext-103-raw-v1" config into ./data/wikitext103, and
   supports truncating the training split to `dataset.max_train_samples`.
"""
import os
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.utils import decode, get_context, get_text_to_classify, group_texts_and_tokenize_data
from utils.detector import Detector


def setup_directories(path):
    os.makedirs(path, exist_ok=True)


def load_raw_data():
    """Load WikiText-103 dataset and filter out empty lines."""
    print("Loading WikiText-103 dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    dataset["train"] = dataset["train"].filter(lambda x: len(x["text"]) > 0)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["text"]) > 0)
    dataset["validation"] = dataset["validation"].filter(lambda x: len(x["text"]) > 0)

    return dataset["train"], dataset["validation"], dataset["test"]


def truncate_to_max_samples(raw_data, max_train_samples):
    """Truncate a raw (pre-tokenization) split to at most max_train_samples rows."""
    if max_train_samples is None:
        return raw_data
    n = min(int(max_train_samples), len(raw_data))
    return raw_data.select(range(n))


def process_dataset(raw_data, tokenizer, block_size=512, input_token_length=256, train=True):
    """Process and tokenize the dataset (identical logic to load_data.py)."""
    print("Processing and tokenizing dataset...")

    processed_dataset = group_texts_and_tokenize_data(
        raw_datasets=raw_data,
        tokenizer=tokenizer,
        column_names=list(raw_data.features),
        block_size=block_size,
        preprocessing_num_workers=4,
        overwrite_cache=True,
    )

    processed_dataset = processed_dataset.map(
        lambda example: decode(example, tokenizer, 'text', 'input_ids'),
        batched=False,
        desc="Decoding full text",
    )

    if train:
        processed_dataset = processed_dataset.map(
            lambda example: get_context(example, input_token_length, input_ids_key="context_input_ids", attention_mask_key="context_attention_mask"),
            batched=False,
            desc="Getting context"
        )

        processed_dataset = processed_dataset.map(
            lambda example: get_text_to_classify(example, input_token_length),
            batched=False,
            desc="Truncating texts to last {input_token_length} tokens"
        )

        processed_dataset = processed_dataset.map(
            lambda example: decode(example, tokenizer, 'context', 'context_input_ids'),
            batched=False,
            desc="Decoding context",
        )

        processed_dataset = processed_dataset.map(
            lambda example: decode(example, tokenizer, 'cls_text', 'cls_input_ids'),
            batched=False,
            desc="Decoding classification text"
        )

    return processed_dataset


def classify_dataset(dataset, detector, tokenizer, block_size=512, input_token_length=256, cfg=None):
    print("Classifying dataset...")
    return dataset.map(
        lambda example: detector.predict_batch(
            example,
            cls_text_key="cls_text",
            max_length=(block_size - input_token_length),
            threshold=cfg.detector.ai_confidence_threshold,
            temperature=cfg.detector.temperature
        ),
        batched=True,
        desc="Classifying texts"
    )


def save_dataset(classified_dataset, output_path):
    print(f"Saving classified dataset to {output_path}...")
    import json
    output = []
    for row in classified_dataset:
        output_row = {'text': row['text']}
        for key in ('context', 'cls_text', 'cls_score', 'cls_confidence'):
            if key in row:
                output_row[key] = row[key]
        output.append(output_row)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)


@hydra.main(config_path="../config", config_name="ours", version_base=None)
def main(cfg: DictConfig):
    block_size = cfg.train.block_size
    input_token_length = block_size - cfg.train.loss_on_last_n_tokens
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    setup_directories(cfg.dataset.path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    print("Loading detector...")
    detector = Detector(
        tokenizer_name=cfg.detector.tokenizer_name,
        detector_path=cfg.detector.model_path,
        device=device
    )

    max_train_samples = cfg.dataset.get("max_train_samples", None)
    if max_train_samples == "None":
        max_train_samples = None

    try:
        train_data, validation_data, test_data = load_raw_data()
        train_data = truncate_to_max_samples(train_data, max_train_samples)

        processed_train = process_dataset(train_data, tokenizer, block_size, input_token_length)
        processed_validation = process_dataset(validation_data, tokenizer, block_size, input_token_length, train=False)
        processed_test = process_dataset(test_data, tokenizer, block_size, input_token_length, train=False)

        classified_train = classify_dataset(processed_train, detector, tokenizer, block_size, input_token_length, cfg)

        save_dataset(classified_train, f"{cfg.dataset.path}/train.json")
        save_dataset(processed_validation, f"{cfg.dataset.path}/validation.json")
        save_dataset(processed_test, f"{cfg.dataset.path}/test.json")

        print("WikiText-103 dataset preparation completed successfully!")

    finally:
        print("Cleaning up...")
        del detector
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
