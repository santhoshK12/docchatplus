import os
import json
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate


# -----------------------
# CONFIG
# -----------------------

@dataclass
class ModelConfig:
    name: str
    pretrained_id: str


MODEL_CONFIGS = [
    ModelConfig(name="bert-base-uncased", pretrained_id="bert-base-uncased"),
    ModelConfig(name="minilm", pretrained_id="microsoft/MiniLM-L12-H384-uncased"),
    ModelConfig(name="mpnet-base", pretrained_id="microsoft/mpnet-base"),
]

OUTPUT_DIR = "models"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "results.json")
MAX_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 1  # use 1 first to test environment


# -----------------------
# DATA PREP
# -----------------------

def load_squad():
    print("📥 Loading SQuAD dataset...")
    return load_dataset("squad")


def prepare_train_features(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            start_positions.append(tokenizer.cls_token_id)
            end_positions.append(tokenizer.cls_token_id)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        # find context token range
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if not (offsets[context_start][0] <= start_char <= offsets[context_end][1]):
            start_positions.append(tokenizer.cls_token_id)
            end_positions.append(tokenizer.cls_token_id)
        else:
            start_token = context_start
            while start_token <= context_end and offsets[start_token][0] <= start_char:
                start_token += 1
            start_positions.append(start_token - 1)

            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= end_char:
                end_token -= 1
            end_positions.append(end_token + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_validation_features(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    offset_mapping = tokenized["offset_mapping"]

    for i in range(len(offset_mapping)):
        context_id = 1
        sample_idx = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_idx])

        new_offsets = []
        seq_ids = tokenized.sequence_ids(i)
        for k, off in enumerate(offset_mapping[i]):
            if seq_ids[k] == context_id:
                new_offsets.append(off)
            else:
                new_offsets.append((0, 0))
        offset_mapping[i] = new_offsets

    tokenized["offset_mapping"] = offset_mapping
    return tokenized


# -----------------------
# METRICS
# -----------------------

metric = evaluate.load("squad")


def compute_squad(preds, refs):
    formatted = [{"id": k, "prediction_text": v} for k, v in preds.items()]
    ref = [{"id": ex["id"], "answers": ex["answers"]} for ex in refs]
    return metric.compute(predictions=formatted, references=ref)


def postprocess_predictions(examples, features, raw_preds, tokenizer):
    start_logits, end_logits = raw_preds
    predictions = {}

    example_to_features = {}
    for i, f in enumerate(features):
        example_to_features.setdefault(f["example_id"], []).append(i)

    for example_id, feat_idxs in example_to_features.items():
        context = None
        for idx, ex in enumerate(examples):
            if ex["id"] == example_id:
                context = ex["context"]
                break

        best_score = None
        best_text = ""

        for idx in feat_idxs:
            s_logits = start_logits[idx]
            e_logits = end_logits[idx]
            offsets = features[idx]["offset_mapping"]

            start = int(s_logits.argmax())
            end = int(e_logits.argmax())
            if end < start:
                end = start

            start_char, _ = offsets[start]
            _, end_char = offsets[end]
            text = context[start_char:end_char]

            score = float(s_logits[start] + e_logits[end])

            if best_score is None or score > best_score:
                best_score = score
                best_text = text

        predictions[example_id] = best_text

    return predictions


# -----------------------
# TRAINING LOOP
# -----------------------

def train_and_eval(cfg, squad):
    print("\n==============================")
    print("🔥 Training:", cfg.name)
    print("==============================")

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_id)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.pretrained_id)

    train_ds = squad["train"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=squad["train"].column_names
    )

    val_examples = squad["validation"]
    val_features = val_examples.map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=val_examples.column_names
    )

    out_dir = os.path.join(OUTPUT_DIR, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    raw_preds = trainer.predict(val_features.remove_columns(["example_id", "offset_mapping"])).predictions
    preds = postprocess_predictions(val_examples, val_features, raw_preds, tokenizer)

    metrics = compute_squad(preds, val_examples)
    print("✅", cfg.name, "metrics:", metrics)
    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    squad = load_squad()

    results = {}
    best_model = None
    best_f1 = -1

    for cfg in MODEL_CONFIGS:
        metrics = train_and_eval(cfg, squad)
        results[cfg.name] = metrics

        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_model = cfg.name

    with open(RESULTS_JSON, "w") as f:
        json.dump({"best_model": best_model, "results": results}, f, indent=2)

    print("\n🏁 DONE! Best model =", best_model)


if __name__ == "__main__":
    main()
