"""
成员3负责: Adversarial training for robustness improvement.

Strategy:
    1. Load the clean fine-tuned model (from Member 1).
    2. Generate adversarial examples from the training set using TextFooler.
    3. Mix adversarial examples with clean examples (50/50 ratio).
    4. Fine-tune the model on the mixed dataset.
    5. Evaluate: compare clean accuracy vs. robust accuracy (accuracy under attack).

Depends on: train/finetune_bert.py must finish first.

Run:
    python defense/adversarial_training.py

Outputs:
    ./checkpoints/bert-imdb-adv/    -- robustly trained model
    ./results/defense/defense_summary.txt

Key metric to report:
    Clean model:    clean_acc=~93%, robust_acc=~10-20% (after TextFooler attack)
    Robust model:   clean_acc=~89%, robust_acc=~50-60% (after same attack)
    => robustness improves substantially at small cost to clean accuracy
"""
import argparse
import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # TF runs on CPU, PyTorch keeps GPU
except Exception:
    pass

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    CHECKPOINT_DIR, DATASET, LEARNING_RATE, MAX_LENGTH,
    PRETRAINED_MODEL_DIR, RESULTS_DIR, TRAIN_BATCH_SIZE,
)

ADV_CHECKPOINT_DIR = CHECKPOINT_DIR + "-adv"


# ---------------------------------------------------------------------------
# Step 1: Generate adversarial examples using TextFooler
# ---------------------------------------------------------------------------

def generate_adversarial_examples(model_dir, dataset_name, num_adv_examples):
    """Run TextFooler on training set to get adversarial texts."""
    print(f"Generating {num_adv_examples} adversarial examples from training split...")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Use train split for adversarial training (not test — that's for evaluation)
    dataset = HuggingFaceDataset(dataset_name, "plain_text", split="train")

    attack = TextFoolerJin2019.build(model_wrapper)
    tmp_csv = os.path.join(RESULTS_DIR, "defense", "adv_train_examples.csv")
    os.makedirs(os.path.dirname(tmp_csv), exist_ok=True)

    attack_args = AttackArgs(
        num_examples=num_adv_examples,
        log_to_csv=tmp_csv,
        csv_coloring_style="plain",
        disable_stdout=True,
    )
    attacker = Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

    # Parse the CSV to extract (adversarial_text, original_label) pairs
    import pandas as pd
    df = pd.read_csv(tmp_csv)
    successful = df[df["result_type"] == "Successful"]
    adv_texts = successful["perturbed_text"].tolist()
    # original_label = original ground truth (the class that was successfully fooled)
    adv_labels = successful["ground_truth_output"].tolist()

    print(f"  Generated {len(adv_texts)} successful adversarial examples.")
    return adv_texts, adv_labels


# ---------------------------------------------------------------------------
# Step 2: Build mixed dataset and retrain
# ---------------------------------------------------------------------------

def tokenize_texts(texts, labels, tokenizer):
    encodings = dict(tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding="max_length"
    ))
    encodings["label"] = labels
    return Dataset.from_dict(encodings)


def main(args):
    out_dir = os.path.join(args.results_dir, "defense")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.adv_save_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)

    # --- Generate adversarial examples (or load existing) ---
    tmp_csv = os.path.join(args.results_dir, "defense", "adv_train_examples.csv")
    if args.skip_generation and os.path.exists(tmp_csv):
        print(f"Skipping generation, loading existing CSV: {tmp_csv}")
        import pandas as pd
        df = pd.read_csv(tmp_csv)
        successful = df[df["result_type"] == "Successful"]
        adv_texts = successful["perturbed_text"].tolist()
        adv_labels = successful["ground_truth_output"].tolist()
        print(f"  Loaded {len(adv_texts)} adversarial examples.")
    else:
        adv_texts, adv_labels = generate_adversarial_examples(
            args.model_dir, args.dataset, args.num_adv_examples
        )

    # --- Build mixed training dataset ---
    print("Building mixed (clean + adversarial) training dataset...")
    raw = load_dataset(args.dataset, "plain_text")
    clean_subset = raw["train"].select(range(args.num_adv_examples))  # same size as adv

    clean_ds = tokenize_texts(
        clean_subset["text"], clean_subset["label"], tokenizer
    )
    adv_ds = tokenize_texts(adv_texts, [int(l) for l in adv_labels], tokenizer)

    mixed_ds = concatenate_datasets([clean_ds, adv_ds]).shuffle(seed=42)
    eval_ds_raw = raw["test"].select(range(500))
    eval_ds = tokenize_texts(eval_ds_raw["text"], eval_ds_raw["label"], tokenizer)

    print(f"  Mixed dataset size: {len(mixed_ds)} (clean: {len(clean_ds)}, adv: {len(adv_ds)})")

    # --- Fine-tune on mixed dataset ---
    print("Fine-tuning on mixed dataset for robustness...")
    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=2)

    import numpy as np
    from sklearn.metrics import accuracy_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    training_args = TrainingArguments(
        output_dir=args.adv_save_dir,
        num_train_epochs=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mixed_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.adv_save_dir)
    tokenizer.save_pretrained(args.adv_save_dir)

    clean_results = trainer.evaluate()
    print(f"\nRobustly trained model clean accuracy: {clean_results['eval_accuracy']:.4f}")
    print(f"Saved to {args.adv_save_dir}")
    print("\nNext step: run baseline_attack.py on this model to measure robust accuracy.")

    summary = (
        f"Clean model dir: {args.model_dir}\n"
        f"Robust model dir: {args.adv_save_dir}\n"
        f"Adversarial examples used in training: {len(adv_texts)}\n"
        f"Robust model clean accuracy: {clean_results['eval_accuracy']:.4f}\n"
        "To get robust accuracy: run attack/baseline_attack.py --model_dir "
        f"{args.adv_save_dir}\n"
    )
    with open(os.path.join(out_dir, "defense_summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=PRETRAINED_MODEL_DIR)
    parser.add_argument("--adv_save_dir", default=ADV_CHECKPOINT_DIR)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--num_adv_examples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip adversarial example generation and load existing CSV.")
    args = parser.parse_args()
    main(args)
