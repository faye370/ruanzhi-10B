"""
组长负责: Semantically-Constrained BERT-Attack (SC-BERT-Attack).

Improvement over BERT-Attack (Li et al., EMNLP 2020):

BERT-Attack generates candidates via MLM but applies no semantic constraint on
the substituted word itself — a word like "good" can be replaced by "bad" if it
fools the classifier, producing adversarial text that is unnatural or
meaning-reversed and easily spotted by humans.

Our improvement: after MLM candidate generation, filter each candidate by
cosine similarity of BERT word embeddings to the original word (threshold >= 0.5).
Only semantically similar candidates are attempted, making substitutions more
natural while still achieving high attack success.

Control group  : BERT-Attack  — WIR + BERT-MLM, no semantic filter
Treatment group: SC-BERT-Attack — WIR + BERT-MLM + embedding similarity filter

Expected improvements in treatment vs control:
  - Higher Semantic Similarity (adversarial text closer to original meaning)
  - Lower / comparable Perturbation Rate
  - ASR may decrease slightly (more constrained search space)

Run:
    python attack/improved_attack.py

Outputs (in ./results/improved/):
    improved_results.json  -- ASR, avg queries, perturbation rate, sem_sim
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import DATASET, PRETRAINED_MODEL_DIR, RESULTS_DIR


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def compute_semantic_similarity(orig_texts, pert_texts):
    if not orig_texts:
        return float("nan")
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        orig_embs = model.encode(orig_texts, convert_to_tensor=True, show_progress_bar=False)
        pert_embs = model.encode(pert_texts, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(orig_embs, pert_embs).diagonal()
        return round(float(sims.mean()), 4)
    except ImportError:
        print("[WARNING] sentence-transformers not installed.")
        return float("nan")


# ---------------------------------------------------------------------------
# Core model utilities
# ---------------------------------------------------------------------------

def predict(model, tokenizer, text, device):
    preds, confs = predict_batch(model, tokenizer, [text], device)
    return preds[0], confs[0]


def predict_batch(model, tokenizer, texts, device):
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=256, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1).tolist()
    confs = probs[range(len(texts)), preds].tolist()
    return preds, confs


def get_mlm_candidates(mlm_model, mlm_tokenizer, words, word_idx, device, top_k=50):
    """Generate substitution candidates using BERT MLM."""
    masked = words.copy()
    masked[word_idx] = mlm_tokenizer.mask_token
    masked_text = " ".join(masked)

    enc = mlm_tokenizer(
        masked_text, return_tensors="pt", truncation=True, max_length=256
    ).to(device)

    mask_positions = (enc.input_ids == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_positions) == 0:
        return []

    with torch.no_grad():
        logits = mlm_model(**enc).logits
    pred_logits = logits[0, mask_positions[0]]
    top_ids = pred_logits.topk(top_k).indices.tolist()

    original = words[word_idx].lower()
    candidates = []
    for tid in top_ids:
        token = mlm_tokenizer.decode([tid]).strip()
        if token.lower() != original and token.isalpha() and "##" not in token:
            candidates.append(token)
        if len(candidates) >= 10:
            break
    return candidates


def filter_by_embedding_similarity(mlm_model, mlm_tokenizer, original_word,
                                   candidates, device, threshold=0.5):
    """Keep only candidates whose BERT word-embedding cosine similarity
    to the original word exceeds `threshold`."""
    if not candidates:
        return candidates

    emb_layer = mlm_model.bert.embeddings.word_embeddings

    def word_emb(word):
        ids = mlm_tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        t = torch.tensor([ids[0]], device=device)
        with torch.no_grad():
            return emb_layer(t)

    orig_emb = word_emb(original_word)
    if orig_emb is None:
        return candidates

    filtered = []
    for cand in candidates:
        cand_emb = word_emb(cand)
        if cand_emb is None:
            continue
        sim = torch.cosine_similarity(orig_emb, cand_emb, dim=-1).item()
        if sim >= threshold:
            filtered.append(cand)

    return filtered if filtered else candidates[:3]


# ---------------------------------------------------------------------------
# Attack core
# ---------------------------------------------------------------------------

def sc_bert_attack(model, tokenizer, text, true_label, device,
                   mlm_model, mlm_tokenizer, use_sem_filter=True):
    """Attack a single example.

    Both groups use WIR ranking + BERT-MLM candidates.
    Treatment group additionally applies embedding similarity filter.
    """
    words = text.split()
    n_words = len(words)
    if n_words < 3:
        return None, 0, 0

    original_pred, original_conf = predict(model, tokenizer, text, device)
    if original_pred != true_label:
        return None, 0, 0

    # Step 1: WIR via batch word deletion
    deletion_texts = [" ".join(words[:i] + words[i + 1:]) for i in range(n_words)]
    confs = []
    _MINI_BATCH = 32
    for _b in range(0, len(deletion_texts), _MINI_BATCH):
        _, _bc = predict_batch(model, tokenizer, deletion_texts[_b:_b + _MINI_BATCH], device)
        confs.extend(_bc)
    queries = 1 + n_words
    importance_array = np.array([original_conf - c for c in confs])

    sorted_indices = np.argsort(importance_array)[::-1]

    # Step 2: greedy substitution
    current_words = words.copy()
    words_changed = 0

    for idx in sorted_indices:
        original_word = current_words[idx]
        synonyms = get_mlm_candidates(mlm_model, mlm_tokenizer, current_words, idx, device)

        if use_sem_filter:
            synonyms = filter_by_embedding_similarity(
                mlm_model, mlm_tokenizer, original_word, synonyms, device, threshold=0.5
            )

        if not synonyms:
            continue

        candidates_texts = []
        for synonym in synonyms:
            candidate = current_words.copy()
            candidate[idx] = synonym
            candidates_texts.append(" ".join(candidate))

        preds, _ = predict_batch(model, tokenizer, candidates_texts, device)
        queries += len(synonyms)

        for j, pred in enumerate(preds):
            if pred != true_label:
                current_words[idx] = synonyms[j]
                words_changed += 1
                return " ".join(current_words), queries, words_changed

        current_words[idx] = original_word

    return None, queries, 0


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(model, tokenizer, dataset, num_examples, device, label,
                   mlm_model, mlm_tokenizer, use_sem_filter):
    results = {"success": 0, "total": 0, "queries": [], "perturb_rates": []}
    orig_texts = []
    pert_texts = []

    for example in tqdm(dataset.select(range(num_examples)), desc=label):
        text = example["text"][:800]
        true_label = example["label"]

        adv, queries, n_changed = sc_bert_attack(
            model, tokenizer, text, true_label, device,
            mlm_model, mlm_tokenizer, use_sem_filter=use_sem_filter,
        )
        results["total"] += 1
        results["queries"].append(queries)

        if adv is not None:
            results["success"] += 1
            n_words = len(text.split())
            results["perturb_rates"].append(n_changed / n_words if n_words > 0 else 0)
            orig_texts.append(text)
            pert_texts.append(adv)

    results["asr"] = results["success"] / results["total"] * 100
    results["avg_queries"] = float(np.mean(results["queries"]))
    results["avg_perturb_rate"] = float(np.mean(results["perturb_rates"])) if results["perturb_rates"] else 0.0
    results["sem_sim"] = compute_semantic_similarity(orig_texts, pert_texts)
    return results


def main(args):
    out_dir = os.path.join(args.results_dir, "improved")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    print("Loading BERT-MLM (bert-base-uncased) for candidate generation...")
    mlm_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased")  # keep on CPU to avoid OOM
    mlm_model.eval()

    dataset = load_dataset(args.dataset, "plain_text", split="test")

    all_results = {}

    # Control: BERT-Attack (WIR + BERT-MLM, no semantic filter)
    print("\n[1/2] BERT-Attack baseline (WIR + BERT-MLM, no filter)...")
    all_results["WIR_baseline"] = run_experiment(
        model, tokenizer, dataset, args.num_examples, device,
        label="BERT-Attack (control)",
        mlm_model=mlm_model, mlm_tokenizer=mlm_tokenizer, use_sem_filter=False,
    )

    # Treatment: SC-BERT-Attack (WIR + BERT-MLM + embedding similarity filter)
    print("\n[2/2] SC-BERT-Attack (WIR + BERT-MLM + sem-filter, our method)...")
    all_results["AWIR_improved"] = run_experiment(
        model, tokenizer, dataset, args.num_examples, device,
        label="SC-BERT-Attack (ours)",
        mlm_model=mlm_model, mlm_tokenizer=mlm_tokenizer, use_sem_filter=True,
    )

    print("\n" + "=" * 80)
    print("IMPROVEMENT COMPARISON")
    print("=" * 80)
    print(f"{'Method':<20} {'ASR':>8} {'Avg Queries':>13} {'Perturb Rate':>14} {'Sem. Similarity':>17}")
    print("-" * 80)
    for name, r in all_results.items():
        print(
            f"{name:<20} {r['asr']:>7.1f}% "
            f"{r['avg_queries']:>13.1f} "
            f"{r['avg_perturb_rate']:>13.1%} "
            f"{r['sem_sim']:>17.4f}"
        )

    output_path = os.path.join(out_dir, "improved_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=PRETRAINED_MODEL_DIR)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()
    main(args)
