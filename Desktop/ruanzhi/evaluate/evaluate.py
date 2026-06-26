"""
成员3负责: Unified evaluation — compute and print the final comparison table.

This script reads results from all three attack experiments and produces
a single table for the report. Run after all attacks are done.

Run:
    python evaluate/evaluate.py

Inputs (must exist first):
    ./results/baseline/textfooler_results.csv        (成员1产出)
    ./results/baseline/bertattack_results.csv         (成员2产出)
    ./results/improved/improved_results.json          (成员4产出)
    ./results/defense/baseline/textfooler_results.csv (成员5产出, optional)
    ./results/defense/baseline/bertattack_results.csv (成员5产出, optional)
    ./results/defense/improved/improved_results.json  (成员5产出, optional)

Output:
    ./results/final_comparison.csv   -- main comparison table for your report
    ./results/defense_comparison.csv -- defense table (generated when member 5 results exist)
"""
import json
import os
import sys

import pandas as pd


def _compute_sem_sim(csv_path):
    """Compute average cosine similarity between original and perturbed texts
    for successful attacks (sentence-transformers, all-MiniLM-L6-v2)."""
    try:
        from sentence_transformers import SentenceTransformer, util
        df = pd.read_csv(csv_path)
        successful = df[df["result_type"] == "Successful"]
        if len(successful) == 0:
            return float("nan")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        orig_embs = model.encode(
            successful["original_text"].tolist(),
            convert_to_tensor=True, show_progress_bar=False
        )
        pert_embs = model.encode(
            successful["perturbed_text"].tolist(),
            convert_to_tensor=True, show_progress_bar=False
        )
        sims = util.cos_sim(orig_embs, pert_embs).diagonal()
        return round(float(sims.mean()), 4)
    except ImportError:
        return float("nan")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import RESULTS_DIR


def parse_textattack_csv(csv_path, method_name):
    """Parse a TextAttack result CSV and return a metrics dict."""
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found. Run baseline_attack.py first.")
        return None

    df = pd.read_csv(csv_path)
    total = len(df)
    successful = df[df["result_type"] == "Successful"]
    failed = df[df["result_type"] == "Failed"]
    skipped = df[df["result_type"] == "Skipped"]

    asr = len(successful) / (total - len(skipped)) * 100 if (total - len(skipped)) > 0 else 0

    avg_words_changed = (
        successful["num_words_changed"].mean()
        if "num_words_changed" in successful.columns and len(successful) > 0
        else float("nan")
    )
    avg_queries = (
        df["num_queries"].mean()
        if "num_queries" in df.columns
        else float("nan")
    )

    sem_sim = _compute_sem_sim(csv_path)

    return {
        "Method": method_name,
        "Total": total,
        "Successful": len(successful),
        "Failed": len(failed),
        "Skipped": len(skipped),
        "ASR (%)": round(asr, 1),
        "Avg Words Changed": round(avg_words_changed, 2),
        "Avg Queries": round(avg_queries, 1),
        "Sem. Similarity": sem_sim,
    }


def _awir_defense_row(json_path, model_label):
    """Return a single defense row dict for the AWIR_improved entry in a json result file."""
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    r = data.get("AWIR_improved")
    if r is None:
        return None
    return {
        "Method": f"{model_label} + SC-BERT-Attack",
        "Total": r["total"],
        "Successful": r["success"],
        "Failed": r["total"] - r["success"],
        "Skipped": 0,
        "ASR (%)": round(r["asr"], 1),
        "Avg Words Changed": round(r.get("avg_perturb_rate", float("nan")), 3),
        "Avg Queries": round(r["avg_queries"], 1),
        "Sem. Similarity": r.get("sem_sim", float("nan")),
    }


def parse_improved_json(json_path):
    """Parse results from improved_attack.py and return rows for AWIR and WIR baseline."""
    if not os.path.exists(json_path):
        print(f"  [SKIP] {json_path} not found. Run improved_attack.py first.")
        return []

    with open(json_path) as f:
        data = json.load(f)

    rows = []
    name_map = {
        "WIR_baseline": "BERT-Attack (control, ours)",
        "AWIR_improved": "SC-BERT-Attack (ours)",
    }
    for key, r in data.items():
        rows.append({
            "Method": name_map.get(key, key),
            "Total": r["total"],
            "Successful": r["success"],
            "Failed": r["total"] - r["success"],
            "Skipped": 0,
            "ASR (%)": round(r["asr"], 1),
            "Avg Words Changed": round(r.get("avg_perturb_rate", float("nan")), 3),
            "Avg Queries": round(r["avg_queries"], 1),
            "Sem. Similarity": r.get("sem_sim", float("nan")),
        })
    return rows


def main():
    print("="*65)
    print("FINAL COMPARISON TABLE")
    print("="*65)

    rows = []

    # Baseline results
    for name, fname in [
        ("TextFooler (Jin 2020)", "baseline/textfooler_results.csv"),
        ("BERT-Attack (Li 2020)", "baseline/bertattack_results.csv"),
    ]:
        r = parse_textattack_csv(os.path.join(RESULTS_DIR, fname), name)
        if r:
            rows.append(r)

    # Improved attack results
    improved_path = os.path.join(RESULTS_DIR, "improved", "improved_results.json")
    rows.extend(parse_improved_json(improved_path))

    if not rows:
        print("No results found. Please run the attack scripts first.")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    output_path = os.path.join(RESULTS_DIR, "final_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nTable saved to {output_path}")

    # Also compute defense comparison if robust model results exist
    defense_rows = []
    for attack_name, fname in [
        ("TextFooler", "textfooler_results.csv"),
        ("BERT-Attack", "bertattack_results.csv"),
    ]:
        clean_path = os.path.join(RESULTS_DIR, "baseline", fname)
        robust_path = os.path.join(RESULTS_DIR, "defense", "baseline", fname)
        clean_r = parse_textattack_csv(clean_path, f"Clean BERT + {attack_name}")
        robust_r = parse_textattack_csv(robust_path, f"Robust BERT + {attack_name}")
        if clean_r:
            defense_rows.append(clean_r)
        if robust_r:
            defense_rows.append(robust_r)

    # AWIR defense rows (成员5产出: results/defense/improved/improved_results.json)
    clean_awir = _awir_defense_row(
        os.path.join(RESULTS_DIR, "improved", "improved_results.json"),
        "Clean BERT"
    )
    robust_awir = _awir_defense_row(
        os.path.join(RESULTS_DIR, "defense", "improved", "improved_results.json"),
        "Robust BERT"
    )
    if clean_awir:
        defense_rows.append(clean_awir)
    if robust_awir:
        defense_rows.append(robust_awir)

    if defense_rows:
        print("\n" + "="*65)
        print("DEFENSE COMPARISON (Clean Model vs Robust Model)")
        print("="*65)
        defense_df = pd.DataFrame(defense_rows)
        print(defense_df[["Method", "ASR (%)", "Avg Queries"]].to_string(index=False))
        defense_path = os.path.join(RESULTS_DIR, "defense_comparison.csv")
        defense_df.to_csv(defense_path, index=False)
        print(f"\nDefense table saved to {defense_path}")


if __name__ == "__main__":
    main()
