"""
基线攻击复现: TextFooler 和 BERT-Attack（均为黑盒攻击）

Run:
    python attack/baseline_attack.py --attack textfooler
    python attack/baseline_attack.py --attack bertattack
    python attack/baseline_attack.py  # 两个都跑

Outputs (in ./results/baseline/):
    textfooler_results.csv   -- Jin et al. AAAI 2020   (black-box)
    bertattack_results.csv   -- Li et al. EMNLP 2020   (black-box)
    baseline_summary.txt     -- printed comparison table

Paper targets to reproduce (IMDB, BERT victim):
    TextFooler: Attack Success Rate ~87%, Avg perturbed words ~6
    BERT-Attack: Attack Success Rate ~90%+
"""
import argparse
import glob
import os
import pickle
import signal
import sys
import threading

# Force TensorFlow (used by USE constraint) to run on CPU so it does not
# compete with PyTorch for GPU memory.  USE is only used for semantic
# filtering (threshold 0.2) so CPU is fast enough.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # PyTorch sees GPU 0
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"     # TF allocates GPU memory on demand

import textattack
import torch
import tqdm

from textattack import AttackArgs, Attacker
from textattack.attack_recipes import BERTAttackLi2020, TextFoolerJin2019
from textattack.attack_results import FailedAttackResult, MaximizedAttackResult, SkippedAttackResult, SuccessfulAttackResult
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.config import ATTACK_NUM_EXAMPLES, DATASET, PRETRAINED_MODEL_DIR, RESULTS_DIR

def configure_bertattack_low_resource(attack, max_candidates=48, keep_use=False):
    """Set BERT-Attack params. USE is removed by default (TF+PyTorch GPU deadlock on Windows).
    Pass keep_use=True when running in bertattack_tf env where TF is configured CPU-only."""
    attack.transformation.max_candidates = max_candidates
    if not keep_use:
        attack.constraints = [
            c for c in attack.constraints
            if type(c).__name__ != "UniversalSentenceEncoder"
        ]
    for idx, constraint in enumerate(attack.constraints):
        if isinstance(constraint, MaxWordsPerturbed):
            attack.constraints[idx] = MaxWordsPerturbed(max_percent=0.4)
    return attack


class AttackTimeoutError(Exception):
    pass


def _handle_attack_timeout(signum, frame):
    raise AttackTimeoutError


class TimeoutAttacker(Attacker):
    def __init__(self, *args, per_example_timeout=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_example_timeout = per_example_timeout

    def _attack(self):
        """Run attacks with a per-example wall-clock timeout."""
        if torch.cuda.is_available():
            self.attack.cuda_()

        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_candidates = self._checkpoint.worklist_candidates
        else:
            if self.attack_args.num_successful_examples:
                num_remaining_attacks = self.attack_args.num_successful_examples
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_successful_examples,
                    self.attack_args.shuffle,
                )
            else:
                num_remaining_attacks = self.attack_args.num_examples
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_examples,
                    self.attack_args.shuffle,
                )

        if not self.attack_args.silent:
            print(self.attack, "\n")

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0, dynamic_ncols=True)
        num_results = self._checkpoint.results_count if self._checkpoint else 0
        num_failures = self._checkpoint.num_failed_attacks if self._checkpoint else 0
        num_skipped = self._checkpoint.num_skipped_attacks if self._checkpoint else 0
        num_successes = self._checkpoint.num_successful_attacks if self._checkpoint else 0

        sample_exhaustion_warned = False
        _has_sigalrm = hasattr(signal, "SIGALRM")
        old_handler = signal.getsignal(signal.SIGALRM) if _has_sigalrm else None
        while worklist:
            idx = worklist.popleft()
            try:
                example, ground_truth_output = self.dataset[idx]
            except IndexError:
                continue
            example = textattack.shared.AttackedText(example)
            if self.dataset.label_names is not None:
                example.attack_attrs["label_names"] = self.dataset.label_names

            try:
                if self.per_example_timeout and _has_sigalrm:
                    signal.signal(signal.SIGALRM, _handle_attack_timeout)
                    signal.alarm(int(self.per_example_timeout))
                    try:
                        result = self.attack.attack(example, ground_truth_output)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                else:
                    # Windows has no SIGALRM; rely on query_budget to cap per-example cost.
                    result = self.attack.attack(example, ground_truth_output)
            except AttackTimeoutError:
                print(
                    f"\n[WARNING] Attack timed out for dataset index {idx} "
                    f"after {self.per_example_timeout}s; marking as failed and continuing.",
                    flush=True,
                )
                original_result, _ = self.attack.goal_function.init_attack_example(
                    example, ground_truth_output
                )
                result = FailedAttackResult(original_result)
            except Exception as e:
                raise e

            if (
                isinstance(result, SkippedAttackResult) and self.attack_args.attack_n
            ) or (
                not isinstance(result, SuccessfulAttackResult)
                and self.attack_args.num_successful_examples
            ):
                if worklist_candidates:
                    next_sample = worklist_candidates.popleft()
                    worklist.append(next_sample)
                else:
                    if not sample_exhaustion_warned:
                        print("[WARNING] Ran out of samples to attack!", flush=True)
                        sample_exhaustion_warned = True
            else:
                pbar.update(1)

            self.attack_log_manager.log_result(result)
            if not self.attack_args.disable_stdout and not self.attack_args.silent:
                print("\n")
            num_results += 1

            if isinstance(result, SkippedAttackResult):
                num_skipped += 1
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                num_successes += 1
            if isinstance(result, FailedAttackResult):
                num_failures += 1
            pbar.set_description(
                f"[Succeeded / Failed / Skipped / Total] "
                f"{num_successes} / {num_failures} / {num_skipped} / {num_results}"
            )

            if (
                self.attack_args.checkpoint_interval
                and len(self.attack_log_manager.results)
                % self.attack_args.checkpoint_interval
                == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    self.attack_args,
                    self.attack_log_manager,
                    worklist,
                    worklist_candidates,
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        print()
        if not self.attack_args.silent and self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()

        if self.attack_args.enable_advance_metrics:
            self.attack_log_manager.enable_advance_metrics = True

        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()


def run_single_attack(
    recipe_class,
    model_wrapper,
    dataset,
    num_examples,
    output_csv,
    per_example_timeout=None,
    num_examples_offset=0,
    checkpoint_interval=1,
    args_max_candidates=8,
    args_keep_use=False,
    args_query_budget=None,
):
    if recipe_class is BERTAttackLi2020 and not args_keep_use:
        # Patch USE.__init__ to no-op before build() so TFHub is never loaded.
        # We remove USE from constraints in configure_bertattack_low_resource anyway.
        from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
        _orig_use_init = UniversalSentenceEncoder.__init__
        UniversalSentenceEncoder.__init__ = lambda self, *a, **kw: None
        attack = recipe_class.build(model_wrapper)
        UniversalSentenceEncoder.__init__ = _orig_use_init
    else:
        attack = recipe_class.build(model_wrapper)
    if recipe_class is BERTAttackLi2020:
        attack = configure_bertattack_low_resource(attack, max_candidates=args_max_candidates, keep_use=args_keep_use)
        print(f"  BERT-Attack params: max_candidates={args_max_candidates}, max_percent=0.4, keep_use={args_keep_use}")
    elif not args_keep_use:
        attack.constraints = [
            c for c in attack.constraints
            if type(c).__name__ != "UniversalSentenceEncoder"
        ]
        print(f"  USE constraint removed (TF GPU not available on Windows)")
    attack_args = AttackArgs(
        num_examples=num_examples,
        num_examples_offset=num_examples_offset,
        log_to_csv=output_csv,
        csv_coloring_style="plain",
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=os.path.join(os.path.dirname(output_csv), "checkpoints",
                                   os.path.splitext(os.path.basename(output_csv))[0]),
        disable_stdout=False,
        query_budget=args_query_budget,
    )
    if args_query_budget:
        print(f"  Query budget: {args_query_budget} per example")

    if per_example_timeout:
        print(f"  Per-example timeout: {per_example_timeout}s")
        attacker = TimeoutAttacker(attack, dataset, attack_args, per_example_timeout=per_example_timeout)
    else:
        attacker = Attacker(attack, dataset, attack_args)

    # Auto-resume from latest valid checkpoint if one exists
    chkpt_dir = os.path.join(os.path.dirname(output_csv), "checkpoints",
                             os.path.splitext(os.path.basename(output_csv))[0])
    chkpt_files = glob.glob(os.path.join(chkpt_dir, "*.ta.chkpt"))
    if chkpt_files:
        # Try checkpoints from newest to oldest, skip any with inconsistent state
        for latest in sorted(chkpt_files, key=os.path.getmtime, reverse=True):
            try:
                with open(latest, "rb") as f:
                    checkpoint = pickle.load(f)
                # Validate consistency before using
                if checkpoint.num_remaining_attacks != len(checkpoint.worklist):
                    print(f"  [SKIP] Inconsistent checkpoint ({checkpoint.results_count} done, "
                          f"remaining={checkpoint.num_remaining_attacks} vs worklist={len(checkpoint.worklist)}): "
                          f"{os.path.basename(latest)}")
                    continue
                # Fix: new checkpoints use attacker.attack_args (num_examples=200) but
                # attack_log_manager starts fresh, so num_remaining = 200-1 ≠ len(worklist).
                # Patch attacker's num_examples to remaining count so verification passes.
                remaining = checkpoint.num_remaining_attacks
                attacker._checkpoint = checkpoint
                attacker.attack_args.num_examples = remaining  # ← key fix, do NOT touch offset
                print(f"  Resuming from checkpoint ({checkpoint.results_count} done, {remaining} remaining): "
                      f"{os.path.basename(latest)}")
                break
            except Exception as e:
                print(f"  [SKIP] Could not load {os.path.basename(latest)}: {e}")
        else:
            print(f"  No valid checkpoint found, starting fresh.")

    results = attacker.attack_dataset()
    return results


def compute_semantic_similarity(csv_path):
    """Compute average cosine similarity between original and perturbed texts
    for successful attacks using sentence-transformers (all-MiniLM-L6-v2).
    Original papers use USE; values may differ slightly."""
    try:
        import pandas as pd
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
        print("[WARNING] sentence-transformers not installed. Run: pip install sentence-transformers")
        return float("nan")
    except Exception as exc:
        print(f"[WARNING] Semantic similarity skipped: {exc}")
        return float("nan")


def summarize_results(csv_path, method_name):
    import pandas as pd
    df = pd.read_csv(csv_path)
    total = len(df)
    success = (df["result_type"] == "Successful").sum()
    asr = success / total * 100 if total > 0 else 0.0

    if "num_words_changed" in df.columns:
        avg_words_changed = df.loc[df["result_type"] == "Successful", "num_words_changed"].mean()
    else:
        avg_words_changed = float("nan")

    print(f"  Computing semantic similarity (sentence-transformers)...")
    sem_sim = compute_semantic_similarity(csv_path)

    print(f"\n[{method_name}]")
    print(f"  Total examples:       {total}")
    print(f"  Successful attacks:   {success}")
    print(f"  Attack Success Rate:  {asr:.1f}%")
    print(f"  Avg words changed:    {avg_words_changed:.1f}")
    print(f"  Semantic Similarity:  {sem_sim:.4f}")
    return {"method": method_name, "total": total, "success": success,
            "asr": asr, "avg_words_changed": avg_words_changed, "sem_sim": sem_sim}


def main(args):
    import torch
    out_dir = os.path.join(args.results_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model from {args.model_dir}")
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU.")
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    print(f"Loading dataset: {args.dataset} (test split, {args.num_examples} examples)")
    dataset = HuggingFaceDataset(args.dataset, "plain_text", split="test")

    summaries = []
    run = args.attack  # "textfooler" | "bertattack" | "all"

    # --- TextFooler (Jin et al., AAAI 2020) ---
    if run in ("textfooler", "all"):
        print("\n" + "="*50)
        print("Running TextFooler (Jin et al., AAAI 2020)...")
        print("="*50)
        tf_csv = os.path.join(out_dir, "textfooler_results.csv")
        run_single_attack(
            TextFoolerJin2019,
            model_wrapper,
            dataset,
            args.num_examples,
            tf_csv,
            num_examples_offset=args.num_examples_offset,
            checkpoint_interval=args.checkpoint_interval,
            args_keep_use=args.keep_use,
            args_query_budget=args.query_budget,
        )
        summaries.append(summarize_results(tf_csv, "TextFooler"))

    # --- BERT-Attack (Li et al., EMNLP 2020) ---
    if run in ("bertattack", "all"):
        print("\n" + "="*50)
        print("Running BERT-Attack (Li et al., EMNLP 2020)...")
        print("="*50)
        ba_csv = os.path.join(out_dir, "bertattack_results.csv")
        run_single_attack(
            BERTAttackLi2020,
            model_wrapper,
            dataset,
            args.num_examples,
            ba_csv,
            args.example_timeout,
            args.num_examples_offset,
            args.checkpoint_interval,
            args_max_candidates=args.max_candidates,
            args_keep_use=args.keep_use,
            args_query_budget=args.query_budget,
        )
        summaries.append(summarize_results(ba_csv, "BERT-Attack"))

    # --- Print comparison table ---
    print("\n" + "="*50)
    print("BASELINE SUMMARY (compare with paper tables)")
    print("="*50)
    print(f"{'Method':<15} {'ASR':>8} {'Avg Words Changed':>20} {'Sem. Similarity':>17}")
    print("-"*63)
    for s in summaries:
        print(f"{s['method']:<15} {s['asr']:>7.1f}% {s['avg_words_changed']:>20.1f} {s['sem_sim']:>17.4f}")

    summary_path = os.path.join(out_dir, "baseline_summary.txt")
    with open(summary_path, "w") as f:
        for s in summaries:
            f.write(f"{s}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=PRETRAINED_MODEL_DIR)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--num_examples", type=int, default=ATTACK_NUM_EXAMPLES)
    parser.add_argument("--num_examples_offset", type=int, default=0)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument(
        "--example_timeout",
        type=int,
        default=300,
        help="Per-example timeout in seconds for BERT-Attack. Use 0 to disable.",
    )
    parser.add_argument(
        "--attack",
        choices=["textfooler", "bertattack", "all"],
        default="all",
        help="Which attack to run. Default: both.",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=8,
        help="Max MLM candidates for BERT-Attack (default: 8).",
    )
    parser.add_argument(
        "--keep_use",
        action="store_true",
        help="Keep Universal Sentence Encoder constraint. Requires TF; use in bertattack_tf env.",
    )
    parser.add_argument(
        "--query_budget",
        type=int,
        default=None,
        help="Max model queries per example. None = unlimited. Use ~1000 for robust model eval.",
    )
    args = parser.parse_args()
    main(args)
