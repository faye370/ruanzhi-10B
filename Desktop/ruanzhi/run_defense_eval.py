"""
Defense evaluation: run Steps 2, 3, 4 sequentially overnight.

Step 2 — TextFooler   on robust model (bertattack_tf env, WITH USE, 200 samples)
Step 3 — BERT-Attack  on robust model (bertattack_tf env, WITH USE, max_candidates=8, 200 samples)
Step 4 — SC-BERT-Attack on robust model (pytorch env, 200 samples)

Run from project root:
    python run_defense_eval.py
"""
import subprocess
import sys
import os
import datetime

LOG_DIR = os.path.join("results", "defense")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Adjust these paths if your conda is installed elsewhere ──────────────────
BERTATTACK_TF_PYTHON = r"D:\anaconda\envs\bertattack_tf\python.exe"
PYTORCH_PYTHON       = r"D:\anaconda\envs\pytorch\python.exe"
# ─────────────────────────────────────────────────────────────────────────────

STEPS = [
    {
        "name": "Step2_TextFooler_RobustModel",
        "python": BERTATTACK_TF_PYTHON,
        "args": [
            "attack/baseline_attack.py",
            "--attack", "textfooler",
            "--model_dir", "checkpoints/bert-imdb-adv",
            "--results_dir", "results/defense",
            "--num_examples", "200",
            "--keep_use",
            "--example_timeout", "600",
        ],
        "log": os.path.join(LOG_DIR, "step2_textfooler_robust.log"),
        "done_if": os.path.join("results", "defense", "baseline", "textfooler_results.csv"),
    },
    {
        "name": "Step3_BERTAttack_RobustModel",
        "python": BERTATTACK_TF_PYTHON,
        "args": [
            "attack/baseline_attack.py",
            "--attack", "bertattack",
            "--model_dir", "checkpoints/bert-imdb-adv",
            "--results_dir", "results/defense",
            "--num_examples", "200",
            "--max_candidates", "8",
            "--example_timeout", "600",
            "--query_budget", "2000",
        ],
        "log": os.path.join(LOG_DIR, "step3_bertattack_robust.log"),
        "done_if": os.path.join("results", "defense", "baseline", "bertattack_results.csv"),
    },
    {
        "name": "Step4_SCBERTAttack_RobustModel",
        "python": PYTORCH_PYTHON,
        "args": [
            "attack/improved_attack.py",
            "--model_dir", "checkpoints/bert-imdb-adv",
            "--results_dir", "results/defense",
            "--num_examples", "200",
        ],
        "log": os.path.join(LOG_DIR, "step4_scbert_robust.log"),
        "done_if": os.path.join("results", "defense", "improved_results.json"),
    },
]


def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(step):
    cmd = [step["python"]] + step["args"]
    print(f"\n{'='*60}")
    print(f"[{ts()}] Starting: {step['name']}")
    print(f"  Log → {step['log']}")
    print(f"  CMD: {' '.join(cmd)}")
    print("="*60, flush=True)

    with open(step["log"], "w", encoding="utf-8") as log_f:
        log_f.write(f"# {step['name']}\n# Started: {ts()}\n# CMD: {' '.join(cmd)}\n\n")
        log_f.flush()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["TFHUB_CACHE_DIR"] = r"C:\Users\fafa\AppData\Local\Temp\tfhub_modules"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
            errors="replace",
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            log_f.write(line)
            log_f.flush()
        proc.wait()

    rc = proc.returncode
    status = "SUCCESS" if rc == 0 else f"FAILED (exit code {rc})"
    print(f"\n[{ts()}] {step['name']}: {status}")
    with open(step["log"], "a", encoding="utf-8") as log_f:
        log_f.write(f"\n# Finished: {ts()} | {status}\n")
    return rc == 0


def main():
    # Fix GBK terminal encoding issues on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"[{ts()}] Defense evaluation starting — 3 steps queued.")
    results = []
    for step in STEPS:
        done_file = step.get("done_if")
        if done_file and os.path.isfile(done_file):
            try:
                with open(done_file, "r", encoding="utf-8", errors="replace") as _f:
                    line_count = sum(1 for _ in _f)
            except Exception:
                line_count = 0
            if line_count > 150:  # header + results; 168+ after checkpoint-resume is enough
                print(f"\n[{ts()}] Skipping {step['name']} — output already exists: {done_file}")
                results.append((step["name"], "SKIPPED"))
                continue
        ok = run_step(step)
        results.append((step["name"], "OK" if ok else "FAILED"))
        if not ok:
            print(f"\n[{ts()}] *** {step['name']} failed — stopping pipeline. ***")
            break

    print(f"\n\n{'='*60}")
    print(f"[{ts()}] SUMMARY")
    print("="*60)
    for name, status in results:
        print(f"  {status:<8} {name}")
    print("="*60)

    summary_path = os.path.join(LOG_DIR, "defense_eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Defense evaluation finished: {ts()}\n\n")
        for name, status in results:
            f.write(f"{status:<8} {name}\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
