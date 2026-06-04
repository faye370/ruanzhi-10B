"""
Watchdog for BERT-Attack defense run.
Monitors CSV growth; if stuck > STUCK_SEC, kills process, skips stuck sample, restarts.
Accumulates CSV rows across restarts so data is never lost.
"""
import collections
import csv
import glob
import os
import pickle
import shutil
import subprocess
import sys
import time

CHKPT_DIR   = r"results\defense\baseline\checkpoints\bertattack_results"
CSV_PATH    = r"results\defense\baseline\bertattack_results.csv"
LOG_PATH    = r"results\defense\step3_bertattack_robust.log"
STUCK_SEC   = 240   # 4 minutes without a new CSV row = stuck

# Step 3
PYTHON_TF   = r"D:\anaconda\envs\bertattack_tf\python.exe"
STEP3_ARGS  = [
    "attack/baseline_attack.py",
    "--attack", "bertattack",
    "--model_dir", "checkpoints/bert-imdb-adv",
    "--results_dir", "results/defense",
    "--num_examples", "200",
    "--max_candidates", "8",
    "--query_budget", "2000",
]

# Step 4
PYTHON_PT   = r"D:\anaconda\envs\pytorch\python.exe"
STEP4_ARGS  = [
    "attack/improved_attack.py",
    "--model_dir", "checkpoints/bert-imdb-adv",
    "--results_dir", "results/defense",
    "--num_examples", "200",
]
STEP4_LOG   = r"results\defense\step4_scbert_robust.log"
STEP4_DONE  = r"results\defense\improved\improved_results.json"
STEP4_CKPT_CTRL = r"results\defense\improved\ckpt_control.json"
STEP4_CKPT_TRT  = r"results\defense\improved\ckpt_treatment.json"
STEP4_STUCK_SEC = 300  # 5 minutes without checkpoint growth = stuck

CWD = os.path.dirname(os.path.abspath(__file__))

# CSV accumulator — collects rows from all restarts so nothing is lost
_csv_accumulated = []   # list of dicts
_csv_fieldnames  = None


def csv_lines():
    try:
        with open(CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def backup_csv():
    """Read current CSV and add new rows to the accumulator (dedup by original_text)."""
    global _csv_accumulated, _csv_fieldnames
    try:
        if not os.path.exists(CSV_PATH):
            return
        with open(CSV_PATH, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if _csv_fieldnames is None and reader.fieldnames:
                _csv_fieldnames = reader.fieldnames
            existing_keys = {r.get("original_text", "") for r in _csv_accumulated}
            new_rows = [r for r in reader if r.get("original_text", "") not in existing_keys]
            _csv_accumulated.extend(new_rows)
        print(f"[{ts()}] CSV backup: +{len(new_rows)} new rows, total accumulated={len(_csv_accumulated)}", flush=True)
    except Exception as e:
        print(f"[{ts()}] CSV backup failed: {e}", flush=True)


def save_accumulated_csv():
    """Write all accumulated rows back to the CSV file."""
    global _csv_accumulated, _csv_fieldnames
    if not _csv_accumulated or not _csv_fieldnames:
        return
    try:
        with open(CSV_PATH, "w", encoding="utf-8", errors="replace", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fieldnames)
            writer.writeheader()
            writer.writerows(_csv_accumulated)
        print(f"[{ts()}] Final CSV written: {len(_csv_accumulated)} rows total", flush=True)
    except Exception as e:
        print(f"[{ts()}] Save accumulated CSV failed: {e}", flush=True)


def total_rows():
    """Accumulated rows + current CSV rows (deduped estimate)."""
    return len(_csv_accumulated) + max(0, csv_lines() - 1)  # -1 for header


def latest_checkpoint():
    files = sorted(glob.glob(os.path.join(CHKPT_DIR, "*.ta.chkpt")),
                   key=os.path.getmtime)
    return files[-1] if files else None


def skip_first_sample(chkpt_path):
    """Remove the first worklist entry and save a new checkpoint."""
    with open(chkpt_path, "rb") as f:
        chk = pickle.load(f)
    wl = list(chk.worklist)
    if not wl:
        return None, None
    skipped_idx = wl.pop(0)
    chk.worklist = collections.deque(wl)
    chk.attack_args.num_examples = len(wl) + chk.results_count
    new_path = os.path.join(CHKPT_DIR, f"skip{skipped_idx}_auto.ta.chkpt")
    with open(new_path, "wb") as f:
        pickle.dump(chk, f)
    return skipped_idx, new_path


def check_done():
    """Return True if BERT-Attack is complete."""
    if total_rows() > 150:
        return True
    latest = latest_checkpoint()
    if latest:
        try:
            with open(latest, "rb") as f:
                chk = pickle.load(f)
            if len(chk.worklist) == 0:
                return True
        except Exception:
            pass
    return False


def ts():
    return time.strftime("%H:%M:%S")


def make_env():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TFHUB_CACHE_DIR"] = r"C:\Users\fafa\AppData\Local\Temp\tfhub_modules"
    return env


def start_process():
    cmd = [PYTHON_TF] + STEP3_ARGS
    print(f"[{ts()}] Starting Step3: {' '.join(cmd)}", flush=True)
    log_f = open(LOG_PATH, "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            cwd=CWD, env=make_env())
    return proc, log_f


def start_step4():
    cmd = [PYTHON_PT] + STEP4_ARGS
    print(f"[{ts()}] Starting Step4: {' '.join(cmd)}", flush=True)
    log_f = open(STEP4_LOG, "a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            cwd=CWD, env=make_env())
    return proc, log_f


def step4_done():
    return os.path.isfile(STEP4_DONE) and os.path.getsize(STEP4_DONE) > 100


def step4_progress_mtime():
    """Return latest mtime of any Step4 checkpoint file, or 0 if none."""
    best = 0
    for p in [STEP4_CKPT_CTRL, STEP4_CKPT_TRT]:
        if os.path.exists(p):
            best = max(best, os.path.getmtime(p))
    return best


def run_step4():
    if step4_done():
        print(f"[{ts()}] Step4 already done, skipping.", flush=True)
        return

    print(f"[{ts()}] === Starting Step 4 (SC-BERT-Attack) ===", flush=True)
    proc, log_f = start_step4()
    last_mtime = step4_progress_mtime() or time.time()
    last_progress = time.time()

    try:
        while True:
            time.sleep(30)
            rc = proc.poll()
            if rc is not None:
                log_f.close()
                if step4_done():
                    print(f"[{ts()}] Step4 completed successfully!", flush=True)
                    break
                else:
                    print(f"[{ts()}] Step4 exited (rc={rc}) but not done. Restarting...", flush=True)
                    proc, log_f = start_step4()
                    last_progress = time.time()
                continue

            cur_mtime = step4_progress_mtime()
            if cur_mtime > last_mtime:
                last_mtime = cur_mtime
                last_progress = time.time()
                print(f"[{ts()}] Step4 progress detected (checkpoint updated)", flush=True)
            else:
                stuck = time.time() - last_progress
                if stuck > STEP4_STUCK_SEC:
                    print(f"[{ts()}] Step4 STUCK {stuck:.0f}s! Killing and restarting...", flush=True)
                    proc.kill()
                    proc.wait()
                    log_f.close()
                    time.sleep(2)
                    proc, log_f = start_step4()
                    last_progress = time.time()
                    last_mtime = step4_progress_mtime() or time.time()
    except KeyboardInterrupt:
        proc.kill()
    finally:
        try:
            log_f.close()
        except Exception:
            pass


def main():
    # Rotate old checkpoints so the attack starts fresh (not resume from empty worklist)
    if os.path.exists(CHKPT_DIR):
        backup_dir = CHKPT_DIR + "_backup_" + time.strftime("%Y%m%d_%H%M%S")
        shutil.move(CHKPT_DIR, backup_dir)
        print(f"[{ts()}] Moved old checkpoints to {os.path.basename(backup_dir)}", flush=True)
    os.makedirs(CHKPT_DIR, exist_ok=True)

    print(f"[{ts()}] Watchdog started (rerun). STUCK_SEC={STUCK_SEC}", flush=True)
    skipped_samples = []
    proc, log_f = start_process()
    last_lines = csv_lines()
    last_progress = time.time()

    try:
        while True:
            time.sleep(30)

            # Process finished on its own?
            rc = proc.poll()
            if rc is not None:
                backup_csv()
                log_f.close()
                if check_done():
                    print(f"[{ts()}] Process exited (rc={rc}), attack complete. Done!", flush=True)
                    break
                else:
                    print(f"[{ts()}] Process exited (rc={rc}) but not done. Restarting...", flush=True)
                    proc, log_f = start_process()
                    last_lines = csv_lines()
                    last_progress = time.time()
                    continue

            cur = csv_lines()
            if cur > last_lines:
                last_lines = cur
                last_progress = time.time()
                print(f"[{ts()}] Progress: CSV {cur} rows (accumulated {len(_csv_accumulated)})", flush=True)
            else:
                stuck_for = time.time() - last_progress
                if stuck_for > STUCK_SEC:
                    print(f"[{ts()}] STUCK {stuck_for:.0f}s! Killing PID {proc.pid}...", flush=True)
                    proc.kill()
                    proc.wait()
                    backup_csv()
                    log_f.close()
                    time.sleep(2)

                    chkpt = latest_checkpoint()
                    if chkpt:
                        skipped, new_chkpt = skip_first_sample(chkpt)
                        if skipped is not None:
                            skipped_samples.append(skipped)
                            print(f"[{ts()}] Skipped sample {skipped}. "
                                  f"Total skipped: {skipped_samples}", flush=True)
                    else:
                        print(f"[{ts()}] No checkpoint found, starting fresh.", flush=True)

                    proc, log_f = start_process()
                    last_lines = csv_lines()
                    last_progress = time.time()

    except KeyboardInterrupt:
        print(f"\n[{ts()}] Watchdog stopped. Killing process...", flush=True)
        backup_csv()
        proc.kill()
        try:
            log_f.close()
        except Exception:
            pass
        save_accumulated_csv()
        return
    finally:
        try:
            log_f.close()
        except Exception:
            pass

    backup_csv()
    save_accumulated_csv()
    print(f"[{ts()}] Step3 done. Skipped samples: {skipped_samples}", flush=True)
    run_step4()


if __name__ == "__main__":
    main()
