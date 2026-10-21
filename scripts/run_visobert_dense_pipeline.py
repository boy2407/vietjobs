import sys, subprocess, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C

PYTHON = "/Users/admin/Documents/TrNghia/VietJob/vietjobs/.venv-dl/bin/python"
PYTHON_MAIN = "/Users/admin/Documents/TrNghia/VietJob/vietjobs/.venv/bin/python"

def run(cmd, desc):
    print(f"\n>>> [START] {desc}...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, env={"PYTHONPATH": "src", "PATH": sys.path[0]})
    if res.returncode != 0:
        print(f"FAILED: {desc} with code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)
    print(f">>> [DONE] {desc} in {(time.time() - t0)/60:.2f}m", flush=True)

# 1. Encode train (dev is already running or done)
train_npy = C.ARTIFACT_DIR / "embeddings" / "visobert-train-raw-len512-last4.npy"
if not train_npy.exists():
    run([PYTHON, "-m", "vietjobs.dl.encode", "--encoder", "visobert", "--max-len", "512",
         "--layers", "4", "--splits", "train", "--task", "category", "--batch", "64"],
        "Encoding ViSoBERT train raw len512 last4")
else:
    print("train_npy already exists, skipping encode.", flush=True)

# 2. Train 5 seeds Dense on CPU
seeds = [42, 43, 44, 45, 46]
for s in seeds:
    run_id = f"dl-cat-visobert-dense-last4-cweff999-ls05-s{s}"
    run([PYTHON, "-m", "vietjobs.dl.train_dl", "--task", "category", "--encoder", "visobert",
         "--max-len", "512", "--layers", "4", "--hidden", "256", "--class-weight",
         "--cw-mode", "effective", "--cw-beta", "0.999", "--label-smoothing", "0.05",
         "--device", "cpu", "--seed", str(s), "--run-id", run_id],
        f"Training {run_id}")

# 3. Calculate metrics
mf1s, accs = [], []
for s in seeds:
    p = C.ARTIFACT_DIR / f"dl-cat-visobert-dense-last4-cweff999-ls05-s{s}" / "metrics.json"
    m = json.loads(p.read_text())
    mf1 = m["metrics"]["f1_macro"]
    acc = m["metrics"]["accuracy"]
    mf1s.append(mf1)
    accs.append(acc)
    print(f"s{s}: Macro-F1 = {mf1:.4f}, Acc = {acc:.4f}", flush=True)

mean_f1 = float(np.mean(mf1s))
sd_f1 = float(np.std(mf1s, ddof=1))
mean_acc = float(np.mean(accs))
sd_acc = float(np.std(accs, ddof=1))

print(f"\n==========================================")
print(f"ViSoBERT + Dense last4 (5 seeds):")
print(f"Macro-F1: {mean_f1:.4f} ± {sd_f1:.4f}")
print(f"Accuracy: {mean_acc:.4f} ± {sd_acc:.4f}")
print(f"==========================================\n", flush=True)

# 4. Update Google Sheet Dashboard_last4
spreadsheet_id = "14yGfv2RII0q69t-WEvcGtqT2J5guujwsb6Llm7vNuQ8"
dash_updates = {
    "range": "Dashboard_last4!A11:L11",
    "values": [
        [
            "dl-cat-visobert-dense-last4-cweff999-ls05",
            "category",
            "ViSoBERT + Dense",
            "last4 + cw(0.999) + ls(0.05)",
            5,
            f"{mean_f1:.4f}".replace(".", ","),
            f"{sd_f1:.4f}".replace(".", ","),
            f"{mean_acc:.4f}".replace(".", ","),
            "",
            "",
            "",
            "5 seed Mac CPU (mốc cũ: 0.5723)"
        ]
    ]
}

subprocess.run([
    "gws", "sheets", "spreadsheets", "values", "update",
    "--params", json.dumps({"spreadsheetId": spreadsheet_id, "range": "Dashboard_last4!A11:L11", "valueInputOption": "USER_ENTERED"}),
    "--json", json.dumps(dash_updates)
], check=True)
print("Updated Google Sheet Dashboard_last4!", flush=True)

# 5. Update local Dashboard_last4.tsv
tsv_path = C.ROOT / "data" / "eda_xlsx" / "Dashboard_last4.tsv"
lines = tsv_path.read_text().splitlines()
for i, line in enumerate(lines):
    if line.startswith("dl-cat-visobert-dense-last4-cweff999-ls05"):
        lines[i] = f"dl-cat-visobert-dense-last4-cweff999-ls05\tcategory\tViSoBERT + Dense\tlast4 + cw(0.999) + ls(0.05)\t5\t{mean_f1:.4f}\t{sd_f1:.4f}\t{mean_acc:.4f}\t\t\t\t5 seed Mac CPU (mốc cũ: 0.5723)"
tsv_path.write_text("\n".join(lines) + "\n")
print(f"Updated {tsv_path}!", flush=True)

# 6. Rebuild ket_qua_chay.xlsx
subprocess.run([PYTHON_MAIN, "scripts/export_runs.py"], check=True)
print("Rebuilt ket_qua_chay.xlsx!", flush=True)
print("ALL DONE!", flush=True)
