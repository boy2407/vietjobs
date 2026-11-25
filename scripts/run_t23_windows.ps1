# T23 trên máy desktop Windows (RTX 5060 8 GB) — bản PowerShell của scripts/vast_run_t23.sh,
# cùng lệnh, cùng tham số, cùng run_id. Bảng quyết định: tasks/T23-train-lai.md.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_t23_windows.ps1 pooled dense
#   powershell -ExecutionPolicy Bypass -File scripts\run_t23_windows.ps1 rnn:cafebert
#   powershell -ExecutionPolicy Bypass -File scripts\run_t23_windows.ps1 chunk
#
# chunk: đối chứng độ công bằng (tác giả, 2026-11-25) — cùng Dense, cùng tham số, nhưng tin dài
# hơn max_len được chia đoạn thay vì cắt bỏ (encode --chunked); run_id mang "sal17chunk".
#
# rnn: cũng chia đoạn (tác giả, 2026-11-25) — token các đoạn nối thành một dãy, không trần độ
# dài, cache lưu gọn không đệm (encode --pooling none --chunked); run_id mang "sal17chunk".
#
# Chạy lại cùng lệnh là chạy tiếp: run đã có metrics.json bị bỏ qua; MTL RNN dùng --resume.
# Không đọc test: chấm test + ext37k là bước riêng, tác giả gõ lệnh (Rule 5).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"; $env:VIETJOBS_SPLITS = "splits-sal17-11c"; $env:PYTHONIOENCODING = "utf-8"
$PY = ".venv-dl\Scripts\python.exe"
New-Item -ItemType Directory -Force logs, artifacts | Out-Null
function Say($m) { $l = "[{0}] {1}" -f (Get-Date -Format "HH:mm"), $m; $l; Add-Content logs\t23.txt $l }

$COMMON = "--last4 --lr 3e-4 --weight-decay 0.01 --dropout 0.3 --epochs 80 --patience 12".Split()
$CAT = "--family masked --class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05".Split()
$SAL = "--beta-sal 0.1 --class-weight".Split()
$MTL = "--class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05 --beta-sal 0.1 --alpha-sal 5.0".Split()
$RNN = "--hidden 100 --conv-channels 50".Split()
$ENCS = "phobert", "cafebert", "xlmr", "visobert", "bamibert"
$LEN = @{ phobert = 256; cafebert = 512; xlmr = 512; visobert = 512; bamibert = 1024 }
$BS = @{ phobert = 64; cafebert = 32; xlmr = 32; visobert = 32; bamibert = 16 }

$free = (Get-PSDrive (Get-Location).Drive.Name).Free / 1GB
if ($free -lt 70) { Say "DUNG: chi con $([int]$free) GB trong o (can >= 70)"; exit 1 }

function Train($rid, $mod, [string[]]$a) {
  if (Test-Path "artifacts\$rid\metrics.json") { Say "BO QUA $rid (da co metrics.json)"; return }
  Say "TRAIN $rid"
  & $PY -m vietjobs.dl.$mod @a --run-id $rid 2>&1 | Tee-Object -Append "logs\$rid.txt" |
    Select-String -Pattern "^\[dl-|Error|Traceback"
  if (-not (Test-Path "artifacts\$rid\metrics.json")) { Say "LOI ${rid}: khong co metrics.json"; exit 1 }
}

function Stage-Pooled {
  foreach ($e in $ENCS) {
    Say "ENCODE pooled last4 masked $e"
    & $PY -m vietjobs.dl.encode --task salary --splits train dev --encoder $e --max-len $LEN[$e] `
      --batch $BS[$e] --last4 --device cuda 2>&1 | Tee-Object "logs\t23-encode-$e.txt"
  }
}

function Stage-Dense {
  foreach ($e in $ENCS) { foreach ($s in 42..46) {
    $base = @("--encoder", $e, "--max-len", $LEN[$e], "--head", "dense") + $COMMON + @("--device", "cpu", "--seed", $s)
    Train "dl-cat-$e-dense-sal17-s$s" train_dl (@("--task", "category") + $base + $CAT)
    Train "dl-sal-$e-dense-sal17-s$s" train_dl (@("--task", "salary") + $base + $SAL)
    Train "dl-mtl-$e-dense-sal17-s$s" train_mtl ($base + $MTL)
  } }
}

function Stage-Chunk {
  foreach ($e in $ENCS) {
    Say "ENCODE pooled last4 masked chunked $e"
    & $PY -m vietjobs.dl.encode --task salary --splits train dev --encoder $e --max-len $LEN[$e] `
      --batch $BS[$e] --last4 --chunked --device cuda 2>&1 | Tee-Object "logs\t23-encode-chunk-$e.txt"
  }
  foreach ($e in $ENCS) { foreach ($s in 42..46) {
    $base = @("--encoder", $e, "--max-len", $LEN[$e], "--head", "dense", "--chunked") + $COMMON + @("--device", "cpu", "--seed", $s)
    Train "dl-cat-$e-dense-sal17chunk-s$s" train_dl (@("--task", "category") + $base + $CAT)
    Train "dl-sal-$e-dense-sal17chunk-s$s" train_dl (@("--task", "salary") + $base + $SAL)
    Train "dl-mtl-$e-dense-sal17chunk-s$s" train_mtl ($base + $MTL)
  } }
}

function Stage-Rnn($e) {
  Say "ENCODE token last4 masked chunked $e ($([int]((Get-PSDrive C).Free/1GB)) GB trong)"
  & $PY -m vietjobs.dl.encode --task salary --splits train dev --encoder $e --max-len $LEN[$e] `
    --batch $BS[$e] --last4 --pooling none --chunked --device cuda 2>&1 | Tee-Object "logs\t23-encode-tok-$e.txt"
  $base = @("--encoder", $e, "--max-len", $LEN[$e], "--head", "rnn", "--chunked") + $COMMON + $RNN + @("--device", "cuda", "--seed", 42)
  Train "dl-cat-$e-rnn-sal17chunk-s42" train_dl (@("--task", "category") + $base + $CAT)
  Train "dl-sal-$e-rnn-sal17chunk-s42" train_dl (@("--task", "salary") + $base + $SAL)
  $rid = "dl-mtl-$e-rnn-sal17chunk-s42"
  if (-not (Test-Path "artifacts\$rid\metrics.json")) {
    Say "TRAIN $rid"
    & $PY -m vietjobs.dl.train_mtl @($base + $MTL) --run-id $rid --resume 2>&1 |
      Tee-Object -Append "logs\$rid.txt" | Select-String -Pattern "^\[dl-|Error|Traceback"
    if (-not (Test-Path "artifacts\$rid\metrics.json")) { Say "LOI MTL RNN $e"; exit 1 }
  }
  # Encoder chạy lần lượt nên chỉ cache token của encoder này còn trên đĩa.
  # Chỉ xoá cache token chia đoạn của đúng encoder này; cache pooled "-chunk.npy" giữ lại.
  $pre = if ($e -eq "phobert") { "" } else { "$e-" }
  foreach ($sp in "train", "dev") { foreach ($k in "tok.f16", "off") {
    Remove-Item "artifacts\embeddings\$env:VIETJOBS_SPLITS\$pre$sp-masked-len$($LEN[$e])-last4-chunk-$k.npy" -EA SilentlyContinue } }
  Say "XONG RNN $e"
}

foreach ($job in $args) {
  switch -Regex ($job) {
    "^pooled$" { Stage-Pooled }
    "^dense$"  { Stage-Dense }
    "^chunk$"  { Stage-Chunk }
    "^rnn$"    { foreach ($e in $ENCS) { Stage-Rnn $e } }
    "^rnn:(.+)$" { Stage-Rnn $Matches[1] }
    default { "job la: $job"; exit 2 }
  }
}
Say "=== XONG T23: $args ==="
