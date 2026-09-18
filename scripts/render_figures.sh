#!/usr/bin/env bash
# Export every ```mermaid block in docs/ to SVG + PNG.
#
# The Markdown files are the single source of truth — they render natively in
# VS Code, GitHub and Obsidian. This script exists only for the cases that
# cannot read Mermaid: pasting a figure into Word, PowerPoint or a printed report.
#
#   ./scripts/render_figures.sh
#
# Needs Node (for npx) and Chrome. Nothing is installed permanently.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="docs/figures"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
mkdir -p "$OUT"

# One entry per mermaid block, as "source-file:figure-name", in document order.
# Every file below must contain EXACTLY the number of blocks named for it —
# the extractor aborts loudly on a mismatch rather than mislabelling a figure.
SOURCES=(
  "docs/00-tong-quan.md:01-suon-tong-the"
  "docs/01-data-audit.md:02-bon-ban-sao-van-ban"
  "docs/02-vietnamese-nlp.md:03-xu-ly-tieng-viet"
  "docs/06-baseline-dl.md:04-kien-truc-hoc-sau"
  "docs/08-ma-nguon.md:06-ma-nguon"
)

NAMES=()
for entry in "${SOURCES[@]}"; do NAMES+=("${entry##*:}"); done

python3 - "$OUT" "${SOURCES[@]}" <<'PY'
import re, sys, pathlib, collections

out, *entries = sys.argv[1:]
per_file = collections.OrderedDict()
for entry in entries:
    src, name = entry.rsplit(":", 1)
    per_file.setdefault(src, []).append(name)

total = 0
for src, names in per_file.items():
    path = pathlib.Path(src)
    if not path.exists():
        sys.exit(f"{src} not found — update SOURCES in scripts/render_figures.sh")
    blocks = re.findall(r'```mermaid\n(.*?)```', path.read_text(encoding='utf-8'), re.S)
    if len(blocks) != len(names):
        sys.exit(f"{src}: {len(blocks)} mermaid blocks but {len(names)} names "
                 f"— update SOURCES in scripts/render_figures.sh")
    for name, body in zip(names, blocks):
        pathlib.Path(out, name + '.mmd').write_text(body, encoding='utf-8')
        total += 1
print(f"extracted {total} blocks from {len(per_file)} files")
PY

for name in "${NAMES[@]}"; do
  npx -y -q @mermaid-js/mermaid-cli@11 -i "$OUT/$name.mmd" -o "$OUT/$name.svg" -q
  read -r w h < <(grep -o 'viewBox="[^"]*"' "$OUT/$name.svg" | head -1 \
                  | sed 's/viewBox="0 0 //;s/"//' | awk '{printf "%d %d\n", $1+1, $2+1}')
  if [ -x "$CHROME" ]; then
    "$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars \
      --default-background-color=FFFFFF --screenshot="$OUT/$name.png" \
      --window-size="$w,$h" --force-device-scale-factor=3 \
      "file://$PWD/$OUT/$name.svg" 2>/dev/null
    echo "  $name  ${w}x${h}  -> svg + png @3x"
  else
    echo "  $name  ${w}x${h}  -> svg only (Chrome not found)"
  fi
  rm -f "$OUT/$name.mmd"
done
