"""Print text outputs of an executed notebook and dump its figures to a folder (dev helper).

  python nb/show.py _dev_run.ipynb [figdir] [--from N]
"""
import base64
import pathlib
import re
import sys

import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
figdir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
if figdir:
    figdir.mkdir(parents=True, exist_ok=True)
for ci, cell in enumerate(nb.cells):
    if cell.cell_type != "code" or ci < start:
        continue
    for oi, o in enumerate(cell.get("outputs", [])):
        if o.output_type == "stream":
            print(f"[{ci}] {o.text.rstrip()}")
        elif o.output_type == "error":
            print(f"[{ci}] ERROR {o.ename}: {o.evalue}")
            print("\n".join(re.sub(r"\x1b\[[0-9;]*m", "", l) for l in o.traceback[-8:]))
        else:
            data = o.get("data", {})
            if "image/png" in data and figdir:
                p = figdir / f"cell{ci:03d}_{oi}.png"
                p.write_bytes(base64.b64decode(data["image/png"]))
                print(f"[{ci}] <figure {p.name}>")
            elif "text/html" in data:
                txt = re.sub(r"<[^>]+>", " ", data["text/html"])
                print(f"[{ci}] <html> {re.sub(r'\\s+', ' ', txt)[:1500]}")
            elif "text/plain" in data:
                print(f"[{ci}] {data['text/plain'][:1500]}")
