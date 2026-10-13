"""Rebuild the notebook from nb/*.py and copy outputs from an executed run into it.

Markdown text can then be refined after the run without re-executing anything. Code cells are matched by
their exact source, in order. The script refuses to write a notebook in which any code cell has no
executed counterpart, because the submission must be one clean Restart & Run All.

  python nb/merge_outputs.py executed.ipynb A2_s4271512.ipynb
"""
import subprocess
import sys
import pathlib

import nbformat

HERE = pathlib.Path(__file__).resolve().parent
executed_path, out_name = sys.argv[1], sys.argv[2]
tmp = "_fresh_build.ipynb"
subprocess.run([sys.executable, str(HERE / "build.py"), "--out", tmp], check=True)
fresh = nbformat.read(HERE.parent / tmp, as_version=4)
done = nbformat.read(executed_path, as_version=4)

norm = lambda s: "\n".join(l.rstrip() for l in s.strip().splitlines())
pool = [c for c in done.cells if c.cell_type == "code"]
i = 0
missing = []
for cell in fresh.cells:
    if cell.cell_type != "code":
        continue
    # The embedded LLM archive is rebuilt with a new gzip timestamp on every build; use the executed copy verbatim.
    embedded = cell.source.startswith("# Raw responses of all LLM runs")
    while i < len(pool) and not (embedded and pool[i].source.startswith("# Raw responses of all LLM runs")) \
            and norm(pool[i].source) != norm(cell.source):
        i += 1
    if embedded and i < len(pool):
        cell.source = pool[i].source
    if i == len(pool):
        missing.append(cell.source[:80])
        break
    cell.outputs = pool[i].outputs
    cell.execution_count = pool[i].execution_count
    i += 1
if missing:
    sys.exit(f"code cell not found in executed notebook (code changed after the run?): {missing[0]!r}")
fresh.metadata = done.metadata
nbformat.write(fresh, HERE.parent / out_name)
(HERE.parent / tmp).unlink()
print("wrote", out_name)
