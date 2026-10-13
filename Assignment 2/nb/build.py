"""Assemble the notebook from nb/NN_*.py (jupytext percent format) and optionally execute it.

  python nb/build.py                 # write A2_s4271512.ipynb (no outputs)
  python nb/build.py --execute       # also execute (set A2_FAST=1 for a quick local smoke test)
  python nb/build.py --upto 03       # only parts up to 03 (for iterative development)
"""
import argparse
import os
import pathlib
import sys

import jupytext
import nbformat
from nbclient import NotebookClient

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("--execute", action="store_true")
ap.add_argument("--upto", default="99")
ap.add_argument("--out", default="A2_s4271512.ipynb")
ap.add_argument("--timeout", type=int, default=-1)
a = ap.parse_args()

def c2_payload():
    """Compact, embeddable archive of every raw LLM response plus the exact prompts."""
    import base64, gzip, json
    sys.path.insert(0, str(ROOT / "c2_llm"))
    import prompts as P
    keep = ["id", "model", "task", "prompt", "run", "temperature", "text", "error", "finish_reason",
            "model_version", "latency_s", "timestamp_utc"]
    recs = []
    for f in sorted((ROOT / "c2_llm" / "outputs").glob("*.jsonl")):
        dataset = f.name.split("__")[0] if f.name.startswith("au_probe__") else "test"
        for line in f.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                rec = {k: r.get(k) for k in keep}
                rec["dataset"] = dataset
                u = r.get("usage") or {}
                rec["tokens_in"] = u.get("promptTokenCount", u.get("prompt_tokens"))
                rec["tokens_out"] = u.get("candidatesTokenCount", u.get("completion_tokens"))
                recs.append(rec)
    payload = {"prompts": {"system_def": P.system_prompt("def"), "system_names": P.system_prompt("names"),
                           "user_template": P.user_prompt("{meme_text}"), "memo_system": P.MEMO_SYSTEM,
                           "memo_user_template": P.memo_user_prompt("{first_half_of_meme}")},
               "records": recs}
    return base64.b64encode(gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf8"), 9)).decode()


parts = sorted(p for p in HERE.glob("[0-9][0-9]_*.py") if p.name[:2] <= a.upto)
src = "\n\n".join(p.read_text(encoding="utf8").strip() for p in parts) + "\n"
if "__AU_PROBE_JSON__" in src:
    src = src.replace("__AU_PROBE_JSON__", (ROOT / "c4_probe" / "au_probe_2026.json").read_text(encoding="utf8").strip())
if "__C2_RAW_B64__" in src:
    src = src.replace("__C2_RAW_B64__", c2_payload())
nb = jupytext.reads(src, fmt="py:percent")
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
out = ROOT / a.out

if a.execute:
    client = NotebookClient(nb, timeout=a.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, out)
else:
    nbformat.write(nb, out)
print("wrote", out, "from", [p.name for p in parts])
