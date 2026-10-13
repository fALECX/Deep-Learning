"""Collect raw zero-shot LLM responses for C2 (run outside the notebook, as the spec allows).

Only raw responses are stored here; parsing, mapping to the 20 techniques and
scoring all happen in the notebook.

Usage:
  python run_llm.py --provider grsai --model gemini-3.1-flash-lite --task classify --prompt def --run 1
  python run_llm.py --provider lmstudio --model ornith-1.5-9b --task memo --run 1

Outputs: outputs/<task>__<model>__<prompt>__run<k>.jsonl, one line per meme, resumable.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

import prompts

HERE = pathlib.Path(__file__).resolve().parent
DATASETS = {"test": HERE.parent / "data" / "test_set_task2.txt",
            "au_probe": HERE.parent / "c4_probe" / "au_probe_2026.json"}
OUT = HERE / "outputs"
GRSAI_URL = "https://api.grsai.com/v1beta/models/{model}:generateContent"
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"


def load_env():
    env = pathlib.Path.home() / ".env"
    for line in env.read_text(encoding="utf8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_grsai(model, system, user, temperature):
    key = os.environ["GRSAI_API_KEY"]
    # grsai silently drops `systemInstruction` (verified: prompt token count did not
    # include it), so for every provider the instructions go into one user turn.
    body = {
        "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
        "generationConfig": {"temperature": temperature},
    }
    r = post(GRSAI_URL.format(model=model), body,
             {"Authorization": f"Bearer {key}", "x-goog-api-key": key}, timeout=300)
    cand = (r.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    return {"text": text, "finish_reason": cand.get("finishReason"),
            "model_version": r.get("modelVersion"), "usage": r.get("usageMetadata"),
            "prompt_feedback": r.get("promptFeedback")}


def call_lmstudio(model, system, user, temperature, reasoning=None):
    body = {"model": model, "temperature": temperature, "max_tokens": 8192,
            "messages": [{"role": "user", "content": f"{system}\n\n{user}"}]}
    if reasoning:
        body["reasoning_effort"] = reasoning
    r = post(LMSTUDIO_URL, body, {}, timeout=1800)
    ch = r["choices"][0]
    msg = ch["message"]
    return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content"),
            "finish_reason": ch.get("finish_reason"), "model_version": r.get("model"),
            "usage": r.get("usage")}


def memo_prefix(text):
    words = text.split()
    return " ".join(words[: len(words) // 2])


def build_items(task, prompt_variant, dataset):
    memes = json.loads(DATASETS[dataset].read_text(encoding="utf8"))
    items = []
    for m in memes:
        if task == "classify":
            items.append((m["id"], prompts.system_prompt(prompt_variant), prompts.user_prompt(m["text"])))
        elif len(m["text"].split()) >= 8:  # memo probe needs a meaningful second half
            items.append((m["id"], prompts.MEMO_SYSTEM, prompts.memo_user_prompt(memo_prefix(m["text"]))))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["grsai", "lmstudio"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["classify", "memo"], default="classify")
    ap.add_argument("--prompt", choices=["def", "names"], default="def")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--dataset", choices=list(DATASETS), default="test")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N memes (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reasoning", default=None, help="lmstudio only, e.g. none (thinking off)")
    ap.add_argument("--every", type=int, default=1, help="systematic subsample: every k-th meme")
    a = ap.parse_args()
    load_env()

    OUT.mkdir(exist_ok=True)
    tag = a.prompt if a.task == "classify" else "memo"
    prefix = "" if a.dataset == "test" else f"{a.dataset}__"
    label = a.model + (f"-reasoning-{a.reasoning}" if a.reasoning else "")
    out = OUT / f"{prefix}{a.task}__{label}__{tag}__run{a.run}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf8").splitlines()
                if l.strip() and json.loads(l).get("error") is None}

    items = [it for i, it in enumerate(build_items(a.task, a.prompt, a.dataset)) if i % a.every == 0 and it[0] not in done]
    if a.limit:
        items = items[: a.limit]
    call = call_grsai if a.provider == "grsai" else (lambda *x: call_lmstudio(*x, reasoning=a.reasoning))
    workers = a.workers if a.provider == "grsai" else 1

    def work(item):
        mid, system, user = item
        rec = {"id": mid, "provider": a.provider, "model": label, "task": a.task, "prompt": tag,
               "run": a.run, "temperature": a.temperature, "error": None}
        for attempt in range(4):
            t0 = time.time()
            try:
                rec.update(call(a.model, system, user, a.temperature))
                rec["error"] = None
                break
            except Exception as e:  # network resets, timeouts, malformed provider responses
                body = e.read().decode()[:500] if isinstance(e, urllib.error.HTTPError) else ""
                rec["error"] = f"{type(e).__name__}: {e} {body}".strip()
                time.sleep(5 * 2 ** attempt)
        rec["latency_s"] = round(time.time() - t0, 2)
        rec["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        return rec

    with out.open("a", encoding="utf8") as f, cf.ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(work, items), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or rec["error"]:
                print(f"{i}/{len(items)} {rec['id']} err={rec['error']}", flush=True)
    print("done", out.name)


if __name__ == "__main__":
    main()
