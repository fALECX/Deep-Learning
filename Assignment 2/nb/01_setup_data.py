# %% [markdown]
# # Litmus: detecting persuasion techniques in meme text
# **COSC2779 Deep Learning, Assignment 2 (Sequence Modelling)** · Student: s4271512
#
# The tool reads the text extracted from a meme and returns
# **(A)** the set of persuasion techniques it uses (20-way multi-label classification) and
# **(B)** where each technique sits in the text (multi-label span tagging).
# **(C)** then answers the curriculum committee's four questions with experiments.
#
# **How to read this notebook.** Every design decision is made at the point where the evidence for it appears.
# Each experiment states its *hypothesis and the decision it feeds* **before** it runs, so it is clear what
# result would have changed our mind. All numbers are produced by this notebook (`Restart & Run All`),
# except the zero-shot LLM responses for C2, which were collected outside it as the specification allows,
# are embedded here, and are parsed and scored here.
#
# | Section | Content |
# |---|---|
# | 0 | Setup, reproducibility, data |
# | 1 | Understanding the data and where it came from |
# | 2 | Evaluation framework: metrics, splits, baselines |
# | 3 | Model: shared encoder with two self-built heads |
# | 4 | Experiments E1–E8: encoder, adaptation strategy, architecture, head, loss, hyper-parameters, thresholds, robustness |
# | 5 | Final model on the held-out test set, error analysis, explainability |
# | 6 | Part C: C1 bias, C2 zero-shot comparison, C3 head agreement, C4 Australian deployment |
# | 7 | Ultimate judgement, limitations, declarations |

# %% [markdown]
# ## 0. Setup
# A fixed seed is set for Python, NumPy and PyTorch, and cuDNN is put into deterministic mode. GPU kernels can
# still introduce tiny non-determinism; for that reason every model comparison below is made across five folds
# rather than on a single run, so conclusions never hinge on one lucky seed.

# %%
import os, sys, json, re, math, time, random, copy, hashlib, itertools, collections, subprocess, gc
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
from sklearn.metrics import f1_score, precision_score, recall_score

hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Local smoke-test switch (tiny encoder, 2 folds, 1 epoch). Always False for the submitted run.
FAST_DEV_RUN = os.environ.get("A2_FAST", "0") == "1"
print("torch", torch.__version__, "| device:", DEVICE,
      torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", "| FAST_DEV_RUN:", FAST_DEV_RUN)

# %% [markdown]
# ### Data
# The assignment subset `data_2026_A2.zip` contains the six official SemEval-2021 Task 6 files for subtasks 1 and 2.
# We verified byte-for-byte that they are identical to the public corpus on GitHub. So if the files are not next to
# the notebook, they are downloaded from a pinned commit, which keeps the notebook runnable anywhere.
# `task1` files hold the label set per meme (Part A) and `task2` files hold the character-offset spans (Part B).

# %%
CORPUS_COMMIT = "796b32fa76b7a5814a48d1870d163058e81e4ce1"
CORPUS_URL = f"https://raw.githubusercontent.com/di-dimitrov/SEMEVAL-2021-task6-corpus/{CORPUS_COMMIT}"
SPLIT_FILES = {"train": "training_set", "dev": "dev_set", "test": "test_set"}


def download(url, path, attempts=6):
    """urlretrieve with retries: public mirrors occasionally time out (seen: HTTP 504 from Zenodo)."""
    import urllib.request
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, path)
            return
        except Exception as exc:
            if i == attempts - 1:
                raise
            print(f"download failed ({exc}); retrying in {15 * (i + 1)} s")
            time.sleep(15 * (i + 1))


def find_data_dir() -> Path:
    for cand in [Path("data"), Path("Assignment 2/data"), Path("/content/data")]:
        if (cand / "training_set_task2.txt").exists():
            return cand
    target = Path("data"); target.mkdir(exist_ok=True)
    for stem in SPLIT_FILES.values():
        for task in (1, 2):
            download(f"{CORPUS_URL}/data/{stem}_task{task}.txt", target / f"{stem}_task{task}.txt")
    download(f"{CORPUS_URL}/techniques_list_task1-2.txt", target / "techniques_list_task1-2.txt")
    return target


DATA_DIR = find_data_dir()
raw = {split: json.loads((DATA_DIR / f"{stem}_task2.txt").read_text(encoding="utf8"))
       for split, stem in SPLIT_FILES.items()}
raw_t1 = {split: json.loads((DATA_DIR / f"{stem}_task1.txt").read_text(encoding="utf8"))
          for split, stem in SPLIT_FILES.items()}

# The 20 text techniques, in the official order (Transfer and Appeal to (Strong) Emotions are image-only).
TECHNIQUES = [
    "Appeal to authority", "Appeal to fear/prejudice", "Black-and-white Fallacy/Dictatorship",
    "Causal Oversimplification", "Doubt", "Exaggeration/Minimisation", "Flag-waving",
    "Glittering generalities (Virtue)", "Loaded Language", "Misrepresentation of Someone's Position (Straw Man)",
    "Name calling/Labeling", "Obfuscation, Intentional vagueness, Confusion",
    "Presenting Irrelevant Data (Red Herring)", "Reductio ad hitlerum", "Repetition", "Slogans", "Smears",
    "Thought-terminating cliché", "Whataboutism", "Bandwagon",
]
T2I = {t: i for i, t in enumerate(TECHNIQUES)}
K = len(TECHNIQUES)
SHORT = {  # compact names for plots
    "Appeal to authority": "Authority", "Appeal to fear/prejudice": "Fear/prejudice",
    "Black-and-white Fallacy/Dictatorship": "Black&white", "Causal Oversimplification": "Causal oversimpl.",
    "Doubt": "Doubt", "Exaggeration/Minimisation": "Exagg./Minim.", "Flag-waving": "Flag-waving",
    "Glittering generalities (Virtue)": "Glittering gen.", "Loaded Language": "Loaded language",
    "Misrepresentation of Someone's Position (Straw Man)": "Straw man", "Name calling/Labeling": "Name calling",
    "Obfuscation, Intentional vagueness, Confusion": "Obfuscation",
    "Presenting Irrelevant Data (Red Herring)": "Red herring", "Reductio ad hitlerum": "Ad hitlerum",
    "Repetition": "Repetition", "Slogans": "Slogans", "Smears": "Smears",
    "Thought-terminating cliché": "Thought-term.", "Whataboutism": "Whataboutism", "Bandwagon": "Bandwagon",
}


def to_frame(split: str) -> pd.DataFrame:
    rows = []
    for m in raw[split]:
        spans = [(l["start"], l["end"], l["technique"]) for l in m["labels"]]
        y = np.zeros(K, dtype=np.float32)
        for _, _, t in spans:
            y[T2I[t]] = 1
        rows.append({"id": m["id"], "split": split, "text": m["text"], "spans": spans, "y": y})
    return pd.DataFrame(rows)


df_all = pd.concat([to_frame(s) for s in SPLIT_FILES], ignore_index=True)
print(df_all.groupby("split").size().to_dict(), "| unknown techniques:",
      {l["technique"] for s in raw for m in raw[s] for l in m["labels"]} - set(TECHNIQUES))
