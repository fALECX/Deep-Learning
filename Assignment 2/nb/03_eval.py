# %% [markdown]
# ## 2. Evaluation framework
#
# ### 2.1 What we measure, and why
# **Part A (labels).**
# * Micro-F1 is the official SemEval metric, so we are comparable with the 2021 leaderboard (best .593, majority
#   baseline .374).
# * Macro-F1 weights the 20 techniques equally. For teaching, a missed *Straw man* matters as much as a missed
#   *Loaded Language*, and macro-F1 exposes a model that ignores the rare techniques.
# * Per-technique F1 tells the teacher which labels to trust.
#
# Precision and recall are reported separately. In a classroom a false accusation ("this meme smears someone") does
# more damage than a missed technique, so precision gets explicit attention.
#
# **Part B (spans).** We use the official SemEval-2021 subtask-2 measure. It gives partial credit for character
# overlap, and only when the technique matches. With predicted spans $p$ and gold spans $g$ of the same technique,
# $P=\frac{1}{|\text{pred}|}\sum_{p}\sum_{g}\frac{|p\cap g|}{|p|}$ and
# $R=\frac{1}{|\text{gold}|}\sum_{g}\sum_{p}\frac{|p\cap g|}{|g|}$. F1 is their harmonic mean.
# We re-implement it so it can run inside the training loop, and check it against the official scorer below.

# %%
def label_scores(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    return {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "micro_p": precision_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_r": recall_score(Y_true, Y_pred, average="micro", zero_division=0),
    }


def per_technique_f1(Y_true, Y_pred) -> pd.Series:
    return pd.Series(f1_score(Y_true, Y_pred, average=None, zero_division=0, labels=range(K)),
                     index=[SHORT[t] for t in TECHNIQUES])


def span_scores(gold: list, pred: list) -> dict:
    """Official SemEval-2021 subtask-2 measure. gold/pred: per meme a list of (start, end, technique)."""
    p_num = r_num = 0.0
    n_pred = sum(len(p) for p in pred); n_gold = sum(len(g) for g in gold)
    for g_spans, p_spans in zip(gold, pred):
        for ps, pe, pt in p_spans:
            for gs, ge, gt in g_spans:
                if pt == gt:
                    inter = max(0, min(pe, ge) - max(ps, gs))
                    p_num += inter / (pe - ps)
                    r_num += inter / (ge - gs)
    p = p_num / n_pred if n_pred else 0.0
    r = r_num / n_gold if n_gold else 0.0
    f1 = 1.0 if n_pred == 0 and n_gold == 0 else (2 * p * r / (p + r) if p > 0 and r > 0 else 0.0)
    return {"span_f1": f1, "span_p": p, "span_r": r}


# %% [markdown]
# ### 2.2 Tokenisation and span ↔ token alignment
# The gold spans are character offsets, and transformers see sub-word tokens. A token is labelled *inside
# technique k* if any of its (whitespace-trimmed) characters fall inside a gold span of *k*. To decode, each
# maximal run of tokens above the threshold for *k* becomes one predicted span, from the first token's first
# character to the last token's last character. Some tokenisers (e.g. DeBERTa) include the leading space in a
# token's offsets, so offsets are trimmed first.

# %%
MAX_LEN = 160  # longest meme is 142 RoBERTa tokens


def trimmed_offsets(text, offsets):
    out = []
    for a, b in offsets:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        out.append((a, b))
    return out


def token_labels(text, offsets, spans):
    """offsets already trimmed; returns (T, K) float array, special tokens (a == b) all-zero."""
    lab = np.zeros((len(offsets), K), dtype=np.float32)
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        for s, e, t in spans:
            if a < e and s < b:
                lab[i, T2I[t]] = 1
    return lab


def decode_spans(offsets, probs, thr, text=None):
    """probs: (T, K) token probabilities; thr: scalar or (K,) thresholds -> list of (start, end, technique).
    If the text is given, spans without any letter or digit (a lone '.' or '!') are dropped: nothing to point at."""
    thr = np.broadcast_to(np.asarray(thr, dtype=np.float32), (K,))
    spans = []
    for k in range(K):
        start = end = None
        for (a, b), p in zip(offsets, probs[:, k]):
            if a == b:          # special / padding token
                continue
            if p > thr[k]:
                start = a if start is None else start
                end = b
            elif start is not None:
                spans.append((start, end, TECHNIQUES[k])); start = None
        if start is not None:
            spans.append((start, end, TECHNIQUES[k]))
    if text is not None:
        spans = [sp for sp in spans if any(c.isalnum() for c in text[sp[0]:sp[1]])]
    return spans


def tokenize_frame(df, tok):
    enc = tok(list(df.text), truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    offs = [trimmed_offsets(t, o) for t, o in zip(df.text, enc["offset_mapping"])]
    labs = [token_labels(t, o, s) for t, o, s in zip(df.text, offs, df.spans)]
    return {"input_ids": enc["input_ids"], "offsets": offs, "token_labels": labs}


# %% [markdown]
# **Two checks before trusting any number.**
# 1. *Representation ceiling.* Convert every gold span to token IO tags with the RoBERTa tokeniser, decode back,
#    and score against the original spans. This is the best Part-B score any IO tagger could reach. It quantifies
#    what the IO decision from Section 1 costs.
# 2. *Metric correctness.* Score the same predictions with our implementation and with the official scorer
#    (`propaganda-techniques-scorer`, the commit pinned by the SemEval repository).

# %%
TOK_CHECK = AutoTokenizer.from_pretrained("roberta-base")
enc_dev = tokenize_frame(df_dev, TOK_CHECK)
roundtrip = [decode_spans(o, l, 0.5) for o, l in zip(enc_dev["offsets"], enc_dev["token_labels"])]
print("IO representation ceiling on the development pool:",
      {k: round(v, 4) for k, v in span_scores(list(df_dev.spans), roundtrip).items()})


def official_span_score(gold_df, pred_spans):
    """Runs the official SemEval-2021 task-2 scorer; returns F1 or None if it cannot be fetched."""
    repo = Path("propaganda-techniques-scorer")
    try:
        if not repo.exists():
            subprocess.run(["git", "clone", "-q", "https://gitlab.com/joedsm/propaganda-techniques-scorer.git",
                            str(repo)], check=True, timeout=120)
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", "05fe8d38c4a729dfea749e401456ef35b8476cf4"],
                           check=True)
        to_json = lambda spans_list: [{"id": i, "labels": [{"start": s, "end": e, "technique": t} for s, e, t in sp]}
                                      for i, sp in zip(gold_df.id, spans_list)]
        Path("_gold.json").write_text(json.dumps(to_json(list(gold_df.spans))), encoding="utf8")
        Path("_pred.json").write_text(json.dumps(to_json(pred_spans)), encoding="utf8")
        Path("_techniques.txt").write_text("\n".join(TECHNIQUES) + "\n", encoding="utf8")
        out = subprocess.run([sys.executable, "task-2-semeval21_scorer.py", "-s", "../_pred.json", "-r", "../_gold.json",
                              "-p", "../_techniques.txt", "-o"], cwd=repo, capture_output=True, text=True, timeout=300,
                             env={**os.environ, "PYTHONUTF8": "1"})
        return float(out.stdout.strip().split("\t")[0])
    except Exception as exc:  # no network / git: the check is skipped, not faked
        print("official scorer unavailable:", type(exc).__name__, exc)
        return None


# Sanity predictions with partial overlaps: shift every gold span by 3 characters and drop every third one.
perturbed = [[(s + 3, e + 3, t) for j, (s, e, t) in enumerate(sp) if j % 3] for sp in df_dev.spans]
for name, pred in [("IO round-trip", roundtrip), ("shifted+dropped", perturbed)]:
    ours = span_scores(list(df_dev.spans), pred)["span_f1"]
    print(f"{name:16s} ours={ours:.6f} official={official_span_score(df_dev, pred)}")

# %% [markdown]
# ### 2.3 Data splits
# * **Test** (200 memes, official split) is held out and used once, in Section 5. Every choice before that (encoder,
#   adaptation, head, loss, hyper-parameters, thresholds) is made on the development pool; otherwise the test number
#   would be optimistically biased.
# * The official dev set (63 memes) is too small for model selection: one meme changes micro-F1 by about 1 point, and
#   several techniques have no dev example. We therefore pool train and dev (750 memes) and use 5-fold
#   cross-validation. Model comparisons are then made on 750 out-of-fold predictions instead of 63, and they are
#   paired: all configurations see identical folds.
# * Folds are built with iterative stratification (Sechidis et al., 2011), which balances all 20 labels across folds
#   at once, so rare techniques appear in every fold where possible.
#
# The test set is still in-distribution (same Facebook groups, months and annotators). Its score estimates
# performance on 2020 US/Canadian Facebook memes, not on an Australian Year 10 feed. That gap is measured separately
# in C4.

# %%
def iterative_stratification(Y: np.ndarray, n_folds: int, seed: int = SEED) -> np.ndarray:
    """Sechidis et al. (2011): assign examples so every label is spread evenly over folds. Returns fold id per row."""
    rng = np.random.RandomState(seed)
    n = len(Y)
    fold = -np.ones(n, dtype=int)
    desired = np.full(n_folds, n / n_folds)
    desired_label = np.tile(Y.sum(0) / n_folds, (n_folds, 1))
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        counts = Y[remaining].sum(0)
        if counts.sum() == 0:  # rows without labels: fill folds by remaining capacity
            for i in rng.permutation(np.where(remaining)[0]):
                f = rng.choice(np.flatnonzero(desired == desired.max())); fold[i] = f; desired[f] -= 1
            break
        lab = np.argmin(np.where(counts > 0, counts, np.inf))       # rarest label still unassigned
        for i in rng.permutation(np.where(remaining & (Y[:, lab] == 1))[0]):
            cand = np.flatnonzero(desired_label[:, lab] == desired_label[:, lab].max())
            if len(cand) > 1:
                cand = cand[desired[cand] == desired[cand].max()]
            f = rng.choice(cand)
            fold[i] = f; remaining[i] = False
            desired[f] -= 1; desired_label[f] -= Y[i]
    return fold


N_FOLDS = 2 if FAST_DEV_RUN else 5
Y_dev = np.stack(df_dev.y)
df_dev["fold"] = iterative_stratification(Y_dev, N_FOLDS)
fold_tab = pd.DataFrame({f: Y_dev[df_dev.fold == f].sum(0) for f in range(N_FOLDS)},
                        index=[SHORT[t] for t in TECHNIQUES]).astype(int)
fold_tab.insert(0, "total", Y_dev.sum(0).astype(int))
display(fold_tab.T.style.set_caption("Memes per technique in each fold (iterative stratification)"))


# %% [markdown]
# ### 2.4 Uncertainty
# With 200 test memes a difference of a few F1 points can be noise. Test-set numbers therefore get a **95% bootstrap
# confidence interval** (1,000 resamples of memes). Model comparisons on the development pool use the **fold-paired
# difference** (mean ± standard deviation over the five folds). A configuration is called better only if it wins
# consistently, not just on average.

# %%
def bootstrap_ci(metric_fn, *arrays, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(arrays[0]); vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(metric_fn(*[[a[i] for i in idx] if isinstance(a, list) else a[idx] for a in arrays]))
    return np.percentile(vals, [2.5, 97.5])


# %% [markdown]
# ### 2.5 Trivial baselines (the floor any model must clear)
# * **Majority:** every meme gets *Loaded Language*, and for spans the whole text is tagged as Loaded Language.
# * **Prior-random:** each technique is predicted independently with its training frequency (averaged over 50 draws).
#   For spans, the whole text gets the sampled techniques.
#
# Both are computed out-of-fold: the frequencies come from the fold's training part only.

# %%
def whole_text_spans(text, techs):
    s = len(text) - len(text.lstrip()); e = len(text.rstrip())
    return [(s, e, t) for t in techs] if e > s else []


rng = np.random.RandomState(SEED)
base_rows = []
for f in range(N_FOLDS):
    tr, va = df_dev[df_dev.fold != f], df_dev[df_dev.fold == f]
    Yva = np.stack(va.y); prior = np.stack(tr.y).mean(0)
    maj = np.zeros_like(Yva); maj[:, T2I["Loaded Language"]] = 1
    maj_sp = [whole_text_spans(t, ["Loaded Language"]) for t in va.text]
    base_rows.append({"baseline": "majority", "fold": f, **label_scores(Yva, maj), **span_scores(list(va.spans), maj_sp)})
    for d in range(50):
        rnd = (rng.rand(*Yva.shape) < prior).astype(int)
        rnd_sp = [whole_text_spans(t, [TECHNIQUES[k] for k in np.where(r)[0]]) for t, r in zip(va.text, rnd)]
        base_rows.append({"baseline": "prior-random", "fold": f, **label_scores(Yva, rnd),
                          **span_scores(list(va.spans), rnd_sp)})
baselines = pd.DataFrame(base_rows).groupby("baseline")[["micro_f1", "macro_f1", "micro_p", "micro_r", "span_f1"]].mean()
display(baselines.style.format("{:.3f}").set_caption("Baselines, 5-fold out-of-fold on the development pool"))
