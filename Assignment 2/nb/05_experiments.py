# %% [markdown]
# ## 4. Experiments on the development pool (5-fold cross-validation)
#
# **Protocol.** The experiments run in a fixed order, and each fixes one design dimension for the ones after it
# (greedy, coordinate-wise search). A full grid is out of reach (3 encoders × 3 adaptations × 4 architectures × … ≈ 500
# configurations × 5 folds), and the dimensions are largely separable. Each experiment states before it runs:
# * **H:** the hypothesis, i.e. what we expect;
# * **Decision rule:** how the result feeds the next step.
#
# **Selection score** = mean of out-of-fold label micro-F1 and span F1, each at its best global threshold. The tool
# needs both outputs, so neither alone decides.
#
# **Tie rule:** a candidate replaces the current choice only if it wins on the selection score and its fold-paired
# advantage is larger than the fold-to-fold standard deviation of that advantage. Otherwise the simpler or cheaper
# option is kept. With 750 memes, differences of about one F1 point are within noise, so this rule prevents tuning on
# noise.
#
# **Robustness check in every run.** Each validation fold is also scored after lower-casing and after sentence-casing
# the text (labels and offsets unchanged). This shows how each choice behaves on text that looks different from the
# 2020 corpus (E8, C4).

# %%
_TOKENIZERS = {}


def get_tok(encoder):
    if encoder not in _TOKENIZERS:
        _TOKENIZERS[encoder] = AutoTokenizer.from_pretrained(ENCODERS[encoder]["hf"])
    return _TOKENIZERS[encoder]


ROBUST_VIEWS = {"lower": str.lower, "sentence": lambda t: t[:1].upper() + t[1:].lower()}
RESULTS = {}


def run_cv(name, keep_models=False, **overrides):
    """5-fold CV of one configuration. Stores out-of-fold predictions and fold-level scores in RESULTS[name]."""
    cfg = {**DEFAULT_CFG, **overrides}
    if name in RESULTS and not keep_models:
        return RESULTS[name]
    tok = get_tok(cfg["encoder"])
    n = len(df_dev)
    label_p = np.zeros((n, K)); token_p = [None] * n; offsets = [None] * n
    robust = {v: np.zeros((n, K)) for v in ROBUST_VIEWS}
    histories, models, t0 = [], [], time.time()
    for f in range(N_FOLDS):
        tr, va = df_dev[df_dev.fold != f], df_dev[df_dev.fold == f]
        out = train_fold(cfg, tr, va, tok, keep_model=True)
        idx = np.where(df_dev.fold == f)[0]
        label_p[idx] = out["label_p"]
        for j, i in enumerate(idx):
            token_p[i], offsets[i] = out["token_p"][j], out["offsets"][j]
        model = out["model"].to(DEVICE)
        for v, fn in ROBUST_VIEWS.items():
            va_v = va.assign(text=va.text.map(fn))
            dl = torch.utils.data.DataLoader(MemeDataset(va_v, tok), batch_size=64, collate_fn=make_collate(tok.pad_token_id))
            robust[v][idx] = predict(model, dl, ENCODERS[cfg["encoder"]]["amp"])[0]
        if keep_models:
            models.append(model.cpu())
        del model, out["model"]; gc.collect()
        histories.append(pd.DataFrame(out["history"]).assign(fold=f))
    thr_l, _ = best_label_thr(Y_dev, label_p)
    thr_s, _ = best_span_thr(list(df_dev.spans), token_p, offsets, list(df_dev.text))
    pred_spans = spans_from_probs(token_p, offsets, thr_s, list(df_dev.text)) if thr_s is not None else [[] for _ in range(n)]
    folds = []
    for f in range(N_FOLDS):
        m = (df_dev.fold == f).values
        row = {"fold": f, **label_scores(Y_dev[m], label_p[m] >= thr_l)}
        row["span_f1"] = span_scores([s for s, k in zip(df_dev.spans, m) if k],
                                     [s for s, k in zip(pred_spans, m) if k])["span_f1"] if thr_s is not None else np.nan
        for v in ROBUST_VIEWS:
            row[f"micro_f1_{v}"] = f1_score(Y_dev[m], robust[v][m] >= thr_l, average="micro", zero_division=0)
        folds.append(row)
    folds = pd.DataFrame(folds)
    folds["select"] = folds[["micro_f1", "span_f1"]].mean(axis=1, skipna=True)
    overall = {**label_scores(Y_dev, label_p >= thr_l),
               "span_f1": span_scores(list(df_dev.spans), pred_spans)["span_f1"] if thr_s is not None else np.nan}
    overall["select"] = np.nanmean([overall["micro_f1"], overall["span_f1"]])
    for v in ROBUST_VIEWS:
        overall[f"micro_f1_{v}"] = f1_score(Y_dev, robust[v] >= thr_l, average="micro", zero_division=0)
    RESULTS[name] = {"cfg": cfg, "label_p": label_p, "token_p": token_p, "offsets": offsets, "robust": robust,
                     "thr_label": thr_l, "thr_span": thr_s, "pred_spans": pred_spans, "folds": folds,
                     "overall": overall, "history": pd.concat(histories), "minutes": (time.time() - t0) / 60,
                     "n_trainable": out["n_trainable"], "n_total": out["n_total"], "models": models}
    o = overall
    print(f"{name:28s} micro={o['micro_f1']:.3f} macro={o['macro_f1']:.3f} span={o['span_f1']:.3f} "
          f"select={o['select']:.3f} | lower={o['micro_f1_lower']:.3f} | {RESULTS[name]['minutes']:.1f} min")
    return RESULTS[name]


def compare(names, reference):
    """Summary table; 'Δ select' is the fold-paired difference to `reference` (mean ± sd over folds)."""
    ref = RESULTS[reference]["folds"]["select"].values
    rows = []
    for nm in names:
        r = RESULTS[nm]; d = r["folds"]["select"].values - ref
        rows.append({"config": nm, "micro_f1": r["overall"]["micro_f1"], "macro_f1": r["overall"]["macro_f1"],
                     "span_f1": r["overall"]["span_f1"], "select": r["overall"]["select"],
                     "Δ select (paired)": f"{d.mean():+.3f} ± {d.std(ddof=1) if len(d) > 1 else 0:.3f}",
                     "folds won": f"{int((d > 0).sum())}/{len(d)}" if nm != reference else "–",
                     "micro_f1 lower-cased": r["overall"]["micro_f1_lower"],
                     "trainable params (M)": r["n_trainable"] / 1e6, "minutes": r["minutes"]})
    return pd.DataFrame(rows).set_index("config")


def choose(current, candidates):
    """Tie rule: switch only if the candidate's paired advantage exceeds its fold-to-fold sd."""
    best = current
    for c in candidates:
        d = RESULTS[c]["folds"]["select"].values - RESULTS[best]["folds"]["select"].values
        if d.mean() > 0 and d.mean() > d.std(ddof=1):
            best = c
    return best


def show(names, reference, title):
    tab = compare(names, reference)
    display(tab.style.format({c: "{:.3f}" for c in ["micro_f1", "macro_f1", "span_f1", "select", "micro_f1 lower-cased"]}
                             | {"trainable params (M)": "{:.1f}", "minutes": "{:.1f}"}).set_caption(title))


# %% [markdown]
# ### E1: Which pretrained encoder?
# Candidates (all ~125–185M parameters, comfortably T4-sized):
# * **RoBERTa-base:** strong general baseline.
# * **Twitter-RoBERTa-2021:** RoBERTa further pre-trained, *without labels*, on tweets from 2020–2021. That is the
#   same era and a similar register to the memes, and it was never trained on persuasion labels.
# * **DeBERTa-v3-base:** usually the strongest base-size encoder on classification benchmarks.
#
# **H:** the social-media encoder helps, because meme language is informal, shouty and topical (COVID, elections).
# DeBERTa may match it through general strength. **Decision rule:** the winner under the tie rule becomes the encoder
# for all later experiments. Ties go to RoBERTa-base (fastest, fp16-stable).

# %%
run_cv("E1 roberta-base", encoder="roberta-base")
run_cv("E1 twitter-roberta-2021", encoder="twitter-roberta-2021")
run_cv("E1 deberta-v3-base", encoder="deberta-v3-base")
show(["E1 roberta-base", "E1 twitter-roberta-2021", "E1 deberta-v3-base"], "E1 roberta-base", "E1: encoder")
BEST = choose("E1 roberta-base", ["E1 twitter-roberta-2021", "E1 deberta-v3-base"])
print("→ chosen:", BEST)


# %% [markdown]
# **E1 result.** DeBERTa-v3 wins on all five folds (paired +0.038 ± 0.033 on the selection score). The gain comes
# mainly from span F1 (0.401 vs. 0.338 for RoBERTa). The tweet-pretrained RoBERTa did **not** help (−0.005 ± 0.016,
# won 2/5 folds), so the hypothesis is rejected: pretraining text from the same era and register did not beat a
# stronger general encoder. DeBERTa has 185M parameters instead of 125M, which is still small for a T4.
