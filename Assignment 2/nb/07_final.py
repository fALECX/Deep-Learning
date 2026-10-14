# %% [markdown]
# ## 5. The final model on the held-out test set
#
# **Final system.** The selected configuration is trained once per fold, and the **five fold models are averaged**
# (a 5-model ensemble). Why not retrain one model on all 750 memes?
# * The ensemble's thresholds were tuned on exactly the kind of model they are applied to (out-of-fold predictions
#   of fold models). A retrained model would have no held-out data left to calibrate them.
# * On 600-meme training sets, single runs vary noticeably with initialisation and data order. Averaging five models
#   reduces that variance (Section 4, seed check).
# * The latency cost is measured below and is acceptable on a T4.
#
# The test set is used **here for the first time**, with every choice already fixed.

# %%
FINAL = run_cv("FINAL", keep_models=True, **FINAL_CFG)
LABEL_THR = fit_thresholds(Y_dev, FINAL["label_p"], PER_CLASS_THR)
SPAN_THR = FINAL["thr_span"]
FINAL_TOK = get_tok(FINAL_CFG["encoder"])
AMP = ENCODERS[FINAL_CFG["encoder"]]["amp"]


def ensemble_predict(df, models=None):
    models = models or FINAL["models"]
    dl = torch.utils.data.DataLoader(MemeDataset(df, FINAL_TOK), batch_size=64, collate_fn=make_collate(FINAL_TOK.pad_token_id))
    lps, tps, atts = [], [], []
    for m in models:
        m.to(DEVICE)
        lp, tp, offs, att = predict(m, dl, AMP)
        lps.append(lp); tps.append(tp); atts.append(att)
        m.cpu()
    label_p = np.mean(lps, 0)
    token_p = [np.mean([tp[i] for tp in tps], 0) if tps[0][i] is not None else None for i in range(len(df))]
    att = [np.mean([a[i] for a in atts], 0) if atts[0][i] is not None else None for i in range(len(df))]
    return {"label_p": label_p, "labels": (label_p >= LABEL_THR).astype(int), "token_p": token_p, "offsets": offs,
            "spans": spans_from_probs(token_p, offs, SPAN_THR, list(df.text)) if SPAN_THR is not None else [[] for _ in range(len(df))],
            "att": att}


TEST = ensemble_predict(df_test)
Y_test = np.stack(df_test.y)
gold_test = list(df_test.spans)

# %%
def test_row(name, Yp, spans):
    s = {**label_scores(Y_test, Yp), **span_scores(gold_test, spans)}
    lo, hi = bootstrap_ci(lambda a, b: f1_score(a, b, average="micro", zero_division=0), Y_test, Yp)
    s["micro_f1 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
    lo, hi = bootstrap_ci(lambda g, p: span_scores(g, p)["span_f1"], gold_test, spans)
    s["span_f1 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
    return pd.Series(s, name=name)


maj = np.zeros_like(Y_test); maj[:, T2I["Loaded Language"]] = 1
single = ensemble_predict(df_test, FINAL["models"][:1])
test_tab = pd.DataFrame([
    test_row("final ensemble (5 models)", TEST["labels"], TEST["spans"]),
    test_row("single fold model (for reference)", single["labels"], single["spans"]),
    test_row("majority baseline", maj, [whole_text_spans(t, ["Loaded Language"]) for t in df_test.text]),
])
test_tab.loc["SemEval-2021 best team (Dimitrov et al.)"] = pd.Series({"micro_f1": .593, "macro_f1": .290, "span_f1": .482})
display(test_tab[["micro_f1", "micro_f1 95% CI", "macro_f1", "micro_p", "micro_r", "span_f1", "span_f1 95% CI",
                  "span_p", "span_r"]].style.format("{:.3f}", na_rep="–", subset=["micro_f1", "macro_f1", "micro_p",
                  "micro_r", "span_f1", "span_p", "span_r"]).set_caption("Held-out official test set (200 memes)"))


# %% [markdown]
# **Test result.**
# * Labels: micro-F1 **0.519** (95% CI 0.469–0.565), macro-F1 0.221. Precision 0.48, recall 0.57.
# * Spans: span F1 **0.405** (95% CI 0.357–0.442).
# * This is clearly above the majority baseline (0.374, and 0.165 for spans). It is below the SemEval-2021 winner
#   (0.593 / 0.482), which used PTC-pretrained ensembles that our rules forbid.
# * It is lower than the cross-validation estimate (0.562). The CV numbers were used to select the configuration and
#   thresholds, so they are optimistic, and the test memes are harder for the rare techniques. 0.52 is the number to
#   expect on memes like these.
# * The ensemble beats a single fold model (0.495 / 0.376).

# %% [markdown]
# ### Per-technique behaviour

# %%
sup = Y_test.sum(0).astype(int)
pt = pd.DataFrame({
    "test support": sup,
    "precision": precision_score(Y_test, TEST["labels"], average=None, zero_division=0),
    "recall": recall_score(Y_test, TEST["labels"], average=None, zero_division=0),
    "F1": f1_score(Y_test, TEST["labels"], average=None, zero_division=0),
    "dev support": Y_dev.sum(0).astype(int),
}, index=[SHORT[t] for t in TECHNIQUES]).sort_values("dev support", ascending=False)
fig, ax = plt.subplots(figsize=(12, 4.2))
x = np.arange(len(pt))
ax.bar(x - 0.27, pt.precision, 0.27, label="precision", color="#4C72B0")
ax.bar(x, pt.recall, 0.27, label="recall", color="#DD8452")
ax.bar(x + 0.27, pt.F1, 0.27, label="F1", color="#55A868")
ax.set_xticks(x, [f"{i}\n(n={n})" for i, n in zip(pt.index, pt["test support"])], rotation=60, ha="right", fontsize=8)
ax.set_title("Test set, per technique (sorted by number of development examples)"); ax.legend()
plt.tight_layout(); plt.show()
SUPPORTED = [t for t in TECHNIQUES if Y_dev[:, T2I[t]].sum() >= MIN_POS]
print("techniques with ≥", MIN_POS, "development examples:", [SHORT[t] for t in SUPPORTED])


# %% [markdown]
# **Reading.** Only a handful of techniques are usable: Loaded Language (F1 0.77), Name calling (0.57), Whataboutism
# (0.50), Exaggeration (0.42), Smears (0.45) and Appeal to authority (0.53, but only 7 test examples). Slogans gets
# F1 0 on test despite 47 development examples. All techniques with fewer than 10 development examples are never
# found. For the classroom this means the tool can support a lesson on a few core techniques, not the whole taxonomy.

# %% [markdown]
# ### Which techniques get confused with which?
# In multi-label output, "confusion" means: when the gold technique *a* is **missed**, which wrong technique *b* was
# predicted for the same meme instead?

# %%
conf = np.zeros((K, K))
for yt, yp in zip(Y_test, TEST["labels"]):
    for a in np.where((yt == 1) & (yp == 0))[0]:
        for b in np.where((yp == 1) & (yt == 0))[0]:
            conf[a, b] += 1
top = np.argsort(-(conf.sum(1) + conf.sum(0)))[:10]
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.imshow(conf[np.ix_(top, top)], cmap="Reds")
for i, a in enumerate(top):
    for j, b in enumerate(top):
        if conf[a, b]:
            ax.text(j, i, int(conf[a, b]), ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(top)), [SHORT[TECHNIQUES[i]] for i in top], rotation=60, ha="right")
ax.set_yticks(range(len(top)), [SHORT[TECHNIQUES[i]] for i in top])
ax.set_xlabel("wrongly predicted instead"); ax.set_ylabel("gold technique that was missed")
ax.set_title("Missed technique → spurious technique on the same meme (test)")
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading.** The errors have a pattern. When the model misses a subtler technique (Slogans, Doubt, Exaggeration,
# Fear), it usually predicts one of the three frequent "attack" techniques instead (Smears, Name calling, Loaded
# Language). The model has learned that a meme is hostile, but not always *how* it argues.

# %% [markdown]
# ### Reading individual outputs
# A sample of test memes with gold and predicted techniques. Spans are shown for one technique each, to keep the
# table readable.

# %%
def fmt_labels(y):
    return ", ".join(SHORT[TECHNIQUES[i]] for i in np.where(y)[0]) or "—"


ex = df_test.assign(gold=[fmt_labels(y) for y in Y_test], predicted=[fmt_labels(y) for y in TEST["labels"]],
                    jaccard=[(a & b).sum() / (a | b).sum() if (a | b).any() else 1.0
                             for a, b in zip(Y_test.astype(bool), TEST["labels"].astype(bool))],
                    predicted_spans=["; ".join(f"{SHORT[t]}: «{txt[s:e][:40]}»" for s, e, t in sp[:3])
                                     for txt, sp in zip(df_test.text, TEST["spans"])])
pick = lambda d, n: d.sample(min(n, len(d)), random_state=0)
sample = pd.concat([pick(ex[ex.jaccard == 1], 3), pick(ex[(ex.jaccard > 0) & (ex.jaccard < 1)], 3), pick(ex[ex.jaccard == 0], 4)])
display(sample[["text", "gold", "predicted", "predicted_spans"]].assign(text=sample.text.str.slice(0, 110)))

# %% [markdown]
# ### Can it run on the school's machine? (size and latency)
# The department's machine is "no better than one T4 (16 GB)". The ensemble is five base-size encoders. We measure
# parameter memory and the wall-clock latency for one meme (batch size 1, the classroom case) on the hardware this
# notebook runs on. A T4 is the Colab default, so on Colab the numbers are measured on the target hardware.

# %%
def latency(models, texts, reps=30):
    ms = []
    for m in models:
        m.to(DEVICE).eval()
    with torch.no_grad():
        for i in range(reps + 3):
            enc = FINAL_TOK([texts[i % len(texts)]], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            tm = enc["attention_mask"].bool()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for m in models:
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP and DEVICE.type == "cuda"):
                    m(enc["input_ids"], enc["attention_mask"], tm)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            if i >= 3:
                ms.append((time.perf_counter() - t0) * 1000)
    for m in models:
        m.cpu()
    return np.median(ms), np.percentile(ms, 95)


params = sum(p.numel() for p in FINAL["models"][0].parameters())
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
lat1 = latency(FINAL["models"][:1], list(df_test.text))
lat5 = latency(FINAL["models"], list(df_test.text))
peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else float("nan")
display(pd.DataFrame({
    "parameters per model (M)": [params / 1e6] * 2,
    "weights in fp16 (GB)": [params * 2 / 1e9, 5 * params * 2 / 1e9],
    "median latency per meme (ms)": [lat1[0], lat5[0]], "p95 latency (ms)": [lat1[1], lat5[1]],
}, index=["single model", "5-model ensemble"]).round(2))
print(f"device: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}; peak GPU memory during the benchmark: {peak:.2f} GB")


# %% [markdown]
# **Deployment fit.** Measured on a Tesla T4: one model answers in **28 ms**, the five-model ensemble in **142 ms**
# per meme. The ensemble's weights take 1.85 GB in fp16. That is far inside a 16 GB T4 and instant from a
# student's point of view. (The 7.6 GB peak includes the models kept in memory for the whole notebook.)
