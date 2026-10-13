# %% [markdown]
# ### C2. Defend the system you built: zero-shot general-purpose LLMs on the same test memes
#
# **Set-up** (responses collected outside this notebook with `c2_llm/run_llm.py`, embedded below, and parsed and
# scored here):
#
# | | Gemini 3.1 Pro | Gemini 3.1 Flash-Lite | Ornith-1.5-9B (Q4_K_M) |
# |---|---|---|---|
# | Role | strongest general model available to us: the committee's "general-purpose AI" | cheap hosted tier | **deployable alternative**: open weights, runs locally, fits a T4 (5.8 GB) |
# | Access | grsai.ai, Gemini-native `generateContent` API | same | LM Studio on a laptop |
# | Can it be used in schools? | **No.** Hosted API; the meme would leave the device | No | Yes |
#
# * **Date:** 12–13 October 2026. Model names and versions are the `modelVersion` / `model` fields of each response.
# * **Sampling:** temperature 1.0 for all models (the recommended default for Gemini 3; lower values are discouraged
#   for this family). Because sampling is stochastic, the main prompt was run **three times** for Gemini Pro, Gemini
#   Flash-Lite and Ornith without reasoning, and we report the spread. The names-only prompt for Pro ran once.
# * **Thinking:** provider default for Gemini. Ornith ran twice: with its reasoning mode switched off
#   (`reasoning_effort = none`, all 200 memes) and switched on (every 4th meme, 50 memes, because each answer takes
#   about 3 minutes on local hardware). A configuration run on a subsample is scored on that subsample only, and our
#   system is compared on the same memes.
# * **Sample:** all **200 test memes** for every other configuration, so the comparison uses the identical memes and
#   identical metrics as our system.
#
# **Making it fair.**
# * **Same information as the human annotators.** The prompt gives the 20 technique names with the paper's own
#   definitions (prompt `def`). A second prompt with names only (`names`) measures how much the definitions matter.
# * **Same output as our system.** The model returns techniques *and the exact words* for each, so both label F1
#   and span F1 can be computed. Fragments are located in the meme text; if they are not verbatim, a fuzzy match is
#   used.
# * **Wrong-shaped answers.** Responses are parsed leniently (code fences and surrounding prose are tolerated).
#   Technique names are mapped to the 20 labels by exact, alias, then fuzzy matching; names outside the taxonomy
#   (e.g. "Transfer") are dropped and counted. A response that cannot be parsed, or a refusal, is reported and scored
#   two ways: as "no techniques" (**strict**: what a deployed tool would show), and by excluding those memes for
#   *both* systems (**lenient**).
# * **Contamination.** The corpus has been public on GitHub since 2021, before these models' training cut-offs, so the
#   memes may have been seen. We test **verbatim memorisation** directly: given the first half of a meme, the model
#   is asked to continue it, and we compare its continuation with the real second half. As a control we compare it
#   with the second half of a *different* meme.
#
# The prompts are printed in full below.

# %%
# Raw responses of all LLM runs (gzip + base64 of JSON, generated from c2_llm/outputs/*.jsonl). One line, ~kB.
C2_RAW_B64 = "__C2_RAW_B64__"

# %%
import base64, gzip, difflib
C2_RAW = json.loads(gzip.decompress(base64.b64decode(C2_RAW_B64)).decode("utf8"))
C2_PROMPTS = C2_RAW["prompts"]
c2 = pd.DataFrame(C2_RAW["records"])
print("records:", len(c2), "| configurations:")
display(c2.groupby(["dataset", "task", "model", "prompt"]).agg(runs=("run", "nunique"), memes=("id", "nunique"),
        errors=("error", lambda e: e.notna().sum()), median_latency_s=("latency_s", "median")))
for k in ["system_def", "system_names", "user_template", "memo_system", "memo_user_template"]:
    print(f"\n===== {k} =====\n{C2_PROMPTS[k]}")

# %%
_norm = lambda s: re.sub(r"[^a-z]+", " ", s.lower().replace("é", "e")).strip()
_ALIASES = {_norm(t): t for t in TECHNIQUES}
_ALIASES.update({_norm(a): t for a, t in [
    ("name calling", "Name calling/Labeling"), ("labeling", "Name calling/Labeling"), ("labelling", "Name calling/Labeling"),
    ("loaded language", "Loaded Language"), ("black and white fallacy", "Black-and-white Fallacy/Dictatorship"),
    ("dictatorship", "Black-and-white Fallacy/Dictatorship"), ("false dilemma", "Black-and-white Fallacy/Dictatorship"),
    ("straw man", "Misrepresentation of Someone's Position (Straw Man)"), ("strawman", "Misrepresentation of Someone's Position (Straw Man)"),
    ("red herring", "Presenting Irrelevant Data (Red Herring)"), ("glittering generalities", "Glittering generalities (Virtue)"),
    ("virtue", "Glittering generalities (Virtue)"), ("thought terminating cliche", "Thought-terminating cliché"),
    ("appeal to fear", "Appeal to fear/prejudice"), ("appeal to prejudice", "Appeal to fear/prejudice"),
    ("exaggeration", "Exaggeration/Minimisation"), ("minimisation", "Exaggeration/Minimisation"),
    ("minimization", "Exaggeration/Minimisation"), ("exaggeration minimization", "Exaggeration/Minimisation"),
    ("flag waving", "Flag-waving"), ("obfuscation", "Obfuscation, Intentional vagueness, Confusion"),
    ("intentional vagueness", "Obfuscation, Intentional vagueness, Confusion"), ("causal oversimplification", "Causal Oversimplification"),
    ("reductio ad hitlerum", "Reductio ad hitlerum"), ("appeal to authority", "Appeal to authority"),
    ("whataboutism", "Whataboutism"), ("bandwagon", "Bandwagon"), ("smears", "Smears"), ("smear", "Smears"),
    ("doubt", "Doubt"), ("slogans", "Slogans"), ("slogan", "Slogans"), ("repetition", "Repetition")]})


def map_technique(name):
    n = _norm(str(name))
    if n in _ALIASES:
        return _ALIASES[n]
    close = difflib.get_close_matches(n, list(_ALIASES), n=1, cutoff=0.8)
    return _ALIASES[close[0]] if close else None


def locate(fragment, text):
    if not fragment:
        return None
    for hay, needle in [(text, fragment), (text.lower(), fragment.lower())]:
        i = hay.find(needle)
        if i >= 0:
            return i, i + len(needle)
    sm = difflib.SequenceMatcher(None, text.lower(), fragment.lower(), autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if blocks and sum(b.size for b in blocks) >= 0.8 * len(fragment):
        return blocks[0].a, blocks[-1].a + blocks[-1].size
    return None


def parse_response(text, meme_text):
    """→ status, label vector, spans, n_unmapped, n_unlocated"""
    y = np.zeros(K, dtype=int)
    if not isinstance(text, str) or not text.strip():
        return "empty/refused", y, [], 0, 0
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        return "no JSON", y, [], 0, 0
    try:
        obj = json.loads(m.group(1))
        items = obj["techniques"]
        assert isinstance(items, list)
    except Exception:
        return "malformed JSON", y, [], 0, 0
    spans, unmapped, unlocated = [], 0, 0
    for it in items:
        if not isinstance(it, dict):
            unmapped += 1; continue
        t = map_technique(it.get("technique", ""))
        if t is None:
            unmapped += 1; continue
        y[T2I[t]] = 1
        loc = locate(str(it.get("fragment", "")), meme_text)
        if loc is None:
            unlocated += 1
        else:
            spans.append((loc[0], loc[1], t))
    return "ok", y, spans, unmapped, unlocated


test_text = dict(zip(df_test.id, df_test.text)); test_pos = {i: n for n, i in enumerate(df_test.id)}
cls = c2[(c2.task == "classify") & (c2.dataset == "test") & c2.id.isin(test_text)].copy()
cls = cls.sort_values("timestamp_utc").groupby(["model", "prompt", "run", "id"]).tail(1)  # last attempt per meme
parsed = cls.apply(lambda r: parse_response(r.text if r.error is None or pd.isna(r.error) else None, test_text[r.id]), axis=1)
cls["status"], cls["y"], cls["spans"], cls["unmapped"], cls["unlocated"] = zip(*parsed)
status_tab = cls.groupby(["model", "prompt", "run"]).agg(
    ok=("status", lambda s: (s == "ok").sum()), failed=("status", lambda s: (s != "ok").sum()),
    unmapped_names=("unmapped", "sum"), fragments_not_found=("unlocated", "sum"))
display(status_tab.style.set_caption("Response shape: how many answers could be used?"))

# %%
def score_config(g, lenient):
    """Strict: every meme the model was asked about counts, failed answers as 'no technique'.
    Lenient: only memes with a usable answer. A configuration run on a subsample is scored on that subsample."""
    g = g.set_index("id").reindex(df_test.id)
    asked = g.status.notna().values
    ok = (g.status == "ok").values
    Yp = np.stack([y if isinstance(y, np.ndarray) else np.zeros(K, int) for y in g.y])
    sp = [s if isinstance(s, list) else [] for s in g.spans]
    keep = ok if lenient else asked
    return {**label_scores(Y_test[keep], Yp[keep]),
            **span_scores([gold_test[i] for i in np.where(keep)[0]], [sp[i] for i in np.where(keep)[0]]),
            "memes": int(keep.sum())}, Yp, sp, keep


rows, PRED_LLM = [], {}
for (model, prompt, run), g in cls.groupby(["model", "prompt", "run"]):
    for lenient in (False, True):
        s, Yp, sp, keep = score_config(g, lenient)
        rows.append({"system": model, "prompt": prompt, "run": run, "scoring": "lenient" if lenient else "strict", **s})
        if not lenient:
            PRED_LLM[(model, prompt, run)] = (Yp, sp, keep)
        ours = {**label_scores(Y_test[keep], TEST["labels"][keep]),
                **span_scores([gold_test[i] for i in np.where(keep)[0]], [TEST["spans"][i] for i in np.where(keep)[0]])}
        rows.append({"system": "OURS (same memes)", "prompt": f"vs {model}/{prompt}", "run": run,
                     "scoring": "lenient" if lenient else "strict", **ours, "memes": int(keep.sum())})
c2_all = pd.DataFrame(rows)
c2_sum = (c2_all[c2_all.system != "OURS (same memes)"]
          .groupby(["system", "prompt", "scoring"])[["micro_f1", "macro_f1", "micro_p", "micro_r", "span_f1"]]
          .agg(lambda x: f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"))
ours_row = pd.DataFrame([{**label_scores(Y_test, TEST["labels"]), **span_scores(gold_test, TEST["spans"])}],
                        index=pd.MultiIndex.from_tuples([("OURS: final ensemble", "–", "–")], names=c2_sum.index.names))
display(pd.concat([ours_row.map(lambda v: f"{v:.3f}"), c2_sum]).style.set_caption(
    "200 test memes, mean ± sd over runs (strict: failed answers = no technique)"))

# %% [markdown]
# **Is the gap real?** We use a paired bootstrap over memes on run 1 of each LLM configuration: our micro-F1 minus
# the LLM's, resampling the same memes for both systems.

# %%
def paired_gap(Yp_llm, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed); d = []
    for _ in range(n_boot):
        i = rng.randint(0, len(Y_test), len(Y_test))
        d.append(f1_score(Y_test[i], TEST["labels"][i], average="micro", zero_division=0)
                 - f1_score(Y_test[i], Yp_llm[i], average="micro", zero_division=0))
    return np.mean(d), np.percentile(d, [2.5, 97.5])


def paired_gap_on(Yp, mask, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed); d = []
    idx_all = np.where(mask)[0]
    for _ in range(n_boot):
        i = rng.choice(idx_all, len(idx_all))
        d.append(f1_score(Y_test[i], TEST["labels"][i], average="micro", zero_division=0)
                 - f1_score(Y_test[i], Yp[i], average="micro", zero_division=0))
    return np.mean(d), np.percentile(d, [2.5, 97.5])


gap_rows = []
for (model, prompt, run), (Yp, sp, keep) in PRED_LLM.items():
    if run == 1:
        m, ci = paired_gap_on(Yp, keep)
        gap_rows.append({"LLM": f"{model} / {prompt}", "memes": int(keep.sum()),
                         "ours − LLM (micro-F1)": m, "95% CI": np.round(ci, 3),
                         "LLM better?": "yes" if ci[1] < 0 else "no" if ci[0] > 0 else "not distinguishable"})
display(pd.DataFrame(gap_rows).style.format({"ours − LLM (micro-F1)": "{:+.3f}"}))

# agreement between repeated runs of the same LLM configuration (stability for a teacher re-uploading a meme)
stab = []
for (model, prompt), g in cls.groupby(["model", "prompt"]):
    runs = sorted(g.run.unique())
    if len(runs) > 1:
        Ys = [PRED_LLM[(model, prompt, r)][0] for r in runs]
        jac = [np.mean([(a & b).sum() / (a | b).sum() if (a | b).any() else 1.0 for a, b in zip(Ys[i], Ys[j])])
               for i, j in itertools.combinations(range(len(runs)), 2)]
        stab.append({"LLM": f"{model} / {prompt}", "runs": len(runs),
                     "mean pairwise Jaccard of label sets": np.mean(jac),
                     "memes with identical output in all runs": np.mean([all((Ys[0][n] == Y_[n]).all() for Y_ in Ys)
                                                                         for n in range(len(Y_test))])})
display(pd.DataFrame(stab).style.format("{:.2f}", subset=["mean pairwise Jaccard of label sets",
                                                          "memes with identical output in all runs"])
        .set_caption("Run-to-run stability (our system is deterministic: 1.00 by construction)"))

# %%
full = [k for k in PRED_LLM if k[2] == 1 and PRED_LLM[k][2].all()]
best_llm = max(full, key=lambda k: f1_score(Y_test, PRED_LLM[k][0], average="micro", zero_division=0))
f_ours = f1_score(Y_test, TEST["labels"], average=None, zero_division=0, labels=range(K))
f_llm = f1_score(Y_test, PRED_LLM[best_llm][0], average=None, zero_division=0, labels=range(K))
order_t = np.argsort(-Y_test.sum(0))
fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(K)
ax.bar(x - 0.2, f_ours[order_t], 0.4, label="our system"); ax.bar(x + 0.2, f_llm[order_t], 0.4, label=f"{best_llm[0]} / {best_llm[1]} (run 1)")
ax.set_xticks(x, [f"{SHORT[TECHNIQUES[i]]} ({int(Y_test[:, i].sum())})" for i in order_t], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("F1 on test"); ax.legend(); ax.set_title("Per-technique F1: our system vs. the best zero-shot LLM configuration")
plt.tight_layout(); plt.show()
n_pred = pd.Series({"gold": Y_test.sum(1).mean(), "ours": TEST["labels"].sum(1).mean(),
                    **{f"{k[0]}/{k[1]}": v[0][v[2]].sum(1).mean() for k, v in PRED_LLM.items() if k[2] == 1}})
print("average number of techniques per meme:", n_pred.round(2).to_dict())

# %% [markdown]
# **Contamination probe (verbatim memorisation).** For each test meme of at least 8 words, the model received the
# first half and was asked to continue it. We report the share of the true second half recovered (longest common
# word sequence / length of the second half). As a control we compare the same continuation with the second half of
# a *different*, randomly paired meme.

# %%
memo = c2[(c2.task == "memo") & (c2.dataset == "test") & c2.id.isin(test_text)].sort_values("timestamp_utc").groupby(["model", "id"]).tail(1)


def lcs_words(a, b):
    a, b = a.lower().split(), b.lower().split()
    if not a or not b:
        return 0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


second_half = {i: " ".join(t.split()[len(t.split()) // 2:]) for i, t in test_text.items()}
rng_m = np.random.RandomState(SEED)
memo_rows = []
for model, g in memo.groupby("model"):
    ids = list(g.id); perm = rng_m.permutation(ids)
    for (i, txt), j in zip(zip(g.id, g.text.fillna("")), perm):
        true, other = second_half[i], second_half[j]
        memo_rows.append({"model": model, "id": i, "recovered (true half)": lcs_words(txt, true) / max(len(true.split()), 1),
                          "recovered (other meme, control)": lcs_words(txt, other) / max(len(other.split()), 1)})
memo_df = pd.DataFrame(memo_rows)
memo_tab = memo_df.groupby("model").agg(memes=("id", "size"), true_half=("recovered (true half)", "mean"),
                                        control=("recovered (other meme, control)", "mean"),
                                        verbatim_50pct=("recovered (true half)", lambda x: (x >= 0.5).mean()))
display(memo_tab.style.format("{:.3f}", subset=["true_half", "control", "verbatim_50pct"]).set_caption(
    "Memorisation probe: share of the unseen second half reproduced (verbatim_50pct = memes with ≥50% reproduced)"))

# %% [markdown]
# **Practical comparison.**

# %%
lat = c2[(c2.task == "classify") & (c2.dataset == "test")].groupby("model").latency_s.median()
display(pd.DataFrame({
    "median latency per meme (s)": {**lat.round(1).to_dict(), "OURS (5-model ensemble)": round(lat5[0] / 1000, 3)},
    "student text leaves the device": {**{m: ("no (local)" if "ornith" in m.lower() else "yes (hosted API)") for m in lat.index},
                                       "OURS (5-model ensemble)": "no"},
    "same input → same output": {**{m: "no (sampled)" for m in lat.index}, "OURS (5-model ensemble)": "yes"},
}))


# %% [markdown]
# **C2 conclusion.**
# * **The strongest general-purpose model is better than ours.** Gemini 3.1 Pro, zero-shot, scores micro-F1 0.574 ±
#   0.007 over three runs and is significantly better on the same 200 memes (our −0.056, 95% CI −0.107 to −0.002). It
#   is much better on rare techniques (macro-F1 0.44 vs. 0.22) and slightly better on spans (0.46 vs. 0.41). With
#   the names-only prompt the gap is not significant. We report this as a finding, not a failure.
# * **Contamination is plausible but limited.** Pro reproduces 16% of the unseen second halves of memes, against 2% for
#   the control, and more than half of the text for 12% of memes, mostly well-known slogans and quotes. It has seen
#   these meme texts or their sources, so its test score is likely somewhat optimistic. The labels themselves are not
#   public in the same form, so this does not explain the whole advantage.
# * **It cannot be deployed.** Every meme would leave the device to a hosted service, which the contract forbids. It is
#   also not reproducible: only 45% of memes get the same labels in all three runs (mean Jaccard 0.82), and it takes
#   18 s per meme.
# * **The deployable alternative is far worse.** Ornith-1.5-9B runs locally on T4-class hardware but scores 0.20
#   without reasoning (we are +0.33 better) and 0.35 with reasoning on 50 memes, at 160 s per meme, which is unusable
#   while a class waits.
# * **Cheap hosted models are worse than ours** (Flash-Lite 0.44, we are +0.09 better).
#
# **Why the department should pay for this system:** among the options that satisfy the privacy constraint, it is the
# most accurate by a wide margin, it answers in 0.14 s, and it gives the same answer every time. The honest caveat:
# a frontier model is better at the task, so the best use of it is *offline*, on public memes only, for example to
# pre-label Australian training data that our local model then learns from.
#
# **Limits of the comparison.** One provider gateway (grsai.ai) whose exact model build we cannot verify beyond the
# reported `modelVersion`; temperature 1.0 as recommended; one prompt family; strict and lenient scoring coincide
# because every response could be parsed.
