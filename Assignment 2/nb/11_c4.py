# %% [markdown]
# ### C4. Will it work here? (Australia, 2026, a Year 10 feed)
# There is no labelled Australian dataset. Instead we build evidence from three directions. Each answers a different
# part of "what goes wrong in the gap":
#
# | Gap | Evidence we can produce | Labels needed? |
# |---|---|---|
# | **Who and what** (US names, parties, institutions → Australian ones) | Replace US entities in the *test* memes with Australian counterparts. The rhetoric is unchanged, so the gold labels stay valid | no new labels |
# | **How it is written** (case, slang, emoji, OCR noise of phone screenshots) | Controlled perturbations of the test memes, gold labels unchanged | no new labels |
# | **What it is about** (2026 issues, teen feeds, non-political memes) | A small **probe set of 44 Australian-style memes**: 30 persuasive (12 aimed at each side + 6 non-partisan) and 14 everyday teen memes | written and labelled for this assignment (see the declaration) |
# | **Language drift** | Out-of-vocabulary rate and sub-word fragmentation on the probe vs. the corpus | none |
#
# The probe set is small, synthetic and labelled by a single annotator. It cannot estimate Australian accuracy
# precisely. What it can show is *failure modes*: false alarms on harmless teen memes, side asymmetries, and
# techniques the model never fires on.

# %%
AU_MAP = [("donald trump", "peter dutton"), ("trump", "dutton"), ("joe biden", "anthony albanese"), ("biden", "albanese"),
          ("obama", "rudd"), ("hillary clinton", "julia gillard"), ("hillary", "julia"), ("clinton", "gillard"),
          ("kamala harris", "penny wong"), ("kamala", "penny"), ("pelosi", "wong"), ("bernie sanders", "adam bandt"),
          ("bernie", "bandt"), ("mcconnell", "littleproud"), ("pence", "ley"), ("aoc", "the greens"),
          ("democrats", "labor"), ("democrat", "labor"), ("republicans", "liberals"), ("republican", "liberal"),
          ("gop", "coalition"), ("maga", "one nation"), ("congress", "parliament"), ("white house", "the lodge"),
          ("president", "prime minister"), ("america", "australia"), ("americans", "australians"),
          ("american", "australian"), ("usa", "australia"), ("u.s.", "australia"), ("fbi", "afp"), ("cnn", "the abc"),
          ("fox news", "sky news")]
_au_re = re.compile(r"(?<![\w.])(" + "|".join(sorted(map(re.escape, dict(AU_MAP)), key=len, reverse=True)) + r")(?![\w])", re.I)
_au = dict(AU_MAP)


def to_australia(text):
    return _au_re.sub(lambda m: match_case(m.group(0), _au.get(fold_key(m.group(0)), m.group(0))), text)


EMOJI = ["💀", "😭", "🤡", "🔥", "😂"]


def ocr_noise(text, rate=0.03, seed=SEED):
    r = np.random.RandomState(seed + len(text)); chars = list(text)
    sub = {"o": "0", "l": "1", "i": "l", "e": "c", "s": "5", "O": "0", "I": "l", "E": "F", "S": "5"}
    for i, c in enumerate(chars):
        if c in sub and r.rand() < rate * 3:
            chars[i] = sub[c]
    return "".join(chars)


PERTURB = {
    "original": lambda t: t,
    "US → AU entities": to_australia,
    "lower-case": str.lower,
    "sentence case": lambda t: t[:1].upper() + t[1:].lower(),
    "teen slang appended": lambda t: t.rstrip() + " fr fr no cap 💀",
    "emoji inserted": lambda t: " ".join(w + (f" {EMOJI[i % 5]}" if i % 4 == 3 else "") for i, w in enumerate(t.split(" "))),
    "OCR noise (~5% chars)": ocr_noise,
    "AU entities + lower-case": lambda t: to_australia(t).lower(),
}
changed_au = np.array([to_australia(t) != t for t in df_test.text])
pert_rows, pert_pred = [], {}
for nm, fn in PERTURB.items():
    r = ensemble_predict(df_test.assign(text=[fn(t) for t in df_test.text]))
    pert_pred[nm] = r["labels"]
    lo, hi = bootstrap_ci(lambda a, b: f1_score(a, b, average="micro", zero_division=0), Y_test, r["labels"])
    pert_rows.append({"perturbation": nm, "micro_f1": f1_score(Y_test, r["labels"], average="micro", zero_division=0),
                      "95% CI": f"[{lo:.3f}, {hi:.3f}]", "macro_f1": f1_score(Y_test, r["labels"], average="macro", zero_division=0),
                      "predicted techniques / meme": r["labels"].sum(1).mean(),
                      "memes whose labels changed": int((r["labels"] != TEST["labels"]).any(1).sum())})
pert = pd.DataFrame(pert_rows).set_index("perturbation")
display(pert.style.format({"micro_f1": "{:.3f}", "macro_f1": "{:.3f}", "predicted techniques / meme": "{:.2f}"})
        .set_caption(f"Test set (200 memes) under controlled shifts; gold labels unchanged. "
                     f"{changed_au.sum()} memes contain a US entity that was replaced."))
m = changed_au
print(f"on the {m.sum()} memes that actually changed under US→AU: micro-F1 {f1_score(Y_test[m], TEST['labels'][m], average='micro'):.3f}"
      f" → {f1_score(Y_test[m], pert_pred['US → AU entities'][m], average='micro'):.3f}")

# %% [markdown]
# **The Australian probe set.**

# %%
AU_PROBE_JSON = r"""__AU_PROBE_JSON__"""
AU_PROBE = json.loads(AU_PROBE_JSON)
probe_rows = []
for p in AU_PROBE:
    spans = []
    for l in p["labels"]:
        s = p["text"].find(l["fragment"]); assert s >= 0, (p["id"], l["fragment"])
        spans.append((s, s + len(l["fragment"]), l["technique"]))
    y = np.zeros(K, np.float32)
    for _, _, t in spans:
        y[T2I[t]] = 1
    probe_rows.append({"id": p["id"], "side": p["side"], "text": p["text"], "spans": spans, "y": y})
df_probe = pd.DataFrame(probe_rows)
PR = ensemble_predict(df_probe)
Y_probe = np.stack(df_probe.y)
pers = (df_probe.side != "benign").values
rows = []
for nm, mask in [("persuasive (30)", pers), ("aimed at left (12)", (df_probe.side == "against-left").values),
                 ("aimed at right (12)", (df_probe.side == "against-right").values),
                 ("non-partisan (6)", (df_probe.side == "non-partisan").values), ("everyday teen memes (14)", ~pers)]:
    rows.append({"subset": nm, **label_scores(Y_probe[mask], PR["labels"][mask]),
                 **span_scores([df_probe.spans[i] for i in np.where(mask)[0]], [PR["spans"][i] for i in np.where(mask)[0]]),
                 "memes with any label shown": f"{int(PR['labels'][mask].any(1).sum())}/{int(mask.sum())}",
                 "gold memes with a technique": f"{int(Y_probe[mask].any(1).sum())}/{int(mask.sum())}"})
display(pd.DataFrame(rows).set_index("subset").style.format("{:.3f}", subset=["micro_f1", "macro_f1", "micro_p", "micro_r", "span_f1"])
        .set_caption("Australian probe set (author-written, single annotator)"))
display(df_probe.assign(gold=[fmt_labels(y) for y in Y_probe], predicted=[fmt_labels(y) for y in PR["labels"]],
                        top_prob=PR["label_p"].max(1).round(2))[["side", "text", "gold", "predicted", "top_prob"]])

# %% [markdown]
# **Language drift and confidence.** Two label-free signals that work on any new meme: how many of its words the
# training corpus never contained, and how confident the model is. A model that is *confidently* wrong on new
# material is the dangerous case for a classroom.

# %%
train_vocab = {w for t in df_dev.text for w in re.findall(r"[a-z']+", t.lower())}


def drift(texts):
    words = [w for t in texts for w in re.findall(r"[a-z']+", t.lower())]
    toks = sum(len(FINAL_TOK.tokenize(" " + w)) for w in words)
    return {"words": len(words), "OOV share (tokens)": np.mean([w not in train_vocab for w in words]),
            "sub-word pieces per word": toks / max(len(words), 1)}


display(pd.DataFrame({"SemEval test (2020, US)": drift(df_test.text), "AU probe: persuasive": drift(df_probe.text[pers]),
                      "AU probe: everyday": drift(df_probe.text[~pers])}).T.style.format(
    {"words": "{:.0f}", "OOV share (tokens)": "{:.3f}", "sub-word pieces per word": "{:.2f}"}))
fig, ax = plt.subplots(figsize=(9, 3.6))
for lbl, p in [("SemEval test", TEST["label_p"]), ("AU probe: persuasive", PR["label_p"][pers]), ("AU probe: everyday", PR["label_p"][~pers])]:
    ax.hist(p.max(1), bins=np.linspace(0, 1, 21), alpha=0.5, density=True, label=lbl)
ax.axvline(np.min(LABEL_THR), color="black", ls="--", lw=0.8, label="lowest decision threshold")
ax.set_xlabel("highest technique probability for the meme"); ax.set_ylabel("density"); ax.legend(fontsize=8)
ax.set_title("How confident is the model on familiar vs. Australian material?"); plt.tight_layout(); plt.show()

# %% [markdown]
# **Does a general LLM transfer better?** The same zero-shot prompts from C2 were run on the probe set. These models
# were trained on far more recent and more Australian text than our 2020 corpus, so this shows how much of our gap is
# about *knowledge of the world* rather than about the task.

# %%
probe_text = dict(zip(df_probe.id, df_probe.text)); probe_pos = {i: n for n, i in enumerate(df_probe.id)}
pc = c2[(c2.task == "classify") & (c2.dataset == "au_probe")].sort_values("timestamp_utc").groupby(["model", "prompt", "run", "id"]).tail(1)
rows = [{"system": "OURS", "run": "–", "persuasive micro_f1": f1_score(Y_probe[pers], PR["labels"][pers], average="micro", zero_division=0),
         "persuasive span_f1": span_scores([df_probe.spans[i] for i in np.where(pers)[0]], [PR["spans"][i] for i in np.where(pers)[0]])["span_f1"],
         "everyday memes flagged": f"{int(PR['labels'][~pers].any(1).sum())}/{int((~pers).sum())}"}]
for (model, prompt, run), g in pc.groupby(["model", "prompt", "run"]):
    Yp = np.zeros((len(df_probe), K), int); sp = [[] for _ in range(len(df_probe))]
    for _, r in g.iterrows():
        st, y, s, _, _ = parse_response(r.text if pd.isna(r.error) else None, probe_text[r.id])
        Yp[probe_pos[r.id]], sp[probe_pos[r.id]] = y, s
    rows.append({"system": f"{model} / {prompt}", "run": run,
                 "persuasive micro_f1": f1_score(Y_probe[pers], Yp[pers], average="micro", zero_division=0),
                 "persuasive span_f1": span_scores([df_probe.spans[i] for i in np.where(pers)[0]], [sp[i] for i in np.where(pers)[0]])["span_f1"],
                 "everyday memes flagged": f"{int(Yp[~pers].any(1).sum())}/{int((~pers).sum())}"})
display(pd.DataFrame(rows).style.format({"persuasive micro_f1": "{:.3f}", "persuasive span_f1": "{:.3f}"})
        .set_caption("Australian probe set: our system vs. zero-shot LLMs"))


# %% [markdown]
# **C4 conclusion: will it work here?**
#
# | Gap | Evidence | Finding |
# |---|---|---|
# | US names and institutions → Australian ones | entity swap on 73 test memes, labels unchanged | no drop (0.531 → 0.541); the model reads the rhetoric, not the names |
# | Writing style of a 2026 feed | lower-case, sentence case, slang, emoji, OCR noise | small drops (0.49–0.50 for slang, emoji, OCR), all inside the CI, but **more labels per meme** (2.5 instead of 2.0 with slang) |
# | New topics and vocabulary | 30 persuasive Australian probe memes | micro-F1 falls to **0.28** (test: 0.52); 27% of words unseen in training (test: 17%) |
# | Harmless teen memes | 14 everyday memes | **8 of 14 get a technique label**, often Loaded Language or Name calling ("Maccas at home", "group project partner") |
# | Does a general LLM transfer better? | same probe set | Gemini Pro 0.75 and 1/14 false alarms; our model 0.28 and 8/14 |
#
# **What goes wrong in the gap.** The model has learned what *2020 US political* persuasion looks like. In front of
# new topics (housing, energy, the under-16 social-media ban) it falls back to the three frequent attack labels. In
# front of ordinary teen humour it sees hostility that is not there. The model is also *confident* on Australian
# material (the confidence histogram looks like the familiar test set), so a teacher cannot rely on low confidence to
# spot these errors.
#
# **What it means in the classroom.**
# * **For the teacher:** the lesson must not assume the labels are right. The tool should frame every output as a
#   question for the class, and the teacher needs a one-click "the tool is wrong here" button that is logged.
# * **For a student whose family's politics appear on screen:** a label like *Smears* on a meme they shared from home
#   can feel like a verdict on their family. The tool must label techniques, never sides or people. It should show the
#   same technique on a meme from the other side whenever possible, and teachers need guidance for that moment.
# * **Accountability:** the department should collect consented, de-identified Australian examples (teacher-chosen,
#   not student uploads), label a few hundred with at least two annotators, and re-evaluate before every term. False
#   alarms on non-political memes should be tracked as their own metric.
#
# **Limits.** The probe set is small (44 memes), synthetic, written and labelled by one annotator with AI assistance.
# It can reveal failure modes but cannot estimate Australian accuracy. The perturbations test style, not meaning.
