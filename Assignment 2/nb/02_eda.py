# %% [markdown]
# ## 1. Understanding the data
#
# **Provenance** (Dimitrov et al., 2021, §4 and App. A). The memes were collected in 2020 from 26 public Facebook
# groups: 23 about US and Canadian politics (left, right, centrist, anti-government, gun control) and 3 about
# COVID-19, vaccines and gender equality. All memes are in English. The text was extracted with Google Vision and
# corrected by hand. Six annotators labelled the text alone, three per meme, before seeing the image. Disagreements
# were settled by discussion with a consolidator, not by majority vote. Agreement of the three annotators with the
# consolidated labels was Krippendorff's α = 0.83, 0.91 and 0.56, so one annotator diverged clearly from the consensus.
#
# Consequences for later sections:
# 1. The gold labels are one team's negotiated reading of US/Canadian political memes from 2020. "Correct" means
#    agreeing with that team, which matters for C1 (whose reading?) and C4 (Australia, 2026, teenagers).
# 2. Labels were assigned from the text alone, which matches our input: we also only get the extracted text.
# 3. Moderate agreement limits the achievable score.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
counts = pd.DataFrame({s: np.stack(df_all[df_all.split == s].y).sum(0) for s in SPLIT_FILES},
                      index=[SHORT[t] for t in TECHNIQUES]).astype(int)
counts = counts.loc[counts.sum(axis=1).sort_values().index]
counts.plot.barh(stacked=True, ax=axes[0], color=["#4C72B0", "#DD8452", "#55A868"], width=0.8)
for i, v in enumerate(counts.sum(axis=1)):
    axes[0].text(v + 3, i, str(v), va="center", fontsize=8)
axes[0].set_title("Memes per technique (subtask 1 labels)"); axes[0].set_xlabel("number of memes")
n_labels = df_all.y.apply(lambda y: int(y.sum()))
n_labels.value_counts().sort_index().plot.bar(ax=axes[1], color="#4C72B0")
axes[1].set_title("Techniques per meme"); axes[1].set_xlabel("number of techniques"); axes[1].set_ylabel("memes")
plt.tight_layout(); plt.show()
print(f"mean techniques/meme = {n_labels.mean():.2f}; memes with none = {(n_labels == 0).sum()} "
      f"({(n_labels == 0).mean():.0%}); train-set counts below 10: "
      f"{[SHORT[t] for t in TECHNIQUES if np.stack(df_all[df_all.split=='train'].y).sum(0)[T2I[t]] < 10]}")

# %% [markdown]
# **Reading.** The label distribution is very long-tailed. Loaded Language, Name calling and Smears account for 64% of
# all labels. Five techniques have fewer than ten training memes, and *Red Herring* has a single one. Two consequences:
# * Micro-F1, the official metric, is dominated by the frequent classes, so we also report macro-F1 and per-technique
#   F1. A tool that only ever says "Loaded Language" would look acceptable on micro-F1.
# * No method can learn a technique from one to four examples. We expect F1 ≈ 0 for those and list them as
#   unsupported (Section 7).

# %%
Y_all = np.stack(df_all.y)
co = Y_all.T @ Y_all
cond = co / np.maximum(np.diag(co)[:, None], 1)  # P(column | row)
order = np.argsort(-np.diag(co))[:12]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(cond[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(order)), [SHORT[TECHNIQUES[i]] for i in order], rotation=60, ha="right")
ax.set_yticks(range(len(order)), [SHORT[TECHNIQUES[i]] for i in order])
for a, i in enumerate(order):
    for b, j in enumerate(order):
        ax.text(b, a, f"{cond[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if cond[i, j] > 0.6 else "black")
ax.set_title("Co-occurrence: P(column technique | row technique), 12 most frequent")
plt.colorbar(im, fraction=0.04); plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading.** Techniques co-occur strongly: a meme with *Name calling* very often also has *Loaded Language*, and
# *Smears* often comes with both. The conditional probabilities stay well below 1, so the dependence is soft. This
# supports 20 sigmoid outputs on a shared encoder: the encoder can learn the correlations, and no technique excludes
# another (as a softmax would).

# %%
def word_len(s, e, text):
    return len(text[s:e].split())


span_rows = []
for _, r in df_all.iterrows():
    clean_len = max(len(r.text.strip()), 1)
    for s, e, t in r.spans:
        span_rows.append({"technique": SHORT[t], "words": word_len(s, e, r.text),
                          "frac_of_text": (e - s) / clean_len, "split": r.split})
spans_df = pd.DataFrame(span_rows)
stats = spans_df.groupby("technique").agg(spans=("words", "size"), median_words=("words", "median"),
                                          whole_text_share=("frac_of_text", lambda f: (f >= 0.9).mean()))
stats = stats.sort_values("median_words")
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
data = [spans_df[spans_df.technique == t].words.values for t in stats.index]
ax[0].boxplot(data, orientation="horizontal", showfliers=False); ax[0].set_yticks(range(1, len(stats) + 1), stats.index)
ax[0].set_xlabel("span length (words)"); ax[0].set_title("How long is a technique's span?")
ax[1].barh(range(1, len(stats) + 1), stats.whole_text_share, color="#DD8452")
ax[1].set_xlabel("share of spans covering ≥90% of the meme text"); ax[1].set_title("Span = (almost) whole meme")
plt.tight_layout(); plt.show()
print(f"{len(spans_df)} spans; {(spans_df.frac_of_text >= 0.9).mean():.0%} of them cover ≥90% of the meme text")

# %% [markdown]
# **Reading.** The techniques split into two kinds:
# * Lexical techniques (Loaded Language, Name calling, Repetition, Slogans) have spans of 1–4 words. A tagger can point
#   at them precisely.
# * Discourse-level techniques (Whataboutism, Doubt, Causal oversimplification, Smears, Black-and-white) usually span
#   most of the meme. "Where is it?" is then answered by "the whole message".
#
# For Part B the evaluation must therefore accept spans that cover the whole text. For C3 we expect the label head and
# the span head to agree easily on lexical techniques and to disagree more on discourse-level ones.

# %%
def overlap_stats(df):
    cross, same_overlap, same_adjacent, same_pairs = 0, 0, 0, 0
    for _, r in df.iterrows():
        sp = r.spans
        for i in range(len(sp)):
            for j in range(i + 1, len(sp)):
                if sp[i][0] < sp[j][1] and sp[j][0] < sp[i][1] and sp[i][2] != sp[j][2]:
                    cross += 1
        by = collections.defaultdict(list)
        for s, e, t in sp:
            by[t].append((s, e))
        for v in by.values():
            v.sort()
            for a, b in zip(v, v[1:]):
                same_pairs += 1
                if b[0] < a[1]:
                    same_overlap += 1
                elif re.fullmatch(r"[\s\W]*", r.text[a[1]:b[0]]):
                    same_adjacent += 1
    return cross, same_pairs, same_overlap, same_adjacent


cross, same_pairs, same_ov, same_adj = overlap_stats(df_all)
n_spans = sum(len(s) for s in df_all.spans)
print(f"overlapping span pairs with DIFFERENT techniques: {cross}")
print(f"consecutive spans of the SAME technique in a meme: {same_pairs}; overlapping: {same_ov}; "
      f"separated only by spaces/punctuation: {same_adj} ({(same_ov + same_adj) / n_spans:.1%} of all spans)")

# %% [markdown]
# **Decision: tagging scheme for Part B.** Spans of different techniques overlap all the time. Standard BIO tagging
# gives every token a single label and cannot represent this. We therefore tag each technique independently: every
# token gets 20 binary "inside technique *k*?" decisions (IO tagging, one sigmoid per technique per token).
#
# IO tagging cannot separate two adjacent spans of the same technique. Per-technique BIO could, at the cost of 3× the
# outputs and sparser B tags. The table above shows that this case affects only about 3% of spans, and the official
# scorer gives partial credit by character overlap, so one merged prediction over two adjacent gold spans still earns
# almost all of their credit. IO is the simpler choice and loses almost nothing; Section 2 checks this numerically.

# %%
dupe_groups = df_all.assign(norm=df_all.text.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()) \
    .groupby("norm").filter(lambda g: len(g) > 1)
show = dupe_groups.assign(labels=dupe_groups.y.apply(lambda y: [SHORT[TECHNIQUES[i]] for i in np.where(y)[0]]))
display(show[["id", "split", "text", "labels"]])
t1_mismatch = sum(set(a["labels"]) != {l["technique"] for l in b["labels"]}
                  for s in SPLIT_FILES for a, b in zip(raw_t1[s], raw[s]))
frag_mismatch = [m["id"] for s in raw for m in raw[s] for l in m["labels"]
                 if m["text"][l["start"]:l["end"]] != l["text_fragment"]]
print("task1 label sets that differ from the techniques present in task2 spans:", t1_mismatch)
print("spans whose stored fragment differs from text[start:end]:", len(frag_mismatch), sorted(set(frag_mismatch)))

# %% [markdown]
# **Findings that change the pipeline**
# * **Leakage in the official split.** Meme `474_batch_2` ("Joe Biden's lifetime accomplishments.") appears in both
#   train and test, with different labels (none vs. *Slogans*). We drop it from the training data, so the test set
#   stays the official one and remains comparable with published results.
# * **Label noise.** Identical texts carry different labels. "The real virus threatening America" is *Loaded
#   Language* once and *Fear/prejudice + Flag-waving* the other time. The "Hey Hillary, the Coronavirus is going to
#   testify against you" meme gains or loses *Loaded Language*. Since the text was annotated without the image,
#   annotator inconsistency is the most plausible explanation. This limits the achievable F1. We keep both copies:
#   the noise is part of the task.
# * **Labels and spans are consistent.** Every task1 label set equals the set of techniques with at least one task2
#   span: a meme has technique *k* if and only if some part of its text is tagged *k*. This motivates a joint model
#   (Section 3) and makes the C3 question well-defined.
# * A few stored `text_fragment`s do not match their offsets. We always use the offsets, as the official scorer does.

# %%
LEAK_IDS = set(df_all[df_all.split == "train"].id) & set(df_all[df_all.split == "test"].id)
print("ids in both train and test:", LEAK_IDS)
df_all = df_all[~((df_all.split == "train") & df_all.id.isin(LEAK_IDS))].reset_index(drop=True)
df_dev = df_all[df_all.split.isin(["train", "dev"])].reset_index(drop=True)   # development pool (train + dev)
df_test = df_all[df_all.split == "test"].reset_index(drop=True)               # untouched until Section 5
if FAST_DEV_RUN:  # local smoke test only: tiny subsets
    df_dev = df_dev.sample(150, random_state=SEED).reset_index(drop=True)
    df_test = df_test.sample(60, random_state=SEED).reset_index(drop=True)
print("development pool:", len(df_dev), "| held-out test:", len(df_test))

# %% [markdown]
# ### Style of the text and political composition
# Two properties of the text matter later. Casing: memes are often set in capitals, which sub-word tokenisers split
# into more, rarer pieces. Political target: which side a meme attacks. For the target we use a simple keyword list.
# It is used only for analysis (C1, C4), never inside the classifier, as the specification requires.

# %%
LEFT_TERMS = r"\b(biden|obama|hillary|clinton|democrat\w*|dems?|liberal\w*|leftist\w*|libtard\w*|kamala|harris|pelosi|bernie|sanders|aoc|antifa|socialis\w*|communis\w*)\b"
RIGHT_TERMS = r"\b(trump\w*|republican\w*|conservative\w*|maga|gop|pence|right[- ]wing\w*|mcconnell|trumpster\w*)\b"


def political_target(text: str) -> str:
    t = text.lower()
    left, right = bool(re.search(LEFT_TERMS, t)), bool(re.search(RIGHT_TERMS, t))
    return "both" if left and right else "left-figures" if left else "right-figures" if right else "neither"


def upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / max(len(letters), 1)


df_all["target"] = df_all.text.apply(political_target)
df_all["mostly_caps"] = df_all.text.apply(upper_ratio) > 0.7
df_dev["target"] = df_dev.text.apply(political_target); df_test["target"] = df_test.text.apply(political_target)
comp = pd.crosstab(df_all.split, df_all.target)
display(comp)
by_side = df_all[df_all.target.isin(["left-figures", "right-figures"])]
rate = pd.DataFrame({side: np.stack(g.y).mean(0) for side, g in by_side.groupby("target")},
                    index=[SHORT[t] for t in TECHNIQUES])
rate["n"] = 0
rate = rate.loc[rate[["left-figures", "right-figures"]].max(axis=1).sort_values(ascending=False).index[:8]]
display(rate.drop(columns="n").style.format("{:.2f}").set_caption(
    "Share of memes carrying each technique, by which side's figures are named (all splits)"))
print(f"memes mostly in capitals: {df_all.mostly_caps.mean():.0%}")

# %% [markdown]
# **Reading.** Memes naming left-wing figures outnumber memes naming right-wing figures roughly 3:2, and about two
# thirds of memes name neither side. The gold technique rates also differ by side. Whether this reflects the memes or
# the annotators' reading of them cannot be separated with this data; either way a model trained on it can inherit
# the asymmetry, which C1 tests. About half of the memes are written mostly in capitals, while a teenager's feed in
# 2026 is mostly mixed case. We test this style shift in E8 and C4.
#
# **Summary: data → design**
#
# | Observation | Design consequence |
# |---|---|
# | 750 development memes, 63 official dev memes | 5-fold cross-validation on train+dev; official test (200) used once |
# | Long tail: 5 techniques < 10 training memes | macro-F1 and per-technique F1 reported; class weighting tested (E5) |
# | Labels co-occur, but softly | 20 sigmoid outputs on a shared encoder |
# | Spans overlap across techniques; same-technique adjacency ≈ 3% | per-technique IO tagging |
# | Label ⇔ at least one span (100%) | joint model; label/span agreement is measurable (C3) |
# | Lexical vs. discourse-level techniques | technique-specific attention in the label head (Section 3) |
# | Leak `474_batch_2`; duplicated texts with conflicting labels | leak removed from training; noise limits achievable F1 |
# | Half the memes in capitals; US politics 2020 | casing robustness (E8); C1 and C4 |
