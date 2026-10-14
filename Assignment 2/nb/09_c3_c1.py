# %% [markdown]
# ## 6. Part C: product evaluation
# All four questions are answered with experiments on the final system. For more statistical power, several
# analyses use **all 950 memes**. The 750 development memes are scored *out-of-fold*: each by the fold model that
# never saw it. The 200 test memes are scored by the ensemble. No model is ever scored on a meme it was trained on.

# %%
def oof_and_test_predict(texts_dev, texts_test):
    """Out-of-fold predictions for (possibly modified) development texts + ensemble predictions for test texts."""
    lab_dev = np.zeros((len(df_dev), K)); tok_dev = [None] * len(df_dev); off_dev = [None] * len(df_dev)
    for f, m in enumerate(FINAL["models"]):
        idx = np.where(df_dev.fold == f)[0]
        sub = df_dev.iloc[idx].assign(text=[texts_dev[i] for i in idx])
        r = ensemble_predict(sub, [m])
        lab_dev[idx] = r["label_p"]
        for j, i in enumerate(idx):
            tok_dev[i], off_dev[i] = r["token_p"][j], r["offsets"][j]
    rt = ensemble_predict(df_test.assign(text=list(texts_test)))
    return (np.concatenate([lab_dev, rt["label_p"]]), tok_dev + rt["token_p"], off_dev + rt["offsets"])


df_both = pd.concat([df_dev.assign(part="dev (out-of-fold)"), df_test.assign(part="test")], ignore_index=True)
Y_both = np.stack(df_both.y)
P_both, TOKP_both, OFF_both = oof_and_test_predict(list(df_dev.text), list(df_test.text))
L_both = (P_both >= LABEL_THR).astype(int)
SPANS_both = spans_from_probs(TOKP_both, OFF_both, SPAN_THR, list(df_both.text))

# %% [markdown]
# ### C3. Does the label head agree with the span head?
# For each meme and technique there are two independent yes/no answers:
# * the **label head** says "the meme uses *k*" (probability ≥ threshold);
# * the **span head** says "these words are *k*" (at least one decoded span of *k*).
#
# We count four cases: both yes, label-only (a label with nothing to point at), span-only (words underlined but no
# label), and both no. For the disagreements we also check **who was right** against the gold annotation. That is
# what decides what the screen should show.
#
# **H:** disagreement concentrates on discourse-level techniques. Their evidence is the whole message, so the span
# head's per-token decisions are less confident than the label head's meme-level decision.

# %%
if FINAL_CFG["label_head"] == "from_spans":
    # The selected architecture derives labels from spans, so its two outputs agree by construction (one reason it
    # may have been chosen). To still answer C3 we measure the best two-head configuration from Section 4 on its
    # 750 out-of-fold predictions.
    two_head = [nm for nm, r in RESULTS.items() if r["cfg"]["label_head"] != "from_spans" and r["thr_span"] is not None]
    src = max(two_head, key=lambda nm: RESULTS[nm]["overall"]["select"])
    R3 = RESULTS[src]
    C3_TEXT, C3_GOLD = df_dev, Y_dev
    C3_L = (R3["label_p"] >= R3["thr_label"]).astype(int)
    C3_SPANS = spans_from_probs(R3["token_p"], R3["offsets"], R3["thr_span"], list(df_dev.text))
    print(f"C3 measured on the two-head configuration '{src}' ({len(C3_TEXT)} out-of-fold memes)")
else:
    C3_TEXT, C3_GOLD, C3_L, C3_SPANS = df_both, Y_both, L_both, SPANS_both
    print(f"C3 measured on the final system ({len(C3_TEXT)} memes: out-of-fold + test)")
pct = lambda a: f"{100 * a.mean():.0f}%" if a.size else "n/a"
span_flag = np.zeros_like(C3_L)
for i, sp in enumerate(C3_SPANS):
    for _, _, t in sp:
        span_flag[i, T2I[t]] = 1
rows = []
for k, t in enumerate(TECHNIQUES):
    lab, spn, gold = C3_L[:, k], span_flag[:, k], C3_GOLD[:, k]
    lo, so = (lab == 1) & (spn == 0), (lab == 0) & (spn == 1)
    rows.append({"technique": SHORT[t], "gold memes": int(gold.sum()), "both": int(((lab == 1) & (spn == 1)).sum()),
                 "label only": int(lo.sum()), "span only": int(so.sum()),
                 "disagree % of flagged": 100 * (lo.sum() + so.sum()) / max(((lab == 1) | (spn == 1)).sum(), 1),
                 "label-only correct %": 100 * gold[lo].mean() if lo.any() else np.nan,
                 "span-only correct %": 100 * gold[so].mean() if so.any() else np.nan,
                 "median gold span / text": np.median([(e - s) / max(len(C3_TEXT.text.iloc[i].strip()), 1)
                                                       for i in range(len(C3_TEXT)) for s, e, tt in C3_TEXT.spans.iloc[i] if tt == t])
                 if gold.any() else np.nan})
c3 = pd.DataFrame(rows).set_index("technique").sort_values("gold memes", ascending=False)
tot = c3[["both", "label only", "span only"]].sum()
lo_all, so_all = (C3_L == 1) & (span_flag == 0), (C3_L == 0) & (span_flag == 1)
print(f"overall: {tot['both']} agreed positives, {tot['label only']} label-only, {tot['span only']} span-only → "
      f"{100 * (tot['label only'] + tot['span only']) / tot.sum():.1f}% of flagged (meme, technique) pairs disagree; "
      f"label-only correct {pct(C3_GOLD[lo_all])}, span-only correct {pct(C3_GOLD[so_all])}, "
      f"both-yes correct {pct(C3_GOLD[(C3_L == 1) & (span_flag == 1)])}")
display(c3.style.format("{:.0f}", na_rep="–").set_caption(f"Label head vs. span head ({len(C3_TEXT)} memes)"))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
show_t = c3[c3["gold memes"] >= 10]
show_t[["both", "label only", "span only"]].plot.bar(stacked=True, ax=ax[0], color=["#55A868", "#C44E52", "#8172B2"])
ax[0].set_title("Where the two heads disagree (techniques with ≥10 gold memes)"); ax[0].set_ylabel("memes")
ax[1].scatter(show_t["median gold span / text"], show_t["disagree % of flagged"], s=show_t["gold memes"])
for nm, r in show_t.iterrows():
    ax[1].annotate(nm, (r["median gold span / text"], r["disagree % of flagged"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("median gold span length / meme length"); ax[1].set_ylabel("disagreement (% of flagged)")
ax[1].set_title("Is disagreement higher for discourse-level techniques?")
plt.tight_layout(); plt.show()
rho = pd.Series(show_t["median gold span / text"]).corr(show_t["disagree % of flagged"], method="spearman")
print(f"Spearman correlation (span length share vs. disagreement) over {len(show_t)} techniques: {rho:.2f}")

# %% [markdown]
# **What should the screen show?** We turn the evidence into a display policy and score each candidate with the
# same label metrics. A label is displayed if the policy says so.

# %%
policies = {
    "label head only": C3_L,
    "only when both agree": C3_L & span_flag,
    "either head": C3_L | span_flag,
}
pol = pd.DataFrame({nm: label_scores(C3_GOLD, P) for nm, P in policies.items()}).T
display(pol.style.format("{:.3f}").set_caption(f"Display policies scored as labels ({len(C3_TEXT)} memes)"))

# %% [markdown]
# **C3 conclusion.** The two heads disagree on 35% of all flagged (meme, technique) pairs: 1,308 times both say yes,
# 583 times only the label head does, 117 times only the span head does.
#
# **Who is right when they disagree?** A label the span head does not support is correct only 29% of the time; an
# underlined span without a label is correct 36% of the time. When both agree, the technique is correct 62% of the
# time. Agreement is therefore a strong signal of reliability. The display-policy table shows the trade-off: showing
# only agreed labels raises precision from 0.52 to 0.62 but lowers recall from 0.60 to 0.49.
#
# **Hypothesis rejected.** We expected more disagreement for whole-message techniques. The opposite holds (Spearman
# −0.50): disagreement is highest for short, rarer techniques such as Fear/prejudice (67%), Exaggeration (58%),
# Slogans (52%) and Glittering generalities (76%). For those, the label head fires but the span head rarely
# underlines anything. Whataboutism and Smears, which cover the whole meme, agree well (27% and 22%).
#
# **What the screen should show.**
# 1. Both agree: show the technique and underline the words (62% correct).
# 2. Label only: show it greyed out as *"possible: the tool can't point to the words, can you?"*, never as a finding
#    (it is wrong 71% of the time).
# 3. Span only: underline in a neutral colour with a question, *"could this be Smears?"*.
#
# This rule never shows a label with nothing to point at, and it fits the classroom goal: the tool prompts students
# to look, the teacher decides.
#
# **Limits.** Measured on 950 in-distribution memes, against one consensus annotation with moderate agreement. On
# Australian memes (C4) both heads are less reliable, so the share of agreed labels will drop.

# %% [markdown]
# ### C1. Does it treat all sides the same?
# "Fair to all sides" can fail in three places, and each needs a different test:
#
# | Where bias could enter | Test | Why this test can detect it |
# |---|---|---|
# | **The labels** (whose reading?) | Gold technique rates for memes naming left- vs. right-wing figures (Section 1) | Shows asymmetries the model can only inherit |
# | **The model's errors** | False-positive and false-negative rates per political target, 950 memes, bootstrap CIs | A tool that over-accuses one side shows a higher FP rate there |
# | **The model's use of names** | **Counterfactual swap:** exchange names/parties (Trump↔Biden, Republicans↔Democrats, …) and re-score the *same* meme | Paired: the rhetoric is held constant, only the side changes. Any change in output is caused by the side alone |
# | **Spurious associations** | **Neutral templates** ("{X} held a press conference today.") with figures, parties and social groups | A neutral sentence contains no technique, so any probability mass is bias attached to the name itself |
#
# The group comparison has little power (about 160 vs. 100 memes, and they differ in content, not only in side).
# The counterfactual test is the main experiment because every meme is its own control.

# %%
def bootstrap_rate(x, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed); x = np.asarray(x, dtype=float)
    return np.percentile([x[rng.randint(0, len(x), len(x))].mean() for _ in range(n_boot)], [2.5, 97.5])


df_both["target"] = df_both.text.apply(political_target)
grp_rows = []
for g in ["left-figures", "right-figures", "neither"]:
    m = (df_both.target == g).values
    fp = ((L_both[m] == 1) & (Y_both[m] == 0)).sum(1); fn = ((L_both[m] == 0) & (Y_both[m] == 1)).sum(1)
    attack = [T2I[t] for t in ["Name calling/Labeling", "Smears", "Loaded Language"]]
    fp_attack = ((L_both[m][:, attack] == 1) & (Y_both[m][:, attack] == 0)).any(1)
    grp_rows.append({"target": g, "memes": int(m.sum()), **label_scores(Y_both[m], L_both[m]),
                     "false positives / meme": fp.mean(), "FP 95% CI": bootstrap_rate(fp),
                     "false negatives / meme": fn.mean(), "FN 95% CI": bootstrap_rate(fn),
                     "spurious attack label (share of memes)": fp_attack.mean(), "attack CI": bootstrap_rate(fp_attack)})
grp = pd.DataFrame(grp_rows).set_index("target")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, (col, ci, title) in zip(ax, [("false positives / meme", "FP 95% CI", "false positives per meme"),
                                    ("false negatives / meme", "FN 95% CI", "missed techniques per meme"),
                                    ("spurious attack label (share of memes)", "attack CI",
                                     "memes wrongly given Name calling/Smears/Loaded")]):
    v = grp[col].values; err = np.abs(np.stack(grp[ci].values).T - v)
    a.bar(grp.index, v, yerr=err, capsize=4, color=["#4C72B0", "#C44E52", "#999999"]); a.set_title(title, fontsize=10)
plt.suptitle("Error rates by which side's figures the meme names (950 memes, 95% bootstrap CI)", y=1.03)
plt.tight_layout(); plt.show()
display(grp[["memes", "micro_f1", "micro_p", "micro_r", "false positives / meme", "false negatives / meme",
             "spurious attack label (share of memes)"]].style.format("{:.3f}").format({"memes": "{:.0f}"}))

# %% [markdown]
# **Counterfactual swap.** Every meme that names a figure or party from the list below is rewritten with the
# counterpart from the other side, preserving capitalisation. The swapped meme is scored by the same model as the
# original (its out-of-fold model, or the ensemble for test memes). We measure:
# * the **flip rate**: how often the displayed label set changes;
# * the **mean change in probability** of each technique, separately for memes that originally named left-wing
#   figures (now naming right-wing ones) and the reverse.
#
# A model that treats sides the same has flip rates near zero and mean changes centred on zero in both directions.
# A model that associates attacks with one side shows changes of *opposite sign* in the two directions.

# %%
SWAP_PAIRS = [("trump", "biden"), ("donald", "joe"), ("republicans", "democrats"), ("republican", "democrat"),
              ("conservatives", "liberals"), ("conservative", "liberal"), ("gop", "dnc"), ("pence", "harris"),
              ("mcconnell", "pelosi"), ("right-wing", "left-wing"), ("right wing", "left wing"),
              ("melania", "jill"), ("ivanka", "hunter"), ("kamala", "mike"), ("obama", "bush")]
_swap_map = {**{a: b for a, b in SWAP_PAIRS}, **{b: a for a, b in SWAP_PAIRS}}
_swap_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _swap_map), key=len, reverse=True)) + r")\b", re.I)


def match_case(src, word):
    return word.upper() if src.isupper() else word.title() if src[:1].isupper() else word


def fold_key(s):
    """Lower-case and strip accents: 'HARRİS'.lower() is 'harri̇s' (dotted capital I), which must still match."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def swap_sides(text):
    return _swap_re.sub(lambda m: match_case(m.group(0), _swap_map.get(fold_key(m.group(0)), m.group(0))), text)


swapped_dev = [swap_sides(t) for t in df_dev.text]; swapped_test = [swap_sides(t) for t in df_test.text]
P_sw, _, _ = oof_and_test_predict(swapped_dev, swapped_test)
changed = np.array([a != b for a, b in zip(list(df_dev.text) + list(df_test.text), swapped_dev + swapped_test)])
direction = df_both.target.values
L_sw = (P_sw >= LABEL_THR).astype(int)
flip = (L_sw != L_both).any(1)
print(f"memes containing a swappable name: {changed.sum()} | label set changed after the swap: {flip[changed].sum()} "
      f"({100 * flip[changed].mean():.1f}%, 95% CI {np.round(100 * bootstrap_rate(flip[changed]), 1)})")
_ex = min(np.where(changed)[0], key=lambda i: len(df_both.text[i]))
print("example:", df_both.text[_ex].replace("\n", " "), "→", (swapped_dev + swapped_test)[_ex].replace("\n", " "))

from scipy.stats import wilcoxon
focus = ["Name calling/Labeling", "Smears", "Loaded Language", "Doubt", "Appeal to fear/prejudice", "Slogans"]
rows = []
for d in ["left-figures", "right-figures"]:
    m = changed & (direction == d)
    for t in focus:
        delta = P_sw[m, T2I[t]] - P_both[m, T2I[t]]
        p = wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
        rows.append({"original target → swapped": f"{d} → {'right' if d.startswith('left') else 'left'}-figures",
                     "technique": SHORT[t], "n": int(m.sum()), "mean Δp": delta.mean(),
                     "95% CI": np.round(bootstrap_rate(delta), 3), "Wilcoxon p": p,
                     "label flips": int((L_sw[m, T2I[t]] != L_both[m, T2I[t]]).sum())})
cf = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(11, 4))
for j, (d, g) in enumerate(cf.groupby("original target → swapped", sort=False)):
    x = np.arange(len(focus)) + (j - 0.5) * 0.35
    err = np.abs(np.stack(g["95% CI"].values).T - g["mean Δp"].values)
    ax.bar(x, g["mean Δp"], 0.35, yerr=err, capsize=3, label=f"{d} (n={g.n.iloc[0]})")
ax.axhline(0, color="black", lw=0.8); ax.set_xticks(np.arange(len(focus)), [SHORT[t] for t in focus])
ax.set_ylabel("mean change in probability"); ax.legend(fontsize=8)
ax.set_title("Counterfactual side swap: change in technique probability (95% bootstrap CI)")
plt.tight_layout(); plt.show()
display(cf.style.format({"mean Δp": "{:+.3f}", "Wilcoxon p": "{:.3f}"}))

# %% [markdown]
# **Control: is it the side, or any name change?** A flip rate on its own cannot tell bias from general sensitivity
# to names. So the same memes are rewritten a second time, with names and party words replaced by **another name
# from the same side** (Trump→Pence, Biden→Obama, Republicans→GOP, liberals→leftists, …). If the side swap flips
# labels clearly more often than the same-side swap, the extra flips are caused by the side.

# %%
CONTROL_PAIRS = [("trump", "pence"), ("mcconnell", "trump"), ("melania", "ivanka"), ("ivanka", "melania"),
                 ("republicans", "gop"), ("republican", "gop"), ("gop", "republicans"), ("conservatives", "right-wingers"),
                 ("conservative", "right-wing"), ("right-wing", "conservative"),
                 ("biden", "obama"), ("obama", "biden"), ("hillary", "pelosi"), ("pelosi", "hillary"), ("clinton", "pelosi"),
                 ("kamala", "hillary"), ("harris", "clinton"), ("bernie", "obama"), ("sanders", "obama"),
                 ("democrats", "dems"), ("democrat", "dem"), ("dems", "democrats"), ("liberals", "leftists"),
                 ("liberal", "leftist"), ("left-wing", "liberal"), ("jill", "michelle"), ("hunter", "chelsea")]
_ctrl_map = dict(CONTROL_PAIRS)
_ctrl_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _ctrl_map), key=len, reverse=True)) + r")\b", re.I)
same_side = lambda t: _ctrl_re.sub(lambda m: match_case(m.group(0), _ctrl_map.get(fold_key(m.group(0)), m.group(0))), t)
ctrl_dev = [same_side(t) for t in df_dev.text]; ctrl_test = [same_side(t) for t in df_test.text]
P_ctrl, _, _ = oof_and_test_predict(ctrl_dev, ctrl_test)
L_ctrl = (P_ctrl >= LABEL_THR).astype(int)
both_changed = changed & np.array([a != b for a, b in zip(list(df_dev.text) + list(df_test.text), ctrl_dev + ctrl_test)])
flip_ctrl = (L_ctrl != L_both).any(1)
attack = [T2I[t] for t in ["Name calling/Labeling", "Smears"]]
ctl = pd.DataFrame({
    "rewrite": ["side swap (other side)", "control (same side)"],
    "memes": [int(both_changed.sum())] * 2,
    "label set changed": [flip[both_changed].mean(), flip_ctrl[both_changed].mean()],
    "95% CI": [np.round(bootstrap_rate(flip[both_changed]), 3), np.round(bootstrap_rate(flip_ctrl[both_changed]), 3)],
    "mean |Δp| Name calling + Smears": [np.abs(P_sw[both_changed][:, attack] - P_both[both_changed][:, attack]).mean(),
                                        np.abs(P_ctrl[both_changed][:, attack] - P_both[both_changed][:, attack]).mean()],
}).set_index("rewrite")
d_flip = flip[both_changed].astype(float) - flip_ctrl[both_changed].astype(float)
display(ctl.style.format({"label set changed": "{:.1%}", "mean |Δp| Name calling + Smears": "{:.3f}"}))
print(f"paired difference in flip rate (side − control): {d_flip.mean():+.3f}, 95% CI {np.round(bootstrap_rate(d_flip), 3)}")

# %% [markdown]
# **Neutral templates.** Sentences that contain no persuasion technique, filled with names of figures, parties and
# social groups from both countries. Any probability the model assigns to a technique here comes from the name.

# %%
TEMPLATES = ["{X} held a press conference today.", "{X} will announce the new budget on Tuesday.",
             "A spokesperson for {X} said the meeting was productive.", "This is a photo of {X}."]
GROUPS = {"US left": ["Joe Biden", "the Democrats", "Kamala Harris"], "US right": ["Donald Trump", "the Republicans", "Mike Pence"],
          "AU left": ["Anthony Albanese", "the Labor Party", "the Greens"],
          "AU right": ["Sussan Ley", "the Liberal Party", "Pauline Hanson"],
          "social groups": ["Muslims", "Christians", "immigrants", "women", "teenagers", "farmers"],
          "control": ["the weather bureau", "the local library", "my neighbour"]}
tmpl_rows = []
for g, names in GROUPS.items():
    for nm in names:
        texts = [t.format(X=nm) for t in TEMPLATES]
        p = ensemble_predict(pd.DataFrame({"text": texts, "spans": [[] for _ in texts], "y": [np.zeros(K, np.float32)] * len(texts)}))
        tmpl_rows.append({"group": g, "name": nm, "mean P(any technique)": (1 - np.prod(1 - p["label_p"], 1)).mean(),
                          "max P(Name calling)": p["label_p"][:, T2I["Name calling/Labeling"]].max(),
                          "max P(Smears)": p["label_p"][:, T2I["Smears"]].max(),
                          "any label shown": int(p["labels"].any(1).sum())})
tmpl = pd.DataFrame(tmpl_rows)
display(tmpl.style.format({c: "{:.3f}" for c in ["mean P(any technique)", "max P(Name calling)", "max P(Smears)"]})
        .background_gradient(subset=["mean P(any technique)"], cmap="Reds").set_caption(
        f"Neutral templates ({len(TEMPLATES)} per name): should be ≈0 everywhere"))


# %% [markdown]
# **C1 conclusion.**
# 1. **The labels themselves are not symmetric.** In the gold data, memes naming left-wing figures carry Smears more
#    often (0.50 vs. 0.39), memes naming right-wing figures more often Glittering generalities (0.15 vs. 0.03). We cannot
#    tell whether this reflects the memes or the annotators' reading; a model trained on it can only inherit it.
# 2. **Group error rates:** memes naming right-wing figures receive more false-positive labels (1.26 per meme vs. 0.94
#    left and 0.90 neither) and more often a wrong "attack" label (60% vs. 45%). The confidence intervals barely
#    overlap, but the groups also differ in content, so this alone does not prove bias.
# 3. **Counterfactual swap (main test).** Swapping the side changes the predicted label set for 32.0% of the
#    266 memes. But swapping to a name of the **same** side changes it for 32.7%. The paired difference is −0.8 points
#    (95% CI −6.4 to +4.9). **We find no evidence that the side itself changes the output**; with 266 paired memes the
#    test could have detected a difference of about 6 points. Within that, the probabilities shift in a consistent
#    direction: left-wing names raise P(Smears) (+0.06, p < 0.001) and right-wing names raise P(Name calling)
#    (+0.03 to +0.04). This mirrors the label asymmetry in point 1, at the level of a few percentage points.
# 4. **Name sensitivity.** The model is very sensitive to names in general: one label set in three changes
#    when any name is replaced. The neutral templates make this concrete: "{politician} held a press
#    conference today" gets P(any technique) of 0.19–0.46, and a visible label in 1–3 of 4 sentences, for politicians of
#    **both** countries and **both** sides. Social groups (Muslims, immigrants, women, …) stay low (0.13–0.17), below
#    neutral controls like "the local library" (0.19–0.27).
#
# **Is it fair enough for a room of teenagers?** It treats the sides alike within the precision our tests allow, but it
# is **not stable enough to be legible**: a student who swaps a name in their meme will see the labels change one
# time in three, and a harmless sentence that only names a politician can be flagged. In class this looks like bias
# even when it is not. Mitigations before deployment: name-masking or name-swap augmentation in training, the C3
# display rule (only agreed labels shown as findings), and teacher materials that explain uncertainty.
#
# **Limits.** Political side is found with a keyword list (missing memes that attack a side without naming it). The
# swap pairs are imperfect counterparts. Group sizes are 102–159. The templates are few and English-only.
