# %% [markdown]
# ### E2: How should the encoder be adapted? (frozen vs. LoRA vs. full fine-tuning)
# * **Frozen encoder:** only the two heads are trained. Cheapest, but the representation cannot learn what
#   persuasion looks like.
# * **LoRA** (Hu et al., 2022), implemented from scratch in Section 3: low-rank adapters (r = 16) in the attention
#   query/value projections; about 1% of the encoder's parameters train. Parameter-efficient, and on small data it may overfit less.
# * **Full fine-tuning:** all parameters train.
#
# LoRA adapters start at zero and usually need a larger learning rate than full fine-tuning, so LoRA is run at two
# rates (5e-4, 2e-3) and the better one represents it. Frozen and full fine-tuning use their standard rates.
#
# **H:** frozen is clearly worse (micro-F1 roughly 0.05–0.10 lower), because general encoders do not linearly separate
# rhetorical techniques. LoRA is within noise of full fine-tuning. **Decision rule:** take full fine-tuning only if it
# beats LoRA under the tie rule. Otherwise take LoRA: it is cheaper to train, and per-school adapters are only a few
# MB, which matters for updating the tool later (C4).

# %%
ENC = RESULTS[BEST]["cfg"]["encoder"]
run_cv("E2 frozen", encoder=ENC, adaptation="frozen", lr_head=1e-3)
run_cv("E2 lora", encoder=ENC, adaptation="lora", lr_head=5e-4)
run_cv("E2 lora lr=2e-3", encoder=ENC, adaptation="lora", lr_head=2e-3)
show([BEST, "E2 frozen", "E2 lora", "E2 lora lr=2e-3"], BEST, f"E2: adaptation strategy (encoder = {ENC})")
LORA = max(["E2 lora", "E2 lora lr=2e-3"], key=lambda nm: RESULTS[nm]["overall"]["select"])
BEST = choose(LORA, [BEST, "E2 frozen"])  # LoRA unless an alternative clearly beats it
print("→ chosen:", BEST)


# %% [markdown]
# <!-- E2-RESULT -->
#
# **E2 result.** Full fine-tuning beats every cheaper strategy on the selection score: frozen −0.022 ± 0.021,
# LoRA −0.034 ± 0.018 (0/5 folds won), LoRA at 2e-3 −0.072 ± 0.051. The gap is largest on macro-F1 (0.24 vs.
# 0.14–0.16): the rare techniques need the encoder itself to change. Our hypothesis ("LoRA within noise") is
# **rejected**. LoRA trains only 1.4M parameters, so it is the right choice when many task adapters must be stored,
# but here accuracy matters more and one 185M model per school machine is cheap. A surprise worth noting: the frozen
# encoder (0.8M trainable parameters) scored slightly *higher* than LoRA. With 600 training memes per fold, LoRA's
# extra capacity seems to add noise rather than signal.

# %% [markdown]
# ### E3: One joint model, or two?
# * **Joint** (current): shared encoder, label head + span head, both losses.
# * **Label-only:** the same model trained without the span loss (λ = 0). Does span supervision help Part A?
# * **Span-only, labels derived:** only the span head is trained, and a label is predicted when any token is tagged
#   (max over tokens). This is the "extend Part A from Part B" design. It cannot disagree by construction.
# * **Derived labels + label loss:** as above, but the max-pooled span logits are also trained against meme labels.
#
# **H:** joint ≥ label-only on label F1, because span supervision tells the encoder which words matter. Derived labels
# are worse on discourse techniques, where no single token is decisive. **Decision rule:** keep the joint model
# unless an alternative wins under the tie rule. The joint model is also what makes C3 a meaningful question.

# %%
base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
run_cv("E3 label-only", **{**base, "span_loss_weight": 0.0})
run_cv("E3 span-only (labels derived)", **{**base, "label_head": "from_spans", "label_loss_weight": 0.0})
run_cv("E3 derived + label loss", **{**base, "label_head": "from_spans"})
show([BEST, "E3 label-only", "E3 span-only (labels derived)", "E3 derived + label loss"], BEST,
     "E3: joint vs. separate objectives (label-only has no span output: compare its micro_f1, not its select score)")
BEST = choose(BEST, ["E3 span-only (labels derived)", "E3 derived + label loss"])
print("→ chosen:", BEST)


# %% [markdown]
# <!-- E3-RESULT -->
#
# **E3 result.** The joint model stays.
# * Removing the span loss (label-only) changes label micro-F1 by about one point (0.560 vs. 0.570) but lowers
#   macro-F1 from 0.24 to 0.18, so the span supervision helps the rare techniques. (Its "select" score is not
#   comparable because it has no span output.)
# * Deriving labels from spans alone is clearly worse on labels (0.494). Many techniques are a property of the whole
#   message, not of single tokens.
# * "Derived + label loss" is +0.009 ± 0.018, within noise, so the tie rule keeps the simpler joint model.

# %% [markdown]
# ### E4: The self-built label head (pooling)
# Label-wise attention (current) versus the two standard alternatives, `[CLS]` and mean pooling, each followed by a
# 2-layer MLP. **H:** label-wise attention ≥ `[CLS]`, with the largest gain on lexical techniques that occupy a few
# words in a longer meme. **Decision rule:** the tie rule. A tie goes to label-wise attention, because it also gives an
# interpretable per-technique attention map for the teacher.

# %%
base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
if base["label_head"] == "label_attention":
    run_cv("E4 cls", **{**base, "label_head": "cls"})
    run_cv("E4 mean", **{**base, "label_head": "mean"})
    show([BEST, "E4 cls", "E4 mean"], BEST, "E4: label head")
    BEST = choose(BEST, ["E4 cls", "E4 mean"])
print("→ chosen:", BEST)


# %% [markdown]
# <!-- E4-RESULT -->
#
# **E4 result.** Label-wise attention is not worse than [CLS] (−0.007 ± 0.017) or mean pooling (−0.005 ± 0.026) on
# the selection score, and it has the highest label macro-F1 (0.24 vs. 0.14 and 0.19). The differences are within
# noise, so our self-built head is kept by the tie rule and for its per-technique attention maps.

# %% [markdown]
# ### E5: Handling class imbalance
# Plain BCE (current) vs. BCE with √-damped positive weights vs. focal loss (γ = 2).
# **H:** weighting raises recall on rare techniques. That should help macro-F1 a little but cost precision, and
# micro-F1 may not move. **Decision rule:** the selection score (micro-weighted) decides via the tie rule. We also
# report macro-F1 because the classroom cares about the tail. If a loss wins clearly on macro-F1 without losing on
# the selection score, we take it.

# %%
base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
run_cv("E5 bce+pos_weight", **{**base, "loss": "bce_pw"})
run_cv("E5 focal", **{**base, "loss": "focal"})
show([BEST, "E5 bce+pos_weight", "E5 focal"], BEST, "E5: loss")
cand = choose(BEST, ["E5 bce+pos_weight", "E5 focal"])
if cand == BEST:  # no win on the selection score: accept a clear macro-F1 gain that does not hurt it
    for c in ["E5 bce+pos_weight", "E5 focal"]:
        d_sel = RESULTS[c]["folds"]["select"].values - RESULTS[BEST]["folds"]["select"].values
        d_mac = RESULTS[c]["folds"]["macro_f1"].values - RESULTS[BEST]["folds"]["macro_f1"].values
        if d_mac.mean() > d_mac.std(ddof=1) and d_sel.mean() > -d_sel.std(ddof=1):
            cand = c; break
BEST = cand
print("→ chosen:", BEST)


# %% [markdown]
# <!-- E5-RESULT -->
#
# **E5 result.** As expected, weighting rare positives barely moves the selection score (−0.002 ± 0.027) but lifts
# macro-F1 from 0.242 to 0.296: the tail techniques start to be found. Focal loss behaves similarly (0.288). The
# macro-F1 rule therefore selects BCE with positive weights. Precision drops (0.56 → 0.53) while recall rises
# (0.58 → 0.60), the trade-off we predicted.

# %% [markdown]
# ### E6: Learning rates and regularisation
# A small, targeted search instead of a large grid. The literature on fine-tuning BERT-style encoders (Mosbach et
# al., 2021) identifies the **encoder learning rate** as the most influential and most unstable choice. The
# **head learning rate** matters because the heads start from scratch. **Dropout** matters on 600 training memes
# per fold. Epochs are not searched: early stopping on the validation fold picks them, and the learning curves
# below show where training saturates.
#
# **H:** the default (encoder 3e-5) is near-optimal; 1e-5 under-fits within 15 epochs and 5e-5 is less stable.
# **Decision rule:** the tie rule, one dimension at a time, in the order encoder LR → head LR → dropout.

# %%
base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
lr_key = "lr_encoder" if base["adaptation"] == "full" else "lr_head"
grid = [1e-5, 5e-5] if lr_key == "lr_encoder" else [2e-4, 1e-3]
names = [run_cv(f"E6 {lr_key}={v:g}", **{**base, lr_key: v}) and f"E6 {lr_key}={v:g}" for v in grid]
show([BEST] + names, BEST, f"E6a: {lr_key}")
BEST = choose(BEST, names)
base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
if lr_key == "lr_encoder":
    names = [run_cv(f"E6 lr_head={v:g}", **{**base, "lr_head": v}) and f"E6 lr_head={v:g}" for v in [1e-4, 2e-3]]
    show([BEST] + names, BEST, "E6b: head learning rate")
    BEST = choose(BEST, names)
    base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
run_cv("E6 dropout=0.3", **{**base, "dropout": 0.3})
show([BEST, "E6 dropout=0.3"], BEST, "E6c: dropout")
BEST = choose(BEST, ["E6 dropout=0.3"])
print("→ chosen:", BEST, RESULTS[BEST]["cfg"])


# %% [markdown]
# <!-- E6-RESULT -->
#
# **E6 result.** No change passes the tie rule. Encoder learning rate 1e-5 clearly underfits (−0.036 ± 0.011);
# 5e-5 is better on 4/5 folds (+0.015 ± 0.015), but its advantage only equals its spread, so the default is kept. A
# head learning rate of 1e-4 is too slow (−0.034); 2e-3 is +0.011 ± 0.012; dropout 0.3 is worse (−0.010). The
# selected configuration is DeBERTa-v3, full fine-tuning, label-wise attention head, joint loss, BCE with positive
# weights, encoder LR 3e-5, head LR 5e-4, dropout 0.1.

# %%
hist = RESULTS[BEST]["history"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for f, h in hist.groupby("fold"):
    ax[0].plot(h.epoch, h.train_loss, marker=".", label=f"fold {f}")
    ax[1].plot(h.epoch, h.val_label_f1, marker=".")
    ax[2].plot(h.epoch, h.val_span_f1, marker=".")
for a, t in zip(ax, ["training loss", "validation label micro-F1", "validation span F1"]):
    a.set_title(t); a.set_xlabel("epoch")
ax[0].legend(fontsize=7); fig.suptitle(f"Learning curves of the selected configuration ({BEST})", y=1.03)
plt.tight_layout(); plt.show()


# %% [markdown]
# <!-- CURVES-RESULT -->
#
# **Reading the curves.** Training loss falls smoothly in every fold. Validation span F1 levels off after about 8
# epochs. Validation label F1 is noisy (one meme moves it by almost a point) and still drifts upwards at epoch 15 in
# some folds, so early stopping never had to stop training. There is no sign of the validation score collapsing
# (overfitting), but a few more epochs might add a little. We note this as a limitation rather than tune it, because
# it would mean selecting on the same noisy folds again.

# %% [markdown]
# ### E7: One threshold or one per technique?
# The model outputs probabilities, and the tool needs yes/no. We compare one global threshold with per-technique
# thresholds, both tuned on out-of-fold predictions. Per-technique thresholds are estimated on few positives for
# rare classes, so they risk overfitting. To measure that honestly we use **nested** tuning: thresholds are fitted on
# four folds' out-of-fold predictions and applied to the fifth.
#
# **H:** per-technique thresholds help only for techniques with enough positives. **Decision rule:** use
# per-technique thresholds for techniques with at least `MIN_POS` development positives, and the global threshold
# otherwise, *only if* this beats the global threshold in the nested comparison.

# %%
MIN_POS = 30
R = RESULTS[BEST]


def fit_thresholds(Y, P, per_class):
    g, _ = best_label_thr(Y, P)
    thr = np.full(K, g)
    if per_class:
        for k in range(K):
            if Y[:, k].sum() >= MIN_POS:
                scores = [f1_score(Y[:, k], P[:, k] >= t, zero_division=0) for t in LABEL_THR_GRID]
                thr[k] = LABEL_THR_GRID[int(np.argmax(scores))]
    return thr


rows = []
for f in range(N_FOLDS):
    fit, ev = (df_dev.fold != f).values, (df_dev.fold == f).values
    for mode in [False, True]:
        thr = fit_thresholds(Y_dev[fit], R["label_p"][fit], mode)
        rows.append({"fold": f, "thresholds": "per-technique" if mode else "global",
                     **label_scores(Y_dev[ev], R["label_p"][ev] >= thr)})
e7 = pd.DataFrame(rows)
display(e7.groupby("thresholds")[["micro_f1", "macro_f1", "micro_p", "micro_r"]].agg(["mean", "std"]).round(3))
d = (e7[e7.thresholds == "per-technique"].micro_f1.values - e7[e7.thresholds == "global"].micro_f1.values)
PER_CLASS_THR = d.mean() > 0 and d.mean() > d.std(ddof=1)
LABEL_THR = fit_thresholds(Y_dev, R["label_p"], PER_CLASS_THR)
print(f"paired Δ micro-F1 (per-technique − global) = {d.mean():+.3f} ± {d.std(ddof=1):.3f} → "
      f"{'per-technique' if PER_CLASS_THR else 'global'} thresholds; span threshold = {R['thr_span']}")


# %% [markdown]
# <!-- E7-RESULT -->
#
# **E7 result.** Per-technique thresholds do not help in the nested comparison (−0.003 ± 0.006 micro-F1). They raise
# recall but cost more precision. One global threshold is used.

# %% [markdown]
# ### E8: Robustness to casing, and casing augmentation
# Every run above was also scored on lower-cased and sentence-cased validation text. About half the training
# memes are in capitals, while students' feeds are mostly mixed case. **H:** performance drops on re-cased text,
# and training with random re-casing (p = 0.5) removes most of the drop at little cost on the original text.
# **Decision rule:** adopt augmentation if it keeps the selection score within noise (tie rule in reverse: it must
# not *lose* by more than one sd) and improves lower-cased micro-F1 by more than one sd. Robustness is a deployment
# requirement, not a tie-breaker.

# %%
base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
run_cv("E8 casing-aug", **{**base, "casing_aug": 0.5})
rob = pd.DataFrame({nm: {"original": RESULTS[nm]["overall"]["micro_f1"],
                         "lower-cased": RESULTS[nm]["overall"]["micro_f1_lower"],
                         "sentence-cased": RESULTS[nm]["overall"]["micro_f1_sentence"]}
                    for nm in [BEST, "E8 casing-aug"]}).T
display(rob.style.format("{:.3f}").set_caption("Label micro-F1 (out-of-fold) under re-casing"))
d_sel = RESULTS["E8 casing-aug"]["folds"]["select"].values - RESULTS[BEST]["folds"]["select"].values
d_low = RESULTS["E8 casing-aug"]["folds"]["micro_f1_lower"].values - RESULTS[BEST]["folds"]["micro_f1_lower"].values
if d_sel.mean() > -d_sel.std(ddof=1) and d_low.mean() > d_low.std(ddof=1):
    BEST = "E8 casing-aug"
print(f"Δ select {d_sel.mean():+.3f} ± {d_sel.std(ddof=1):.3f}; Δ lower-cased micro-F1 {d_low.mean():+.3f} ± "
      f"{d_low.std(ddof=1):.3f} → chosen: {BEST}")


# %% [markdown]
# <!-- E8-RESULT -->
#
# **E8 result.** The hypothesis is rejected in an informative way: the model is **already robust to casing**
# (0.562 original, 0.554 lower-cased, 0.560 sentence-cased), so augmentation has nothing to fix (Δ lower-cased
# −0.001 ± 0.017). DeBERTa's tokenizer handles capitals well enough. Augmentation is not adopted.

