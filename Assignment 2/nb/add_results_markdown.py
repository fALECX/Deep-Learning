"""Insert result interpretations (markdown only) after the final run. Code cells are not touched, so the
executed outputs can be merged back unchanged (nb/merge_outputs.py). Idempotent via a marker per block."""
import pathlib

NB = pathlib.Path(__file__).resolve().parent


def md(text):
    return "\n\n# %% [markdown]\n" + "\n".join(("# " + l) if l else "#" for l in text.strip().splitlines()) + "\n"


INSERTS = {
    "05_experiments.py": [
        ('print("→ chosen:", BEST)\n', "E1-RESULT", """
**E1 result.** DeBERTa-v3 wins on all five folds (paired +0.038 ± 0.033 on the selection score). The gain comes
mainly from span F1 (0.401 vs. 0.338 for RoBERTa). The tweet-pretrained RoBERTa did **not** help (−0.005 ± 0.016,
won 2/5 folds), so the hypothesis is rejected: pretraining text from the same era and register did not beat a
stronger general encoder. DeBERTa has 185M parameters instead of 125M, which is still small for a T4.
"""),
    ],
    "06_a_experiments2.py": [
        ('BEST = choose(LORA, [BEST, "E2 frozen"])  # LoRA unless an alternative clearly beats it\nprint("→ chosen:", BEST)\n', "E2-RESULT", """
**E2 result.** Full fine-tuning beats every cheaper strategy on the selection score: frozen −0.022 ± 0.021,
LoRA −0.034 ± 0.018 (0/5 folds won), LoRA at 2e-3 −0.072 ± 0.051. The gap is largest on macro-F1 (0.24 vs.
0.14–0.16): the rare techniques need the encoder itself to change. Our hypothesis ("LoRA within noise") is
**rejected**. LoRA trains only 1.4M parameters, so it is the right choice when many task adapters must be stored,
but here accuracy matters more and one 185M model per school machine is cheap. A surprise worth noting: the frozen
encoder (0.8M trainable parameters) scored slightly *higher* than LoRA. With 600 training memes per fold, LoRA's
extra capacity seems to add noise rather than signal.
"""),
        ('BEST = choose(BEST, ["E3 span-only (labels derived)", "E3 derived + label loss"])\nprint("→ chosen:", BEST)\n', "E3-RESULT", """
**E3 result.** The joint model stays.
* Removing the span loss (label-only) changes label micro-F1 by about one point (0.560 vs. 0.570) but lowers
  macro-F1 from 0.24 to 0.18, so the span supervision helps the rare techniques. (Its "select" score is not
  comparable because it has no span output.)
* Deriving labels from spans alone is clearly worse on labels (0.494). Many techniques are a property of the whole
  message, not of single tokens.
* "Derived + label loss" is +0.009 ± 0.018, within noise, so the tie rule keeps the simpler joint model.
"""),
        ('    BEST = choose(BEST, ["E4 cls", "E4 mean"])\nprint("→ chosen:", BEST)\n', "E4-RESULT", """
**E4 result.** Label-wise attention is not worse than [CLS] (−0.007 ± 0.017) or mean pooling (−0.005 ± 0.026) on
the selection score, and it has the highest label macro-F1 (0.24 vs. 0.14 and 0.19). The differences are within
noise, so our self-built head is kept by the tie rule and for its per-technique attention maps.
"""),
        ('BEST = cand\nprint("→ chosen:", BEST)\n', "E5-RESULT", """
**E5 result.** As expected, weighting rare positives barely moves the selection score (−0.002 ± 0.027) but lifts
macro-F1 from 0.242 to 0.296: the tail techniques start to be found. Focal loss behaves similarly (0.288). The
macro-F1 rule therefore selects BCE with positive weights. Precision drops (0.56 → 0.53) while recall rises
(0.58 → 0.60), the trade-off we predicted.
"""),
        ('print("→ chosen:", BEST, RESULTS[BEST]["cfg"])\n', "E6-RESULT", """
**E6 result.** No change passes the tie rule. Encoder learning rate 1e-5 clearly underfits (−0.036 ± 0.011);
5e-5 is better on 4/5 folds (+0.015 ± 0.015), but its advantage only equals its spread, so the default is kept. A
head learning rate of 1e-4 is too slow (−0.034); 2e-3 is +0.011 ± 0.012; dropout 0.3 is worse (−0.010). The
selected configuration is DeBERTa-v3, full fine-tuning, label-wise attention head, joint loss, BCE with positive
weights, encoder LR 3e-5, head LR 5e-4, dropout 0.1.
"""),
        ('fig.suptitle(f"Learning curves of the selected configuration ({BEST})", y=1.03)\nplt.tight_layout(); plt.show()\n', "CURVES-RESULT", """
**Reading the curves.** Training loss falls smoothly in every fold. Validation span F1 levels off after about 8
epochs. Validation label F1 is noisy (one meme moves it by almost a point) and still drifts upwards at epoch 15 in
some folds, so early stopping never had to stop training. There is no sign of the validation score collapsing
(overfitting), but a few more epochs might add a little. We note this as a limitation rather than tune it, because
it would mean selecting on the same noisy folds again.
"""),
        ("f\"{'per-technique' if PER_CLASS_THR else 'global'} thresholds; span threshold = {R['thr_span']}\")\n", "E7-RESULT", """
**E7 result.** Per-technique thresholds do not help in the nested comparison (−0.003 ± 0.006 micro-F1). They raise
recall but cost more precision. One global threshold is used.
"""),
        ('      f"{d_low.std(ddof=1):.3f} → chosen: {BEST}")\n', "E8-RESULT", """
**E8 result.** The hypothesis is rejected in an informative way: the model is **already robust to casing**
(0.562 original, 0.554 lower-cased, 0.560 sentence-cased), so augmentation has nothing to fix (Δ lower-cased
−0.001 ± 0.017). DeBERTa's tokenizer handles capitals well enough. Augmentation is not adopted.
"""),
    ],
    "06_b_e9.py": [
        ('print("→ chosen:", BEST)\n', "E9-RESULT", """
**E9 result.** PTC intermediate training does not help overall (−0.001 ± 0.014). Per technique the picture is
mixed: it helps several techniques that the PTC annotates (Loaded Language, Name calling, Flag-waving, Doubt,
Exaggeration, Fear) but hurts others (Appeal to authority, Repetition, Causal oversimplification). News prose seems
to teach the *vocabulary* of propaganda but not the *meme format*. The extra data is therefore not used in the final
model, and no external data enters the final system.
"""),
    ],
    "06_c_summary.py": [
        ('print("sd across seeds:", seed_tab.std().round(3).to_dict())\n', "SEED-RESULT", """
**Noise level.** Changing only the seed moves the selection score by a standard deviation of 0.007 and macro-F1 by
0.015. Most design differences above (E3–E9) are of the same size. They should be read as "no measurable
difference", which is exactly why the tie rule kept the simpler options. Only E1 (encoder), E2 (adaptation) and the
learning-rate extremes in E6 are clearly larger than this noise.
"""),
    ],
    "07_final.py": [
        ('                  "micro_r", "span_f1", "span_p", "span_r"]).set_caption("Held-out official test set (200 memes)"))\n', "TEST-RESULT", """
**Test result.**
* Labels: **micro-F1 0.519 (95% CI 0.469–0.565)**, macro-F1 0.221. Precision 0.48, recall 0.57.
* Spans: **span F1 0.405 (95% CI 0.357–0.442)**.
* This is clearly above the majority baseline (0.374, and 0.165 for spans). It is below the SemEval-2021 winner
  (0.593 / 0.482), which used PTC-pretrained ensembles that our rules forbid.
* It is **lower than the cross-validation estimate (0.562)**. Two reasons: the CV numbers were used to select the
  configuration and thresholds, so they are optimistic; and the test memes are harder for the tail techniques. This
  gap is the honest price of a clean protocol. **0.52 is the number to expect on memes like these.**
* The ensemble beats a single fold model (0.495 / 0.376), as intended.
"""),
        ('print("techniques with ≥", MIN_POS, "development examples:", [SHORT[t] for t in SUPPORTED])\n', "PERTECH-RESULT", """
**Reading.** Only a handful of techniques are usable: Loaded Language (F1 0.77), Name calling (0.57), Whataboutism
(0.50), Exaggeration (0.42), Smears (0.45) and Appeal to authority (0.53, but only 7 test examples). Slogans gets
F1 0 on test despite 47 development examples. All techniques with fewer than 10 development examples are never
found. For the classroom this means the tool can support a lesson on a few core techniques, not the whole taxonomy.
"""),
        ('ax.set_title("Missed technique → spurious technique on the same meme (test)")\nplt.tight_layout(); plt.show()\n', "CONF-RESULT", """
**Reading.** The errors have a pattern. When the model misses a subtler technique (Slogans, Doubt, Exaggeration,
Fear), it usually predicts one of the three frequent "attack" techniques instead (Smears, Name calling, Loaded
Language). The model has learned that a meme is hostile, but not always *how* it argues.
"""),
        ('print(f"device: {torch.cuda.get_device_name(0) if DEVICE.type == \'cuda\' else \'CPU\'}; peak GPU memory during the benchmark: {peak:.2f} GB")\n', "LATENCY-RESULT", """
**Deployment fit.** Measured on a Tesla T4: one model answers in **28 ms**, the five-model ensemble in **142 ms**
per meme. The ensemble's weights take 1.85 GB in fp16. That is far inside a 16 GB T4 and instant from a
student's point of view. (The 7.6 GB peak includes the models kept in memory for the whole notebook.)
"""),
    ],
    "08_explain.py": [
        ('fig.suptitle("Integrated Gradients for the predicted technique (red = pushes towards it; boxed = gold span)", y=1.02)\nplt.tight_layout(); plt.show()\n', "IG-RESULT", """
**Explainability result.** Overall, 45% of the positive attribution falls on gold-span tokens, against 42% for a
uniform spread. For short, lexical techniques the concentration is a little stronger (Loaded Language 0.28 vs.
0.22), and for whole-message techniques the two are equal by construction. So the label head does **not** rely
mainly on the exact words the annotators marked: it uses the whole meme. For a teacher this means the IG heatmap
alone is not a reliable "here is the technique" pointer. The span head, which is trained for that job, should be
the pointer, and C3 below checks how often the two agree.
"""),
    ],
}

for fname, items in INSERTS.items():
    p = NB / fname
    s = p.read_text(encoding="utf8")
    for anchor, marker, text in items:
        if f"<!-- {marker} -->" in s:
            continue
        i = s.find(anchor)
        assert i >= 0, (fname, anchor[:60])
        j = i + len(anchor)
        s = s[:j] + md(f"<!-- {marker} -->\n{text}") + s[j:]
    p.write_text(s, encoding="utf8")
    print("ok", fname)
