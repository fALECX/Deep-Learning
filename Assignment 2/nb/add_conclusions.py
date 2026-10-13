"""Replace / add the Part C conclusions and the ultimate judgement (markdown only). Idempotent."""
import pathlib

NB = pathlib.Path(__file__).resolve().parent


def md(text):
    return "\n\n# %% [markdown]\n" + "\n".join(("# " + l) if l else "#" for l in text.strip().splitlines()) + "\n"


def as_comment(text):
    return "\n".join(("# " + l) if l else "#" for l in text.strip().splitlines()) + "\n"


C3 = """
**C3 conclusion.** The two heads disagree on **35% of all flagged (meme, technique) pairs**: 1,308 times both say
yes, 583 times only the label head does, 117 times only the span head does.

**Who is right when they disagree?** A label the span head does not support is correct only **29%** of the time; an
underlined span without a label is correct **36%** of the time. When both agree, the technique is correct **62%** of
the time. Agreement is therefore a strong signal of reliability, and the display-policy table shows the trade-off:
showing only agreed labels raises precision from 0.52 to **0.62** but lowers recall from 0.60 to 0.49.

**Our hypothesis was wrong.** We expected more disagreement for whole-message techniques. The opposite holds
(Spearman −0.50): disagreement is highest for short, rarer techniques such as Fear/prejudice (67%), Exaggeration
(58%), Slogans (52%) and Glittering generalities (76%). For those, the label head fires but the span head rarely
dares to underline anything. Whataboutism and Smears, which cover the whole meme, agree well (27% and 22%).

**What the screen should show.**
1. **Both agree:** show the technique and underline the words. This is the default teaching moment (62% correct).
2. **Label only:** show it greyed out as *"possible: the tool can't point to the words, can you?"*. Never show it as a
   finding: it is wrong 71% of the time.
3. **Span only:** underline in a neutral colour with a question, *"could this be Smears?"*.

This rule never produces a label with nothing to point at, and it matches the classroom goal: the tool prompts
students to look, the teacher decides.

**Limits.** Measured on 950 in-distribution memes, against one consensus annotation with moderate agreement. On
Australian memes (C4) both heads are less reliable, so the share of agreed labels will drop.
"""

C1 = """
**C1 conclusion.**
1. **The labels themselves are not symmetric.** In the gold data, memes naming left-wing figures carry Smears more
   often (0.50 vs. 0.39), memes naming right-wing figures more often Glittering generalities (0.15 vs. 0.03). We cannot
   tell whether this reflects the memes or the annotators' reading; a model trained on it can only inherit it.
2. **Group error rates:** memes naming right-wing figures receive more false-positive labels (1.26 per meme vs. 0.94
   left and 0.90 neither) and more often a wrong "attack" label (60% vs. 45%). The confidence intervals barely
   overlap, but the groups also differ in content, so this alone does not prove bias.
3. **Counterfactual swap, the decisive test:** swapping the side changes the predicted label set for 32.0% of the
   266 memes. But swapping to a name of the **same** side changes it for 32.7%. The paired difference is −0.8 points
   (95% CI −6.4 to +4.9). **We find no evidence that the side itself changes the output**; with 266 paired memes the
   test could have detected a difference of about 6 points. Within that, the probabilities shift in a consistent
   direction: left-wing names raise P(Smears) (+0.06, p < 0.001) and right-wing names raise P(Name calling)
   (+0.03 to +0.04). This mirrors the label asymmetry in point 1, at the level of a few percentage points.
4. **The real problem is a different one: the model is very sensitive to *names* in general.** One label set in three
   changes when any name is replaced. The neutral templates make this concrete: "{politician} held a press
   conference today" gets P(any technique) of 0.19–0.46, and a visible label in 1–3 of 4 sentences, for politicians of
   **both** countries and **both** sides. Social groups (Muslims, immigrants, women, …) stay low (0.13–0.17), below
   neutral controls like "the local library" (0.19–0.27).

**Is it fair enough for a room of teenagers?** It treats the sides alike within the precision our tests allow, but it
is **not stable enough to be legible**: a student who swaps a name in their meme will see the labels change one
time in three, and a harmless sentence that only names a politician can be flagged. In class this looks like bias
even when it is not. Mitigations before deployment: name-masking or name-swap augmentation in training, the C3
display rule (only agreed labels shown as findings), and teacher materials that explain uncertainty.

**Limits.** Political side is found with a keyword list (missing memes that attack a side without naming it). The
swap pairs are imperfect counterparts. Group sizes are 102–159. The templates are few and English-only.
"""

C2 = """
**C2 conclusion.**
* **The strongest general-purpose model is better than ours.** Gemini 3.1 Pro, zero-shot, scores micro-F1 0.574 ±
  0.007 over three runs and is significantly better on the same 200 memes (our −0.056, 95% CI −0.107 to −0.002). It
  is much better on rare techniques (macro-F1 0.44 vs. 0.22) and slightly better on spans (0.46 vs. 0.41). With
  the names-only prompt the gap is not significant. We report this as a finding, not a failure.
* **Contamination is plausible but limited.** Pro reproduces 16% of the unseen second halves of memes, against 2% for
  the control, and more than half of the text for 12% of memes, mostly well-known slogans and quotes. It has seen
  these meme texts or their sources, so its test score is likely somewhat optimistic. The labels themselves are not
  public in the same form, so this does not explain the whole advantage.
* **It cannot be deployed.** Every meme would leave the device to a hosted service, which the contract forbids. It is
  also not reproducible: only 45% of memes get the same labels in all three runs (mean Jaccard 0.82), and it takes
  18 s per meme.
* **The deployable alternative is far worse.** Ornith-1.5-9B runs locally on T4-class hardware but scores 0.20
  without reasoning (we are +0.33 better) and 0.35 with reasoning on 50 memes, at 160 s per meme, which is unusable
  while a class waits.
* **Cheap hosted models are worse than ours** (Flash-Lite 0.44, we are +0.09 better).

**Why the department should pay for this system:** among the options that satisfy the privacy constraint, it is the
most accurate by a wide margin, it answers in 0.14 s, and it gives the same answer every time. The honest caveat:
a frontier model is better at the task, so the best use of it is *offline*, on public memes only, for example to
pre-label Australian training data that our local model then learns from.

**Limits of the comparison.** One provider gateway (grsai.ai) whose exact model build we cannot verify beyond the
reported `modelVersion`; temperature 1.0 as recommended; one prompt family; strict and lenient scoring coincide
because every response could be parsed.
"""

C4 = """
**C4 conclusion: will it work here?**

| Gap | Evidence | Finding |
|---|---|---|
| US names and institutions → Australian ones | entity swap on 73 test memes, labels unchanged | no drop (0.531 → 0.541); the model reads the rhetoric, not the names |
| Writing style of a 2026 feed | lower-case, sentence case, slang, emoji, OCR noise | small drops (0.49–0.50 for slang, emoji, OCR), all inside the CI, but **more labels per meme** (2.5 instead of 2.0 with slang) |
| New topics and vocabulary | 30 persuasive Australian probe memes | micro-F1 falls to **0.28** (test: 0.52); 27% of words unseen in training (test: 17%) |
| Harmless teen memes | 14 everyday memes | **8 of 14 get a technique label**, often Loaded Language or Name calling ("Maccas at home", "group project partner") |
| Does a general LLM transfer better? | same probe set | Gemini Pro 0.75 and 1/14 false alarms; our model 0.28 and 8/14 |

**What goes wrong in the gap.** The model has learned what *2020 US political* persuasion looks like. In front of
new topics (housing, energy, the under-16 social-media ban) it falls back to the three frequent attack labels. In
front of ordinary teen humour it sees hostility that is not there. The model is also *confident* on Australian
material (the confidence histogram looks like the familiar test set), so a teacher cannot rely on low confidence to
spot these errors.

**What it means in the classroom.**
* **For the teacher:** the lesson must not assume the labels are right. The tool should frame every output as a
  question for the class, and the teacher needs a one-click "the tool is wrong here" button that is logged.
* **For a student whose family's politics appear on screen:** a label like *Smears* on a meme they shared from home
  can feel like a verdict on their family. The tool must label techniques, never sides or people. It should show the
  same technique on a meme from the other side whenever possible, and teachers need guidance for that moment.
* **Accountability:** the department should collect consented, de-identified Australian examples (teacher-chosen,
  not student uploads), label a few hundred with at least two annotators, and re-evaluate before every term. False
  alarms on non-political memes should be tracked as their own metric.

**Limits.** The probe set is small (44 memes), synthetic, written and labelled by one annotator with AI assistance.
It can reveal failure modes but cannot estimate Australian accuracy. The perturbations test style, not meaning.
"""

JUDGEMENT = """
**Recommendation: do not deploy this system as an autonomous technique detector next term. Deploy it, if at all,
as a teacher-led discussion aid for a few core techniques, with the C3 display rule, and plan a re-evaluation on
Australian data.**

**Why this model is the best available under the constraints.**
* It is the best of the configurations we tested on clean, paired cross-validation: DeBERTa-v3, fully fine-tuned,
  our label-wise attention head, joint label + span training, class-weighted loss, a five-fold ensemble.
* On the held-out test set it reaches micro-F1 0.52 and span F1 0.41: clearly above the baselines, below the 2021
  state of the art that used now-forbidden propaganda pretraining.
* It satisfies every constraint: it runs locally, answers in 0.14 s on a T4 using under 2 GB of weights, is
  deterministic, and uses no persuasion-trained checkpoint and no external data.
* Among systems that may legally run in a classroom, nothing we tested comes close: a local 9B LLM reaches 0.20.

**Why it is not ready to run unsupervised.**
1. **Few techniques work.** Only Loaded Language, Name calling, Smears, Whataboutism, Exaggeration and (on very few
   examples) Appeal to authority reach F1 ≥ 0.4. Twelve of the twenty techniques are almost never found.
2. **It over-accuses.** Precision is 0.48: about half of all shown labels are wrong. Missed subtle techniques turn
   into extra "attack" labels.
3. **It is sensitive to names.** One in three label sets change when a name is swapped, and neutral sentences about
   politicians can be flagged. We found no side-specific bias, but students will experience the instability as bias.
4. **It does not transfer.** On Australian-style memes accuracy roughly halves, and 8 of 14 harmless teen memes are
   flagged, while the model stays confident.
5. **A general model is better at the task** (Gemini Pro, 0.57 on test and 0.75 on the Australian probe set), but it
   is not allowed on student data.

**Realistic chance of success.** In a classroom where the teacher uses the output as a prompt ("the tool thinks
this is Name calling; which words would do that?"), the tool can work for the core techniques today. As a system
that names the techniques in a student's meme by itself, it would mislead students in a large share of cases.

**What would change this verdict.**
* Label 300–500 Australian memes (teacher-curated, public sources, at least two annotators) and fine-tune on them.
* Use the frontier LLM **offline on public memes only** to pre-label more Australian training data, then distil it
  into the local model. No student data would leave the device.
* Augment training with name masking and benign everyday memes to cut the name sensitivity and the false alarms.
* Re-run C1–C4 on that data before every term, and track teacher overrides as a live quality metric.

### 7.3 Limitations
* **Small, noisy data:** 750 development memes; duplicated texts carry different labels; inter-annotator agreement
  α ≈ 0.77. Rare techniques have 1–5 training examples.
* **Selection optimism:** cross-validation was used for every choice, so its 0.56 overstates the 0.52 found on test.
  The test set itself is only 200 memes (CI ±0.05).
* **Training budget:** validation label F1 was still rising slowly at epoch 15 in some folds.
* **Text only:** the image is ignored, although many techniques live in it (by design of the task).
* **Evaluation reach:** everything except C4 is measured on 2020 US/Canadian Facebook memes. The Australian evidence
  is synthetic and small. Political side is detected with keywords.
* **Reproducibility:** GPU kernels are not bit-deterministic. A re-run gives the same conclusions, but individual
  numbers can move by about the seed standard deviation (≈0.01).
"""

p = NB / "09_c3_c1.py"; s = p.read_text(encoding="utf8")
a = s.index("# %% [markdown]\n# **C3 conclusion and display rule.**")
b = s.index("# %% [markdown]\n# ### C1. Does it treat all sides the same?")
s = s[:a] + "# %% [markdown]\n" + as_comment(C3) + "\n" + s[b:]
if "**C1 conclusion.**" not in s:
    s = s.rstrip("\n") + "\n" + md(C1)
p.write_text(s, encoding="utf8")

p = NB / "10_c2.py"; s = p.read_text(encoding="utf8")
if "**C2 conclusion.**" not in s:
    s = s.rstrip("\n") + "\n" + md(C2)
p.write_text(s, encoding="utf8")

p = NB / "11_c4.py"; s = p.read_text(encoding="utf8")
if "**C4 conclusion" not in s:
    s = s.rstrip("\n") + "\n" + md(C4)
p.write_text(s, encoding="utf8")

p = NB / "12_judgement.py"; s = p.read_text(encoding="utf8")
assert "# __JUDGEMENT__" in s
s = s.replace("# __JUDGEMENT__\n", as_comment(JUDGEMENT), 1)
p.write_text(s, encoding="utf8")
print("ok")
