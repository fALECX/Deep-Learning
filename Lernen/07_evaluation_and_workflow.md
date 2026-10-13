# 07 · Evaluation and the practical workflow

> [!NOTE]
> **Colour code for all formulas:** $\textcolor{#3b82f6}{\text{input / data}}$ · $\textcolor{#e8833a}{\text{learned weights}}$ · $\textcolor{#22a06b}{\text{model output}}$ · $\textcolor{#e5484d}{\text{true label}}$ · $\textcolor{#a855f7}{\text{setting you choose}}$

**Why this matters.** A model is only as good as the evidence for it. Most mistakes in practice are not in the network. They are in the data split, the metric or the way results are compared. Both assignments and many exam case studies are graded on exactly this.

**Course material:** Week 1 slides (`Week 1/W1_Lecture1_Part2_v2.pdf`): *Generalization*, *Under-fitting and Over-fitting*, *Hyperparameters and Validation Sets*, *Cross-Validation*, *ML Evaluation Framework*. Week 1 notebook (`Week 1/L1_Week1_MLBasics.ipynb`), especially *The Golden Rule of Data Splitting*. Week 2 notebook, *Managing Dataset Imbalance* and *Performance Measures*. Week 6 notebook (`Week 6/L6_Week6_IterativeWorkflow.ipynb`): Steps 1–4 and *Hyperparameter tuning*. Week 7 tutorial (`Week 7/COSC2779_Tute7_ans.pdf`) on splits, metrics and reading learning curves.


> [!TIP]
> **Play with it** (interactive, in the browser). Do this before or after reading; seeing it move is the fastest way in.
> - [MLU-Explain: train, test and validation sets](https://mlu-explain.github.io/train-test-validation/): why the test set is only touched once, as a scrolling story.
> - [MLU-Explain: cross-validation](https://mlu-explain.github.io/cross-validation/): animated k-fold splits.
> - [MLU-Explain: precision and recall](https://mlu-explain.github.io/precision-recall/): drag the threshold and watch precision and recall trade off.
> - [MLU-Explain: bias and variance](https://mlu-explain.github.io/bias-variance/): under- and overfitting as model complexity grows.
> - [MLU-Explain: double descent](https://mlu-explain.github.io/double-descent/): the modern twist on overfitting from the week 1 slides.

---

## 1. The workflow (Week 6)

1. **Define the goal and the metric.** What decision will the model support? Which error is worse?
2. **Build a simple baseline.** For example, always predict the most common class. Now every later number has a reference.
3. **Set up instrumentation.** Log training and validation loss and metrics every epoch. Plot them.
4. **Read the curves and decide the next move.** Underfitting? Overfitting? Unstable?
5. **Change one thing at a time** and measure again.
6. **Test once at the end.**

## 2. Data splits

| Split | Used for | Touched how often |
|---|---|---|
| Training | fitting the weights | every step |
| Validation | choosing models, hyperparameters, thresholds, epochs | many times |
| Test | final, unbiased estimate | **once**, at the very end |

**The golden rule:** no decision may use the test set. If you pick the model that scores best on the test set, the test score is no longer an honest estimate. It is optimistic.

**The validation and test data should look like the deployment data.** If the model will be used on images from microscope C, then validation and test must contain images from C. Training can contain all microscopes. In A2 the test set is 2020 US memes. So it says little about Australian memes in 2026. That gap is studied separately in C4.

> [!NOTE]
> **From the course · Week 7 tutorial, (a).** 10,000 cell images from microscopes A (5,000), B (2,500) and C (2,500). The doctors will use the model on microscope C. How do you split?

<details>
<summary><b>Show the answer</b></summary>

Roughly 90 / 5 / 5. Validation and test must both come from **C**, because that is the deployment data. Training should also contain C images (more than val/test), plus all of A and B.

</details>

## 3. Cross-validation

With little data, one validation split is noisy. One lucky or unlucky split can decide the result.

**k-fold cross-validation:**
1. Split the data into $k$ parts (folds), for example $k = 5$.
2. Train $k$ models. Each time, one fold is the validation set and the other $k − 1$ are for training.
3. Every example is predicted exactly once by a model that never saw it. These are **out-of-fold** predictions.
4. Report the mean and the spread over folds:

$$
\bar{s} = \frac{1}{\textcolor{#a855f7}{k}}\sum_{f=1}^{k} s_f \qquad\qquad \text{sd} = \sqrt{\frac{1}{k-1}\sum_{f=1}^{k}(s_f - \bar{s})^2}
$$

<sub>$s_f$ = score on fold $f$ · $\bar s$ = mean score · sd = standard deviation across folds, a measure of how noisy the estimate is.</sub>

![5-fold cross-validation: five training runs, each with a different validation fold](img/07_kfold.png)
<sub>5-fold cross-validation: five training runs, each with a different validation fold.</sub>

**Stratification:** make every fold have the same class proportions. For multi-label data, simple stratification does not work. **Iterative stratification** (Sechidis et al., 2011) assigns the rarest label first, then the next rarest, so that each label is spread evenly.

A2 uses 5-fold iterative stratification on 750 memes. The official 63-meme dev set is too small: one meme changes the F1 by about one point.

## 4. Data leakage

Leakage means information from the evaluation data gets into training. The results then look better than they are. Common forms:
- The **same example** in train and test. A2 found one: meme `474_batch_2` is in both. It was removed from training.
- **Near-duplicates** (the same meme with a different crop or capitalisation).
- Computing normalisation statistics on **all** data instead of only training data.
- **Information from the future** in time series.
- Tuning on the test set (see the golden rule).

A sign of leakage: suspiciously good validation results, or validation loss far below training loss.

## 5. Metrics

**Confusion counts for one class:**

| | truly yes | truly no |
|---|---|---|
| **predicted yes** | TP (true positive) | FP (false alarm) |
| **predicted no** | FN (miss) | TN (true negative) |

**Definitions:**

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\qquad
\text{Precision} = \frac{TP}{TP + FP}
\qquad
\text{Recall} = \frac{TP}{TP + FN}
\qquad
F_1 = \frac{2 \cdot \text{P} \cdot \text{R}}{\text{P} + \text{R}}
$$

<sub>**Precision**: when the model says yes, how often is it right? · **Recall**: of all true cases, how many did it find? · **F1**: harmonic mean; it is high only if both are high. · **Accuracy** is misleading with imbalance: if 95% of emails are not spam, "never spam" scores 95%.</sub>

**Worked example.** 10 memes truly use *Smears*. The model flags 8 memes, and 6 of them are correct.
- TP = 6, FP = 2, FN = 4.
- Precision = 6/8 = 0.75. Recall = 6/10 = 0.60.
- $F_1 = \dfrac{2 \cdot 0.75 \cdot 0.60}{0.75 + 0.60} = \dfrac{0.90}{1.35} = 0.67$.

![The worked example as a picture](img/07_precision_recall.png)
<sub>The worked example as a picture. Precision looks at the green circle, recall at the red one.</sub>

**Several classes: micro versus macro.**

$$
\text{micro-}F_1 = F_1\Big(\sum_k TP_k,\ \sum_k FP_k,\ \sum_k FN_k\Big)
\qquad\qquad
\text{macro-}F_1 = \frac{1}{K}\sum_{k=1}^{K} F_{1,k}
$$

<sub>micro = add up the counts over all classes first, then compute one F1, so frequent classes dominate · macro = compute F1 per class $k$, then average, so every class counts the same and rare classes matter.</sub>

Example from A2: always predicting *Loaded Language* gives micro-F1 0.38 but macro-F1 0.03. Micro-F1 alone would hide that the model knows only one class.

**Per-class scores and the confusion matrix** show which classes work and which get confused. In multi-label problems you can build a "confusion" table differently: when gold class *a* was missed, which wrong class *b* was predicted for the same example?

> [!NOTE]
> **From the course · Week 7 tutorial, (c) and Practice exam, C.1 (d).** (1) Which metric for pixel-wise cell segmentation? (2) Accuracy is good overall, but how do you see whether every one of six classes works and which ones get confused?

<details>
<summary><b>Show the answer</b></summary>

1. A pixel-wise measure that handles class imbalance, e.g. **F1**. Cell pixels and background pixels are not balanced, so accuracy would mislead.
2. **Per-class** precision/recall/F1 shows which classes work. A **confusion matrix** shows which classes are mixed up.

</details>

**Choose the metric from the costs.** In a classroom, falsely accusing a meme of smearing someone is worse than missing a technique. So precision deserves extra attention there.

## 6. Reading learning curves

![The four curve shapes you must recognise](img/07_learning_curves.png)
<sub>The four curve shapes you must recognise. Blue = training loss, orange = validation loss.</sub>

| What you see | Diagnosis | What to try |
|---|---|---|
| Training and validation both poor, close together | underfitting | bigger model, train longer, better features or a pretrained model, check the learning rate |
| Training very good, validation clearly worse; validation loss rises again | overfitting | early stopping, dropout, weight decay, data augmentation, more data, smaller model, freeze layers, lower learning rate |
| Both losses jump up and down | learning rate too high, unstable training | lower learning rate, warm-up, gradient clipping, larger batches |
| Validation loss below training loss | not necessarily a bug | dropout and other regularisation act only during training; training loss is averaged during the epoch while validation loss is measured at its end; the validation set may be easier; or there is leakage |

Fine-tuning a large pretrained model often overfits within one or two epochs. Early stopping is essential.

> [!NOTE]
> **From the course · Week 7 tutorial, (f) and (g).** (1) Training and validation loss both oscillate. What do you change? (2) The final validation loss is below the training loss. Why can that happen?

<details>
<summary><b>Show the answer</b></summary>

1. Lower the **learning rate** (see the right panel of the learning-curve figure).
2. Regularisation such as dropout is active only in training; training loss is averaged during the epoch while validation loss is measured at its end; the validation set may be easier; or there is leakage.

</details>

## 7. Hyperparameter tuning

Hyperparameters are settings you choose, not weights the model learns: learning rate, batch size, dropout, number of layers.
- Tune them on validation data or with cross-validation.
- The **learning rate** usually matters most. Tune it first.
- **Grid search** tries all combinations. **Random search** samples combinations and often finds good values faster.
- **Greedy (coordinate-wise) search** tunes one setting at a time and fixes it before the next. It is cheap but can miss interactions. A2 uses it, with a pre-stated decision rule for each step.

## 8. Comparing models fairly

**Noise.** With small data, results change with the random seed and the split. A difference of one F1 point can be pure chance.

**Paired comparison.** Evaluate both models on the same folds. Look at the difference fold by fold:

$$
d_f = s_f^{(A)} - s_f^{(B)}, \qquad \text{A is better only if}\;\; \bar{d} > \text{sd}(d_f)
$$

<sub>$s_f^{(A)}, s_f^{(B)}$ = scores of models A and B on fold $f$ · $d_f$ = their difference on that fold · $\bar d$ = mean difference. A hard fold is hard for both models, so this noise cancels. The rule "mean advantage larger than its spread" is the tie rule used in A2.</sub>

**Bootstrap confidence interval** for a test score:
1. Draw $n$ test examples with replacement from the $n$ test examples.
2. Compute the metric on this resample.
3. Repeat 1,000 times.
4. The 2.5th and 97.5th percentiles form a 95% interval.

It shows how much the score would move with a different test sample of the same size.

**Baselines.** Always report a trivial baseline (majority class, random by class frequency). Without it, "F1 = 0.56" means nothing.

## 9. Class imbalance

When some classes are rare:
- Use precision, recall and F1, not accuracy.
- **Weighted loss:** give positive examples of rare classes a larger weight:

$$
L = -\Big[\,\textcolor{#a855f7}{\alpha_1}\,\textcolor{#e5484d}{y}\,\log \textcolor{#22a06b}{p} + \textcolor{#a855f7}{\alpha_0}\,(1 - \textcolor{#e5484d}{y})\,\log(1 - \textcolor{#22a06b}{p})\,\Big]
$$

<sub>$\alpha_1$ = weight for positive examples (large for rare classes) · $\alpha_0$ = weight for negative examples. A2 sets $\alpha_1 = \sqrt{N_{\text{neg}}/N_{\text{pos}}}$ and $\alpha_0 = 1$; the square root is gentler than the raw ratio.</sub>

- **Focal loss:** multiplies the loss by a factor that shrinks for easy examples:

$$
L_{\text{focal}} = -(1 - p_t)^{\textcolor{#a855f7}{\gamma}}\,\log p_t
$$

<sub>$p_t$ = predicted probability of the true class · $\gamma$ = focus strength (often 2). If the model is already sure ($p_t \approx 1$), the factor is ≈ 0, so easy examples barely count.</sub>

- **Resampling:** show rare examples more often.
- **Tune the thresholds.**

## 10. Common mistakes

- Using the test set to choose the threshold or the epoch.
- Reporting accuracy on imbalanced data.
- Reporting only micro-F1 when rare classes matter.
- Declaring a winner from a difference smaller than the seed-to-seed noise.
- Forgetting that in-distribution test data does not prove the model works in a new setting.

## 11. How this appears in A2

- Official test set used once. 5-fold iterative stratified CV for all choices.
- Leakage found and fixed (`474_batch_2`). Label noise documented (same text, different labels).
- Micro-F1, macro-F1, per-technique F1, span F1 (official SemEval metric, verified against the official scorer).
- Majority and prior-random baselines.
- Paired fold comparison with a tie rule. Seed-variance check. Bootstrap intervals on the test set.

## Watch and read

- StatQuest, *Cross validation*: https://www.youtube.com/watch?v=fSytzGwwBVw
- StatQuest, *Bias and variance*: https://www.youtube.com/watch?v=EuBBz3bI-aA
- StatQuest, *Regularization part 1: Ridge (L2)*: https://www.youtube.com/watch?v=Q81RR3yKn30
- Google ML crash course, *Accuracy, precision, recall*: https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
- Google ML crash course, *Thresholds and the confusion matrix*: https://developers.google.com/machine-learning/crash-course/classification/thresholding
- Google ML crash course, *Overfitting*: https://developers.google.com/machine-learning/crash-course/overfitting/overfitting
- scikit-learn guide, *Cross-validation* (clear diagrams): https://scikit-learn.org/stable/modules/cross_validation.html
- Goodfellow, Bengio, Courville, *Deep Learning*, chapter 11 *Practical methodology* (free online): https://www.deeplearningbook.org/

---

## Cheat sheet

> [!IMPORTANT]
> **The 9 things to remember**
> 1. Workflow: **goal + metric → baseline → instrument → read curves → change one thing → test once.**
> 2. Train = fit, validation = choose, **test = once**. Val/test must look like deployment.
> 3. Small data → **k-fold CV**; multi-label → iterative stratification.
> 4. Leakage: duplicates, near-duplicates, global normalisation, future info, tuning on test.
> 5. P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R). Accuracy lies under imbalance.
> 6. **Micro** = pool counts (frequent classes rule). **Macro** = average per class (rare classes count).
> 7. Curves: both bad → underfit; big gap → overfit; zig-zag → LR too high; val < train → regularisation, timing, easier val, or leakage.
> 8. Compare **paired** on the same folds; a win must beat the noise. Report bootstrap CIs.
> 9. Imbalance: weighted BCE ($\alpha_1$ up for rare positives), focal loss, resampling, tuned thresholds.

> [!TIP]
> **Memory hooks**
> - **Precision = "how Pure are my alarms?" Recall = "how many did I Recover?"**
> - **Test set = a sealed exam paper.** If you peek, your grade means nothing.
> - **Micro = majority rules. Macro = every class gets one vote.**
> - **Learning curves as a doctor's chart:** two low lines = weak patient (underfit); lines drifting apart = memorising patient (overfit); a fever curve = learning rate too hot.
> - **Bootstrap = "what if I had drawn a different class of 200 students?"**

> [!TIP]
> **How to learn this chapter**
> - **Compute** P, R and F1 for: 20 true cases, 25 flagged, 15 correct. (P = 0.60, R = 0.75, F1 = 0.67.)
> - **Sketch the four learning-curve shapes** from section 6 and write the diagnosis under each.
> - **Find the leak:** list three ways a time-series model could see the future by accident.
