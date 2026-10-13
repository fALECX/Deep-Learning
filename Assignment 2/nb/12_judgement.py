# %% [markdown]
# ## 7. Ultimate judgement
#
# ### 7.1 The evidence in one table
# Every number below is computed earlier in this notebook. They are collected here so the judgement can be checked
# against them.

# %%
card = pd.Series({
    "Final configuration": BEST,
    "Test micro-F1 (95% CI)": f"{test_tab.loc['final ensemble (5 models)', 'micro_f1']:.3f} {test_tab.loc['final ensemble (5 models)', 'micro_f1 95% CI']}",
    "Test macro-F1": f"{test_tab.loc['final ensemble (5 models)', 'macro_f1']:.3f}",
    "Test precision / recall (labels)": f"{test_tab.loc['final ensemble (5 models)', 'micro_p']:.3f} / {test_tab.loc['final ensemble (5 models)', 'micro_r']:.3f}",
    "Test span F1 (95% CI)": f"{test_tab.loc['final ensemble (5 models)', 'span_f1']:.3f} {test_tab.loc['final ensemble (5 models)', 'span_f1 95% CI']}",
    "SemEval-2021 best (labels / spans)": "0.593 / 0.482",
    "Majority baseline on test (labels)": f"{test_tab.loc['majority baseline', 'micro_f1']:.3f}",
    "Techniques with usable F1 (≥0.4) on test": ", ".join(pt.index[pt.F1 >= 0.4]),
    "Best zero-shot LLM (strict, run 1)": f"{best_llm[0]} / {best_llm[1]}: micro-F1 "
                                          f"{f1_score(Y_test, PRED_LLM[best_llm][0], average='micro', zero_division=0):.3f}",
    "Label/span heads disagree": f"{100 * (tot['label only'] + tot['span only']) / tot.sum():.1f}% of flagged pairs",
    "Counterfactual side swap: label sets changed": f"{100 * flip[changed].mean():.1f}% of {changed.sum()} memes",
    "Same-side control swap: label sets changed": f"{100 * flip_ctrl[both_changed].mean():.1f}% (side swap on the same memes: {100 * flip[both_changed].mean():.1f}%)",
    "Everyday AU teen memes wrongly flagged": f"{int(PR['labels'][~pers].any(1).sum())}/{int((~pers).sum())}",
    "AU probe persuasive micro-F1": f"{f1_score(Y_probe[pers], PR['labels'][pers], average='micro', zero_division=0):.3f}",
    "Latency per meme, 5-model ensemble (median)": f"{lat5[0]:.0f} ms on {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}",
}, name="value")
display(card.to_frame())

# %% [markdown]
# ### 7.2 Judgement
# **Recommendation: do not deploy this system as an autonomous technique detector next term. Deploy it, if at all,
# as a teacher-led discussion aid for a few core techniques, with the C3 display rule, and plan a re-evaluation on
# Australian data.**
#
# **Why this model is the best available under the constraints.**
# * It is the best of the configurations we tested on clean, paired cross-validation: DeBERTa-v3, fully fine-tuned,
#   our label-wise attention head, joint label + span training, class-weighted loss, a five-fold ensemble.
# * On the held-out test set it reaches micro-F1 0.52 and span F1 0.41: clearly above the baselines, below the 2021
#   state of the art that used now-forbidden propaganda pretraining.
# * It satisfies every constraint: it runs locally, answers in 0.14 s on a T4 using under 2 GB of weights, is
#   deterministic, and uses no persuasion-trained checkpoint and no external data.
# * Among systems that may legally run in a classroom, nothing we tested comes close: a local 9B LLM reaches 0.20.
#
# **Why it is not ready to run unsupervised.**
# 1. **Few techniques work.** Only Loaded Language, Name calling, Smears, Whataboutism, Exaggeration and (on very few
#    examples) Appeal to authority reach F1 ≥ 0.4. Twelve of the twenty techniques are almost never found.
# 2. **It over-accuses.** Precision is 0.48: about half of all shown labels are wrong. Missed subtle techniques turn
#    into extra "attack" labels.
# 3. **It is sensitive to names.** One in three label sets change when a name is swapped, and neutral sentences about
#    politicians can be flagged. We found no side-specific bias, but students will experience the instability as bias.
# 4. **It does not transfer.** On Australian-style memes accuracy roughly halves, and 8 of 14 harmless teen memes are
#    flagged, while the model stays confident.
# 5. **A general model is better at the task** (Gemini Pro, 0.57 on test and 0.75 on the Australian probe set), but it
#    is not allowed on student data.
#
# **Realistic chance of success.** In a classroom where the teacher uses the output as a prompt ("the tool thinks
# this is Name calling; which words would do that?"), the tool can work for the core techniques today. As a system
# that names the techniques in a student's meme by itself, it would mislead students in a large share of cases.
#
# **What would change this verdict.**
# * Label 300–500 Australian memes (teacher-curated, public sources, at least two annotators) and fine-tune on them.
# * Use the frontier LLM **offline on public memes only** to pre-label more Australian training data, then distil it
#   into the local model. No student data would leave the device.
# * Augment training with name masking and benign everyday memes to cut the name sensitivity and the false alarms.
# * Re-run C1–C4 on that data before every term, and track teacher overrides as a live quality metric.
#
# ### 7.3 Limitations
# * **Small, noisy data:** 750 development memes; duplicated texts carry different labels; inter-annotator agreement
#   α ≈ 0.77. Rare techniques have 1–5 training examples.
# * **Selection optimism:** cross-validation was used for every choice, so its 0.56 overstates the 0.52 found on test.
#   The test set itself is only 200 memes (CI ±0.05).
# * **Training budget:** validation label F1 was still rising slowly at epoch 15 in some folds.
# * **Text only:** the image is ignored, although many techniques live in it (by design of the task).
# * **Evaluation reach:** everything except C4 is measured on 2020 US/Canadian Facebook memes. The Australian evidence
#   is synthetic and small. Political side is detected with keywords.
# * **Reproducibility:** GPU kernels are not bit-deterministic. A re-run gives the same conclusions, but individual
#   numbers can move by about the seed standard deviation (≈0.01).

# %% [markdown]
# ## 8. Declarations
#
# **Pretrained weights (downloaded, run locally).** `microsoft/deberta-v3-base`, `roberta-base` and
# `cardiffnlp/twitter-roberta-base-2021-124m`. All three were pretrained with self-supervised objectives only (masked or
# replaced-token prediction). None was trained on persuasion-, propaganda- or rhetoric-labelled data.
#
# **Additional public data.** The Propaganda Techniques Corpus (SemEval-2020 Task 11; Da San Martino et al., 2020;
# Zenodo record 3952415, CC-BY 4.0) was used in experiment E9 only. It contains news articles, not memes. No meme of
# our corpus occurs in it (checked in E9), so it cannot contain the labels of our evaluation set. Whether it is part
# of the final model is decided by E9 and printed there.
#
# **Zero-shot comparison (C2).** Responses were collected outside the notebook with `c2_llm/run_llm.py` and are
# embedded above.
# - Gemini 3.1 Pro and Gemini 3.1 Flash-Lite, through the grsai.ai gateway (Gemini-native `generateContent` API). The
#   version is recorded from the `modelVersion` field of each response.
# - Ornith-1.5-9B (Q4_K_M GGUF), run locally in LM Studio, with and without its reasoning mode.
# - Dates, temperature and full prompts are printed in Section 6. Only public test memes were sent to hosted models.
#   No student data exists in this project.
#
# **Australian probe set (C4).** The 44 probe memes were written for this assignment and labelled by a single
# annotator, with the help of an AI assistant, and checked by the student. They are synthetic. They are used only to
# look for failure modes, not to estimate accuracy.
#
# **Use of AI assistance.** An AI coding assistant (Claude, Anthropic) helped to write code and draft text. All design
# decisions, experiments and conclusions were reviewed by the student, who takes responsibility for them.
#
# ## 9. References
# - Dimitrov, D. et al. (2021). SemEval-2021 Task 6: Detection of persuasion techniques in texts and images. *SemEval-2021*, 70–98.
# - Da San Martino, G. et al. (2020). SemEval-2020 Task 11: Detection of propaganda techniques in news articles. *SemEval-2020*.
# - He, P., Gao, J. and Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR*.
# - Liu, Y. et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
# - Loureiro, D. et al. (2022). TimeLMs: Diachronic language models from Twitter. *ACL demo*.
# - Hu, E. et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
# - Mullenbach, J. et al. (2018). Explainable prediction of medical codes from clinical text (CAML, label-wise attention). *NAACL*.
# - Lin, T.-Y. et al. (2017). Focal loss for dense object detection. *ICCV*.
# - Sechidis, K., Tsoumakas, G. and Vlahavas, I. (2011). On the stratification of multi-label data. *ECML PKDD*.
# - Phang, J., Févry, T. and Bowman, S. (2018). Sentence encoders on STILTs: supplementary training on intermediate labeled-data tasks. *arXiv:1811.01088*.
# - Mosbach, M., Andriushchenko, M. and Klakow, D. (2021). On the stability of fine-tuning BERT. *ICLR*.
# - Sundararajan, M., Taly, A. and Yan, Q. (2017). Axiomatic attribution for deep networks (Integrated Gradients). *ICML*.
# - Ribeiro, M. T. et al. (2020). Beyond accuracy: behavioral testing of NLP models with CheckList. *ACL*.
# - Caruana, R. (1997). Multitask learning. *Machine Learning*, 28, 41–75.
