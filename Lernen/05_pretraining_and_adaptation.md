# 05 · Pretraining, foundation models and adaptation (transfer learning, LoRA)

> [!NOTE]
> **Colour code for all formulas:** $\textcolor{#3b82f6}{\text{input / data}}$ · $\textcolor{#e8833a}{\text{learned weights}}$ · $\textcolor{#22a06b}{\text{model output}}$ · $\textcolor{#e5484d}{\text{true label}}$ · $\textcolor{#a855f7}{\text{setting you choose}}$ · $\textcolor{#64748b}{\text{frozen weights}}$

**The problem.** Good deep models need a lot of labelled data. Labels are expensive. A2 has only 750 labelled memes. The solution is to learn general knowledge first from huge amounts of unlabelled data, and then adapt it with the few labels we have.

**Course material:** Week 5 notebook, *The transfer learning pipeline* with its seven steps (`Week 5/L5_Week5_TransferLearning.ipynb`). Week 9 slides (`Week 9/main_week9_representation_foundation.pdf`), from *Transfer Learning* to *Bias and responsible use*. Week 9 notebook (`Week 9/Lecture_Week09_SSL_LoRA.ipynb`): *Part 2, Self-supervised learning* and *Part 3, Feature extraction, fine-tuning and LoRA*. Week 9 tutorial (`Week 9/COSC2779_Tute9_tex_answers.pdf`). Week 9 lab (`Week 9/week9_foundation_models_lab.ipynb`): *Part 1 decision framework*, *Part 5 LoRA* with *Other PEFT methods*, *Part 7 scenario reasoning*. Week 5 and Week 8 labs: *linear probing, then fine-tuning*.


> [!TIP]
> **Play with it** (interactive, in the browser). Do this before or after reading; seeing it move is the fastest way in.
> - [Jay Alammar: The illustrated BERT](https://jalammar.github.io/illustrated-bert/): pictures of pretraining (masked words) and then fine-tuning with a new head.
> - [Transformer Explainer](https://poloclub.github.io/transformer-explainer/): see what a pretrained model already 'knows' before any fine-tuning: its next-word probabilities.

---

## 1. Transfer learning: the basic idea

A network trained on a large task learns features that are useful for many other tasks.
- A CNN trained on ImageNet learns edges in early layers, textures in middle layers and object parts in late layers.
- A language model trained on billions of words learns grammar, word meaning and some world knowledge.

**Transfer learning** reuses those learned weights for a new task with less data.

The standard recipe (Week 5, same idea for text in Week 9):
1. Pick a related source task with lots of data.
2. Get a model trained on it (usually download it).
3. Find the feature layer, the last layer before the task-specific output.
4. Remove the old output layer ("head"). Attach a new head for your task.
5. Freeze the pretrained layers.
6. Train only the new head.
7. Optionally unfreeze some top layers and fine-tune them with a **small learning rate**.

**Why a small learning rate for pretrained layers?** Their weights are already good. Large updates would destroy what they learned, especially early in training while the new head is still random. The Week 5 notebook uses a 10 times smaller learning rate for fine-tuning than for training the head.

**Linear probing.** The labs (Weeks 5, 6 and 8) call steps 5–6 **linear probing**: the backbone is frozen and only a linear head is trained on its features. It is the same as feature extraction with a linear head. Step 7 is **fine-tuning**: continue with the *same* model, unfreeze the backbone (or its top layers), make a new optimiser with a much smaller learning rate, and train on.

> [!NOTE]
> **From the course · Practice exam, A.2 and C.1 (e).** Which steps belong to fine-tuning in transfer learning? How would you adapt an ImageNet-pretrained ResNet to six waste categories?

<details>
<summary><b>Show the answer</b></summary>

- Replace the classification head with a new 6-class head.
- Freeze the pretrained layers and train the head first.
- Then optionally unfreeze some **top** layers and fine-tune them with a **smaller learning rate**. Both head and base weights can be trained in this phase. Keeping every base layer frozen forever is feature extraction, not fine-tuning.

</details>

## 2. Self-supervised learning (SSL)

How do you pretrain without labels? You let the data label itself. Hide part of the input and ask the model to predict it.

**Masked language modelling (MLM),** used by BERT: mask about 15% of the tokens and predict them from the rest. "The [MASK] sat on the mat" → "cat".

$$
L_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log \textcolor{#22a06b}{p}\big(\textcolor{#e5484d}{x_t} \mid \textcolor{#3b82f6}{\mathbf{x}_{\setminus \mathcal{M}}}\big)
$$

<sub>$\mathcal{M}$ = the set of masked positions · $x_t$ = the original token at position $t$ (the "free" label) · $\mathbf{x}_{\setminus\mathcal{M}}$ = the sentence with those tokens hidden · $p(\cdot)$ = the model's predicted probability. It is just cross-entropy on the hidden words.</sub>

**Next-token prediction,** used by GPT: predict each token from all previous ones.

$$
L_{\text{LM}} = -\sum_{t} \log \textcolor{#22a06b}{p}\big(\textcolor{#e5484d}{x_t} \mid \textcolor{#3b82f6}{x_1, \dots, x_{t-1}}\big)
$$

<sub>$t$ = position in the text · $x_t$ = the true token at position $t$ · $x_1 \dots x_{t-1}$ = all tokens before it · $p(\cdot \mid \cdot)$ = the model's probability for $x_t$ given ("$\mid$") the earlier tokens · $\sum_t$ = add up over all positions.</sub>

**Contrastive learning,** used by CLIP and SimCLR: pull matching pairs together (an image and its caption, two crops of the same photo) and push non-matching pairs apart.

The labels are free, so the model can train on billions of examples. To solve these tasks well, it has to learn grammar, meaning and context.

**Proxy (pretext) tasks for images** (Week 9 slides). The idea is the same: invent a task whose answer is already in the data.
- **Inpainting:** black out a region and reconstruct it. The slides' medical example: blank out a small region of a CT scan and ask the network to recreate the original scan. Unlabelled scans are plentiful, labelled ones are not.
- **Rotation prediction:** rotate an image by 0°, 90°, 180° or 270° and predict the angle. To know which way is "up", the network must recognise objects.
- **Temporal order** ("shuffle and learn"): show video frames and ask whether they are in the right order. This teaches motion and how things change.
- **Denoising and masking:** corrupt the input and reconstruct it (the denoising autoencoder of doc 11, the masked ECG signals in the Week 11 lab).

![Four proxy tasks: rotation, inpainting, temporal order, masked tokens](img/05_ssl_pretext.png)
<sub>Four proxy tasks. In each one the correct answer comes from the data itself, so no human has to label anything.</sub>

**The SSL recipe:** (1) train on the proxy task with lots of unlabelled data; (2) throw away the proxy head; (3) use the learned encoder for the real task with transfer learning (section 1) and the few labels you have. Self-supervision needs a large *unlabelled* dataset. Transfer learning needs a pretrained model on a *related* task. Both answer the same question: how to train deep models without many labels.

## 3. Foundation models

A **foundation model** is a large model pretrained on broad data at scale, usually with self-supervision, that can be adapted to many downstream tasks. Examples: BERT, GPT, Gemini, CLIP.

Why they dominate:
- **Few labels needed.** The model already "knows" language or images.
- **Pay once, reuse often (amortised cost).** Pretraining is very expensive but done once. Adapting is cheap, and the cost is shared across every downstream task.
- **They work across tasks and even across modalities** (text, images, audio). The field moved from one bespoke model per task to one adaptable backbone.

**Scaling laws.** When you make the model, the data and the compute larger, the test loss falls smoothly and predictably. It follows a **power law**: on a plot with logarithmic axes it is a straight line. So "bigger" reliably helps, and labs can predict how good a larger model will be before training it.

**Emergent abilities.** Some abilities are nearly absent in small models and appear quite suddenly beyond a certain scale. The standard example is **in-context learning**: the model learns a task from a few examples written into the prompt, without any weight update.

![Scaling law and emergent ability](img/05_scaling_law.png)
<sub>Left: loss falls as a straight line on a log–log plot (a power law). Right: an emergent ability stays at chance level, then jumps. Schematic shapes, not real measurements.</sub>

## 4. The adaptation spectrum, from cheapest to most expensive

| Strategy | What is trained | Advantages | Disadvantages |
|---|---|---|---|
| **Zero-shot** | nothing; you only write a prompt | no labels, instant baseline | sensitive to wording; usually worse than a tuned model; output format can break; inherits the model's blind spots |
| **Feature extraction** | only a new head; the backbone is frozen | fast; few parameters; robust on tiny data | the representation cannot adapt to the task |
| **PEFT, e.g. LoRA** | a small number of extra parameters; the backbone is frozen | about 1% of parameters; small memory; one tiny adapter per task | slightly less capacity than full fine-tuning |
| **Full fine-tuning** | every weight | usually the best accuracy with enough data | expensive; one full model copy per task; can overfit; **catastrophic forgetting** |

![What is trained in each strategy](img/05_adaptation_spectrum.png)
<sub>What is trained in each strategy. Grey layers are frozen, orange ones are trained.</sub>

**Catastrophic forgetting:** fine-tuning on a new task can overwrite what the model knew before.

> [!NOTE]
> **From the course · Week 9 tutorial, (d)–(f) and Practice exam, B.3.** 2,000 labelled emails, a modest GPU. Which adaptation strategy, and why? What are the pros and cons of feature extraction vs. full fine-tuning?

<details>
<summary><b>Show the answer</b></summary>

- Try **feature extraction and/or LoRA** first: few trainable parameters, cheap, less overfitting on 2,000 examples. Run zero-shot as a quick baseline.
- **Feature extraction:** + fast, robust on small data; − the representation cannot adapt.
- **Full fine-tuning:** + best accuracy with enough data; − expensive, one model copy per task, can overfit, catastrophic forgetting.
- Always validate empirically. (In A2, full fine-tuning won even on 750 memes.)

</details>

**Which to pick?** It depends on data size, compute budget and how many tasks you serve.
- Tiny data and little compute: feature extraction or LoRA.
- Enough data: full fine-tuning usually wins.
- Many tasks on one server: LoRA, because each task needs only a small adapter.
- Always check by experiment, and always run zero-shot as a cheap baseline.

## 5. LoRA in detail

**Observation:** when you fine-tune, the change in each weight matrix, $\Delta W$, has a simple structure. It can be approximated by a matrix of low rank.

**LoRA** (low-rank adaptation) keeps the pretrained matrix frozen and learns only the change, in a factored form:

$$
\boxed{\; W = \textcolor{#64748b}{W_0} + \Delta W = \textcolor{#64748b}{W_0} + \frac{\textcolor{#a855f7}{\alpha}}{\textcolor{#a855f7}{r}}\,\textcolor{#e8833a}{B}\,\textcolor{#e8833a}{A} \;}
\qquad
\textcolor{#64748b}{W_0} \in \mathbb{R}^{d \times k},\;\; \textcolor{#e8833a}{B} \in \mathbb{R}^{d \times r},\;\; \textcolor{#e8833a}{A} \in \mathbb{R}^{r \times k},\;\; \textcolor{#a855f7}{r} \ll \min(d, k)
$$

<sub>$W_0$ = pretrained weight matrix, frozen · $A$, $B$ = the two small trainable matrices · $r$ = rank, the "width" of the update (for example 8 or 16) · $\alpha$ = scaling factor that sets the strength of the update · $d, k$ = the matrix's output and input sizes.</sub>

**Simply put:** do not touch the big pretrained matrix. Learn a small correction next to it and add it on.

Applied to an input $\mathbf{x}$, the layer computes:

$$
\textcolor{#22a06b}{\mathbf{y}} = \textcolor{#64748b}{W_0}\,\textcolor{#3b82f6}{\mathbf{x}} + \frac{\textcolor{#a855f7}{\alpha}}{\textcolor{#a855f7}{r}}\,\textcolor{#e8833a}{B}\big(\textcolor{#e8833a}{A}\,\textcolor{#3b82f6}{\mathbf{x}}\big)
$$

<sub>The original path is untouched. A thin side path first squeezes $\mathbf{x}$ down to $r$ numbers ($A\mathbf{x}$) and then expands it back ($B\cdot$).</sub>

![LoRA for one 768 × 768 matrix: the frozen path stays as it is; a thin trainable side path (A then B) adds the update](img/05_lora.png)
<sub>LoRA for one 768 × 768 matrix: the frozen path stays as it is; a thin trainable side path (A then B) adds the update.</sub>

**Parameter count, worked example.** For a 768 × 768 matrix:

$$
\underbrace{768 \times 768}_{\text{full fine-tuning}} = 589{,}824
\qquad\text{vs.}\qquad
\underbrace{\textcolor{#a855f7}{r}\,(d + k) = 16 \times (768 + 768)}_{\text{LoRA},\ r = 16} = 24{,}576 \;\;(\approx 4\%)
$$

<sub>$d \times k$ = 768 × 768, the size of the original matrix · $r$ = LoRA rank (16) · $r(d+k)$ = values in $A$ ($r \times k$) plus values in $B$ ($d \times r$).</sub>

Applied only to the query and value matrices in each layer, LoRA trains well under 1% of the whole model.

> [!NOTE]
> **From the course · Week 9 tutorial, (g)–(h).** Why is full fine-tuning costly when you serve many tasks, how does LoRA fix it, and what does the rank $r$ control?

<details>
<summary><b>Show the answer</b></summary>

- Full fine-tuning stores a full copy of every weight per task. LoRA freezes $W_0$ and learns $\Delta W = BA$ with $r \ll \min(d,k)$; adapters are tiny, swappable and mergeable (no extra latency).

<sub>$W_0$ = the pretrained weight matrix (frozen) · $\Delta W$ ("delta W") = the change learned for this task · $B$, $A$ = the two small trainable matrices · $r$ = rank · $\ll$ = "much smaller than" · $d, k$ = the matrix's two sizes.</sub>
- $r$ sets the capacity of the update and the parameter count $r(d+k)$ per layer: too small underfits, larger gives a bit more accuracy with diminishing returns.

</details>

More details:
- **Initialisation.** $B$ starts at zero, so $BA = 0$. Training starts exactly from the pretrained model.
- **Rank $r$** sets the capacity of the update. Too small and the model underfits. Larger is a bit better, with diminishing returns and more parameters.
- **No extra cost at inference.** After training, $BA$ can be added into $W_0$ once. The model then has the original size and speed.
- **Where to put it.** Usually the query and value projections of attention.

In A2, LoRA is implemented from scratch (`LoRALinear` in the notebook). The Week 9 notebook does the same.

**Why does a low rank suffice?** Research found that the update $\Delta W$ needed for fine-tuning has a low **intrinsic rank**: it changes the matrix only along a few directions. Fine-tuning adapts what the model already knows; it does not rebuild it.

**Why the query and value matrices?** This is a lab question. In the LoRA paper, with a fixed budget of trainable parameters, adapting $W_Q$ and $W_V$ together worked best. They control *what each token looks for* and *what it passes on*. It is a good default, not a law: adapting all linear layers also works and is common today.

### Other PEFT methods

LoRA is one member of a family of **parameter-efficient fine-tuning (PEFT)** methods. All keep the pretrained weights frozen and train a few new parameters. They differ in *where* the new parameters sit.

![Where each PEFT method adds its trainable parameters](img/05_peft_methods.png)
<sub>Grey = frozen pretrained Transformer block; orange = the only trained parameters.</sub>

| Method | What is trained | Trainable share (Week 9 lab) | Extra cost at inference |
|---|---|---|---|
| **Prompt tuning** | a few learned "virtual token" vectors prepended to the input | < 0.1% | almost none (a few extra tokens) |
| **Prefix tuning** | learned key/value vectors added to the attention of every layer | ~0.1–1% | small |
| **Adapters** | small bottleneck layers (down-project, non-linearity, up-project) inserted in each block | ~2–5% | small, permanent (extra layers) |
| **LoRA** | low-rank update $BA$ on existing weight matrices | ~0.5–2% | **none** after merging |
| **QLoRA** | LoRA on top of a frozen model stored in 4-bit numbers | like LoRA | none for LoRA; the base stays 4-bit |

**QLoRA** stores the frozen base model in 4-bit precision instead of 16 or 32 bits. That cuts its memory by a factor of 4 to 8, so large models can be fine-tuned on a single GPU.

**Why is full fine-tuning so memory-hungry?** Training needs, for every weight, the weight itself, its gradient and (with Adam) two extra optimiser numbers. PEFT needs gradients and optimiser state only for the few new parameters.

## 6. Intermediate and domain-adaptive training

Sometimes there is related labelled or unlabelled data that is closer to the target than general text.
- **Domain-adaptive pretraining:** continue self-supervised training on text from the target domain. Twitter-RoBERTa is RoBERTa after extra MLM training on tweets.
- **Intermediate-task training:** first fine-tune on a related labelled task, then on the target task. A2's experiment E9 does this with the PTC corpus (propaganda in news articles), then fine-tunes on memes.

## 7. Zero-shot with large language models

You can prompt a generative model: "Which of these 20 techniques does this meme use? Answer in JSON."
- **Few-shot / in-context learning:** add a few solved examples to the prompt. The model picks up the pattern from them. There is still **no weight update**; the examples only live in the prompt.
- **CLIP zero-shot for images:** embed the image and a set of text prompts such as "a photo of a dog", "a photo of a cat". Pick the class whose text embedding is closest to the image embedding. No training on dogs and cats needed.
- No training data needed.
- Results depend on the prompt wording.
- Answers can come in the wrong shape and need careful parsing.
- With temperature above 0, the same input can give different answers.
- **Data contamination:** if public test data was in the model's training data, its score is too optimistic.
- Hosted models send the data to an outside server, which may be forbidden (privacy).

## 8. Risks of foundation models

- **Inherited bias** from web-scale training data. This can cause different error rates for different groups.
- **Distribution shift.** The model can fail confidently on inputs unlike its training data.
- **Shortcut learning** and **hallucination** for generative models.
- **Mitigations:** evaluate per group (disaggregated metrics), curate the adaptation data, keep humans in the loop for important decisions.

## 9. Common mistakes

- "LoRA makes inference slower." No, the adapter can be merged. (Classic adapters, however, do add a little permanent cost.)
- "Few-shot prompting trains the model." No, the weights do not change; the examples only sit in the prompt.
- Confusing **transfer learning** (needs a model pretrained on a related task) with **self-supervised learning** (needs lots of unlabelled data and a proxy task).
- "With little data, always freeze." Not always. Test it. In A2 full fine-tuning won even with 750 examples.
- Using the same learning rate for a fresh head and for pretrained layers. The head needs a larger one.
- Comparing methods unfairly. Each method needs its usual hyperparameters. LoRA, for example, typically needs a larger learning rate than full fine-tuning.

## 10. How this appears in A2

- Pretrained encoder: DeBERTa-v3 (pretrained by its authors with self-supervision, no persuasion labels).
- Experiment E2 compares frozen encoder, LoRA and full fine-tuning on the same folds.
- Experiment E9 tests intermediate training on PTC news data.
- C2 compares our trained model with zero-shot LLMs (Gemini Pro, Gemini Flash-Lite, Ornith 9B).

## Watch and read

**Official course videos** (from the Canvas week pages; Canvas Studio links need your RMIT login). Watch these first: the exam is built on this material.
- Week 9: IBM Technology, *Is fine-tuning still needed? LLMs, RAG and LoRA*: https://www.youtube.com/watch?v=-W2JdSl1v48
- Week 9: SH AI Academy, *Parameter-efficient fine-tuning explained* (the Week 9 before-class video): https://www.youtube.com/watch?v=-c2aob9hH10
- **Week 5 pre-recorded lectures** (CNN architectures and transfer learning for images): listed in [doc 10](10_cnns_and_vision.md#watch-and-read).
- Week 10: Ari Seff, *How ChatGPT is trained*, shows pretraining and adaptation of a large language model: listed in [doc 11](11_generative_models.md#watch-and-read).

**More videos and readings:**

- Jay Alammar, *The illustrated BERT, ELMo and co.* (how NLP cracked transfer learning): https://jalammar.github.io/illustrated-bert/
- Umar Jamil, *BERT explained: training, inference, fine-tuning*: https://www.youtube.com/watch?v=90mGPxR2GgY
- Andrej Karpathy, *Intro to large language models* (1 h talk): https://www.youtube.com/watch?v=zjkBMFhNj_g
- Sebastian Raschka, *Parameter-efficient LLM fine-tuning with LoRA* (blog): https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html
- Hugging Face PEFT docs, *LoRA conceptual guide*: https://huggingface.co/docs/peft/conceptual_guides/lora
- Hugging Face course, *Fine-tuning a pretrained model*: https://huggingface.co/learn/llm-course/chapter3/1
- LoRA paper (Hu et al., 2021): https://arxiv.org/abs/2106.09685
- Hugging Face PEFT docs, *Adapters* (LoRA, prefix and prompt tuning compared): https://huggingface.co/docs/peft/conceptual_guides/adapter
- Lilian Weng, *Self-supervised representation learning* (proxy tasks for images, video and text): https://lilianweng.github.io/posts/2019-11-10-self-supervised/
- Papers: rotation prediction https://arxiv.org/abs/1803.07728 · shuffle and learn https://arxiv.org/abs/1603.08561 · scaling laws (Kaplan et al.) https://arxiv.org/abs/2001.08361 · emergent abilities (Wei et al.) https://arxiv.org/abs/2206.07682 · adapters https://arxiv.org/abs/1902.00751 · prefix tuning https://arxiv.org/abs/2101.00190 · prompt tuning https://arxiv.org/abs/2104.08691 · QLoRA https://arxiv.org/abs/2305.14314
- BERT paper (Devlin et al., 2018): https://arxiv.org/abs/1810.04805
- *Dive into Deep Learning*, NLP pretraining: https://d2l.ai/chapter_natural-language-processing-pretraining/index.html

---

## Cheat sheet

> [!IMPORTANT]
> **The 12 things to remember**
> 1. Transfer recipe: **new head → freeze → train head → (unfreeze top, small LR)**.
> 2. Self-supervision = labels from the data itself: masked tokens (BERT), next token (GPT), matching pairs (CLIP).
> 3. Foundation model = pretrained at scale, adapted to many tasks.
> 4. Spectrum, cheap → expensive: **zero-shot → feature extraction → LoRA → full fine-tuning**.
> 5. LoRA: $W = W_0 + BA$, $W_0$ frozen, $r \ll d$, params $= r(d+k)$ per matrix, $B$ starts at 0.
> 6. LoRA adds **no inference cost** (merge $BA$ into $W_0$).
> 7. Full fine-tuning: best with enough data, but expensive and can forget.
> 8. Zero-shot LLMs: prompt-sensitive, format can break, stochastic, may be contaminated, may break privacy rules. Few-shot = examples in the prompt, no weight update.
> 9. Image proxy tasks: **inpainting, rotation, temporal order**, denoising. Recipe: proxy task on unlabelled data → keep the encoder → transfer.
> 10. **Scaling law:** loss falls as a power law of size/data/compute (straight on log–log). **Emergent:** abilities like in-context learning appear only at scale.
> 11. PEFT family: **prompt tuning** (virtual tokens), **prefix tuning** (K/V per layer), **adapters** (bottleneck layers, small inference cost), **LoRA** (low-rank, mergeable), **QLoRA** (LoRA on a 4-bit base).
> 12. Linear probing = frozen backbone + linear head; then fine-tune with a much smaller LR.

> [!TIP]
> **Memory hooks**
> - **Transfer learning = hiring an experienced chef.** You do not teach them to cook. You only teach your restaurant's menu (the new head).
> - **Feature extraction vs fine-tuning = renting a flat vs renovating it.** Renting: you only bring furniture (the head). Renovating: you may knock down walls (change the backbone).
> - **LoRA = sticky notes on a textbook.** The book ($W_0$) stays untouched. You add small notes ($BA$). Later you can copy the notes into the book (merge).
> - **"Low rank, low cost."**
> - **Catastrophic forgetting = cramming for one exam and forgetting last semester.**

> [!TIP]
> **How to learn this chapter**
> - **Compute** the LoRA parameter count for a 1024 × 4096 matrix with $r = 8$. (Answer: $8 \times 5120 = 40{,}960$.)
> - **Order the four strategies** by cost from memory, and give one pro and one con for each.
> - **Explain to a friend** why masked-word prediction teaches a model grammar without any labels.
> - **Invent a proxy task** for unlabelled ECG signals or satellite images, and say what the network must learn to solve it.
> - **Sketch one Transformer block** and mark where prompt tuning, prefix tuning, adapters and LoRA put their parameters.
