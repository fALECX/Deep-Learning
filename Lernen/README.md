# Learning material: COSC2779 Deep Learning

Start here. The material assumes no prior knowledge. For the exam, follow the day plan in section 0.

## 0. Day-by-day plan until the exam (Tue 27 October, 12:30)

**One course week per day, in course order.** The chapters are sorted by topic, so some days use only parts of a chapter. The official videos of each week are listed at the top of the chapter's **Watch and read** section. File paths are from the project folder (`Deep Learning/`).

**The routine for every day (about 4 hours):**
1. **Warm-up, 15 min:** answer yesterday's questions in [QUESTIONS_AND_SUMMARY.md](QUESTIONS_AND_SUMMARY.md) again, without looking.
2. **Watch** the official videos of the week (Canvas videos at 1.25× speed are fine).
3. **Read** the listed sections with pen and paper. Keep the *Show the answer* boxes closed until you have written your own answer.
4. **Tutorial:** answer the question PDF on paper, 2–4 sentences per answer. Then compare with the answer PDF.
5. **Self-test:** the listed questions in QUESTIONS_AND_SUMMARY.
6. **Cheat sheet:** add 3–5 candidate lines to one running list. On 26 October you only condense it.
7. Tick off every section and task as you finish it. If the boxes cannot be clicked in your preview, open the file as text (right-click the tab → *Reopen Editor With…* → *Text Editor*) and change `[ ]` to `[x]`.

### Thu 15 Oct · Week 1: machine-learning fundamentals
- [ ] Videos: Week 1 pre-recorded lecture ([doc 07](07_evaluation_and_workflow.md#watch-and-read)). Optional: the Hinton documentary.
- **Read [07](07_evaluation_and_workflow.md)**, one box per section:
  - [ ] 0. What machine learning is (Week 1)
  - [ ] 2. Data splits
  - [ ] 3. Cross-validation
  - [ ] 4. Data leakage
  - [ ] 5. Metrics
  - [ ] 9. Class imbalance
- **Read [01](01_neural_network_basics.md)**, only this part (Tutorial 2 asks about it):
  - [ ] 8. Overfitting, underfitting, regularisation (only the start, up to 8.1)
  - [ ] 8.1 The bias–variance trade-off
- [ ] Tutorial 2: `Week 2/COSC2779_Tute2.pdf` → answers `Week 2/COSC2779_Tute2_sol.pdf`.
- [ ] Questions: 07 no. 3, 4, 7, 8, 9, 10 · 01 no. 6, 11.

### Fri 16 Oct · Week 2: feed-forward networks
- [ ] Videos: Week 2 parts 1–6 ([doc 01](01_neural_network_basics.md#watch-and-read)).
- **Read [01](01_neural_network_basics.md)**, one box per section:
  - [ ] 1. What a model is
  - [ ] 2. One neuron
  - [ ] 3. Layers and networks
  - [ ] 4. The output layer and the loss must match the task
- [ ] Tutorial 3 (it reviews feed-forward networks): `Week 3/COSC2779_Tute3.pdf` → `Week 3/COSC2779_Tute3_sol.pdf`.
- [ ] Questions: 01 no. 1, 4.

### Sat 17 Oct · Week 3: optimisation and regularisation (heavier day)
- [ ] Videos: Week 3 parts 1–9 ([doc 01](01_neural_network_basics.md#watch-and-read)).
- **Read [01](01_neural_network_basics.md)**, one box per section:
  - [ ] 5. Gradient descent
  - [ ] 6. Backpropagation (work through the to-do list and the animation in the Week 4 tutorial Q1 box)
  - [ ] 7. Practical training details
  - [ ] 7.1 Initialisation
  - [ ] 7.2 Normalisation
  - [ ] 7.3 Steep cliffs and exploding gradients
  - [ ] 8. Overfitting, underfitting, regularisation (the rest)
  - [ ] 8.2 Regularisation
  - [ ] 9. Common mistakes
  - [ ] 10. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- [ ] Tutorial 4 (backprop by hand, saddle points, BatchNorm, dropout, weight decay): `Week 4/COSC2779_Tute4.pdf` → `Week 4/COSC2779_Tute4_tex.pdf`.
- [ ] Questions: 01 no. 2, 3, 5, 7, 8, 9, 10.

### Sun 18 Oct · Week 4: convolutional networks
- [ ] Videos: Week 4 parts 1–7 ([doc 10](10_cnns_and_vision.md#watch-and-read)).
- **Read [10](10_cnns_and_vision.md)**, one box per section:
  - [ ] 1. Two ideas that make CNNs work
  - [ ] 2. The convolution operation
  - [ ] 3. Stacking layers: from edges to objects
  - [ ] 4. Pooling
  - [ ] 5. Special convolutions
- [ ] Tutorial 5 (convolution and pooling by hand, shapes, parameters, receptive field): `Week 5/COSC2779_5.pdf` → `Week 5/COSC2779_5_tex-1.pdf`.
- [ ] Questions: 10 no. 1, 2, 3, 5.

### Mon 19 Oct · Week 5: CNN architectures, detection, transfer learning
- [ ] Videos: Week 5 parts 1–5 ([doc 10](10_cnns_and_vision.md#watch-and-read)).
- **Read [10](10_cnns_and_vision.md)**, one box per section:
  - [ ] 6. Architectures
  - [ ] 6.1 LeNet
  - [ ] 6.2 AlexNet (2012): CNNs win
  - [ ] 6.3 VGG (2014): small filters, more depth
  - [ ] 6.4 GoogLeNet / Inception (2014): wide and efficient
  - [ ] 6.5 ResNet (2015): go very deep
  - [ ] 6.6 MobileNet: CNNs for small devices
  - [ ] 6.7 Vision Transformers
  - [ ] 6.8 Encoder–decoder / U-Net
  - [ ] 7. Vision tasks: classification, detection, segmentation
  - [ ] 7.1 One object: classification plus localisation
  - [ ] 7.2 Many objects: object detection
  - [ ] 7.3 Segmentation
  - [ ] 8. Transfer learning for images
  - [ ] 9. Data augmentation for images
- **Read [05](05_pretraining_and_adaptation.md)**, one box per section:
  - [ ] 1. Transfer learning: the basic idea
- [ ] Tutorial 6 (Inception, ResNet, GAP, object detection): `Week 6/COSC2779_6.pdf` → `Week 6/COSC2779_6_tex.pdf`.
- [ ] Questions: 10 no. 4, 6, 7, 8, 9, 10 · 05 no. 5.

### Tue 20 Oct · Week 6: practical methodology (lighter day, use it as a buffer)
- [ ] Videos: Week 6 parts 1–3 ([doc 07](07_evaluation_and_workflow.md#watch-and-read)).
- **Read [07](07_evaluation_and_workflow.md)**, one box per section:
  - [ ] 1. The workflow (Week 6)
  - [ ] 6. Reading learning curves
  - [ ] 7. Hyperparameter tuning
  - [ ] 8. Comparing models fairly
  - [ ] 10. Common mistakes
  - [ ] 11. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- **Read [10](10_cnns_and_vision.md)**, one box per section:
  - [ ] 10. Several images per example (Assignment 1)
  - [ ] 11. Common mistakes
  - [ ] 12. How this connects to other topics
  - [ ] Cheat sheet at the end of the chapter
- [ ] Tutorial 7 (methodology of a CNN project): `Week 7/COSC2779_Tute7_tex.pdf` → `Week 7/COSC2779_Tute7_ans.pdf`.
- [ ] Questions: 07 no. 1, 2, 5, 6 · 10 no. 11.
- [ ] Catch up on anything left from the first five days. Write a first cheat-sheet draft for Weeks 1–6.

### Wed 21 Oct · Week 7: text and recurrent networks
- [ ] Videos: Week 7 parts 1–2 and the two AI Hacker videos ([doc 03](03_sequences_and_rnns.md#watch-and-read)).
- **Read [02](02_text_to_numbers.md)**, one box per section:
  - [ ] 1. Step one: tokenisation
  - [ ] 2. Special tokens
  - [ ] 3. Padding and attention masks
  - [ ] 4. Offsets: mapping tokens back to characters
  - [ ] 5. Step two: from token IDs to vectors
  - [ ] 6. Static versus contextual embeddings
  - [ ] 7. Common mistakes
  - [ ] 8. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- **Read [03](03_sequences_and_rnns.md)**, one box per section:
  - [ ] 1. What makes sequence data special
  - [ ] 2. Task shapes
  - [ ] 3. The recurrent neural network (RNN)
  - [ ] 4. Training: backpropagation through time (BPTT)
  - [ ] 5. Vanishing and exploding gradients
  - [ ] 6. LSTM: long short-term memory
  - [ ] 7. GRU: gated recurrent unit
  - [ ] 8. Variants
  - [ ] 9. Encoder–decoder (seq2seq) and its bottleneck
  - [ ] 10. Token classification with RNNs
  - [ ] 11. RNNs versus Transformers in one table
  - [ ] 12. Common mistakes
  - [ ] 13. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- [ ] No tutorial this week: answer the *From the course* boxes in 02 and 03 yourself before opening them.
- [ ] Questions: all of 02 and 03.

### Thu 22 Oct · Week 8: Transformers
- [ ] Videos: the three Transformer videos ([doc 04](04_attention_and_transformers.md#watch-and-read)).
- **Read [04](04_attention_and_transformers.md)**, one box per section:
  - [ ] 1. The idea of attention in one sentence
  - [ ] 2. Queries, keys and values
  - [ ] 3. Scaled dot-product attention
  - [ ] 4. Multi-head attention
  - [ ] 5. Positional encoding
  - [ ] 6. The Transformer block
  - [ ] 7. Three ways to use the block
  - [ ] 8. Getting outputs for a task
  - [ ] 9. Why Transformers beat RNNs
  - [ ] 10. The cost: O(n²)
  - [ ] 11. Transformers for images: ViT
  - [ ] 12. Common mistakes
  - [ ] 13. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- **Read [06](06_task_types_multilabel_tagging.md)**, one box per section (much of it you know from A2):
  - [ ] 1. Four kinds of classification
  - [ ] 2. Thresholds in multi-label problems
  - [ ] 3. Token classification (sequence tagging)
  - [ ] 4. When BIO is not enough: overlapping spans
  - [ ] 5. Sub-words and labels
  - [ ] 6. Optional: a CRF layer on top
  - [ ] 7. Multi-task learning
  - [ ] 8. Self-test of your understanding
  - [ ] 9. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- [ ] Tutorial 8: `Week 8/COSC2779_Tute8_tex.pdf` → `Week 8/COSC2779_Tute8_tex_answers.pdf`.
- [ ] Questions: all of 04 and 06.

### Fri 23 Oct · Week 9: foundation models and adaptation
- [ ] Videos: the two Week 9 videos ([doc 05](05_pretraining_and_adaptation.md#watch-and-read)).
- **Read [05](05_pretraining_and_adaptation.md)**, one box per section:
  - [ ] 2. Self-supervised learning (SSL)
  - [ ] 3. Foundation models
  - [ ] 4. The adaptation spectrum, from cheapest to most expensive
  - [ ] 5. LoRA in detail
  - [ ] 6. Intermediate and domain-adaptive training
  - [ ] 7. Zero-shot with large language models
  - [ ] 8. Risks of foundation models
  - [ ] 9. Common mistakes
  - [ ] 10. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- **Read [09](09_bias_shift_deployment.md)**, one box per section:
  - [ ] 1. Where bias comes from
  - [ ] 2. Measuring bias
  - [ ] 3. Distribution shift
  - [ ] 4. Deployment constraints
  - [ ] 5. Presenting outputs responsibly
  - [ ] 6. Common mistakes
  - [ ] 7. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- [ ] Tutorial 9: `Week 9/COSC2779_Tute9.pdf` → `Week 9/COSC2779_Tute9_tex_answers.pdf`.
- [ ] Questions: 05 no. 1, 2, 3, 4, 6, 7, 8, 9 · all of 09.

### Sat 24 Oct · Week 10: generative models
- [ ] Videos: Week 10 parts 1–4 and the two Ari Seff videos ([doc 11](11_generative_models.md#watch-and-read)).
- **Read [11](11_generative_models.md)**, one box per section:
  - [ ] 1. Supervised, unsupervised, generative
  - [ ] 2. Autoencoders (AE)
  - [ ] 3. Variational autoencoders (VAE)
  - [ ] 4. Generative adversarial networks (GAN)
  - [ ] 5. Diffusion models
  - [ ] 6. Comparison
  - [ ] 7. One-to-many problems
  - [ ] 8. Generating text and speech
  - [ ] 9. Common mistakes
  - [ ] Cheat sheet at the end of the chapter
- [ ] Tutorial 10, timed like an exam case study (25 min): `Week 7/COSC2779_Tute7.pdf` (wrongly named, its title says Week 10) → `Week 7/COSC2779_Tute7_questions_answers.pdf`.
- [ ] Questions: all of 11.

### Sun 25 Oct · Weeks 11–12: interpretability (heavier day)
- [ ] Videos: Week 11 parts 1–4 and the TCAV talk ([doc 08](08_interpretability.md#watch-and-read)).
- **Read [08](08_interpretability.md)**, one box per section:
  - [ ] 1. Basic vocabulary
  - [ ] 2. Why explanations matter
  - [ ] 3. Feature visualisation (mostly for CNNs)
  - [ ] 4. Attribution methods: which parts of the input mattered?
  - [ ] 4.1 Occlusion and leave-one-out (perturbation)
  - [ ] 4.2 Saliency maps (gradients)
  - [ ] 4.3 CAM and Grad-CAM (CNNs)
  - [ ] 4.4 Integrated Gradients (IG)
  - [ ] 4.5 LIME (model-agnostic)
  - [ ] 4.6 SHAP and Shapley values (model-agnostic)
  - [ ] 5. Concepts instead of pixels
  - [ ] 6. Explaining Transformers
  - [ ] 6.1 Attention rollout
  - [ ] 6.2 Gradient-weighted attention (Chefer et al., 2021)
  - [ ] 6.3 What do the layers know? Probing and the logit lens
  - [ ] 6.4 A taste of mechanistic interpretability
  - [ ] 6.5 Choosing a method for Transformers
  - [ ] 7. Explaining vision–language models (VLMs)
  - [ ] 8. Is an explanation faithful? How to test it
  - [ ] 9. Common mistakes
  - [ ] 10. How this appears in A2
  - [ ] Cheat sheet at the end of the chapter
- [ ] Tutorial 11: `Week 11/COSC2779_Tute11_Q.pdf` → `Week 11/COSC2779_Tute11_QA.pdf`.
- [ ] Questions: all of 08.
- [ ] Optional: the Week 12 revision slides `Week 12/DeepLearninig_Lecture12_2150.pdf` (16 slides) as a check that you know the headline of every week.

### Mon 26 Oct · Mock exam and cheat sheet
- [ ] Practice exam on paper, timed, no notes: `Practice_Exam_DL_nosolutions.pdf` (about 30–40 min) → check with `Practice_Exam_DL.pdf`. In multiple choice, wrong ticks cost marks: tick only what you are sure of.
- [ ] Redo every question in QUESTIONS_AND_SUMMARY that you got wrong earlier.
- [ ] Write the final cheat sheet: one A4 page, one side, handwritten. First the formulas the exam names (attention, LoRA, weighted BCE, Integrated Gradients, Shapley values), then output-size and parameter formulas, the learning-curve diagnosis table, the adaptation spectrum and the comparison table of generative models.
- [ ] Pack: student card, cheat sheet, pen, laptop and phone, both charged (you need them for the 20-minute upload to Canvas after writing). Sleep early.

### Tue 27 Oct · Exam
- Morning: read only your cheat sheet. No new material.
- Be at your room in Building 80 by 12:00 to check in. Writing 12:30–14:30, then 20 minutes to transfer your answers to Canvas.

## 1. Get the overview first

Open the **course map**. It shows every topic of the course, top to bottom, grouped by course block. Each topic lists its key concepts and what it is used for.
- Online: https://claude.ai/artifact/MZcjghLkenLmGLV639CtBj
- Local copy: [course-map.html](course-map.html) (open it in a browser)

## 2. The chapters (sorted by topic)

The numbers follow the order in which A2 needed the topics, not the course weeks. To learn for the exam, use the day plan in section 0.

| # | Chapter | Course weeks | Needed for A2 |
|---|---|---|---|
| 01 | [Neural network basics](01_neural_network_basics.md) | 2–3 | yes, foundation for everything |
| 02 | [From text to numbers](02_text_to_numbers.md) | 7–9 | yes |
| 03 | [Sequences and RNNs](03_sequences_and_rnns.md) | 7 | task shapes; exam |
| 04 | [Attention and Transformers](04_attention_and_transformers.md) | 8 | yes, our model |
| 05 | [Pretraining and adaptation](05_pretraining_and_adaptation.md) | 5, 9 | yes, E2, E9, C2 |
| 06 | [Framing the task](06_task_types_multilabel_tagging.md) | 2, 7–8 | yes, Part A and B |
| 07 | [Evaluation and workflow](07_evaluation_and_workflow.md) | 1–2, 6 | yes, the whole method |
| 08 | [Interpretability](08_interpretability.md) | 11–12 | yes, IG and C3 |
| 09 | [Bias, shift and deployment](09_bias_shift_deployment.md) | 9, 12 | yes, Part C |
| 10 | [CNNs and computer vision](10_cnns_and_vision.md) | 4–6, 7 tute, A1 | no (A1 and exam) |
| 11 | [Generative models](11_generative_models.md) | 10, 11 lab | no (exam) |

## 3. Test yourself

[QUESTIONS_AND_SUMMARY.md](QUESTIONS_AND_SUMMARY.md) has a short summary and self-test questions for each chapter. Answers are folded away. Write your answer first, then check.

## 4. Course videos

The pre-recorded lectures and videos from the Canvas week pages are listed at the top of each chapter's **Watch and read** section, under *Official course videos*. Canvas Studio links need your RMIT login. [deep-learning-videos.md](deep-learning-videos.md) has the full list by week, with the chapter each video belongs to.

| Weeks | Chapter with the videos |
|---|---|
| 1, 6 | [07 Evaluation and workflow](07_evaluation_and_workflow.md#watch-and-read) |
| 2, 3 | [01 Neural network basics](01_neural_network_basics.md#watch-and-read) |
| 4, 5 | [10 CNNs and computer vision](10_cnns_and_vision.md#watch-and-read) |
| 7 | [03 Sequences and RNNs](03_sequences_and_rnns.md#watch-and-read) |
| 8 | [04 Attention and Transformers](04_attention_and_transformers.md#watch-and-read) |
| 9 | [05 Pretraining and adaptation](05_pretraining_and_adaptation.md#watch-and-read) |
| 10 | [11 Generative models](11_generative_models.md#watch-and-read) |
| 11 | [08 Interpretability](08_interpretability.md#watch-and-read) |

Setup videos from the Welcome and Week 1 pages (tools only, not exam content): [Google Colab](https://www.youtube.com/watch?v=inN8seMm7UI) · [Anaconda and conda](https://www.youtube.com/watch?v=YJC6ldI3hWk) · [Jupyter Notebook](https://www.youtube.com/watch?v=HW29067qVWk) · [Python 3 crash course](https://www.youtube.com/watch?v=DcEfZeR71qg).

## Colour code in the formulas

Every formula uses the same colours, so you can see at a glance what is data and what is learned:
$	extcolor{#3b82f6}{	ext{input / data}}$ · $	extcolor{#e8833a}{	ext{learned weights}}$ · $	extcolor{#22a06b}{	ext{model output}}$ · $	extcolor{#e5484d}{	ext{true label}}$ · $	extcolor{#a855f7}{	ext{setting you choose}}$ (· $	extcolor{#64748b}{	ext{frozen weights}}$ in chapter 05).
Under each formula a small line explains every symbol.

## How each chapter is built

1. **The problem:** why the topic exists.
2. **Course material:** the exact lecture file and section to read alongside.
3. **The explanation,** from first principles, with a worked example.
4. **Common mistakes:** typical exam traps.
5. **How it appears in A2.**
6. **Watch and read:** checked links to videos and readings.
7. **Cheat sheet** at the end: the most important points, memory hooks, and a short learning exercise. Copy the best lines onto your handwritten A4 exam sheet.

## Exam facts

- Tuesday 27 October 2026, 12:30–14:50, RMIT Building 80. Arrive 30 minutes early.
- 120 minutes of writing. One handwritten A4 cheat sheet, one side.
- Three parts: multiple choice (wrong ticks cost marks), short answers (2–4 sentences), case studies with sub-questions.
- The exam can cover anything from the lectures, tutorials, labs and both assignments.
