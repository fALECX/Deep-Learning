"""Insert 'From the course' examples (tutorials, lectures, practice exam) with fold-out answers. Idempotent."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def ex(source, question, answer):
    return (f"> [!NOTE]\n> **From the course · {source}.** {question}\n\n"
            f"<details>\n<summary><b>Show the answer</b></summary>\n\n{answer.strip()}\n\n</details>\n\n")


ACTIVATION_EXPLAINER = r'''### What an activation function really is (in plain terms)

Yes: it is a **fixed function that you choose before training**. It is not learned. After a layer has computed its weighted sums $z$ (one number per neuron), the same function is applied to **each number separately**. Then the result goes to the next layer.

![What the activation does to a layer's numbers](img/01_activation_applied.png)
<sub>Four neurons of one layer. Left: their weighted sums $z$. Middle: ReLU sets negatives to 0. Right: sigmoid squashes every value into 0…1.</sub>

```python
import torch
z = torch.tensor([-2.0, 0.5, 3.0, -0.7])   # weighted sums of 4 neurons (W·x + b)
torch.relu(z)      # tensor([0.0000, 0.5000, 3.0000, 0.0000])
torch.sigmoid(z)   # tensor([0.1192, 0.6225, 0.9526, 0.3318])
```

**Why not something simple like $f(z) = 4z$?** Because $4z$ is linear: it only scales. Stack two layers with it and you get $4\,W_2\,(4\,W_1 x) = (16\,W_2 W_1)\,x$, which is again just one matrix times $x$. Ten such layers are still one straight line, so they cannot solve XOR. The activation must **bend** somewhere. ReLU bends at 0, sigmoid and tanh bend smoothly.

<details>
<summary><b>Check it yourself in code</b> (two linear layers collapse into one)</summary>

```python
import torch
W1, W2 = torch.randn(5, 2), torch.randn(1, 5)
x = torch.randn(2)
two_layers = W2 @ (4 * (W1 @ x))          # "activation" f(z) = 4z

print(torch.allclose(W2 @ (4 * (W1 @ x)), (4 * W2 @ W1) @ x))   # True: no extra power
```

</details>

'''

BLOCKS = {
    "01_neural_network_basics.md": [
        ("**Why the activation matters.**", ACTIVATION_EXPLAINER, "What an activation function really is"),
        ("## 3. Layers and networks", ex("Week 3 tutorial, Q2",
            "A network has one hidden layer $a = \\sigma_1(w_1x_1 + w_2x_2 + b)$ and output $y = \\sigma_2(w_3 a + b_2)$. Show that if $\\sigma_1$ is linear, $\\sigma_1(z) = \\alpha z$, the network is just a one-layer perceptron.",
            r"""
Substitute $a = \alpha(w_1x_1 + w_2x_2 + b)$ into the output:

$$y = \sigma_2\big(\alpha w_3 w_1\,x_1 + \alpha w_3 w_2\,x_2 + (\alpha w_3 b + b_2)\big) = \sigma_2(\omega_1 x_1 + \omega_2 x_2 + \beta)$$

with new weights $\omega_1 = \alpha w_3 w_1$, $\omega_2 = \alpha w_3 w_2$ and bias $\beta = \alpha w_3 b + b_2$. That is exactly one neuron. The hidden layer added nothing.
"""), "Week 3 tutorial, Q2"),
        ("**Where these losses come from.**", ex("Week 3 tutorial, Q6",
            "A network must say whether an image contains a cat and/or a dog. Some images contain both. Which output activation and loss?",
            r"""
Two valid designs:
- **Two sigmoid outputs** (cat yes/no, dog yes/no) and the sum of two binary cross-entropies. This is multi-label, exactly like A2 with 20 techniques instead of 2.
- **Four softmax outputs** for the combinations (none, cat, dog, both) with categorical cross-entropy. This works for 2 labels, but with 20 labels it would need $2^{20}$ classes, so it does not scale.
"""), "Week 3 tutorial, Q6"),
        ("**Mini-batch SGD.**", ex("Week 4 tutorial, Q2",
            "What goes wrong if the learning rate is too high, and if it is too low? How do you see it?",
            r"""
- **Too high:** the loss does not converge and can even grow. In the loss-vs-iteration plot it oscillates wildly (right panel of the figure above).
- **Too low:** the loss falls very slowly, almost in a straight line, and may not reach a good minimum in time (left panel).
- Always plot the loss against the iterations to tell which case you have.
"""), "Week 4 tutorial, Q2"),
        ("Backpropagation reuses intermediate results", ex("Week 4 tutorial, Q1",
            "Input $x = [1, 2]$, target $y = 3$, MSE loss, three ReLU hidden units, linear output, learning rate 0.1, all biases 0. Hidden weights: unit 1 $(2, 1)$, unit 2 $(1, -2)$, unit 3 $(1, 2)$. Output weights $(-1, 3, 2)$. Compute the gradients.",
            r"""
**Forward pass**
- $h_1 = \text{ReLU}(2\cdot1 + 1\cdot2) = 4$
- $h_2 = \text{ReLU}(1\cdot1 - 2\cdot2) = \text{ReLU}(-3) = 0$  ← a "dead" unit
- $h_3 = \text{ReLU}(1\cdot1 + 2\cdot2) = 5$
- $\hat y = -1\cdot4 + 3\cdot0 + 2\cdot5 = 6$, loss $= (6-3)^2 = 9$

**Backward pass** (chain rule, $\partial L/\partial\hat y = 2(\hat y - y) = 6$)
- Output weights: $\partial L/\partial w_k = 6\cdot h_k$ → $24,\ 0,\ 30$; output bias $6$.
- Hidden unit 1: $6 \cdot (-1) = -6$, times its inputs → $-6,\ -12$; bias $-6$.
- Hidden unit 2: ReLU was 0, its slope is 0 → all gradients $0$. It does not learn this step.
- Hidden unit 3: $6 \cdot 2 = 12$, times inputs → $12,\ 24$; bias $12$.

**Update:** $w \leftarrow w - 0.1\cdot\text{gradient}$, e.g. output weight 1: $-1 - 0.1\cdot24 = -3.4$.
"""), "Week 4 tutorial, Q1"),
        ("- **Local minima and saddle points.**", ex("Week 4 tutorial, Q3 and Q4",
            "What is a saddle point, and why can SGD help? And: sigmoid units saturate. Does switching to tanh fix it?",
            r"""
- **Saddle point:** the gradient is zero, but it is neither a minimum nor a maximum (it curves up in one direction and down in another). Plain gradient descent can stall there. SGD's noisy updates push it off.
- **Saturation:** the slope of sigmoid is tiny for large positive or negative $z$, so gradients vanish. **tanh does not fix it**: $\tanh(z) = 2\sigma(2z) - 1$ is just a stretched sigmoid with the same flat ends. Its one advantage is that its outputs are centred around 0. ReLU avoids saturation for positive $z$.
"""), "Week 4 tutorial, Q3 and Q4"),
        ("Dropout is only active during training.", ex("Week 4 tutorial, Q6",
            "Why does dropout act as a regulariser?",
            r"""
- **Model averaging:** with $H$ neurons, each step trains one of $2^H$ thinned networks. At test time the full network acts like an average of all of them.
- **No co-adaptation:** a neuron cannot rely on a specific partner neuron being present, so each one must learn something useful on its own.
"""), "Week 4 tutorial, Q6"),
        ("## 9. Common mistakes", ex("Week 2 tutorial, Q6 and Practice exam C.1 (b)–(c)",
            "(1) What happens if you tune the regularisation strength $\\lambda$ by looking at the training loss? (2) A CNN gets 72% train / 70% validation accuracy, after a change 96% / 78%. Diagnose both.",
            r"""
1. The training loss is lowest with $\lambda = 0$ (no penalty) and maximum capacity. You would always pick that, which is the overfitting solution. Tune $\lambda$ on **validation** data.
2. **72 / 70:** both low, small gap → underfitting; add capacity. **96 / 78:** large gap → overfitting; add augmentation, dropout or weight decay. Compare with the fitting figure in section 8.
"""), "Week 2 tutorial, Q6"),
    ],
    "02_text_to_numbers.md": [
        ("## 2. Special tokens", ex("Week 8 tutorial, (a)",
            "Customer messages must go into a Transformer. Why not just split on whitespace and one-hot encode each word?",
            r"""
- Whitespace words give a huge vocabulary and an out-of-vocabulary problem for product names and typos. **Sub-word tokenisation** has a fixed vocabulary and splits unknown words into known pieces.
- One-hot vectors are huge, sparse and treat all words as equally different. **Dense learned embeddings** are small and capture similarity.
- Add special tokens: [CLS] at the front as a summary slot, [SEP] at the end.
"""), "Week 8 tutorial, (a)"),
        ("**Averaging word vectors**", ex("Week 9 tutorial, (a)",
            "Legal emails: why is a contextual representation from a pretrained Transformer better than averaging word2vec/GloVe vectors?",
            r"""
- word2vec/GloVe give **one vector per word**. "bond" (rental deposit) and "bond" (financial) get the same vector.
- A Transformer computes each word's vector **from the whole email**, so meaning adapts to context.
- Averaging also throws away word order. Self-attention plus positional encodings keep it.
"""), "Week 9 tutorial, (a)"),
        ("**Truncation**", ex("Week 8 tutorial, (h)",
            "Messages have very different lengths, but training uses batches. What must attention handle, and how?",
            r"""
Pad every message in the batch to the same length, then use a **padding mask**: set the attention scores of padded positions to $-\infty$ before the softmax. Real tokens then give padding a weight of exactly 0.
"""), "Week 8 tutorial, (h)"),
    ],
    "03_sequences_and_rnns.md": [
        ("Rule of thumb: count the outputs you need.", ex("Practice exam, A.3",
            "Predict the genre of short music clips of different lengths. Which RNN shape fits?",
            r"""
**Many-to-one.** A whole clip (a sequence of any length) goes in, one genre label comes out. The varying length does not make it many-to-many; the number of outputs decides.
"""), "Practice exam, A.3"),
        ("**Attention** fixes this.", ex("Practice exam, B.2",
            "In a recurrent encoder–decoder without attention, the decoder only gets the final encoder state. What limitation does this cause, and how does attention help?",
            r"""
One fixed-size vector must hold the whole input. For long sequences it cannot keep everything, so quality drops. Attention lets the decoder look back at **all** encoder states at every step and weight the relevant ones, instead of relying on the last state only.
"""), "Practice exam, B.2"),
        ("## 11. RNNs versus Transformers in one table", ex("Week 7 lecture notebook, Part 2",
            "The notebook builds a recurrent tagger that outputs a label for every step and visualises true vs. predicted tags. Which shape is that, and what does a bidirectional version add?",
            r"""
Aligned many-to-many (one output per input step). A bidirectional RNN gives every step context from both sides, which helps whenever a tag depends on words that come later.
"""), "Week 7 lecture notebook, Part 2"),
    ],
    "04_attention_and_transformers.md": [
        ("## 5. Positional encoding", ex("Week 8 tutorial, (d)",
            "Why use multi-head attention instead of a single attention?",
            r"""
One attention gives one weighting per token. Several heads can attend to several relations at once (syntax, coreference, local vs. long-range). Each head has its own $W_Q, W_K, W_V$ with $d_k = d/h$; the heads are concatenated and mixed by $W_O$, so the cost is about the same as one full-width attention.
"""), "Week 8 tutorial, (d)"),
        ("## 6. The Transformer block", ex("Week 8 tutorial, (e)",
            "Self-attention is a set operation. Why is that a problem for customer messages, and how is it fixed?",
            r"""
"cancel my plan" and "plan my cancellation" contain the same words but mean different things. Without order, attention sees the same bag. Fix: add a **positional encoding** (sinusoidal or learned) to every token embedding.
"""), "Week 8 tutorial, (e)"),
        ("## 9. Why Transformers beat RNNs", ex("Week 8 tutorial, (f) and (g)",
            "From the same encoder output, produce (i) one intent label per message and (ii) an entity tag for every word. Which model family would you start from?",
            r"""
- (i) Take the final **[CLS]** vector → linear layer → softmax over the intents.
- (ii) Apply one shared linear layer to **every token vector** → a tag per token (BIO). The vectors are contextual, so the same word can get different tags.
- Start from a pretrained **encoder-only** model (BERT-style): both are understanding tasks that need left and right context. Fine-tune it with the two heads. This is the same design as A2.
"""), "Week 8 tutorial, (f) and (g)"),
        ("## 11. Transformers for images: ViT", ex("Week 8 tutorial, (k)",
            "Some transcripts are thousands of tokens long and training runs out of memory. Why, and what can you do?",
            r"""
Attention compares every token with every other one: memory and compute grow with $n^2$. Truncate or split long texts into chunks, or use an efficient/sparse attention model built for long inputs (e.g. Longformer).
"""), "Week 8 tutorial, (k)"),
    ],
    "05_pretraining_and_adaptation.md": [
        ("## 2. Self-supervised learning (SSL)", ex("Practice exam, A.2 and C.1 (e)",
            "Which steps belong to fine-tuning in transfer learning? How would you adapt an ImageNet-pretrained ResNet to six waste categories?",
            r"""
- Replace the classification head with a new 6-class head.
- Freeze the pretrained layers and train the head first.
- Then optionally unfreeze some **top** layers and fine-tune them with a **smaller learning rate**. Both head and base weights can be trained in this phase. Keeping every base layer frozen forever is feature extraction, not fine-tuning.
"""), "Practice exam, A.2"),
        ("**Which to pick?**", ex("Week 9 tutorial, (d)–(f) and Practice exam, B.3",
            "2,000 labelled emails, a modest GPU. Which adaptation strategy, and why? What are the pros and cons of feature extraction vs. full fine-tuning?",
            r"""
- Try **feature extraction and/or LoRA** first: few trainable parameters, cheap, less overfitting on 2,000 examples. Run zero-shot as a quick baseline.
- **Feature extraction:** + fast, robust on small data; − the representation cannot adapt.
- **Full fine-tuning:** + best accuracy with enough data; − expensive, one model copy per task, can overfit, catastrophic forgetting.
- Always validate empirically. (In A2, full fine-tuning won even on 750 memes.)
"""), "Week 9 tutorial, (d)"),
        ("More details:", ex("Week 9 tutorial, (g)–(h)",
            "Why is full fine-tuning costly when you serve many tasks, how does LoRA fix it, and what does the rank $r$ control?",
            r"""
- Full fine-tuning stores a full copy of every weight per task. LoRA freezes $W_0$ and learns $\Delta W = BA$ with $r \ll \min(d,k)$; adapters are tiny, swappable and mergeable (no extra latency).
- $r$ sets the capacity of the update and the parameter count $r(d+k)$ per layer: too small underfits, larger gives a bit more accuracy with diminishing returns.
"""), "Week 9 tutorial, (g)"),
    ],
    "06_task_types_multilabel_tagging.md": [
        ("**Multi-label loss**", ex("Week 3 tutorial, Q6",
            "Images may contain a cat, a dog, or both. Output activation and loss?",
            r"""
Two sigmoid outputs with two binary cross-entropies (multi-label), or four softmax outputs for the combinations. The sigmoid version scales to many labels; the combination version explodes ($2^K$ classes).
"""), "Week 3 tutorial, Q6"),
        ("## 4. When BIO is not enough: overlapping spans", ex("Week 8 tutorial, (f)",
            "Mark every product, location and account ID in a message (a label for every word). How?",
            r"""
Apply one shared linear layer + softmax to every token vector and predict a BIO tag per token. This works because entities do not overlap there. In A2 they do, so we switch to one sigmoid per technique per token.
"""), "Week 8 tutorial, (f)"),
    ],
    "07_evaluation_and_workflow.md": [
        ("## 3. Cross-validation", ex("Week 7 tutorial, (a)",
            "10,000 cell images from microscopes A (5,000), B (2,500) and C (2,500). The doctors will use the model on microscope C. How do you split?",
            r"""
Roughly 90 / 5 / 5. Validation and test must both come from **C**, because that is the deployment data. Training should also contain C images (more than val/test), plus all of A and B.
"""), "Week 7 tutorial, (a)"),
        ("**Choose the metric from the costs.**", ex("Week 7 tutorial, (c) and Practice exam, C.1 (d)",
            "(1) Which metric for pixel-wise cell segmentation? (2) Accuracy is good overall, but how do you see whether every one of six classes works and which ones get confused?",
            r"""
1. A pixel-wise measure that handles class imbalance, e.g. **F1**. Cell pixels and background pixels are not balanced, so accuracy would mislead.
2. **Per-class** precision/recall/F1 shows which classes work. A **confusion matrix** shows which classes are mixed up.
"""), "Week 7 tutorial, (c)"),
        ("## 7. Hyperparameter tuning", ex("Week 7 tutorial, (f) and (g)",
            "(1) Training and validation loss both oscillate. What do you change? (2) The final validation loss is below the training loss. Why can that happen?",
            r"""
1. Lower the **learning rate** (see the right panel of the learning-curve figure).
2. Regularisation such as dropout is active only in training; training loss is averaged during the epoch while validation loss is measured at its end; the validation set may be easier; or there is leakage.
"""), "Week 7 tutorial, (f)"),
    ],
    "08_interpretability.md": [
        ("### 4.2 Saliency maps (gradients)", ex("Week 11 tutorial, (b)",
            "Grad-CAM heatmaps of a melanoma classifier highlight a ruler and ink marks next to the lesion. What does that show, and what do you do?",
            r"""
**Shortcut learning:** suspicious lesions are photographed with rulers more often, so the model uses the ruler. It will fail where that correlation does not hold. Clean or crop the images, collect data with markings in both classes, retrain, and check again with the explanation and with a test set split by "marking / no marking".
"""), "Week 11 tutorial, (b)"),
        ("### 4.5 LIME (model-agnostic)", ex("Week 11 tutorial, (h)",
            "Gradient × input on a text model gives near-zero scores for every token when the model is very confident. Why, and how does Integrated Gradients fix it?",
            r"""
The output is saturated, so the local gradient is almost zero even for important tokens. IG integrates the gradient along the path from a baseline (e.g. all [PAD]) to the input; on the way it passes non-saturated regions. Its scores sum to $f(x) - f(x')$, and the baseline defines "important compared to what".
"""), "Week 11 tutorial, (h)"),
        ("Leave-one-out for comparison:", ex("Week 11 tutorial, (c) and (d)",
            "The worked example above is exactly tutorial question (d). Question (c) asks: the classifier is only available as an API (no weights, no gradients). Can you use Grad-CAM? How would LIME work?",
            r"""
No Grad-CAM: it needs internal feature maps and gradients. Use LIME: segment the image into superpixels, switch random superpixels off, send each perturbed image to the API, weight the samples by closeness to the original, fit a sparse linear model; the largest positive weights are the explanation.
"""), "Week 11 tutorial, (c) and (d)"),
        ("## 7. Is an explanation faithful? How to test it", ex("Week 11 tutorial, (g) and (i)",
            "(1) A colleague shows attention weights as the explanation of a text classifier. Justified? (2) A vision–language model writes a fluent explanation of its own decision. Is that an explanation for free?",
            r"""
1. Only partly. Attention shows where information flows, not how much each token mattered; it ignores values, residuals and feed-forward layers. Better: attention rollout, gradient-weighted attention, IG or SHAP over tokens.
2. No. It is generated text, optimised to sound right. **Plausible is not faithful**; treat it as a claim and test it (with open weights you can check it with IG or relevance maps).
"""), "Week 11 tutorial, (g) and (i)"),
    ],
    "09_bias_shift_deployment.md": [
        ("## 3. Distribution shift", ex("Week 9 tutorial, (j)",
            "Triage of legal emails is high-stakes and fairness-sensitive. Name two risks of using a foundation model and one mitigation.",
            r"""
- **Inherited bias** from web-scale pretraining → different error rates for groups (dialect, names, postcodes).
- **Distribution shift / shortcut learning / hallucination** → confident but wrong routing on unusual emails.
- **Mitigation:** disaggregated metrics per group, curated adaptation data, and human oversight for the decision.
"""), "Week 9 tutorial, (j)"),
        ("## 4. Deployment constraints", ex("Week 12 lab, Parts 3, 8 and 9",
            "The lab checks a sentiment model by changing single words (counterfactuals) and by comparing groups. What is the logic of a counterfactual test?",
            r"""
Change only one attribute of the input (a name, a group word) and keep everything else. If the prediction moves, that attribute caused it. This is the same idea as A2's Trump↔Biden swap in C1.
"""), "Week 12 lab"),
    ],
    "10_cnns_and_vision.md": [
        ("## 2. The convolution operation", ex("Practice exam, A.1, B.1 and C.1 (a)",
            "Which properties make CNNs better than fully connected layers for images?",
            r"""
**Sparse connectivity** (local features), **parameter sharing** (the same filter everywhere) and **pooling** (less redundancy, some shift invariance). Together they use the spatial structure of images and need far fewer parameters. Dense connectivity is a property of fully connected layers, not of CNNs.
"""), "Practice exam, A.1"),
        ("## 3. Stacking layers: from edges to objects", ex("Week 5 tutorial, Q3 and Q4",
            "LeNet-style network on a 32 × 32 × 1 input: conv 6@5×5 → avg-pool 2 → conv 16@5×5 → avg-pool 2 → flatten → dense 120 → dense 84 → dense 10. Output shapes, parameters, and the receptive field of the second conv layer?",
            r"""
| Layer | Output shape | Parameters |
|---|---|---|
| conv 6@5×5 | 28 × 28 × 6 | 6 · (5·5·1 + 1) = 156 |
| avg-pool 2 | 14 × 14 × 6 | 0 |
| conv 16@5×5 | 10 × 10 × 16 | 16 · (5·5·6 + 1) = 2,416 |
| avg-pool 2 | 5 × 5 × 16 | 0 |
| flatten | 400 | 0 |
| dense 120 | 120 | 120 · (400 + 1) = 48,120 |
| dense 84 | 84 | 84 · (120 + 1) = 10,164 |
| dense 10 | 10 | 10 · (84 + 1) = 850 |

Total ≈ 61,706 parameters. Receptive field of the second conv: $5 + (2-1) + (5-1)\cdot 2 = 14$, so **14 × 14** pixels (18 × 18 if the first conv had dilation 2).
"""), "Week 5 tutorial, Q3"),
        ("- **Average pooling**", ex("Week 5 tutorial, Q2",
            "Apply 3 × 3 max pooling with stride 2 to a 7 × 7 input. How large is the output?",
            r"""
$(7 - 3)/2 + 1 = 3$, so a **3 × 3** output. Each value is the maximum of one 3 × 3 window; the windows overlap by one row/column because the stride (2) is smaller than the window (3).
"""), "Week 5 tutorial, Q2"),
        ("- **Encoder–decoder / U-Net**", ex("Week 6 tutorial, Q1–Q3",
            "(1) Why does Inception-v2 replace a 5 × 5 conv by two 3 × 3 convs? (2) How does ResNet shrink the spatial size? (3) Why do global-average-pooling models have far fewer parameters than VGG?",
            r"""
1. Two stacked 3 × 3 convs see the same 5 × 5 area but cost less (5·5 = 25 vs 2·9 = 18 weights per channel pair) and add an extra non-linearity.
2. Some residual blocks use **strided** convolutions, with a strided 1 × 1 conv on the skip path so the shapes still match.
3. GAP turns each feature map into one number, so the big bank of fully connected layers is replaced by one small layer.
"""), "Week 6 tutorial"),
        ("## 9. Data augmentation for images", ex("Week 7 tutorial, (b), (d) and (e)",
            "Cell segmentation: which augmentations, which architecture and which loss?",
            r"""
- Augment with crops, noise, contrast changes, blur, flips and rotations (all keep the mask valid if applied to the mask too).
- Architecture: an encoder–decoder such as **U-Net**, which works well with little medical data.
- Loss: binary cross-entropy per pixel, **weighted** for class imbalance: $L = -\sum \big[\alpha_1 y\log\hat y + \alpha_0(1-y)\log(1-\hat y)\big]$.
"""), "Week 7 tutorial, (b)"),
    ],
    "11_generative_models.md": [
        ("## 3. Variational autoencoders (VAE)", ex("Practice exam, A.4",
            "Which of these are issues of a basic autoencoder: it can memorise the data; it cannot reliably generate new samples; it cannot reconstruct its input; it needs adversarial training?",
            r"""
The first two. It **can** reconstruct (that is its job), and it does **not** need adversarial training (that is GANs). It can memorise, and its unstructured latent space makes random samples decode to nonsense.
"""), "Practice exam, A.4"),
        ("**Uses:**", ex("Week 10 lecture, ISS example",
            "Telemetry from the International Space Station: how can an autoencoder detect anomalies without anomaly labels?",
            r"""
Train it only on normal telemetry. Normal patterns reconstruct well. An unusual pattern was never seen, so it reconstructs badly. Flag inputs whose reconstruction error is above a threshold chosen on validation data.
"""), "Week 10 lecture"),
    ],
}

for doc, items in BLOCKS.items():
    p = ROOT / doc
    s = p.read_text(encoding="utf8")
    for anchor, block, marker in items:
        if marker in s and ("From the course · " + marker) in s or (marker == "What an activation function really is" and marker in s):
            continue
        i = s.find(anchor)
        assert i >= 0, (doc, anchor)
        s = s[:i] + block + s[i:]
        print("inserted", marker, "->", doc)
    p.write_text(s, encoding="utf8")
