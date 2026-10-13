# Summaries and self-test questions

Use this after reading a chapter. First read the summary. Then answer the questions on paper, without looking. Open the answer only after you have written yours.

The questions are in three styles, like the exam: true/false statements (for multiple choice with penalties), short answers of 2–4 sentences, and small case studies.

---

## 01 · Neural network basics

**Summary**
- A neuron computes a weighted sum plus bias, then applies an activation.
- Without non-linear activations, a deep network is just one linear function.
- The output layer and the loss must match the task: regression → MSE; one class of K → softmax + cross-entropy; any number of K → sigmoid per class + BCE.
- Training = gradient descent on the loss. Backpropagation computes the gradients with the chain rule.
- Adam/AdamW is the default optimiser. The learning rate is the most important setting.
- Overfitting is fought with weight decay, dropout, augmentation, early stopping and smaller models.

**Questions**

1. True or false: stacking ten linear layers without activations lets a network learn XOR.
<details><summary>Answer</summary>False. Ten linear layers collapse into one linear map. XOR needs a non-linear boundary.</details>

2. True or false: dropout is switched off at test time.
<details><summary>Answer</summary>True. All neurons are used at test time.</details>

3. Your loss jumps up and down and never settles. Name the most likely cause and two fixes.
<details><summary>Answer</summary>The learning rate is too high. Lower it, add a warm-up, use gradient clipping or larger batches.</details>

4. Compute the binary cross-entropy for a true label of 0 when the model predicts p = 0.8.
<details><summary>Answer</summary>−log(1 − 0.8) = −log(0.2) ≈ 1.61. A confident wrong answer gives a large loss.</details>

5. Why are the weights initialised randomly and not all to zero?
<details><summary>Answer</summary>Equal weights get equal gradients, so all neurons in a layer would stay identical. Random values break this symmetry.</details>

---

## 02 · From text to numbers

**Summary**
- Tokenisation cuts text into tokens with IDs. Modern models use sub-word pieces (BPE, WordPiece). There is no out-of-vocabulary problem.
- Special tokens: [CLS] (summary of the sequence), [SEP], [PAD], [MASK].
- Batches need padding. An attention mask makes the model ignore it.
- Offsets map tokens back to character positions. They are needed for character-level labels.
- Embeddings are dense learned vectors. Static embeddings (word2vec, GloVe) give one vector per word. Contextual ones (BERT) give a new vector per sentence.

**Questions**

1. True or false: a sub-word tokeniser can encode a word it never saw during training.
<details><summary>Answer</summary>True. It splits the word into known pieces, down to single characters if needed.</details>

2. Why is one-hot encoding a poor input representation for words?
<details><summary>Answer</summary>The vectors are huge and sparse, and every word is equally distant from every other word. No similarity is captured.</details>

3. Give one sentence where a static embedding fails and explain why.
<details><summary>Answer</summary>"She sat by the bank of the river." Static embeddings give "bank" one vector for both river bank and money bank. The meaning depends on context, which they ignore.</details>

4. What happens if you train a Transformer on padded batches without an attention mask?
<details><summary>Answer</summary>Real tokens attend to padding tokens. Padding then changes the representations and the predictions, depending on how much padding a batch has.</details>

---

## 03 · Sequences and RNNs

**Summary**
- Task shapes: one-to-many, many-to-one, aligned many-to-many, encoder–decoder.
- An RNN keeps a hidden state and uses the same weights at every step.
- Training uses backpropagation through time. Gradients can vanish or explode over many steps.
- Clipping fixes exploding gradients only. LSTMs and GRUs reduce vanishing gradients with gates and an additive cell update.
- Bidirectional RNNs use both left and right context. They suit tagging, not generation.
- The encoder–decoder bottleneck (one summary vector) motivated attention.

**Questions**

1. True or false: gradient clipping solves the vanishing gradient problem.
<details><summary>Answer</summary>False. It only caps large gradients. It cannot enlarge tiny ones.</details>

2. Which task shape fits "label each word of a sentence with its part of speech"?
<details><summary>Answer</summary>Aligned many-to-many: one output per input token.</details>

3. Explain in 2–3 sentences why the LSTM cell state helps with long sequences.
<details><summary>Answer</summary>The cell state is updated by adding new information, gated by the forget and input gates. When the forget gate stays near 1, information and gradients pass through many steps almost unchanged. A plain RNN instead multiplies by the same matrix at every step, so the signal shrinks or explodes.</details>

4. Why can a bidirectional RNN not be used directly to generate text word by word?
<details><summary>Answer</summary>Its backward direction needs future words. When generating, those words do not exist yet.</details>

---

## 04 · Attention and Transformers

**Summary**
- Attention: each token's new vector is a weighted average of all value vectors. The weights come from query–key dot products.
- Attention(Q, K, V) = softmax(QKᵀ / √d_k) V. The scaling keeps the softmax out of saturation.
- Multi-head attention learns several relations in parallel at about the same cost.
- Attention ignores order, so positional encodings are added.
- Blocks: attention + feed-forward, each with a residual connection and layer norm.
- Encoder-only for understanding, decoder-only for generation (causal mask), encoder–decoder for sequence to sequence (cross-attention).
- Cost grows with the square of the length.

**Questions**

1. True or false: without positional encoding, a Transformer encoder gives the same set of output vectors for "dog bites man" and "man bites dog".
<details><summary>Answer</summary>True. Self-attention is permutation-equivariant. The outputs are the same vectors in a different order, so a [CLS] or average summary is identical.</details>

2. True or false: in a decoder, the causal mask prevents tokens from attending to padding.
<details><summary>Answer</summary>False. The causal mask hides future tokens. The padding mask hides padding.</details>

3. A team must classify customer emails and also highlight product names in them. Which Transformer family fits, and how do you get both outputs?
<details><summary>Answer</summary>Encoder-only, because both are understanding tasks that need context from both sides. Use the [CLS] (or pooled) vector with a classifier for the email label. Apply a shared classifier to every token vector for the highlighting.</details>

4. Explain what goes wrong without the √d_k scaling.
<details><summary>Answer</summary>Dot products grow with the dimension. Large scores push the softmax towards one-hot outputs, where its gradients are almost zero. Training slows or stalls.</details>

5. Your Transformer runs out of memory on documents with 20,000 tokens. Why, and what can you do?
<details><summary>Answer</summary>Attention builds an n × n matrix, so memory grows quadratically with length. Split or truncate the documents, or use an efficient-attention model built for long inputs.</details>

---

## 05 · Pretraining and adaptation

**Summary**
- Transfer learning reuses features learned on a big task. Recipe: new head, freeze, train head, then optionally fine-tune top layers with a small learning rate.
- Self-supervised learning creates labels from the data itself: masked tokens, next tokens, contrastive pairs.
- Foundation models are pretrained at scale and adapted to many tasks.
- Adaptation spectrum: zero-shot < feature extraction < LoRA/PEFT < full fine-tuning, in cost and usually in flexibility.
- LoRA: W = W₀ + BA with small rank r. Only A and B train. Mergeable, so no extra inference cost.
- Risks: inherited bias, distribution shift, contamination, hallucination.

**Questions**

1. True or false: in LoRA, the pretrained weight matrix is updated during training.
<details><summary>Answer</summary>False. W₀ is frozen. Only the low-rank factors A and B are trained.</details>

2. A 1024 × 1024 weight matrix gets a LoRA adapter with r = 8. How many parameters does the adapter train?
<details><summary>Answer</summary>8 · (1024 + 1024) = 16,384, compared with 1,048,576 for the full matrix.</details>

3. A company has millions of unlabelled support tickets and 3,000 labelled ones. Describe how the unlabelled tickets can help.
<details><summary>Answer</summary>Use them for self-supervised training, for example masked-token prediction, to adapt a pretrained language model to the ticket domain. Then fine-tune it, or train a head on it, with the 3,000 labels.</details>

4. Give one advantage and one disadvantage of full fine-tuning compared with feature extraction.
<details><summary>Answer</summary>Advantage: the representation adapts to the task, which usually gives the best accuracy with enough data. Disadvantage: it is expensive, needs a full model copy per task, can overfit small data and can forget pretrained knowledge.</details>

5. Why should pretrained layers get a smaller learning rate than the new head?
<details><summary>Answer</summary>Their weights already encode useful features. Large updates, especially while the new head is still random, would destroy them.</details>

---

## 06 · Framing the task

**Summary**
- Multi-class: exactly one label, softmax + cross-entropy, argmax.
- Multi-label: any number of labels, sigmoid per class + BCE, threshold per class.
- Token tagging: the same head on every token. BIO separates neighbouring entities.
- BIO gives one tag per token, so it cannot represent overlapping spans. Then use one yes/no decision per class per token.
- Multi-task learning: one encoder, several heads, summed losses. It helps when the tasks share knowledge and data is scarce.

**Questions**

1. A skin image can show any combination of 7 findings. Which output layer and loss?
<details><summary>Answer</summary>7 sigmoid outputs with binary cross-entropy for each. The findings are not mutually exclusive.</details>

2. True or false: in multi-label classification, a threshold of 0.5 is always optimal.
<details><summary>Answer</summary>False. The best threshold depends on the class balance and on the costs of errors. Tune it on validation data.</details>

3. Why does A2 not use standard BIO tagging for Part B?
<details><summary>Answer</summary>The same words often belong to several techniques at once. BIO gives each token one tag, so it cannot represent the overlaps.</details>

---

## 07 · Evaluation and workflow

**Summary**
- Workflow: goal and metric, baseline, instrument, read curves, change one thing, test once.
- Test data is touched once. Validation and test should match the deployment setting.
- Use cross-validation with small data. Use iterative stratification for multi-label data.
- Check for leakage: duplicates, near-duplicates, future information, global normalisation.
- Precision, recall, F1. Micro-F1 is dominated by frequent classes. Macro-F1 weighs every class equally.
- Learning curves tell underfitting, overfitting and instability apart.
- Compare models in pairs on the same folds. Report bootstrap intervals.

**Questions**

1. Training F1 is 0.62 and validation F1 is 0.60. Both are well below what you need. Diagnosis and one fix?
<details><summary>Answer</summary>Underfitting: the model is poor even on training data and the gap is small. Increase capacity, train longer, or start from a stronger pretrained model.</details>

2. Training F1 is 0.97 and validation F1 is 0.71. Diagnosis and two fixes?
<details><summary>Answer</summary>Overfitting: a large gap between training and validation. Early stopping, dropout, weight decay, augmentation, more data or a smaller or partly frozen model.</details>

3. True or false: micro-F1 treats a rare class and a frequent class as equally important.
<details><summary>Answer</summary>False. That is macro-F1. Micro-F1 pools all counts, so frequent classes dominate.</details>

4. A colleague chose the decision threshold that maximised F1 on the test set. What is wrong, and how should it be done?
<details><summary>Answer</summary>The test set was used for a decision, so the test score is now optimistic. Choose the threshold on validation or out-of-fold predictions, then evaluate the test set once.</details>

5. Model A beats model B by 0.8 F1 points on average over 5 folds. The fold-wise differences have a standard deviation of 2.0 points. What do you conclude?
<details><summary>Answer</summary>The difference is within the noise. There is no evidence that A is better. Prefer the simpler or cheaper model.</details>

6. Your validation loss is below your training loss. Give two harmless explanations and one harmful one.
<details><summary>Answer</summary>Harmless: dropout is active only during training; training loss is averaged during the epoch while validation loss is measured at its end; the validation set may be easier. Harmful: leakage between training and validation data.</details>

---

## 08 · Interpretability

**Summary**
- Local versus global. Post-hoc versus interpretable by design. Model-specific versus model-agnostic.
- Explanations serve trust, debugging (shortcut learning) and fairness.
- Gradient × input fails when the model is saturated. Integrated Gradients averages gradients along a path from a baseline. Its attributions sum to f(x) − f(baseline).
- Grad-CAM needs feature maps and gradients. LIME and SHAP need only inputs and outputs.
- Shapley values share interaction effects fairly. Leave-one-out double-counts them.
- Attention weights are not a full explanation.
- Test faithfulness with deletion tests, model randomisation and counterfactuals.

**Questions**

1. True or false: Grad-CAM can explain a model that you can only access through a prediction API.
<details><summary>Answer</summary>False. Grad-CAM needs internal feature maps and gradients. Use a model-agnostic method such as LIME or SHAP.</details>

2. Compute the Shapley values for two features with v(∅) = 0, v({A}) = 0.2, v({B}) = 0.4, v({A,B}) = 1.0.
<details><summary>Answer</summary>φ_A = ½[(0.2 − 0) + (1.0 − 0.4)] = ½[0.2 + 0.6] = 0.4. φ_B = ½[(0.4 − 0) + (1.0 − 0.2)] = ½[0.4 + 0.8] = 0.6. Check: 0.4 + 0.6 = 1.0 = v({A,B}) − v(∅).</details>

3. Gradient × input gives near-zero scores for every word when the model is 99% sure. Why, and what fixes it?
<details><summary>Answer</summary>The output is saturated, so the local gradient is almost zero even for important words. Integrated Gradients accumulates gradients along a path from a baseline, through regions where the model is not saturated.</details>

4. A heatmap for a "wolf" classifier highlights snow, not the animal. What does this show, and what do you do?
<details><summary>Answer</summary>Shortcut learning: wolves in the training data mostly appear on snow. Collect or create data that breaks the correlation (wolves without snow, dogs on snow), retrain, and check again with the explanation method and a targeted test set.</details>

5. Name two tests that check whether an explanation is faithful.
<details><summary>Answer</summary>Deletion test: removing the most important inputs first should drop the score faster than a random order. Model randomisation: with random weights the explanation should change a lot.</details>

---

## 09 · Bias, shift and deployment

**Summary**
- Bias can come from the data and labels, from shortcuts the model learns, and from the deployment population.
- Measure per group, with confidence intervals. Small groups give weak evidence.
- Counterfactual tests change only the group attribute in the same input. They are paired and powerful.
- Template tests use neutral sentences. Any prediction there is bias attached to a name.
- Distribution shift: covariate, label/concept, temporal. Without target labels, use controlled transformations, a small probe set and label-free signals.
- Deployment needs privacy, fitting hardware and latency, stability, monitoring and human oversight.

**Questions**

1. True or false: if overall accuracy is high, the model is fair to all groups.
<details><summary>Answer</summary>False. Overall accuracy can hide large differences between groups. Compute metrics per group.</details>

2. Why is a counterfactual swap test usually stronger than comparing error rates between two groups?
<details><summary>Answer</summary>Each example is compared with itself, so differences in content between the groups cancel out. Only the swapped attribute changes. This needs far fewer examples to detect an effect.</details>

3. A sentiment model trained on product reviews from 2018 will be used on social media posts from 2026. Name three things that may go wrong and one way to get evidence for each without new labels.
<details><summary>Answer</summary>New slang and spelling (check the out-of-vocabulary rate and sub-word fragmentation). New topics and names (swap names in labelled data and measure the change). Different writing style such as emoji or all-caps (apply these changes to labelled data and measure the change). Also check whether the model's confidence stays high on the new posts.</details>

4. Your best model is a large hosted LLM, but the users are minors whose data may not leave the device. What do you do?
<details><summary>Answer</summary>The hosted model cannot be deployed. Use it only as an offline benchmark on public data. Deploy a local model that fits the hardware, and report the accuracy gap honestly.</details>


---

## 10 · CNNs and computer vision

**Summary**
- Sparse connectivity and parameter sharing: a small filter slides over the image. Far fewer weights, and a pattern is recognised anywhere.
- Output size = (input − kernel + 2 · padding) / stride + 1. A filter spans all input channels.
- Layers build up from edges to textures to object parts. The receptive field grows with depth.
- Pooling shrinks feature maps and gives some shift invariance.
- ResNet adds residual connections. U-Net (encoder–decoder with skip connections) does segmentation.
- Transfer learning: pretrained ImageNet backbone, new head, freeze, then fine-tune top layers with a small learning rate.

**Questions**

1. True or false: in a convolutional layer, every output neuron is connected to every input pixel.
<details><summary>Answer</summary>False. Each output looks only at a small local patch (sparse connectivity).</details>

2. A 64 × 64 input, kernel 5, padding 0, stride 1. What is the output width?
<details><summary>Answer</summary>(64 − 5 + 0) / 1 + 1 = 60.</details>

3. How many weights (with biases) does a conv layer with 32 filters of size 3 × 3 have on a 16-channel input?
<details><summary>Answer</summary>32 · (3 · 3 · 16 + 1) = 32 · 145 = 4,640.</details>

4. You must mark every tumour pixel in medical scans and have 800 labelled scans. Which architecture and which loss?
<details><summary>Answer</summary>An encoder–decoder such as U-Net, which outputs a label per pixel and works with little data. A loss that handles the imbalance between few tumour pixels and many background pixels, such as weighted binary cross-entropy or a Dice-based loss.</details>

5. Name two reasons why max pooling is used.
<details><summary>Answer</summary>It reduces the size of the feature maps (less compute and fewer parameters later) and it gives some invariance to small shifts of features.</details>

---

## 11 · Generative models

**Summary**
- Autoencoder: encoder → small latent code → decoder, trained to reconstruct the input. Useful for compression, denoising and anomaly detection. Cannot reliably generate.
- VAE: the encoder outputs a distribution, pushed towards N(0, I). Sample z and decode to generate. Often blurry.
- GAN: a generator fools a discriminator. Sharp samples, unstable training, mode collapse.
- Diffusion: a fixed forward process adds noise; a learned U-Net removes it step by step. Stable training, high quality, slow sampling.

**Questions**

1. True or false: an autoencoder needs labelled data.
<details><summary>Answer</summary>False. The input is its own target.</details>

2. Why can you not simply decode a random vector with a basic autoencoder to get a new realistic image?
<details><summary>Answer</summary>Its latent space has no enforced structure. Training codes sit in scattered regions, and a random vector usually falls in a gap that decodes to nonsense.</details>

3. How would you use an autoencoder to detect faulty machines from sensor data?
<details><summary>Answer</summary>Train it only on data from healthy machines. At run time, compute the reconstruction error. Unusual (faulty) patterns reconstruct badly, so an error above a threshold flags an anomaly.</details>

4. In a diffusion model, which process is learned: adding noise or removing it?
<details><summary>Answer</summary>Removing it. The forward noising process is fixed. The network learns the reverse, usually by predicting the added noise.</details>

5. Give one strength and one weakness of GANs compared with diffusion models.
<details><summary>Answer</summary>Strength: sampling is fast (one forward pass). Weakness: training is unstable and can suffer from mode collapse.</details>
