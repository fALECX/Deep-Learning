"""Insert figure references into the learning docs (idempotent: skips figures already present)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (doc, image, caption, anchor text: the figure is inserted on its own lines just BEFORE this text)
FIGS = [
    ("01_neural_network_basics.md", "01_activations", "The three activations you meet most. Sigmoid squashes to 0…1, tanh to −1…1, ReLU cuts off negatives.", "**Why the activation matters.**"),
    ("01_neural_network_basics.md", "01_xor", "XOR: blue where x₁ and x₂ have the same sign, red where they differ. A linear model can only draw one straight line and stays at 50%. One hidden layer with ReLU bends the boundary and separates all four clusters.", "## 3. Layers and networks"),
    ("01_neural_network_basics.md", "01_gradient_descent", "The same loss, three learning rates. Each dot is one update step, starting at w = −1.", "**Mini-batch SGD.**"),
    ("01_neural_network_basics.md", "01_fitting", "Grey = the true pattern, blue = noisy training points, green = the model. Too simple misses the pattern; too flexible chases the noise.", "**L2 regularisation / weight decay**"),
    ("02_text_to_numbers.md", "02_tokenisation", "The same text cut three ways. Red tokens are rare and risk becoming [UNK].", "**How BPE (byte-pair encoding) builds its vocabulary.**"),
    ("02_text_to_numbers.md", "02_padding_mask", "Each row is one query token. Columns of [PAD] get −∞, so after the softmax their weight is 0.", "**Dynamic padding**"),
    ("02_text_to_numbers.md", "02_embedding_space", "A 2-D sketch of an embedding space. Related words cluster, and a relation like male → female is roughly the same arrow everywhere.", "- The weakness: one vector per word"),
    ("03_sequences_and_rnns.md", "03_task_shapes", "Blue = inputs, grey = the recurrent cell over time, green = outputs. Count the green boxes to name the shape.", "Rule of thumb: count the outputs you need."),
    ("03_sequences_and_rnns.md", "03_unrolled_rnn", "An RNN unrolled over four words. Every cell is the same network with the same weights; only the memory h changes.", "## 4. Training: backpropagation through time (BPTT)"),
    ("03_sequences_and_rnns.md", "03_vanishing", "Multiply 50 factors and the result either disappears or explodes. Log scale on the y-axis.", "Numeric intuition:"),
    ("03_sequences_and_rnns.md", "03_lstm", "The cell state runs along the top like a conveyor belt. Gates (purple) only scale or add to it.", "**Why this helps.** The cell state"),
    ("04_attention_and_transformers.md", "04_attention_it", "Illustrative attention weights for the token \"it\": most of the weight goes to \"animal\".", "## 2. Queries, keys and values"),
    ("04_attention_and_transformers.md", "04_qkv_flow", "The full computation of one attention layer, left to right.", "Step by step for one token $i$:"),
    ("04_attention_and_transformers.md", "04_softmax_scaling", "The same four scores, with and without scaling. Unscaled, the softmax becomes nearly one-hot.", "## 4. Multi-head attention"),
    ("04_attention_and_transformers.md", "04_positional_encoding", "Rows = positions, columns = dimensions. Left columns change fast with position, right columns slowly.", "- **Sinusoidal** (above)"),
    ("04_attention_and_transformers.md", "04_encoder_block", "One encoder block. The dashed green lines are the residual connections.", "A full model stacks many blocks"),
    ("04_attention_and_transformers.md", "04_masks", "Green = allowed, −∞ = blocked. Left: the encoder ignores padding. Right: the decoder cannot look ahead.", "For A2 we need understanding"),
    ("05_pretraining_and_adaptation.md", "05_adaptation_spectrum", "What is trained in each strategy. Grey layers are frozen, orange ones are trained.", "**Catastrophic forgetting:**"),
    ("05_pretraining_and_adaptation.md", "05_lora", "LoRA for one 768 × 768 matrix: the frozen path stays as it is; a thin trainable side path (A then B) adds the update.", "**Parameter count, worked example.**"),
    ("06_task_types_multilabel_tagging.md", "06_softmax_vs_sigmoid", "Same raw scores, two output functions. Softmax spreads one unit of probability; sigmoid lets all three true techniques pass the threshold.", "- Softmax makes the $K$ probabilities compete."),
    ("06_task_types_multilabel_tagging.md", "07_threshold", "Moving the threshold trades precision against recall. The dashed F1 curve peaks in between.", "- The best threshold is tuned on validation data"),
    ("06_task_types_multilabel_tagging.md", "06_overlap_tagging", "Each technique gets its own row of yes/no decisions per word. The rows overlap freely.", "This is called IO tagging per technique"),
    ("07_evaluation_and_workflow.md", "07_kfold", "5-fold cross-validation: five training runs, each with a different validation fold.", "**Stratification:**"),
    ("07_evaluation_and_workflow.md", "07_precision_recall", "The worked example as a picture. Precision looks at the green circle, recall at the red one.", "**Several classes: micro versus macro.**"),
    ("07_evaluation_and_workflow.md", "07_learning_curves", "The four curve shapes you must recognise. Blue = training loss, orange = validation loss.", "| What you see | Diagnosis | What to try |"),
    ("08_interpretability.md", "08_saturation_ig", "Left: on the flat part of the curve the gradient is almost 0, even though the input matters. Right: IG walks from the baseline to the input and collects the gradient (orange) on the way, where the curve is steep.", "- **Why it works:** along the path"),
    ("08_interpretability.md", "08_shapley", "The two orders of adding A and B. Shapley averages what each part adds over both orders.", "Leave-one-out for comparison:"),
    ("08_interpretability.md", "08_deletion", "A faithful explanation removes the important tokens first, so the score falls faster than with a random order.", "- **Model randomisation (sanity check):**"),
    ("09_bias_shift_deployment.md", "09_counterfactual", "A counterfactual pair. Illustrative numbers: a small change is fine, a large one points to bias.", "- The rhetoric stays the same."),
    ("09_bias_shift_deployment.md", "09_shift", "Left: deployment inputs drift away from the training inputs. Right: the same word means something else (illustrative).", "**How to get evidence without labelled target data:**"),
    ("10_cnns_and_vision.md", "10_convolution", "The worked example on a full 5 × 5 image. Each output cell is one placement of the filter.", "**Settings:**"),
    ("10_cnns_and_vision.md", "10_pooling", "2 × 2 max pooling: each coloured window keeps only its largest value.", "- It halves the width and height"),
    ("10_cnns_and_vision.md", "10_unet", "U-Net: shrink to understand, grow back to locate, and copy details across with skip connections.", "## 7. Vision tasks"),
    ("11_generative_models.md", "11_latent_spaces", "Each colour is one kind of training example. Only the VAE's space is filled, so random samples decode to something sensible.", "Trade-off: VAE samples are often a bit blurry"),
    ("11_generative_models.md", "11_gan", "The GAN game. Both networks train in turns.", "Training alternates:"),
    ("11_generative_models.md", "11_diffusion", "Forward process on a toy image. Generation runs the same strip from right to left with the learned denoiser.", "**Reverse process (learned).**"),
]

for doc, img, caption, anchor in FIGS:
    p = ROOT / doc
    s = p.read_text(encoding="utf8")
    tag = f"(img/{img}.png)"
    if tag in s:
        continue
    i = s.find(anchor)
    assert i >= 0, (doc, anchor)
    block = f"![{caption.split('.')[0]}](img/{img}.png)\n<sub>{caption}</sub>\n\n"
    s = s[:i] + block + s[i:]
    p.write_text(s, encoding="utf8")
    print("inserted", img, "into", doc)
