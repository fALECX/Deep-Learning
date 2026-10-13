"""Insert the extra chapter-01 figures and an interactive 'Play with it' box into every chapter (idempotent)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

FIGS = [
    ("01_neural_network_basics.md", "01_neuron.png", "One neuron as a picture: inputs, weights, sum, activation, output.", "Common activation functions:"),
    ("01_neural_network_basics.md", "01_mlp.png", "A feed-forward network with two hidden layers. Every line is one weight that training adjusts.", "- A **feed-forward network** (also called MLP)"),
    ("01_neural_network_basics.md", "01_loss_landscape.png", "With two weights, the loss is a landscape. Gradient descent (red) always steps downhill. Real networks have millions of weights, but the picture is the same idea.", "- $\\eta$ too small: training is very slow."),
    ("01_neural_network_basics.md", "01_optimisers.gif", "Animation: plain gradient descent creeps along the flat valley, SGD wobbles because each mini-batch gives a noisy slope, momentum builds up speed, Adam adapts its step per weight.", "**Learning-rate schedules.**"),
    ("01_neural_network_basics.md", "01_backprop.png", "Forward pass (blue) computes the loss. Backward pass (red) multiplies the local slopes to get the gradient for w.", "Backpropagation reuses intermediate results"),
    ("01_neural_network_basics.md", "01_dropout.png", "Dropout: in each training step a random set of neurons is switched off (×). The network cannot rely on any single neuron.", "Dropout is only active during training."),
]

PLAY = {
    "01_neural_network_basics.md": [
        ("TensorFlow Playground", "https://playground.tensorflow.org/", "pick the XOR dataset (top-left, 'exclusive or'). Remove all hidden layers and press play: it fails. Add one hidden layer with ReLU: it works. Then set the learning rate to 3 and watch it explode."),
        ("Distill: Why momentum really works", "https://distill.pub/2017/momentum/", "drag the step-size and momentum sliders and watch the path zig-zag or glide."),
        ("MLU-Explain: neural networks", "https://mlu-explain.github.io/neural-networks/", "a scrolling visual story of neurons, layers and training."),
        ("MLU-Explain: bias and variance", "https://mlu-explain.github.io/bias-variance/", "see underfitting and overfitting change as model complexity grows."),
        ("Loss landscape gallery", "https://losslandscape.com/", "3-D renders of real network loss surfaces."),
    ],
    "02_text_to_numbers.md": [
        ("Tiktokenizer", "https://tiktokenizer.vercel.app/", "type 'loser', 'LOSER' and 'unbelievably' and watch how they split into tokens."),
        ("Tokenizer playground (Hugging Face)", "https://huggingface.co/spaces/Xenova/the-tokenizer-playground", "compare the tokenisers of different models on the same sentence."),
        ("TensorFlow Embedding Projector", "https://projector.tensorflow.org/", "search for 'king' and see its nearest neighbours in a real embedding space, rotated in 3-D."),
    ],
    "03_sequences_and_rnns.md": [
        ("Distill: Visualizing memorization in RNNs", "https://distill.pub/2019/memorization-in-rnns/", "hover over a sentence and see which earlier words an LSTM or GRU still uses."),
        ("Distill: Attention and augmented RNNs", "https://distill.pub/2016/augmented-rnns/", "animated diagrams of encoder–decoder RNNs and how attention looks back."),
        ("Chris Olah: Understanding LSTM networks", "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", "the step-by-step LSTM diagrams most courses copy."),
    ],
    "04_attention_and_transformers.md": [
        ("Transformer Explainer", "https://poloclub.github.io/transformer-explainer/", "type your own sentence and click through embeddings, Q/K/V, attention scores, softmax and the next-word prediction of a real GPT-2."),
        ("LLM visualisation (Brendan Bycroft)", "https://bbycroft.net/llm", "a 3-D walk through every matrix of a small GPT, step by step."),
        ("Jay Alammar: The illustrated Transformer", "https://jalammar.github.io/illustrated-transformer/", "the classic picture-by-picture explanation."),
        ("BertViz", "https://github.com/jessevig/bertviz", "a tool that draws attention between tokens for every head and layer (runs in a notebook)."),
    ],
    "05_pretraining_and_adaptation.md": [
        ("Jay Alammar: The illustrated BERT", "https://jalammar.github.io/illustrated-bert/", "pictures of pretraining (masked words) and then fine-tuning with a new head."),
        ("Transformer Explainer", "https://poloclub.github.io/transformer-explainer/", "see what a pretrained model already 'knows' before any fine-tuning: its next-word probabilities."),
    ],
    "06_task_types_multilabel_tagging.md": [
        ("MLU-Explain: logistic regression", "https://mlu-explain.github.io/logistic-regression/", "watch the sigmoid curve and the decision threshold move."),
        ("Hugging Face: token classification", "https://huggingface.co/learn/llm-course/chapter7/2", "worked NER example with the label alignment shown token by token."),
    ],
    "07_evaluation_and_workflow.md": [
        ("MLU-Explain: train, test and validation sets", "https://mlu-explain.github.io/train-test-validation/", "why the test set is only touched once, as a scrolling story."),
        ("MLU-Explain: cross-validation", "https://mlu-explain.github.io/cross-validation/", "animated k-fold splits."),
        ("MLU-Explain: precision and recall", "https://mlu-explain.github.io/precision-recall/", "drag the threshold and watch precision and recall trade off."),
        ("MLU-Explain: bias and variance", "https://mlu-explain.github.io/bias-variance/", "under- and overfitting as model complexity grows."),
        ("MLU-Explain: double descent", "https://mlu-explain.github.io/double-descent/", "the modern twist on overfitting from the week 1 slides."),
    ],
    "08_interpretability.md": [
        ("Distill: Visualizing the impact of feature attribution baselines", "https://distill.pub/2020/attribution-baselines/", "change the IG baseline and see the explanation change."),
        ("Distill: Feature visualization", "https://distill.pub/2017/feature-visualization/", "what single neurons of a CNN 'look for'."),
        ("CNN Explainer", "https://poloclub.github.io/cnn-explainer/", "see the feature maps that Grad-CAM weights."),
    ],
    "09_bias_shift_deployment.md": [
        ("Google ML crash course: fairness", "https://developers.google.com/machine-learning/crash-course/fairness", "short visual lessons on types of bias and how to measure them."),
        ("MLU-Explain", "https://mlu-explain.github.io/", "visual stories on evaluation; read the precision–recall one with groups in mind."),
    ],
    "10_cnns_and_vision.md": [
        ("CNN Explainer", "https://poloclub.github.io/cnn-explainer/", "click any neuron of a real CNN to see the convolution, ReLU and pooling it computes."),
        ("3-D CNN visualisation (Adam Harley)", "https://adamharley.com/nn_vis/cnn/3d.html", "draw a digit and watch every layer light up."),
        ("2-D CNN visualisation (Adam Harley)", "https://www.cs.ryerson.ca/~aharley/vis/conv/", "the flat version, easier to follow."),
    ],
    "11_generative_models.md": [
        ("GAN Lab", "https://poloclub.github.io/ganlab/", "train a small GAN in the browser and watch generator and discriminator fight."),
        ("Diffusion Explainer", "https://poloclub.github.io/diffusion-explainer/", "step through how Stable Diffusion turns noise into an image from a prompt."),
        ("TensorFlow Embedding Projector", "https://projector.tensorflow.org/", "explore what a latent space looks like in 3-D."),
    ],
}

for doc, img, caption, anchor in FIGS:
    p = ROOT / doc; s = p.read_text(encoding="utf8")
    if f"(img/{img})" in s:
        continue
    i = s.find(anchor); assert i >= 0, (doc, anchor)
    s = s[:i] + f"![{caption.split('.')[0]}](img/{img})\n<sub>{caption}</sub>\n\n" + s[i:]
    p.write_text(s, encoding="utf8"); print("inserted", img)

for doc, items in PLAY.items():
    p = ROOT / doc; s = p.read_text(encoding="utf8")
    if "**Play with it**" in s:
        continue
    box = "> [!TIP]\n> **Play with it** (interactive, in the browser). Do this before or after reading; seeing it move is the fastest way in.\n" + \
          "\n".join(f"> - [{name}]({url}): {what}" for name, url, what in items) + "\n\n"
    i = s.find("\n---\n")
    assert i >= 0, doc
    s = s[:i + 1] + "\n" + box + s[i + 1:]
    p.write_text(s, encoding="utf8"); print("play box", doc)
