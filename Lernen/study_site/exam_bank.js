// Exam-style tasks in the format of the COSC2779 final exam.
// src "practice" = official practice exam (Lernen/Practice_Exam_DL (1).pdf), copied word for word.
// src "extra"    = written in the same style to cover the other topics.
// ch = chapter ids (01–11) the task belongs to. Part A: correct = indices into options.
window.EXAM = {
  A: [
    // ---------- official practice exam ----------
    { id: "A.1", src: "practice", ch: ["10"], q: "Which of the following are properties of convolutional neural networks?",
      options: ["Sparse connectivity (feature extraction usually happens locally)", "Parameter sharing (the same operation is applied at different locations)", "Dense connectivity (every output is connected to all input locations)", "Pooling helps reduce redundant information and provides some level of invariance to translations"],
      correct: [0, 1, 3], why: "Dense connectivity describes fully connected layers. A conv output sees only a local patch." },
    { id: "A.2", src: "practice", ch: ["10", "05"], q: "During fine-tuning in transfer learning, which of the following are correct steps?",
      options: ["Unfreeze some top layers of the base model", "Train the unfrozen base model layers with a smaller learning rate", "Both the head and base model weights can be trained", "All layers in the base model must remain frozen"],
      correct: [0, 1, 2], why: "Fine-tuning means unfreezing (some) base layers. Keeping all frozen is feature extraction." },
    { id: "A.3", src: "practice", ch: ["03"], q: "You are given the task of predicting the genre of short music clips of different lengths. Which Recurrent Neural Network (RNN) architecture is most suitable for this task?",
      options: ["One-to-one", "One-to-many", "Many-to-one", "Many-to-many"],
      correct: [2], why: "A sequence of audio frames goes in, one genre label comes out." },
    { id: "A.4", src: "practice", ch: ["11"], q: "Which of the following are issues with a basic autoencoder?",
      options: ["It can memorize (over-fit) the training data", "It cannot reliably generate novel samples (e.g. images) because its latent space distribution is unconstrained", "It cannot reconstruct its input", "It requires adversarial training"],
      correct: [0, 1], why: "Reconstruction is exactly what it is trained for. Adversarial training belongs to GANs." },
    { id: "A.5", src: "practice", ch: ["01"], q: "Which of the following statements about activation functions are correct?",
      options: ["Sigmoid outputs are always strictly between 0 and 1.", "ReLU outputs are always greater than or equal to 0.", "Sigmoid has a large gradient when its inputs have large positive or large negative values.", "ReLU has a large gradient when its inputs have large negative values."],
      correct: [0, 1], why: "Sigmoid saturates at both ends (gradient ≈ 0). ReLU has gradient 0 for negative inputs." },
    { id: "A.6", src: "practice", ch: ["03"], q: "Which of the following statements about GRUs and LSTMs are correct?",
      options: ["Both have built-in attention mechanisms.", "Both are designed to help capture long-range dependencies.", "Neither has a cell state.", "Both can process sequential data."],
      correct: [1, 3], why: "No attention inside either. The LSTM has a cell state; the GRU does not." },
    { id: "A.7", src: "practice", ch: ["11"], q: "Which of the following statements about GANs are correct?",
      options: ["Training GANs involves training both a generator and a discriminator.", "GANs are notoriously difficult to train.", "A trained GAN requires both the generator and discriminator to generate new images.", "Convolutional neural networks can be used in both the generator and discriminator of GANs."],
      correct: [0, 1, 3], why: "At generation time only the generator is needed." },
    { id: "A.8", src: "practice", ch: ["11"], q: "Which of the following statements about diffusion models are correct?",
      options: ["Diffusion models generate new samples (e.g. images) through multiple denoising steps.", "Diffusion models are generally unstable to train.", "Generating new samples (e.g. images) using diffusion models is generally fast.", "Diffusion models generally generate more diverse images compared to GANs."],
      correct: [0, 3], why: "Training is stable (a simple noise-prediction loss). Sampling is slow: many steps." },

    // ---------- extra: 01 neural network basics ----------
    { id: "X-A01", src: "extra", ch: ["01", "06"], q: "Which pairings of output layer and loss are correct?",
      options: ["One class out of K (single-label): softmax + cross-entropy", "Any number of K labels (multi-label): one sigmoid per class + binary cross-entropy", "Predicting a house price: sigmoid + binary cross-entropy", "The softmax outputs of one example sum to 1"],
      correct: [0, 1, 3], why: "A price is regression: linear output + MSE. Sigmoid squeezes into (0, 1)." },
    { id: "X-A02", src: "extra", ch: ["01"], q: "Which statements about regularisation are correct?",
      options: ["Dropout stays active at test time.", "With L2 weight decay each weight is multiplied by (1 − ηλ) before the gradient step.", "Early stopping uses the validation loss to decide when to stop training.", "Data augmentation mainly reduces the bias of a model."],
      correct: [1, 2], why: "Dropout is switched off at test time. Augmentation fights overfitting, so it reduces variance." },
    { id: "X-A03", src: "extra", ch: ["01"], q: "Which statements about initialisation and BatchNorm are correct?",
      options: ["If all weights start at zero, the neurons of a layer receive identical gradients and stay identical.", "He initialisation (variance 2/n) is designed for ReLU networks.", "At test time BatchNorm normalises with the statistics of the current test batch.", "BatchNorm has two learned parameters per channel, γ (scale) and β (shift)."],
      correct: [0, 1, 3], why: "At test time BatchNorm uses running averages collected during training (model.eval())." },
    { id: "X-A04", src: "extra", ch: ["01"], q: "Which statements about optimisers are correct?",
      options: ["Momentum averages past gradients and damps oscillations.", "Adam adapts the step size for every parameter separately.", "A larger batch size always gives better generalisation.", "Mini-batch SGD uses a subset of the training data for each update."],
      correct: [0, 1, 3], why: "'Always' is the trap. Very large batches often generalise worse." },
    { id: "X-A05", src: "extra", ch: ["01"], q: "Which statements about backpropagation are correct?",
      options: ["Backpropagation applies the chain rule from the loss back to every weight.", "Vanishing gradients are more likely in deep networks with sigmoid activations.", "The derivative of ReLU for a positive input is 1.", "Backpropagation only works for networks with one hidden layer."],
      correct: [0, 1, 2], why: "Sigmoid's derivative is at most 0.25, so products over many layers shrink." },

    // ---------- extra: 07 evaluation and workflow ----------
    { id: "X-A06", src: "extra", ch: ["07"], q: "Which of the following cause data leakage or a biased test estimate?",
      options: ["Fitting a feature scaler on the whole dataset before splitting.", "Choosing hyperparameters by their score on the test set.", "Putting all images of one patient into the same split.", "Randomly shuffling a time series before splitting it into train and test."],
      correct: [0, 1, 3], why: "A group (patient-level) split is the fix, not the problem." },
    { id: "X-A07", src: "extra", ch: ["07"], q: "A dataset has 95 % negatives and 5 % positives. Which statements are correct?",
      options: ["A model that always predicts 'negative' reaches 95 % accuracy.", "Recall = TP / (TP + FN).", "Precision = TP / (TP + FN).", "F1 is the harmonic mean of precision and recall."],
      correct: [0, 1, 3], why: "Precision = TP / (TP + FP)." },
    { id: "X-A08", src: "extra", ch: ["07"], q: "Which readings of learning curves are correct?",
      options: ["Training loss keeps falling while validation loss rises: overfitting.", "Training and validation loss are both high and close together: underfitting.", "Validation loss below training loss always means there is a bug.", "Early stopping keeps the weights of the epoch with the lowest validation loss."],
      correct: [0, 1, 3], why: "Dropout and augmentation are active only during training, so validation loss can be lower without a bug." },
    { id: "X-A09", src: "extra", ch: ["07"], q: "Which statements about handling class imbalance are correct?",
      options: ["A weighted loss gives the rare class more weight.", "Oversampling should be done before the train/test split.", "SMOTE creates new minority examples by interpolating between neighbours.", "Moving the decision threshold trades precision against recall."],
      correct: [0, 2, 3], why: "Oversampling before the split copies examples into the test set: leakage." },

    // ---------- extra: 10 CNNs ----------
    { id: "X-A10", src: "extra", ch: ["10"], q: "A conv layer has 10 filters of size 5 × 5, stride 1, no padding, applied to a 32 × 32 × 3 input. Which statements are correct?",
      options: ["The output is 28 × 28 × 10.", "The layer has 760 parameters (including biases).", "A max-pooling layer after it adds learnable weights.", "With 'same' padding and stride 1 the output would stay 32 × 32."],
      correct: [0, 1, 3], why: "(32 − 5)/1 + 1 = 28. Parameters = 5·5·3·10 + 10 = 760. Pooling has no weights." },
    { id: "X-A11", src: "extra", ch: ["10"], q: "Which statements about CNN architectures are correct?",
      options: ["ResNet's skip connections let gradients flow through very deep networks.", "A 1 × 1 convolution can reduce the number of channels.", "VGG uses large 11 × 11 filters in all layers.", "Depthwise separable convolutions (MobileNet) reduce computation."],
      correct: [0, 1, 3], why: "VGG's idea is many small 3 × 3 filters." },
    { id: "X-A12", src: "extra", ch: ["10"], q: "Which statements about detection and segmentation are correct?",
      options: ["IoU = area of overlap / area of union.", "Non-maximum suppression removes boxes that overlap strongly with a higher-scoring box.", "Semantic segmentation predicts one label for the whole image.", "U-Net connects encoder and decoder layers with skip connections."],
      correct: [0, 1, 3], why: "Segmentation gives one label per pixel." },
    { id: "X-A13", src: "extra", ch: ["10"], q: "Which statements about receptive fields are correct (stride 1, no dilation)?",
      options: ["Two stacked 3 × 3 conv layers have a 5 × 5 receptive field.", "Three stacked 3 × 3 conv layers have a 7 × 7 receptive field.", "Two 3 × 3 layers need fewer weights than one 5 × 5 layer with the same channels.", "Pooling makes the receptive field of later layers smaller."],
      correct: [0, 1, 2], why: "Each 3 × 3 layer adds 2. 2·9 = 18 < 25 weights per channel pair. Pooling enlarges it." },

    // ---------- extra: 02 text to numbers ----------
    { id: "X-A14", src: "extra", ch: ["02"], q: "Which statements about turning text into numbers are correct?",
      options: ["Sub-word tokenisation avoids the out-of-vocabulary problem.", "One-hot vectors capture how similar two words are.", "The attention mask marks padding tokens so the model ignores them.", "word2vec gives 'bank' the same vector in every sentence."],
      correct: [0, 2, 3], why: "All one-hot vectors are equally far apart. word2vec is static, BERT is contextual." },

    // ---------- extra: 03 RNNs ----------
    { id: "X-A15", src: "extra", ch: ["03"], q: "Which statements about recurrent networks are correct?",
      options: ["An RNN uses the same weights at every time step.", "Backpropagation through time unrolls the network over the time steps.", "Gradient clipping helps against exploding gradients.", "A vanilla RNN processes all time steps of a sequence in parallel."],
      correct: [0, 1, 2], why: "Each hidden state needs the previous one, so an RNN is sequential." },
    { id: "X-A16", src: "extra", ch: ["03"], q: "Which pairings of task and RNN shape are correct?",
      options: ["Tagging every word of a sentence with a part-of-speech label: many-to-many (aligned)", "Machine translation: many-to-many (encoder–decoder)", "Image captioning: one-to-many", "Sentiment of a product review: one-to-many"],
      correct: [0, 1, 2], why: "A review (many tokens) gives one sentiment: many-to-one." },

    // ---------- extra: 04 attention ----------
    { id: "X-A17", src: "extra", ch: ["04"], q: "Which statements about scaled dot-product self-attention are correct?",
      options: ["The scores QKᵀ are divided by √d_k.", "Softmax is taken over the keys, so each query's weights sum to 1.", "The cost grows linearly with the sequence length n.", "Without positional encoding, self-attention ignores word order."],
      correct: [0, 1, 3], why: "Every token attends to every token: n² scores." },
    { id: "X-A18", src: "extra", ch: ["04"], q: "Which statements about Transformer models are correct?",
      options: ["BERT is encoder-only and sees context on both sides of a token.", "GPT uses a causal mask, so a token cannot attend to later tokens.", "All heads of multi-head attention share one set of Q, K, V weights.", "Each Transformer block contains residual connections and layer normalisation."],
      correct: [0, 1, 3], why: "Every head has its own projections; that is what lets heads specialise." },

    // ---------- extra: 06 task types ----------
    { id: "X-A19", src: "extra", ch: ["06"], q: "Which statements about multi-label classification and tagging are correct?",
      options: ["Each class gets its own sigmoid output.", "A threshold per class can be tuned on the validation set.", "One BIO tag sequence can represent overlapping spans of different labels.", "Softmax makes the classes compete for probability mass."],
      correct: [0, 1, 3], why: "BIO gives one tag per token, so overlaps need one tag sequence per label (or sigmoid per class per token)." },

    // ---------- extra: 05 pretraining and adaptation ----------
    { id: "X-A20", src: "extra", ch: ["05"], q: "Which statements about adapting a pretrained model are correct?",
      options: ["Feature extraction freezes the backbone and trains only a new head.", "LoRA trains small low-rank matrices while the pretrained weights stay frozen.", "Full fine-tuning trains fewer parameters than LoRA.", "Zero-shot prompting needs no labelled training examples."],
      correct: [0, 1, 3], why: "Full fine-tuning trains all parameters: the most expensive option." },
    { id: "X-A21", src: "extra", ch: ["05"], q: "LoRA with rank r = 8 is applied to a 1000 × 1000 weight matrix W. Which statements are correct?",
      options: ["LoRA trains 8 · (1000 + 1000) = 16,000 parameters for this matrix.", "Full fine-tuning of this matrix trains 1,000,000 parameters.", "After training, BA can be merged into W, so inference is not slower.", "A larger rank r always reduces overfitting."],
      correct: [0, 1, 2], why: "A larger r means more trainable parameters, so more capacity, not less overfitting." },
    { id: "X-A22", src: "extra", ch: ["05"], q: "Which statements about self-supervised learning are correct?",
      options: ["Masked language modelling predicts hidden tokens from their context.", "Self-supervised learning needs a human label for every example.", "Contrastive learning pulls two augmented views of the same image together.", "Foundation models are pretrained on large unlabelled data and adapted to many tasks."],
      correct: [0, 2, 3], why: "The labels come from the data itself; that is the point of SSL." },

    // ---------- extra: 11 generative ----------
    { id: "X-A23", src: "extra", ch: ["11"], q: "Which statements about variational autoencoders are correct?",
      options: ["The encoder outputs a mean and a variance for each latent dimension.", "The KL term pushes the latent codes towards the prior N(0, I).", "The reparameterisation trick z = μ + σ·ε lets gradients flow through the sampling step.", "A VAE needs a discriminator network."],
      correct: [0, 1, 2], why: "The discriminator belongs to GANs." },
    { id: "X-A24", src: "extra", ch: ["11"], q: "Which statements comparing generative models are correct?",
      options: ["GANs can suffer from mode collapse.", "Diffusion sampling needs many network evaluations.", "A basic autoencoder is trained with a reconstruction loss.", "A GAN gives an explicit likelihood for every sample."],
      correct: [0, 1, 2], why: "GANs are implicit models: no likelihood." },
    { id: "X-A25", src: "extra", ch: ["11"], q: "Which statements about generating text and images are correct?",
      options: ["Greedy decoding always picks the most probable next token.", "A higher sampling temperature makes the output more random.", "Classifier-free guidance mixes a conditional and an unconditional noise prediction.", "An autoregressive language model generates all tokens of a sentence in one step."],
      correct: [0, 1, 2], why: "Autoregressive = one token at a time, each conditioned on the ones before." },

    // ---------- extra: 08 interpretability ----------
    { id: "X-A26", src: "extra", ch: ["08"], q: "Which statements about explanation methods are correct?",
      options: ["A saliency map uses the gradient of the output with respect to the input.", "Integrated Gradients averages gradients along a path from a baseline to the input.", "Attention weights are always a faithful explanation of a Transformer's decision.", "LIME fits a simple surrogate model around one prediction."],
      correct: [0, 1, 3], why: "Attention shows where information flows, not what caused the output." },
    { id: "X-A27", src: "extra", ch: ["08"], q: "Which statements are correct?",
      options: ["Integrated Gradients attributions sum to F(x) − F(baseline).", "Integrated Gradients needs a baseline input.", "Grad-CAM uses the feature maps of the last convolutional layer.", "Occlusion needs the gradients of the model."],
      correct: [0, 1, 2], why: "Occlusion only needs forward passes: hide a patch, watch the score." },
    { id: "X-A28", src: "extra", ch: ["08"], q: "Which statements about Shapley values are correct?",
      options: ["A feature's Shapley value averages its marginal contribution over all orders in which features are added.", "The Shapley values of all features add up to f(x) − f(empty set).", "Exact Shapley values are cheap to compute for hundreds of features.", "KernelSHAP is model-agnostic."],
      correct: [0, 1, 3], why: "Exact computation needs all 2ⁿ coalitions." },
    { id: "X-A29", src: "extra", ch: ["08"], q: "How can you test whether an explanation is faithful?",
      options: ["Delete the most important features first; a faithful explanation makes the score drop fast.", "Randomise the model weights; a faithful explanation should change.", "Ask people whether the heat map looks plausible; if yes, it is faithful.", "Train a probing classifier on hidden states to test what information they hold."],
      correct: [0, 1, 3], why: "Plausible to humans ≠ faithful to the model." },

    // ---------- extra: 09 bias and shift ----------
    { id: "X-A30", src: "extra", ch: ["09"], q: "Which statements about bias, shift and deployment are correct?",
      options: ["Covariate shift: the input distribution changes, P(y | x) stays the same.", "Label shift: the class proportions change.", "High accuracy on a held-out test set proves that the model is not biased.", "Quantisation and distillation can make a model smaller for deployment."],
      correct: [0, 1, 3], why: "Bias needs a per-group test; one overall number can hide it." }
  ],

  B: [
    // ---------- official practice exam ----------
    { id: "B.1", src: "practice", ch: ["10"], q: "You are designing a neural network for image classification. A colleague suggests replacing all convolutional layers with fully connected layers. Why might convolutional layers be more suitable for image data?",
      a: "Convolutional layers use sparse connectivity and parameter sharing. They extract features locally and apply the same operation at different locations. This exploits the spatial structure of image data, thus requiring fewer parameters than fully connected layers, and allowing CNNs to learn image features more effectively and potentially achieve better performance." },
    { id: "B.2", src: "practice", ch: ["03", "04"], q: "In a recurrent encoder–decoder model without attention, the decoder relies on the final encoder state to represent the entire input sequence. What limitation can this cause, particularly for long sequences, and how does attention help to address this limitation?",
      a: "The final encoder state may not capture all important information from the input sequence, particularly for long sequences. Attention allows the decoder to look back at all encoder states (not just the last one). This gives the decoder access to information from different parts of the input sequence, helping it focus on the most relevant parts of the input when generating each output." },
    { id: "B.3", src: "practice", ch: ["05"], q: "You have a pre-trained Transformer and a small labeled dataset for a new classification task. You want to adapt the model while training as few parameters as possible. Would you use feature extraction or full fine-tuning? Briefly explain why and how you would do this.",
      a: "Feature extraction is more suitable. We freeze the backbone (pre-trained Transformer model), use its outputs as fixed features, and train only a small task-specific head for the classification task. This requires training fewer parameters, is faster than full fine-tuning, and is suitable for small datasets." },
    { id: "B.4", src: "practice", ch: ["01", "07"], q: "You train a feedforward neural network using stochastic gradient descent (SGD), but the training loss oscillates significantly and does not seem to decrease and converge. What might cause this problem and how could you address it?",
      a: "One possible cause is that the learning rate is set too high, causing the model's parameters to overshoot good solutions during optimization. To address this, we can reduce the learning rate or use a learning rate schedule that gradually decreases it during training." },
    { id: "B.5", src: "practice", ch: ["11"], q: "You train a basic autoencoder on a dataset of images. After training, you want to generate a new image by randomly sampling a latent code z and passing it through the decoder. What problem might you encounter and how could you overcome it?",
      a: "For a basic autoencoder, the distribution of latent codes is not explicitly constrained, so randomly sampling a latent code may produce an unrealistic image. To address this, we can use a Variational Autoencoder (VAE) which encourages the latent codes to follow a known prior distribution (e.g. a standard normal distribution). This allows us to sample latent codes from that distribution and pass them through the decoder to generate more realistic images." },
    { id: "B.6", src: "practice", ch: ["05"], q: "You need to adapt the same large foundation model to multiple downstream tasks, but storing a separate fully fine-tuned model for each task would be too expensive. What approach would you use and why?",
      a: "I would use a Parameter-Efficient Fine-Tuning (PEFT) technique such as LoRA (Low-Rank Adaptation). LoRA keeps the pre-trained foundation model's weights frozen and trains only a small number of parameters through low-rank updates (LoRA parameters) for each task. This reduces the number of trainable parameters and allows us to store only one copy of the foundation model, together with the task-specific LoRA parameters for each task." },

    // ---------- extra ----------
    { id: "X-B01", src: "extra", ch: ["01"], q: "A colleague builds a 10-layer network without any activation functions and wonders why it cannot learn the XOR problem. Explain what goes wrong and how to fix it.",
      a: "Without activations every layer is a linear map, and a chain of linear maps is again one linear map. So the 10-layer network can only draw a straight decision boundary, but XOR needs a non-linear one. Adding a non-linear activation such as ReLU after each hidden layer fixes this." },
    { id: "X-B02", src: "extra", ch: ["01"], q: "Your 30-layer network with sigmoid activations trains very slowly, and the weights of the first layers barely change. What is the likely cause and how could you address it?",
      a: "This is the vanishing gradient problem: backpropagation multiplies many small sigmoid derivatives (at most 0.25), so the gradient reaching the first layers is almost zero. Switching to ReLU with He initialisation keeps the gradient scale stable. BatchNorm or residual connections also help the gradient reach the early layers." },
    { id: "X-B03", src: "extra", ch: ["07"], q: "Why do we need both a validation set and a separate test set? What goes wrong if you tune hyperparameters on the test set?",
      a: "The validation set is used to choose hyperparameters and to decide when to stop training. If we tune on the test set, the chosen model is fitted to that test set, so its test score is optimistically biased. The test set must be used only once, at the end, to estimate performance on new data." },
    { id: "X-B04", src: "extra", ch: ["07"], q: "A classifier for a rare disease (2 % positive) reaches 98 % accuracy. Why can this be misleading, and what would you report and change instead?",
      a: "Predicting 'healthy' for everyone already gives 98 % accuracy, so the number says nothing about finding sick patients. Report recall, precision and F1 for the positive class (or PR-AUC) and a confusion matrix. To improve recall, use a weighted loss for the rare class or lower the decision threshold, tuned on the validation set." },
    { id: "X-B05", src: "extra", ch: ["10"], q: "Explain how residual (skip) connections make it possible to train very deep CNNs such as ResNet.",
      a: "A residual block outputs F(x) + x, so the layers only have to learn the change to their input. If a block is not useful it can learn F ≈ 0 and pass the input through, so adding depth does not make the network worse. The identity path also lets the gradient flow directly back to early layers, which reduces vanishing gradients." },
    { id: "X-B06", src: "extra", ch: ["10"], q: "You have only 800 labelled photos of plant diseases. Which data augmentations would you use, and what must you be careful about?",
      a: "Random crops, flips, small rotations and colour jitter create new training views and reduce overfitting. Each augmentation must keep the label correct: for example, strong colour changes could hide the yellowing that defines a disease. Augment only the training set, never the validation or test set." },
    { id: "X-B07", src: "extra", ch: ["03"], q: "Why does a vanilla RNN struggle to learn long-range dependencies, and how does an LSTM help?",
      a: "Backpropagation through time multiplies the gradient by the recurrent weights at every step, so over long sequences it vanishes (or explodes). An LSTM has a cell state that is updated additively and controlled by gates. The forget gate can keep information unchanged over many steps, so the gradient can flow back much further." },
    { id: "X-B08", src: "extra", ch: ["04"], q: "In scaled dot-product attention the scores QKᵀ are divided by √d_k. Why?",
      a: "Dot products of long random vectors grow with the dimension d_k; their variance is about d_k. Large scores push the softmax into saturation, where one weight is almost 1 and the gradients are almost 0. Dividing by √d_k keeps the scores around unit variance, so the softmax stays smooth and training is stable." },
    { id: "X-B09", src: "extra", ch: ["04"], q: "Why are Transformers faster to train than RNNs on long texts, and what is the cost of this?",
      a: "An RNN must compute the hidden states one after another, while self-attention processes all tokens of a sequence in parallel on a GPU. Every token also reaches every other token in one step, so long-range information does not fade. The cost is that attention compares all pairs of tokens, so time and memory grow with n²." },
    { id: "X-B10", src: "extra", ch: ["06"], q: "A meme can show several persuasion techniques at once. Why would you use sigmoid outputs with binary cross-entropy instead of softmax, and how do you turn the outputs into predictions?",
      a: "This is multi-label classification: labels are not mutually exclusive. Softmax makes classes compete and forces the probabilities to sum to 1, while one sigmoid per class gives an independent yes/no probability. Each probability is compared with a threshold (e.g. 0.5, or one tuned per class on the validation set) to decide which labels to predict." },
    { id: "X-B11", src: "extra", ch: ["05"], q: "You fine-tune a general-domain BERT on a small set of clinical notes and get poor results. What is domain-adaptive pretraining and why might it help?",
      a: "Domain-adaptive pretraining continues the self-supervised pretraining task (masked language modelling) on a large amount of unlabelled clinical text before fine-tuning. The model learns the vocabulary and style of the domain without needing labels. Fine-tuning on the small labelled set then starts from representations that fit the domain better." },
    { id: "X-B12", src: "extra", ch: ["11"], q: "What is mode collapse in GANs and why does it happen?",
      a: "Mode collapse means the generator produces only a few kinds of outputs, for example the same face again and again, instead of the full variety of the data. It happens because the generator only has to fool the current discriminator, so finding one output that works is enough. Fixes include minibatch discrimination, Wasserstein losses or switching to diffusion models, which cover the data more fully." },
    { id: "X-B13", src: "extra", ch: ["08"], q: "Integrated Gradients needs a baseline. What is a baseline, why does it matter, and what would you use for images and for text?",
      a: "The baseline is an input that means 'no information', and IG attributes the difference F(x) − F(baseline) to the input features. A bad baseline changes the attributions, because features that equal the baseline get zero credit. Common choices are a black (or blurred) image for images and the embedding of [PAD] or zeros for text." },
    { id: "X-B14", src: "extra", ch: ["09"], q: "A skin-lesion classifier trained on photos from one hospital performs much worse at another hospital. What is the likely problem and how could you detect and reduce it?",
      a: "This is distribution shift, most likely covariate shift: the new hospital uses other cameras, lighting or patient groups. Detect it by monitoring input statistics and by labelling a small sample from the new site to measure performance. Reduce it with data from the new site (fine-tuning), stronger augmentation, or training on several hospitals." },
    { id: "X-B15", src: "extra", ch: ["02"], q: "Sentences in one batch have different lengths. What do padding and the attention mask do, and what happens if you forget the mask?",
      a: "Padding fills short sentences with [PAD] tokens so that all sequences in a batch have the same length and fit into one tensor. The attention mask is 1 for real tokens and 0 for [PAD], so attention gives the padding zero weight. Without the mask the model attends to the padding, so the same sentence gets different outputs depending on the batch." }
  ],

  C: [
    { id: "C.1", src: "practice", ch: ["10", "07"], title: "Developing a CNN Image Classifier",
      scenario: "You are developing a deep learning system to classify images of recyclable waste into six categories. You have several thousand labelled RGB images and decide to develop a CNN-based classifier.",
      parts: [
        { label: "(a) Choosing a CNN", q: "A team member suggests using a fully connected neural network directly on the image pixels instead of a CNN. Why is a CNN more suitable for this task? Give two reasons.",
          a: "CNNs are more suitable for images because they have several important properties:\n• Sparse connectivity: feature extraction happens locally.\n• Parameter sharing: the same operation is applied at different locations.\nThese properties exploit the spatial structure of image data, thus requiring fewer parameters than fully connected layers, and allowing CNNs to learn image features more effectively and potentially achieve better performance." },
        { label: "(b) Initial training", q: "You split your dataset into training, validation, and test sets. For your first attempt, you train a CNN from scratch and obtain the following results:\n• Training accuracy: 72%\n• Validation accuracy: 70%\nBoth training and validation performance are lower than you expected. Does this result mainly suggest overfitting or underfitting? Give one change you could try to address this problem.",
          a: "This result mainly suggests underfitting, because the model does not perform well even on the training set, and there is only a small difference between training and validation performance. One possible approach is to increase the model capacity, for example by adding more layers or filters." },
        { label: "(c) Improving generalization", q: "After increasing the capacity of the CNN and training it again, you obtain:\n• Training accuracy: 96%\n• Validation accuracy: 78%\nWhat problem does this result suggest? Briefly explain why and give two techniques that could help address this problem.",
          a: "This result suggests overfitting, because the model performs very well on the training set but there is a large gap between the training and validation performance. Possible techniques that can address this issue include data augmentation, dropout, and regularization." },
        { label: "(d) Evaluating the classification performance", q: "After improving the model, you obtain good overall validation accuracy. However, you want to understand whether the CNN performs well across all six waste categories, and which categories it tends to confuse. What additional evaluation results could you examine, and what information would they provide?",
          a: "We can examine per-class performance (e.g., accuracy, precision, recall) and a confusion matrix. Per-class performance shows us whether the model performs well or poorly on each individual waste category. A confusion matrix compares the ground-truth and predicted categories, allowing us to identify which categories are commonly confused with each other." },
        { label: "(e) Using a pretrained model", q: "You later obtain access to an ImageNet-pretrained ResNet. Instead of continuing to train your CNN from scratch, describe how you could adapt the pretrained ResNet to this classification task.",
          a: "We can use transfer learning. We replace the original classification head of the pretrained ResNet with a new classification head for the six waste categories. Initially, we freeze the pretrained layers and train only the new classification head using our labelled data. Then, if we want to further improve the performance, we can unfreeze some of the pretrained layers of the pretrained ResNet and continue to fine-tune them on our labelled dataset using a smaller learning rate." }
      ] },

    { id: "X-C1", src: "extra", ch: ["06", "07", "05", "08", "04"], title: "Detecting Harmful Comments",
      scenario: "A forum wants to flag comments for five kinds of harm (insult, threat, obscene, identity attack, spam). One comment can have several labels or none. You have 12,000 labelled comments; 'threat' appears in only 1 % of them. You have one GPU with 16 GB of memory and a pretrained Transformer encoder (DeBERTa-base).",
      parts: [
        { label: "(a) Framing the output", q: "Which output layer and loss would you use? Explain why softmax is not suitable.",
          a: "Use five sigmoid outputs (one per label) with binary cross-entropy. The labels are not mutually exclusive, and a comment can have none. Softmax forces the probabilities to sum to 1, so labels would compete and 'no label' could not be expressed." },
        { label: "(b) The rare label", q: "After training, the model never predicts 'threat'. Give two changes that address this, and say which metric you would report for this label.",
          a: "Give 'threat' a larger weight in the loss (weighted BCE, pos_weight ≈ negatives / positives ≈ 99) and tune a separate decision threshold for it on the validation set. Report recall, precision and F1 for 'threat' (or PR-AUC), not accuracy, because 99 % accuracy is reached by never predicting it." },
        { label: "(c) Adapting the encoder", q: "Full fine-tuning runs out of GPU memory. Name an approach that fits and explain how it works.",
          a: "Use LoRA: freeze the pretrained weights and add small trainable low-rank matrices B·A (rank r ≪ d) to the attention projections. Only these matrices and the new head get gradients and optimiser states, which cuts memory a lot. Results are usually close to full fine-tuning." },
        { label: "(d) Reading the training run", q: "Training loss falls steadily. Validation macro-F1 rises until epoch 3 and then falls while validation loss rises. What is happening and what would you do?",
          a: "The model starts to overfit after epoch 3: it fits the training comments better but generalises worse. Use early stopping on validation macro-F1 (keep the epoch-3 weights). Add regularisation such as dropout or weight decay, or lower the learning rate." },
        { label: "(e) Explaining a decision", q: "Moderators want to know which words made the model flag a comment. Which method would you use and how would you check that the explanation is trustworthy?",
          a: "Use Integrated Gradients on the token embeddings with a [PAD] (or zero) embedding baseline; it gives each token a score, and the scores sum to F(x) − F(baseline). Check faithfulness with a deletion test: remove the highest-scored tokens first and the predicted probability should drop quickly, faster than when random tokens are removed." }
      ] },

    { id: "X-C2", src: "extra", ch: ["11"], title: "Generating Product Images",
      scenario: "An online shop wants to generate new, realistic images of shoes (64 × 64 RGB) for design ideas. You have 50,000 shoe photos without any other labels.",
      parts: [
        { label: "(a) First attempt", q: "You train a basic autoencoder, then decode random latent vectors z. The images look like noise. Why, and which model fixes this?",
          a: "A basic autoencoder only learns to reconstruct; nothing constrains where the codes lie in latent space, so a random z usually falls in an empty region the decoder never saw. A VAE adds a KL term that pushes the codes towards N(0, I), so sampling z from N(0, I) gives meaningful images." },
        { label: "(b) Blurry results", q: "The VAE samples look realistic but blurry. Name a model family that gives sharper images and explain how it is trained.",
          a: "A GAN: a generator turns noise into images and a discriminator tries to tell real from generated images. Both are trained against each other, so the generator is pushed to produce images the discriminator cannot tell apart from real photos, which gives sharp details." },
        { label: "(c) Training problems", q: "Your GAN produces almost the same shoe for every z and the losses swing wildly. Name the two problems.",
          a: "The repeated shoe is mode collapse: the generator found a few outputs that fool the discriminator and ignores the rest of the data. The swinging losses show unstable adversarial training: the two networks keep over-correcting each other instead of converging." },
        { label: "(d) Diffusion instead", q: "A colleague proposes a diffusion model. Give one advantage and one disadvantage compared with the GAN.",
          a: "Advantage: training is stable (a simple loss that predicts the added noise) and samples are more diverse, with no mode collapse. Disadvantage: generation is slow, because an image needs many denoising steps, each one a full pass through the network." },
        { label: "(e) Controlling the output", q: "Designers now want to type 'red running shoe' and get matching images. What must change in the model?",
          a: "Make the model conditional: feed a text embedding of the prompt into the denoising network (e.g. through cross-attention), trained on image–caption pairs. With classifier-free guidance the model mixes a conditional and an unconditional prediction, and a guidance weight controls how strongly the image follows the text." }
      ] },

    { id: "X-C3", src: "extra", ch: ["07", "09", "08", "10"], title: "A Skin-Lesion Classifier in a New Clinic",
      scenario: "You build a CNN that classifies dermatoscope photos as benign or malignant. The data: 9,000 photos from 3,000 patients of one hospital (several photos per patient); 10 % of photos are malignant.",
      parts: [
        { label: "(a) Splitting the data", q: "How should you split the data, and what goes wrong with a random split by photo?",
          a: "Split by patient (group split): all photos of one patient go into the same split. A random split puts photos of the same patient into training and test, so the model can recognise the patient or the lesion, and the test score is too optimistic (leakage)." },
        { label: "(b) Choosing the metric", q: "Which error is worse here, and how does this change the metric and the threshold you use?",
          a: "Missing a malignant lesion (false negative) is worse than a false alarm. So report recall (sensitivity) for 'malignant' together with precision, and choose a lower threshold on the validation set that reaches the required recall." },
        { label: "(c) After deployment", q: "In a second clinic, recall drops from 0.90 to 0.65. The clinic uses a different camera. What has happened and how would you respond?",
          a: "This is distribution (covariate) shift: the inputs look different, while the link between lesion and diagnosis is the same. Label a sample of the new clinic's photos to measure performance, then fine-tune on them or retrain with images from several cameras and with colour and lighting augmentation." },
        { label: "(d) Fairness", q: "How would you check whether the model works equally well for patients with darker skin?",
          a: "Measure recall, false positive rate and false negative rate separately for each skin-tone group, with confidence intervals, because groups may be small. If one group is worse, collect more data for it, reweight it in training, and report the per-group results." },
        { label: "(e) A suspicious shortcut", q: "You suspect the model uses the ruler that doctors place next to suspicious lesions. How can you test this?",
          a: "Use Grad-CAM (or occlusion) to see which image regions drive the 'malignant' score. If the heat map lights up on the ruler, test it directly: cover the ruler or paste it into benign images and watch whether the prediction changes. Then fix the data (crop or remove rulers) and retrain." }
      ] },

    { id: "X-C4", src: "extra", ch: ["03", "04", "07"], title: "Predicting Machine Failure from Sensor Streams",
      scenario: "A factory records 4 sensor values per second from each machine. For every 10-minute window (600 time steps) you must predict whether the machine fails within the next hour. You have 2 years of data from 40 machines.",
      parts: [
        { label: "(a) Task shape", q: "Which RNN task shape fits this problem and what does the output layer look like?",
          a: "Many-to-one: a sequence of 600 steps goes in, one prediction comes out. Use the last hidden state (or a pooled state) into one sigmoid output trained with binary cross-entropy." },
        { label: "(b) Validation", q: "How would you split the data for validation, and why not use random k-fold cross-validation?",
          a: "Split by time: train on earlier months and validate on later ones (time-series / rolling-origin validation), ideally also holding out whole machines. Random k-fold mixes future windows into training, so the model sees the future and the score is too optimistic." },
        { label: "(c) Long sequences", q: "A vanilla RNN learns nothing useful from events early in the window. Explain why and what you would change.",
          a: "Over 600 steps the gradient is multiplied by the recurrent weights again and again and vanishes, so early events get no learning signal. Use an LSTM or GRU, whose gated, additively updated state keeps information and gradients over many steps." },
        { label: "(d) Exploding gradients", q: "During training, the loss sometimes jumps to NaN. What is the likely cause and the standard fix?",
          a: "Exploding gradients: the repeated multiplication through time makes some gradients huge, and one step throws the weights far away. The standard fix is gradient clipping (limit the gradient norm), possibly with a lower learning rate." },
        { label: "(e) A Transformer instead", q: "Would a Transformer encoder work here? Give one advantage and one cost.",
          a: "Yes. Self-attention lets every time step reach every other step directly and runs in parallel, so long-range patterns are easier to learn and training is faster on a GPU. The cost is n² attention: 600 steps give 360,000 scores per head and layer, so memory grows fast; downsampling the window into patches helps." }
      ] }
  ]
};
