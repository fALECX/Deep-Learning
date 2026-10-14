# %% [markdown]
# ## 3. Model
#
# ### 3.1 Architecture
# ```
# meme text
#    │
#    ▼
# pretrained encoder (frozen / LoRA / fully fine-tuned)
#    │  contextual token vectors H (T × d)
#    ├────────────────────────────────┐
#    ▼                                ▼
# LABEL HEAD (Part A, self-built)     SPAN HEAD (Part B, self-built)
# one learned query per technique     per token: d → 256 → 20 sigmoids
# attends over the tokens             ("is this token inside technique k?")
# → 20 summaries → 20 sigmoids
# ```
# **Why one shared encoder with two heads (multi-task)?**
# * There are only 750 training memes. Both tasks need the same skill, recognising persuasive language, so sharing
#   the encoder lets each task regularise the other (Caruana, 1997). The span loss also tells the encoder which words
#   carry each label, which the meme-level label alone cannot.
# * The data showed that a meme has a technique iff it has a span for it. One model keeps both outputs together, and
#   their remaining disagreement can be measured (C3).
# * One encoder pass serves both outputs, which halves latency and memory on the school's T4.
#
# The alternatives (two separate models; labels derived as "any token tagged") are tested in E3.
#
# **Why technique-specific (label-wise) attention in the label head?** Section 1 showed that lexical techniques sit in
# a few words while discourse techniques span the whole meme. A single `[CLS]` vector must summarise all of them at
# once. Label-wise attention (as in CAML, Mullenbach et al., 2018) gives each technique its own query vector, so *Name
# calling* can attend to the insult and *Whataboutism* to the "what about…" clause. The attention weights also show
# which words the head used (explainability, C3). E4 compares it with `[CLS]` and mean pooling.
#
# **The self-built segment** is both heads plus the joint loss below. They are designed, initialised and trained from
# scratch, and their variants are tuned in Section 4.

# %%
ENCODERS = {
    "roberta-base": {"hf": "roberta-base", "amp": True},
    # RoBERTa further pre-trained (MLM only, no labels) on 2020-2021 tweets: same era and register as the memes.
    "twitter-roberta-2021": {"hf": "cardiffnlp/twitter-roberta-base-2021-124m", "amp": True},
    # Stronger general encoder (disentangled attention). Mixed precision is used; the gradient scaler guards overflow.
    "deberta-v3-base": {"hf": "microsoft/deberta-v3-base", "amp": True},
}
if FAST_DEV_RUN:
    ENCODERS = {k: {"hf": "distilroberta-base", "amp": True} for k in ENCODERS}


class LabelAttentionHead(nn.Module):
    """Label-wise attention: each technique k has a query q_k; logit_k = w_k · sum_t softmax_t(q_k·key_t) h_t + b_k."""

    def __init__(self, d, k, dropout):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.query = nn.Parameter(torch.randn(k, d) * d ** -0.5)
        self.out_w = nn.Parameter(torch.randn(k, d) * d ** -0.5)
        self.out_b = nn.Parameter(torch.zeros(k))
        self.drop = nn.Dropout(dropout)

    def forward(self, H, mask):
        scores = torch.einsum("btd,kd->bkt", self.key(self.drop(H)), self.query) / math.sqrt(H.size(-1))
        att = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min).softmax(-1)
        pooled = torch.einsum("bkt,btd->bkd", att, H)
        return (self.drop(pooled) * self.out_w).sum(-1) + self.out_b, att


class PooledHead(nn.Module):
    """Baseline label heads: [CLS] vector or masked mean of token vectors, then a 2-layer MLP."""

    def __init__(self, d, k, dropout, pooling):
        super().__init__()
        self.pooling = pooling
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, k))

    def forward(self, H, mask):
        if self.pooling == "cls":
            v = H[:, 0]
        else:
            m = mask.unsqueeze(-1).to(H.dtype)
            v = (H * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(v), None


class LoRALinear(nn.Module):
    """LoRA (Hu et al., 2022): y = W x + (alpha / r) * B A x, with W frozen and only A (r x d_in), B (d_out x r) trained.
    B starts at zero, so training starts exactly from the pretrained encoder."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale, self.drop = alpha / r, nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def add_lora(encoder, r, targets):
    for p in encoder.parameters():
        p.requires_grad = False
    for name, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha=2 * r))


class PersuasionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # .float(): some checkpoints (e.g. DeBERTa-v3) are stored in fp16 and recent transformers keep that dtype.
        self.encoder = AutoModel.from_pretrained(ENCODERS[cfg["encoder"]]["hf"]).float()
        d = self.encoder.config.hidden_size
        if cfg.get("init_encoder") == "ptc":  # encoder after intermediate training on PTC news (E9)
            self.encoder.load_state_dict(PTC_ENCODER_STATE[cfg["encoder"]])
        if cfg["adaptation"] == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif cfg["adaptation"] == "lora":
            add_lora(self.encoder, cfg["lora_r"], targets=("query", "value", "query_proj", "value_proj"))
        head = cfg["label_head"]
        self.label_head = (LabelAttentionHead(d, K, cfg["dropout"]) if head == "label_attention"
                           else PooledHead(d, K, cfg["dropout"], head) if head in ("cls", "mean") else None)
        self.span_head = (nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(d, 256), nn.GELU(),
                                        nn.Dropout(cfg["dropout"]), nn.Linear(256, K))
                          if cfg["span_loss_weight"] > 0 or head == "from_spans" else None)

    def forward(self, input_ids, attention_mask, token_mask):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        span_logits = self.span_head(H) if self.span_head is not None else None
        att = None
        if self.label_head is not None:
            label_logits, att = self.label_head(H, attention_mask.bool())
        else:  # "from_spans": a meme has technique k iff some token is tagged k  ->  max over tokens
            label_logits = span_logits.masked_fill(~token_mask[..., None], -1e4).max(1).values
        return label_logits, span_logits, att


# %% [markdown]
# ### 3.2 Objective
# $\mathcal{L} = \alpha\,\mathcal{L}_{\text{label}} + \lambda\,\mathcal{L}_{\text{span}}$, with α = λ = 1 unless an
# ablation switches one task off.
# Both terms are binary cross-entropies (multi-label, one sigmoid per technique).
# * The **span term** averages over real tokens only (special and padding tokens are masked).
# * **Class imbalance** (E5) is handled by one of: plain BCE; BCE with a per-technique positive weight
#   $w_k=\sqrt{N_{neg,k}/N_{pos,k}}$ (square root, so a technique with 4 examples does not get a 180× weight that
#   destabilises training); or focal loss (Lin et al., 2017), which down-weights easy negatives.

# %%
def bce_loss(logits, targets, pos_weight=None, focal_gamma=0.0, mask=None):
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if focal_gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = loss * (1 - p_t) ** focal_gamma
    if mask is not None:
        loss = loss[mask]
    return loss.mean()


# %% [markdown]
# ### 3.3 Data pipeline and training loop
# * **Dynamic padding** per batch: memes are short (median 28 tokens), so padding to 160 would waste ~80% of compute.
# * **Optional casing augmentation** (E8): with probability *p* a training meme is shown upper-cased, lower-cased or
#   sentence-cased. Changing case does not move character offsets, so the spans stay valid.
# * **AdamW**, linear warm-up (10%) and decay, gradient clipping at 1.0, mixed precision on GPU.
#   Encoder and heads get separate learning rates: the heads start from random weights and need larger steps.
# * **Model selection per fold**: after every epoch we score the validation fold. We keep the epoch with the best
#   average of label micro-F1 and span F1 (each at its best threshold on that fold), with early stopping after
#   `patience` epochs without improvement. Because this epoch choice uses the validation fold, out-of-fold numbers
#   are slightly optimistic. They are used only to *compare* configurations; the unbiased number is the test set.

# %%
def recase(text, rng):
    r = rng.rand()
    return text.upper() if r < 1 / 3 else text.lower() if r < 2 / 3 else text[:1].upper() + text[1:].lower()


class MemeDataset(torch.utils.data.Dataset):
    def __init__(self, df, tok, casing_aug=0.0, seed=SEED):
        self.df, self.tok, self.casing_aug = df.reset_index(drop=True), tok, casing_aug
        self.rng = np.random.RandomState(seed)
        self.cache = tokenize_frame(self.df, tok)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self.casing_aug > 0 and self.rng.rand() < self.casing_aug:
            row = self.df.iloc[[i]].copy()
            row["text"] = recase(row.text.iloc[0], self.rng)
            enc = tokenize_frame(row, self.tok)
            ids, offs, lab = enc["input_ids"][0], enc["offsets"][0], enc["token_labels"][0]
        else:
            ids, offs, lab = self.cache["input_ids"][i], self.cache["offsets"][i], self.cache["token_labels"][i]
        return {"ids": ids, "offs": offs, "lab": lab, "y": self.df.y.iloc[i], "idx": i}


def make_collate(pad_id):
    def collate(batch):
        T = max(len(b["ids"]) for b in batch)
        n = len(batch)
        ids = torch.full((n, T), pad_id, dtype=torch.long)
        att = torch.zeros((n, T), dtype=torch.long)
        tmask = torch.zeros((n, T), dtype=torch.bool)
        tlab = torch.zeros((n, T, K))
        for j, b in enumerate(batch):
            L = len(b["ids"])
            ids[j, :L] = torch.tensor(b["ids"]); att[j, :L] = 1
            tmask[j, :L] = torch.tensor([a < c for a, c in b["offs"]])
            tlab[j, :L] = torch.from_numpy(b["lab"])
        y = torch.from_numpy(np.stack([b["y"] for b in batch]))
        return {"input_ids": ids, "attention_mask": att, "token_mask": tmask, "token_labels": tlab, "y": y,
                "idx": [b["idx"] for b in batch], "offs": [b["offs"] for b in batch]}
    return collate


@torch.no_grad()
def predict(model, loader, amp):
    model.eval()
    label_p, token_p, offsets, atts = [], [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and DEVICE.type == "cuda"):
            ll, sl, att = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["token_mask"].to(DEVICE))
        label_p.append(torch.sigmoid(ll.float()).cpu().numpy())
        for j, offs in enumerate(b["offs"]):
            L = len(offs)
            token_p.append(torch.sigmoid(sl[j, :L].float()).cpu().numpy() if sl is not None else None)
            offsets.append(offs)
            atts.append(att[j, :, :L].float().cpu().numpy() if att is not None else None)
    return np.concatenate(label_p), token_p, offsets, atts


LABEL_THR_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)
SPAN_THR_GRID = np.round(np.arange(0.20, 0.81, 0.10), 2)


def best_label_thr(Y, P):
    scores = [f1_score(Y, P >= t, average="micro", zero_division=0) for t in LABEL_THR_GRID]
    i = int(np.argmax(scores)); return LABEL_THR_GRID[i], scores[i]


def spans_from_probs(token_p, offsets, thr, texts=None):
    texts = texts if texts is not None else [None] * len(offsets)
    return [decode_spans(o, p, thr, t) if p is not None else [] for o, p, t in zip(offsets, token_p, texts)]


def best_span_thr(gold_spans, token_p, offsets, texts=None):
    if token_p[0] is None:
        return None, float("nan")
    scores = [span_scores(gold_spans, spans_from_probs(token_p, offsets, t, texts))["span_f1"] for t in SPAN_THR_GRID]
    i = int(np.argmax(scores)); return SPAN_THR_GRID[i], scores[i]


DEFAULT_CFG = {
    "encoder": "roberta-base", "adaptation": "full", "lora_r": 16, "label_head": "label_attention",
    "label_loss_weight": 1.0, "span_loss_weight": 1.0, "loss": "bce", "focal_gamma": 2.0,
    "lr_encoder": 3e-5, "lr_head": 5e-4, "weight_decay": 0.01, "dropout": 0.1,
    "epochs": 15, "patience": 4, "batch_size": 16, "warmup": 0.1, "casing_aug": 0.0, "init_encoder": None,
    "seed": SEED,
}
if FAST_DEV_RUN:
    DEFAULT_CFG.update(epochs=1, patience=1)


def train_fold(cfg, df_tr, df_va, tok, keep_model=False):
    seed_everything(cfg["seed"])
    amp = ENCODERS[cfg["encoder"]]["amp"] and DEVICE.type == "cuda"
    collate = make_collate(tok.pad_token_id)
    g = torch.Generator(); g.manual_seed(cfg["seed"])
    dl_tr = torch.utils.data.DataLoader(MemeDataset(df_tr, tok, cfg["casing_aug"], cfg["seed"]),
                                        batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    dl_va = torch.utils.data.DataLoader(MemeDataset(df_va, tok), batch_size=64, collate_fn=collate)
    model = PersuasionModel(cfg).to(DEVICE)
    Ytr = np.stack(df_tr.y)
    pos, neg = Ytr.sum(0), len(Ytr) - Ytr.sum(0)
    pw = torch.tensor(np.sqrt(neg / np.maximum(pos, 1)), dtype=torch.float32, device=DEVICE) \
        if cfg["loss"] == "bce_pw" else None
    gamma = cfg["focal_gamma"] if cfg["loss"] == "focal" else 0.0
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": head_params, "lr": cfg["lr_head"]}]
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg["lr_encoder"] if cfg["adaptation"] == "full" else cfg["lr_head"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(dl_tr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    Yva, gold_va = np.stack(df_va.y), list(df_va.spans)
    best, best_state, history, bad = -1, None, [], 0
    for epoch in range(cfg["epochs"]):
        model.train(); t0 = time.time(); tr_loss = 0.0
        for b in dl_tr:
            b_dev = {k: v.to(DEVICE) for k, v in b.items() if torch.is_tensor(v)}
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                ll, sl, _ = model(b_dev["input_ids"], b_dev["attention_mask"], b_dev["token_mask"])
                loss = cfg["label_loss_weight"] * bce_loss(ll.float(), b_dev["y"], pw, gamma)
                if sl is not None and cfg["span_loss_weight"] > 0:
                    loss = loss + cfg["span_loss_weight"] * bce_loss(sl.float(), b_dev["token_labels"], None, gamma,
                                                                     mask=b_dev["token_mask"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tr_loss += loss.item() * len(b["idx"])
        lp, tp, offs, _ = predict(model, dl_va, amp)
        _, f_lab = best_label_thr(Yva, lp)
        _, f_span = best_span_thr(gold_va, tp, offs, list(df_va.text))
        score = np.nanmean([f_lab, f_span])
        history.append({"epoch": epoch + 1, "train_loss": tr_loss / len(df_tr), "val_label_f1": f_lab,
                        "val_span_f1": f_span, "secs": time.time() - t0})
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = (lp, tp, offs)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    out = {"label_p": best_pred[0], "token_p": best_pred[1], "offsets": best_pred[2], "history": history,
           "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "n_total": sum(p.numel() for p in model.parameters())}
    if keep_model:
        out["model"] = model.cpu()
    else:
        del model
    gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    return out
