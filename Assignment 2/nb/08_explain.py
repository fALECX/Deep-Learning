# %% [markdown]
# ### Explainability: what does the label head look at?
# A label is only useful in class if we can show *why*. We use two complementary views on the final models:
# * **Integrated Gradients** (Sundararajan et al., 2017). This is model-agnostic: the technique's logit is integrated
#   along a straight path from a neutral input (every token embedding replaced by the padding embedding) to the real
#   input, giving each token a share of the logit. It satisfies *completeness*: the attributions sum to the change
#   in the logit.
# * **Label-wise attention weights** from our own head: where each technique's query looked.
#
# **Quantitative test.** If the label head relies on the persuasive words, attribution mass should concentrate inside
# the **gold span** of that technique. For every correctly predicted (meme, technique) pair on the test set we compute
# the share of positive IG attribution that falls on gold-span tokens. We compare it with the share of tokens that
# are in the gold span: this is what a head that looks everywhere uniformly would get.

# %%
def integrated_gradients(model, text, k, steps=32):
    enc = FINAL_TOK([text], return_tensors="pt", truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
    offs = trimmed_offsets(text, enc.pop("offset_mapping")[0].tolist())
    enc = enc.to(DEVICE); model.to(DEVICE).eval()
    emb_layer = model.encoder.get_input_embeddings()
    x = emb_layer(enc["input_ids"]).detach()
    x0 = emb_layer(torch.full_like(enc["input_ids"], FINAL_TOK.pad_token_id)).detach()
    tmask = torch.tensor([[a < b for a, b in offs]], device=DEVICE)
    grads = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps + 1, device=DEVICE)[1:]:
        xi = (x0 + a * (x - x0)).requires_grad_(True)
        H = model.encoder(inputs_embeds=xi, attention_mask=enc["attention_mask"]).last_hidden_state
        if model.label_head is not None:
            logit = model.label_head(H, enc["attention_mask"].bool())[0][0, k]
        else:
            logit = model.span_head(H)[0, :, k].masked_fill(~tmask[0], -1e4).max()
        grads += torch.autograd.grad(logit, xi)[0]
    attr = ((x - x0) * grads / steps).sum(-1)[0].detach().cpu().numpy()
    model.cpu()
    return attr, offs


ig_rows, ig_examples = [], []
explain_model = FINAL["models"][0]
for i, (text, spans) in enumerate(zip(df_test.text, gold_test)):
    for k in np.where((Y_test[i] == 1) & (TEST["labels"][i] == 1))[0]:
        attr, offs = integrated_gradients(explain_model, text, k)
        real = np.array([a < b for a, b in offs])
        inside = np.array([any(a < e and s < b for s, e, t in spans if t == TECHNIQUES[k]) for a, b in offs]) & real
        pos = np.clip(attr, 0, None) * real
        if pos.sum() > 0 and inside.sum() > 0:
            ig_rows.append({"technique": SHORT[TECHNIQUES[k]], "attribution in gold span": pos[inside].sum() / pos.sum(),
                            "tokens in gold span": inside.sum() / real.sum()})
        if len(ig_examples) < 4 and TECHNIQUES[k] in ("Name calling/Labeling", "Loaded Language", "Slogans", "Doubt") \
                and len(text.split()) < 30 and TECHNIQUES[k] not in [e[2] for e in ig_examples]:
            ig_examples.append((text, offs, TECHNIQUES[k], attr, inside))
ig = pd.DataFrame(ig_rows)
ig_tab = ig.groupby("technique").agg(n=("technique", "size"), attribution_in_span=("attribution in gold span", "mean"),
                                     uniform_expectation=("tokens in gold span", "mean")).sort_values("n", ascending=False)
ig_tab.loc["ALL"] = [len(ig), ig["attribution in gold span"].mean(), ig["tokens in gold span"].mean()]
display(ig_tab.style.format({"attribution_in_span": "{:.2f}", "uniform_expectation": "{:.2f}"})
        .set_caption("Share of positive IG attribution on gold-span tokens vs. share of tokens in the span"))

# %%
fig, axes = plt.subplots(len(ig_examples), 1, figsize=(13, 1.3 * len(ig_examples)))
for ax, (text, offs, tech, attr, inside) in zip(np.atleast_1d(axes), ig_examples):
    toks = [(text[a:b], v, ins) for (a, b), v, ins in zip(offs, attr, inside) if a < b]
    vmax = max(abs(v) for _, v, _ in toks) or 1
    xpos = 0.0
    for w, v, ins in toks:
        ax.text(xpos, 0.5, w, fontsize=9, va="center", fontweight="bold" if ins else "normal",
                bbox=dict(facecolor=plt.cm.RdBu_r(0.5 + 0.5 * v / vmax), edgecolor="black" if ins else "none",
                          boxstyle="round,pad=0.15", linewidth=0.8))
        xpos += 0.011 * (len(w) + 2)
    ax.set_xlim(0, max(xpos, 1)); ax.axis("off"); ax.set_title(f"{SHORT[tech]}", loc="left", fontsize=9)
fig.suptitle("Integrated Gradients for the predicted technique (red = pushes towards it; boxed = gold span)", y=1.02)
plt.tight_layout(); plt.show()


# %% [markdown]
# **Explainability result.** Overall, 45% of the positive attribution falls on gold-span tokens, against 42% for a
# uniform spread. For short, lexical techniques the concentration is a little stronger (Loaded Language 0.28 vs.
# 0.22), and for whole-message techniques the two are equal by construction. So the label head does **not** rely
# mainly on the exact words the annotators marked: it uses the whole meme. For a teacher this means the IG heatmap
# alone is not a reliable "here is the technique" pointer. The span head, which is trained for that job, should be
# the pointer, and C3 below checks how often the two agree.
