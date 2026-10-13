# %% [markdown]
# ### E9: Additional public data, the Propaganda Techniques Corpus (PTC)
# **Declared external data.** The PTC is SemEval-2020 Task 11 (Da San Martino et al., 2020; Zenodo record 3952415,
# CC-BY 4.0): 371 English **news articles** (2017–2019) with 6,129 character-level technique spans, annotated with the
# same family of guidelines as our corpus.
# * **Why it might help:** about 3× more span annotations than our 750 memes (2,083 spans), for the same techniques.
# * **Why it might not:** news prose is long, formal and mixed-case. Memes are short, shouty and depend on images. The
#   PTC taxonomy is also coarser: two of its classes merge several of ours (*Whataboutism + Straw Man + Red Herring*;
#   *Bandwagon + Reductio ad hitlerum*), and it has no *Smears*, *Glittering generalities* or *Obfuscation*.
# * **No leakage:** the PTC contains news articles, not memes. Below we check that no meme of our corpus (≥ 6 words)
#   occurs in it verbatim, so it cannot contain the labels of our evaluation set.
# * **Compliance:** this is *data* that we train on ourselves. No checkpoint trained by anyone else on propaganda
#   labels is used.
#
# **How it is used:** intermediate training (Phang et al., 2018). The encoder plus both heads are first trained on PTC
# sentences for 2 epochs. Techniques that the PTC does not annotate cleanly are **masked out** of the loss, i.e.
# treated as neither positive nor negative. The resulting encoder weights are the starting point for the normal
# 5-fold training on memes.
#
# **H:** a small gain on techniques that are frequent in the PTC (Loaded Language, Name calling, Doubt, Exaggeration),
# partly cancelled by the domain shift. **Decision rule:** adopt it only if it wins under the tie rule.

# %%
import tarfile, urllib.request
PTC_URL = "https://zenodo.org/api/records/3952415/files/datasets-v2.tgz/content"
PTC_MAP = {"Loaded_Language": "Loaded Language", "Name_Calling,Labeling": "Name calling/Labeling", "Repetition": "Repetition",
           "Doubt": "Doubt", "Exaggeration,Minimisation": "Exaggeration/Minimisation",
           "Appeal_to_fear-prejudice": "Appeal to fear/prejudice", "Flag-Waving": "Flag-waving",
           "Causal_Oversimplification": "Causal Oversimplification", "Appeal_to_Authority": "Appeal to authority",
           "Slogans": "Slogans", "Black-and-White_Fallacy": "Black-and-white Fallacy/Dictatorship",
           "Thought-terminating_Cliches": "Thought-terminating cliché"}  # merged PTC classes are deliberately not mapped
PTC_KNOWN = np.array([t in PTC_MAP.values() for t in TECHNIQUES])


def load_ptc():
    path = Path("ptc_datasets-v2.tgz")
    if not path.exists():
        download(PTC_URL, path)
    arts, labels = {}, collections.defaultdict(list)
    with tarfile.open(path) as tf:
        for mem in tf.getmembers():
            if mem.name.startswith("datasets/train-articles/") and mem.name.endswith(".txt"):
                arts[re.findall(r"article(\d+)", mem.name)[0]] = tf.extractfile(mem).read().decode("utf8")
            elif mem.name == "datasets/train-task2-TC.labels":
                for line in tf.extractfile(mem).read().decode("utf8").splitlines():
                    aid, tech, s, e = line.split("\t")
                    if tech in PTC_MAP:
                        labels[aid].append((int(s), int(e), PTC_MAP[tech]))
    rows = []
    for aid, text in arts.items():
        start = 0
        for line in text.split("\n"):
            end = start + len(line)
            if line.strip():
                sp = [(max(s, start) - start, min(e, end) - start, t) for s, e, t in labels[aid] if s < end and e > start]
                sp = [x for x in sp if x[1] > x[0]]
                y = np.zeros(K, np.float32)
                for _, _, t in sp:
                    y[T2I[t]] = 1
                rows.append({"id": f"ptc{aid}_{start}", "text": line, "spans": sp, "y": y})
            start = end + 1
    return pd.DataFrame(rows)


df_ptc = load_ptc()
_meme_norm = [re.sub(r"\s+", " ", t.lower().strip()) for t in df_all.text if len(t.split()) >= 6]
_ptc_blob = re.sub(r"\s+", " ", " ".join(df_ptc.text).lower())
print(f"PTC sentences: {len(df_ptc)}, with a mapped technique: {(np.stack(df_ptc.y).sum(1) > 0).sum()} | "
      f"memes (≥6 words) found verbatim in the PTC: {sum(m in _ptc_blob for m in _meme_norm)}")
_has = np.stack(df_ptc.y).sum(1) > 0
df_ptc_train = pd.concat([df_ptc[_has], df_ptc[~_has].sample(int(_has.sum()), random_state=SEED)]) \
    .sample(frac=1, random_state=SEED).reset_index(drop=True)
if FAST_DEV_RUN:
    df_ptc_train = df_ptc_train.head(200)
PTC_ENCODER_STATE = {}


def pretrain_on_ptc(encoder, epochs=2, lr=3e-5):
    seed_everything()
    cfg = {**DEFAULT_CFG, "encoder": encoder, "adaptation": "full", "label_head": "label_attention", "init_encoder": None}
    tok, amp = get_tok(encoder), ENCODERS[encoder]["amp"] and DEVICE.type == "cuda"
    dl = torch.utils.data.DataLoader(MemeDataset(df_ptc_train, tok), batch_size=32, shuffle=True,
                                     collate_fn=make_collate(tok.pad_token_id))
    model = PersuasionModel(cfg).to(DEVICE)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": lr},
                             {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 5e-4}],
                            weight_decay=0.01)
    steps = epochs * len(dl); sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    known = torch.tensor(PTC_KNOWN, device=DEVICE)
    for ep in range(epochs):
        model.train(); tot = 0.0
        for b in dl:
            bd = {k: v.to(DEVICE) for k, v in b.items() if torch.is_tensor(v)}
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                ll, sl, _ = model(bd["input_ids"], bd["attention_mask"], bd["token_mask"])
                loss = bce_loss(ll.float(), bd["y"], mask=known.expand_as(ll)) + bce_loss(
                    sl.float(), bd["token_labels"], mask=bd["token_mask"][..., None] & known)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f"PTC intermediate training, epoch {ep + 1}: mean loss {tot / len(dl):.4f}")
    PTC_ENCODER_STATE[encoder] = {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
    del model; gc.collect()


base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
pretrain_on_ptc(base["encoder"])
run_cv("E9 + PTC intermediate training", **{**base, "init_encoder": "ptc"})
show([BEST, "E9 + PTC intermediate training"], BEST, "E9: additional data (PTC)")
pt_gain = pd.DataFrame({nm: f1_score(Y_dev, RESULTS[nm]["label_p"] >= RESULTS[nm]["thr_label"], average=None, zero_division=0)
                        for nm in [BEST, "E9 + PTC intermediate training"]}, index=[SHORT[t] for t in TECHNIQUES])
pt_gain["annotated in PTC"] = PTC_KNOWN
display(pt_gain.sort_values("annotated in PTC", ascending=False).style.format(
    "{:.3f}", subset=[BEST, "E9 + PTC intermediate training"]).set_caption("Out-of-fold F1 per technique"))
BEST = choose(BEST, ["E9 + PTC intermediate training"])
print("→ chosen:", BEST)


# %% [markdown]
# <!-- E9-RESULT -->
#
# **E9 result.** PTC intermediate training does not help overall (−0.001 ± 0.014). Per technique the picture is
# mixed: it helps several techniques that the PTC annotates (Loaded Language, Name calling, Flag-waving, Doubt,
# Exaggeration, Fear) but hurts others (Appeal to authority, Repetition, Causal oversimplification). News prose seems
# to teach the *vocabulary* of propaganda but not the *meme format*. The extra data is therefore not used in the final
# model, and no external data enters the final system.
