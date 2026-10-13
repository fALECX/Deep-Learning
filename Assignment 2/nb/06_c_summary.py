# %% [markdown]
# ### How large is the noise? (seed variance)
# The tie rule uses fold-to-fold variation. As a second check, the selected configuration is re-trained with two
# more seeds on the same folds. If seeds alone move the score by as much as the differences between design choices
# above, those choices should be read as "no measurable difference", not as wins.

# %%
base = {k: v for k, v in RESULTS[BEST]["cfg"].items() if k != "seed"}
seed_names = [BEST] + [run_cv(f"seed {s}", **{**base, "seed": s}) and f"seed {s}" for s in (1, 2)]
seed_tab = pd.DataFrame({nm: RESULTS[nm]["overall"] for nm in seed_names}).T[["micro_f1", "macro_f1", "span_f1", "select"]]
display(seed_tab.style.format("{:.3f}").set_caption("Same configuration, three seeds"))
print("sd across seeds:", seed_tab.std().round(3).to_dict())


# %% [markdown]
# <!-- SEED-RESULT -->
#
# **Noise level.** Changing only the seed moves the selection score by a standard deviation of 0.007 and macro-F1 by
# 0.015. Most design differences above (E3–E9) are of the same size. They should be read as "no measurable
# difference", which is exactly why the tie rule kept the simpler options. Only E1 (encoder), E2 (adaptation) and the
# learning-rate extremes in E6 are clearly larger than this noise.

# %% [markdown]
# ### Summary of the development experiments

# %%
summary = pd.DataFrame({nm: {**r["overall"], "trainable (M)": r["n_trainable"] / 1e6, "minutes": r["minutes"]}
                        for nm, r in RESULTS.items()}).T
summary = summary[["micro_f1", "macro_f1", "micro_p", "micro_r", "span_f1", "select", "micro_f1_lower",
                   "trainable (M)", "minutes"]]
summary.loc["baseline: majority"] = baselines.loc["majority"]
summary.loc["baseline: prior-random"] = baselines.loc["prior-random"]
display(summary.style.format("{:.3f}", na_rep="–").format({"trainable (M)": "{:.1f}", "minutes": "{:.1f}"}, na_rep="–")
        .highlight_max(subset=["micro_f1", "macro_f1", "span_f1", "select"], color="#d4edda")
        .set_caption("All development runs (out-of-fold, 750 memes)"))
FINAL_CFG = {k: v for k, v in RESULTS[BEST]["cfg"].items()}
print("FINAL configuration:", BEST, FINAL_CFG)
