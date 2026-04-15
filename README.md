# BERT Ensembles via Attention-Weight Topology

> **TL;DR** — Research project exploring whether **Representation Topological
> Divergence (RTD) of attention weights** can guide ensemble weighting for
> fine-tuned BERT models on IMDB sentiment classification, and whether combining
> weak (1-layer) and strong (6-layer) DistilBERT models produces better
> ensembles than strong models alone.

**Key finding:** Mixing weak 1-layer DistilBERT models with the full 5-model
strong ensemble improved accuracy beyond the strong-only baseline — suggesting
that ensemble diversity can come from model *capacity* as well as from
different random seeds.

**Tech stack:** Python · PyTorch · Hugging Face Transformers · Bayesian
Optimization · Quadratic Programming (cvxopt / qpsolvers) · RTD (Representation
Topological Divergence)

---

## Status

This repository contains the original research notebooks from 2023. It is
currently being refactored into a reusable Python library — see
[Refactor roadmap](#refactor-roadmap) below. For now, each notebook is
self-contained and was developed to run on Google Colab with model checkpoints
stored in Google Drive.

---

## Research question

Standard ensemble weighting methods optimize either for accuracy on a held-out
set (prone to overfitting) or for prediction correlation (which measures output
similarity but not *internal* diversity). This project asks:

> Can we measure model diversity directly from the attention weights — before
> looking at predictions at all — and use that signal to weight ensembles?

The approach uses **Representation Topological Divergence** (RTD, Trofimov et
al.) applied to the attention weight matrices of the final transformer layer,
producing a correlation matrix over the ensemble. This matrix is then fed into
a quadratic program that minimises correlation-based posterior variance under
simplex constraints — yielding ensemble weights that favour models with
dissimilar internal representations.

## Methodology

The experiments fine-tune **DistilBERT** on **IMDB** sentiment classification
(25k train / 12.5k val / 12.5k test) and compare three ensemble weighting
strategies:

| Method | What it optimizes | Tool |
|---|---|---|
| **Accuracy (Bayesian opt)** | Direct ensemble accuracy on validation | `bayesian-optimization` |
| **Prediction correlation (QP)** | Posterior variance from output correlation matrix | `cvxopt` / `qpsolvers` |
| **Attention-RTD correlation (QP)** | Posterior variance from RTD-based correlation of attention weights | `rtd` |

Models are evaluated with **accuracy-rejection curves** (an uncertainty
calibration metric) and area-under-rejection (AUC), which measure how well the
ensemble's confidence ranks correctly vs. incorrectly classified examples.

**Model variants:**

- **Strong models** — 5 full DistilBERT fine-tunes with different random seeds
  (42, 109, 121, 122, 123)
- **Weak models** — 3 DistilBERT fine-tunes with all but the first transformer
  layer removed, then re-trained (seeds 42, 43, 123)

## Key findings

1. **Attention-RTD weighting is competitive with prediction-correlation
   weighting** without ever looking at the model's outputs, suggesting
   attention geometry is a viable proxy for ensemble diversity.
2. **Weak + strong mixtures outperform strong-only ensembles** in several
   configurations — replacing one strong model with a weak one did not hurt
   accuracy and sometimes improved the accuracy-rejection AUC.
3. **Optimal subsets** of strong models (e.g. 3 out of 5) can match or exceed
   the full 5-model ensemble when weights are chosen via the QP solvers.

## Repository structure

```
NLP_ensembles/
├── BERT_training.ipynb          Fine-tune one strong DistilBERT on IMDB
├── BERT_training_weak.ipynb     Truncate to 1 layer, fine-tune (the weak models)
├── BERT_ensembles.ipynb         5-model strong ensemble with all 3 weighting methods
├── BERT_ensembles_rtd.ipynb     Same as above but with RTD-based correlation
├── BERT_pairs_comparison.ipynb  Exhaustive pairwise comparison (C(5,2) pairs)
├── models_subset.ipynb          Search over subsets of strong models
└── weak_bert_testing.ipynb      Weak + strong mixture experiments
```

Each notebook currently includes its own data loading, tokenization, and helper
functions (this duplication is what the refactor targets).

## Reproducing the experiments

> ⚠️ The notebooks depend on model checkpoints stored in Google Drive at
> `gdrive/MyDrive/results_bert_{1..5}/` and
> `gdrive/MyDrive/results_bert_weak_{1..3}/`. I am currently exporting these to
> a public Hugging Face Hub organisation so the notebooks run without
> Drive access — see [Refactor roadmap](#refactor-roadmap).

**In the meantime, to re-train from scratch on Colab:**

1. Open `BERT_training.ipynb` in Colab, select GPU runtime.
2. Update the Drive path in the `drive.mount()` cell to your own Drive.
3. Run all cells — one run produces one strong model. Re-run 5 times with seeds
   `[42, 109, 121, 122, 123]` to reproduce the strong set.
4. Repeat with `BERT_training_weak.ipynb` for the weak models (seeds
   `[42, 43, 123]`).
5. Once checkpoints exist, any of the ensemble notebooks can be run directly.

Expected training time per strong model: ~20 minutes on a Colab T4.

## Refactor roadmap

This repository is being converted from a notebook-only research project into a
reusable library:

- [x] Write proper README (this file)
- [ ] Add `pyproject.toml`, `requirements.txt`, `.gitignore`, `LICENSE`
- [ ] Extract shared code into `src/nlp_ensembles/` modules:
  - `data.py`, `models.py`, `training.py`
  - `ensembles.py`, `correlations.py`, `metrics.py`
  - `optimization.py`
- [ ] Rewrite notebooks as clean demos that import from the library
- [ ] Publish trained checkpoints to Hugging Face Hub
- [ ] Add CLI scripts for training and evaluation
- [ ] Add a small results table (saved predictions + metrics) to the repo

When the refactor is complete, the typical usage will look like:

```python
from nlp_ensembles import load_strong_ensemble, optimize_weights, evaluate

models = load_strong_ensemble(seeds=[42, 109, 121, 122, 123])
weights = optimize_weights(models, method="attention_rtd", val_data=val)
metrics = evaluate(models, weights, test_data=test)
```

## References

- Trofimov, I. et al. *Representation Topological Divergence*. The RTD method
  used for attention-weight comparison.
- Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding*. NAACL 2019.
- Sanh, V. et al. *DistilBERT, a distilled version of BERT*. NeurIPS 2019
  Workshop.

## Author

Polina Proskura — [github.com/polinaproskura](https://github.com/polinaproskura)

Research conducted in 2023.
