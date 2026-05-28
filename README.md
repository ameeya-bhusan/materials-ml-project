# Battery Cathode Voltage Prediction: Composition, Structure, and Honest Uncertainty

[![CI](https://github.com/ameeya-bhusan/materials-ml-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ameeya-bhusan/materials-ml-project/actions/workflows/ci.yml)

An end-to-end machine-learning pipeline that predicts the **average insertion voltage**
of lithium-ion battery cathodes from Materials Project data, built to demonstrate
applied data-science work where **evaluation rigor matters more than headline scores**.

The project moves from raw API data to a deployed screening demo, and at every stage
the emphasis is on testing claims honestly: justified data cleaning, leakage-aware
evaluation, multi-seed deep learning, and uncertainty estimates that are *validated*
rather than assumed.

## Headline findings

- **Composition alone predicts voltage surprisingly well.** A gradient-boosting model on
  Magpie composition descriptors reaches ~0.52 V MAE under chemistry-grouped evaluation,
  composition captures most of the predictable signal.
- **Crystal structure helps, but modestly.** A CGCNN graph neural network improves on the
  composition baseline by ~3% MAE on identical, honest evaluation, a real but small gain
  that only became clear after correcting three layers of evaluation optimism.
- **Ensemble uncertainty was uninformative — and I proved it.** Deep-ensemble disagreement
  did not correlate with prediction error (Spearman 0.007). Conformal prediction still
  delivered valid 90% coverage, but with constant-width intervals.
- **Uncertainty-guided active learning *underperformed* random** acquisition, a coherent
  consequence of the uninformative uncertainty signal.

## Project structure

| Path | Contents |
|------|----------|
| `notebooks/01_explore_electrodes.ipynb` | Data ingestion from Materials Project + physics-based cleaning |
| `notebooks/02_featurize_baseline.ipynb` | Magpie featurization + LightGBM baseline + leakage-aware evaluation |
| `notebooks/03_structures.ipynb` | Crystal structure retrieval |
| `notebooks/04_cgcnn_structure.ipynb` | CGCNN graph neural network (trained on GPU) |
| `notebooks/05_uncertainty_active_learning.ipynb` | Conformal calibration + active-learning study |
| `src/materials_ml/` | Reusable data-ingestion + cleaning module (tested) |
| `app.py` | Streamlit screening demo with calibrated intervals |
| `blog/` | Four write-ups documenting the work and reasoning |

## Write-ups

1. [Building a clean battery-materials dataset](blog/01-building-the-dataset.md) — and why I kept the "bad" entries
2. [How well can composition alone predict voltage?](blog/02-composition-baseline.md) — and how I almost fooled myself with the wrong split
3. [Does crystal structure actually help?](blog/03-cgcnn-structure-model.md) — three corrections that shrank a 19% improvement to a real 3%
4. [Honest uncertainty and why active learning backfired](blog/04-uncertainty-and-active-learning.md) — and why active learning backfired

## Reproduce

```bash
# 1. Install (uses uv)
uv sync --dev
uv pip install -e .

# 2. Run tests
uv run pytest -v

# 3. Train the screening model (needs data/processed/li_electrodes_clean.parquet)
uv run python train_screening_model.py

# 4. Launch the screening demo
uv run streamlit run app.py
```

Data files are not committed (gitignored). Notebook 01 regenerates the cleaned dataset
from the Materials Project API (requires a free API key set as `MP_API_KEY`).

## Stack

Python · pandas · scikit-learn · LightGBM · PyTorch · PyTorch Geometric · pymatgen ·
matminer · conformal prediction · Streamlit · pytest · GitHub Actions CI

## Notes

Built as a portfolio project. The CGCNN was trained on Kaggle's free GPU; all other work
runs on CPU. The active-learning study uses the composition model for tractability. For
research/demo use only — not a production voltage predictor.