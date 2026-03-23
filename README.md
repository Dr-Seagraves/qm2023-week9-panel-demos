# Week 9: Panel Data Demos

Teaching demonstrations for **QM 2023 — Statistics II / Data Analytics**, Week 9: Panel Data Architecture.

## Contents

| File | Description |
|------|-------------|
| `panel_fixed_effects.py` | Main demo: simulated panel data, within vs. between variation, Pooled OLS bias, Fixed Effects estimation, diagnostic plots |
| `panel_data_fundamentals.ipynb` | Interactive Jupyter notebook walking through panel data concepts step by step |
| `*.png` | Pre-generated figures from the demo script |

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Demo

```bash
python panel_fixed_effects.py
```

**What it covers:**

- Simulated panel data (100 firms, 10 years) with known true coefficients
- Within-firm vs. between-firm variation decomposition
- Why Pooled OLS is biased when unobserved firm effects are present
- How Fixed Effects estimation recovers the true coefficients
- Diagnostic plots: residuals, Q-Q, firm effects comparison

## Generated Figures

| Figure | Shows |
|--------|-------|
| `between_firm_variation.png` | Cross-sectional heterogeneity across firms |
| `within_firm_variation.png` | Time-series variation within selected firms |
| `coefficient_comparison.png` | True vs. Pooled OLS vs. Fixed Effects coefficients |
| `firm_fixed_effects.png` | True vs. estimated firm-level intercepts |
| `residuals_vs_fitted.png` | FE model residual scatter |
| `residual_distribution.png` | Histogram of FE residuals |
| `qq_plot.png` | Q-Q plot for normality assessment |
| `panel_diagnostics.png` | Combined 2x2 diagnostic panel |

## Dependencies

- numpy
- pandas
- matplotlib
- statsmodels
- linearmodels (optional — falls back to manual demeaning if unavailable)
