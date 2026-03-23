"""
Week 9: Panel Data and Fixed Effects
=====================================
This script demonstrates:
1. Simulated panel data (multiple firms over time)
2. Within-group vs between-group variation
3. Pooled OLS vs Fixed Effects estimation
4. Entity fixed effects using linearmodels/statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Output directory: a new "figures" folder
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Try to import linearmodels, fall back to statsmodels
try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("linearmodels not available, using statsmodels")

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.api import het_breuschpagan
import statsmodels.api as sm

np.random.seed(42)

# =============================================================================
# 1. CREATE SIMULATED PANEL DATA
# =============================================================================
print("=" * 60)
print("WEEK 9: PANEL DATA AND FIXED EFFECTS")
print("=" * 60)

# Parameters
n_firms = 100       # Number of firms (entities)
n_years = 10        # Number of time periods
n_obs = n_firms * n_years

# Generate entity (firm) and time identifiers
firms = np.repeat(np.arange(1, n_firms + 1), n_years)
years = np.tile(np.arange(1, n_years + 1), n_firms)

# Generate firm-specific fixed effects (unobserved heterogeneity)
# These represent time-invariant firm characteristics (e.g., management quality, location)
firm_fe = np.random.normal(0, 2, n_firms)
entity_effect = firm_fe[firms - 1]

# Generate time fixed effects (common shocks to all firms)
time_fe = np.array([0.5 * (t - 1) for t in range(1, n_years + 1)])  # Upward trend
time_effect = time_fe[years - 1]

# Generate covariates
# X1: Firm size (time-varying, correlated with firm FE - endogeneity!)
# X2: R&D spending (time-varying, exogenous)
# X3: Industry competition (time-varying)
x1 = np.random.normal(5, 1, n_obs) + 0.5 * entity_effect + np.random.normal(0, 0.5, n_obs)
x2 = np.random.exponential(1, n_obs)
x3 = np.random.uniform(0, 1, n_obs)

# Generate error term
epsilon = np.random.normal(0, 1, n_obs)

# True data-generating process:
# Y = 1*X1 + 2*X2 - 1.5*X3 + firm_fe + time_effect + epsilon
y = 1 * x1 + 2 * x2 - 1.5 * x3 + entity_effect + time_effect + epsilon

# Create DataFrame
df = pd.DataFrame({
    'firm_id': firms,
    'year': years,
    'y': y,
    'x1_size': x1,
    'x2_rnd': x2,
    'x3_competition': x3
})

print(f"\nPanel Data Summary:")
print(f"  - Number of firms: {n_firms}")
print(f"  - Number of years: {n_years}")
print(f"  - Total observations: {n_obs}")
print(f"\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

# =============================================================================
# 2. WITHIN-GROUP VS BETWEEN-GROUP VARIATION
# =============================================================================
print("\n" + "=" * 60)
print("WITHIN-GROUP VS BETWEEN-GROUP VARIATION")
print("=" * 60)

# Within-group variation: deviation from firm mean (removes firm FE)
df_x1_within = df.groupby('firm_id')['x1_size'].transform(lambda x: x - x.mean())
df_x1_between = df.groupby('firm_id')['x1_size'].transform('mean')

# Calculate variances
var_total_x1 = df['x1_size'].var()
var_within_x1 = df_x1_within.var()
var_between_x1 = df_x1_between.var()

print(f"\nX1 (Firm Size) Variation Decomposition:")
print(f"  - Total variance:       {var_total_x1:.4f}")
print(f"  - Within-group variance: {var_within_x1:.4f} ({100*var_within_x1/var_total_x1:.1f}%)")
print(f"  - Between-group variance: {var_between_x1:.4f} ({100*var_between_x1/var_total_x1:.1f}%)")

# Same for Y
df_y_within = df.groupby('firm_id')['y'].transform(lambda x: x - x.mean())
var_total_y = df['y'].var()
var_within_y = df_y_within.var()

print(f"\nY (Outcome) Variation Decomposition:")
print(f"  - Total variance:       {var_total_y:.4f}")
print(f"  - Within-group variance: {var_within_y:.4f} ({100*var_within_y/var_total_y:.1f}%)")

# =============================================================================
# 3. POOLED OLS VS FIXED EFFECTS
# =============================================================================
print("\n" + "=" * 60)
print("POOLED OLS VS FIXED EFFECTS")
print("=" * 60)

# Prepare data
X = df[['x1_size', 'x2_rnd', 'x3_competition']]
X = sm.add_constant(X)
y = df['y']

# Pooled OLS (ignores panel structure - biased when firm FE matter)
pooled_model = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['firm_id']})

print("\n--- POOLED OLS (Naive, ignores firm effects) ---")
print(f"  const:       {pooled_model.params['const']:.4f}")
print(f"  x1_size:    {pooled_model.params['x1_size']:.4f} (true: 1.0)")
print(f"  x2_rnd:     {pooled_model.params['x2_rnd']:.4f} (true: 2.0)")
print(f"  x3_competition: {pooled_model.params['x3_competition']:.4f} (true: -1.5)")
print(f"  R-squared:  {pooled_model.rsquared:.4f}")

# Fixed Effects (entity fixed effects - absorbs firm FE)
if HAS_LINEARMODELS:
    # Using linearmodels
    df_panel = df.set_index(['firm_id', 'year'])
    X_fe = df_panel[['x1_size', 'x2_rnd', 'x3_competition']]
    y_panel = df_panel['y']
    
    fe_model = PanelOLS(y_panel, X_fe, entity_effects=True).fit(cov_type='clustered')
    
    print("\n--- FIXED EFFECTS (entity FE, absorbs firm heterogeneity) ---")
    print(f"  x1_size:    {fe_model.params['x1_size']:.4f} (true: 1.0)")
    print(f"  x2_rnd:     {fe_model.params['x2_rnd']:.4f} (true: 2.0)")
    print(f"  x3_competition: {fe_model.params['x3_competition']:.4f} (true: -1.5)")
    print(f"  R-squared:  {fe_model.rsquared:.4f}")
    print(f"  F-statistic: {fe_model.f_statistic.stat:.4f}")
else:
    # Manual fixed effects using demeaning (LSDV approach)
    # Demean by firm
    y_demeaned = df.groupby('firm_id')['y'].transform(lambda x: x - x.mean())
    X_demeaned = df.groupby('firm_id')[['x1_size', 'x2_rnd', 'x3_competition']].transform(lambda x: x - x.mean())
    X_demeaned = sm.add_constant(X_demeaned)
    
    fe_model = OLS(y_demeaned, X_demeaned).fit()
    
    print("\n--- FIXED EFFECTS (entity FE, absorbs firm heterogeneity) ---")
    print(f"  x1_size:    {fe_model.params['x1_size']:.4f} (true: 1.0)")
    print(f"  x2_rnd:     {fe_model.params['x2_rnd']:.4f} (true: 2.0)")
    print(f"  x3_competition: {fe_model.params['x3_competition']:.4f} (true: -1.5)")
    print(f"  R-squared:  {fe_model.rsquared:.4f}")

# =============================================================================
# 4. CREATE DIAGNOSTIC PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING DIAGNOSTIC PLOTS")
print("=" * 60)

# Precompute shared data for plots
firm_means = df.groupby('firm_id')['x1_size'].mean()
firm_stds = df.groupby('firm_id')['x1_size'].std()
vars_to_plot = ['x1_size', 'x2_rnd', 'x3_competition']
true_vals = [1.0, 2.0, -1.5]
pooled_vals = [pooled_model.params[v] for v in vars_to_plot]
if HAS_LINEARMODELS:
    fe_vals = [fe_model.params[v] for v in vars_to_plot]
    residuals = fe_model.resids
else:
    fe_vals = [fe_model.params[v] for v in vars_to_plot]
    residuals = fe_model.resid

# --- Combined 2x2 figure (kept as before) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Week 9: Panel Data & Fixed Effects Diagnostics', fontsize=14, fontweight='bold')

ax1 = axes[0, 0]
ax1.bar(range(20), firm_means.values[:20], yerr=firm_stds.values[:20],
        color='steelblue', alpha=0.7, capsize=3)
ax1.set_xlabel('Firm ID')
ax1.set_ylabel('X1 (Size)')
ax1.set_title('Between-Firm Variation in X1\n(Each bar = firm mean +/- std)')
ax1.axhline(df['x1_size'].mean(), color='red', linestyle='--', label='Overall mean')
ax1.legend()

ax2 = axes[0, 1]
sample_firms = [1, 25, 50]
for firm in sample_firms:
    firm_data = df[df['firm_id'] == firm]
    ax2.plot(firm_data['year'], firm_data['x1_size'], marker='o', label=f'Firm {firm}', alpha=0.7)
ax2.set_xlabel('Year')
ax2.set_ylabel('X1 (Size)')
ax2.set_title('Within-Firm Variation Over Time\n(Selected firms)')
ax2.legend()
ax2.set_xticks(range(1, n_years + 1))

ax3 = axes[1, 0]
x_pos = np.arange(len(vars_to_plot))
width = 0.25
ax3.bar(x_pos - width, true_vals, width, label='True', color='green', alpha=0.7)
ax3.bar(x_pos, pooled_vals, width, label='Pooled OLS', color='red', alpha=0.7)
ax3.bar(x_pos + width, fe_vals, width, label='Fixed Effects', color='blue', alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(vars_to_plot)
ax3.set_ylabel('Coefficient Estimate')
ax3.set_title('Coefficient Comparison\n(Pooled OLS biased, FE recovers true values)')
ax3.legend()
ax3.axhline(0, color='black', linewidth=0.5)

ax4 = axes[1, 1]
ax4.scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
ax4.axhline(0, color='red', linestyle='--')
ax4.set_xlabel('Observation')
ax4.set_ylabel('Residual')
ax4.set_title('Residuals from Fixed Effects Model')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'panel_diagnostics.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'panel_diagnostics.png'}")
plt.close(fig)

# -------------------------------------------------------------------------
# INDIVIDUAL DIAGNOSTIC FIGURES (one per chart)
# -------------------------------------------------------------------------

# --- 1. Between-firm variation in X1 ---
fig_between, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(20), firm_means.values[:20], yerr=firm_stds.values[:20],
       color='steelblue', alpha=0.7, capsize=3)
ax.set_xlabel('Firm ID')
ax.set_ylabel('X1 (Size)')
ax.set_title('Between-Firm Variation in X1\n(Each bar = firm mean +/- std)')
ax.axhline(df['x1_size'].mean(), color='red', linestyle='--', label='Overall mean')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'between_firm_variation.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'between_firm_variation.png'}")
plt.close(fig_between)

# --- 2. Within-firm variation over time ---
fig_within, ax = plt.subplots(figsize=(8, 5))
for firm in sample_firms:
    firm_data = df[df['firm_id'] == firm]
    ax.plot(firm_data['year'], firm_data['x1_size'], marker='o', label=f'Firm {firm}', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('X1 (Size)')
ax.set_title('Within-Firm Variation Over Time\n(Selected firms)')
ax.legend()
ax.set_xticks(range(1, n_years + 1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'within_firm_variation.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'within_firm_variation.png'}")
plt.close(fig_within)

# --- 3. Coefficient comparison (Pooled OLS vs Fixed Effects) ---
fig_coeff, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(vars_to_plot))
width = 0.25
ax.bar(x_pos - width, true_vals, width, label='True', color='green', alpha=0.7)
ax.bar(x_pos, pooled_vals, width, label='Pooled OLS', color='red', alpha=0.7)
ax.bar(x_pos + width, fe_vals, width, label='Fixed Effects', color='blue', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(vars_to_plot)
ax.set_ylabel('Coefficient Estimate')
ax.set_title('Coefficient Comparison\n(Pooled OLS biased, FE recovers true values)')
ax.legend()
ax.axhline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'coefficient_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'coefficient_comparison.png'}")
plt.close(fig_coeff)

# --- 4. Residuals from FE model (scatter) ---
fig_resid, ax = plt.subplots(figsize=(8, 5))
ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Observation')
ax.set_ylabel('Residual')
ax.set_title('Residuals from Fixed Effects Model')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'residuals_vs_fitted.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'residuals_vs_fitted.png'}")
plt.close(fig_resid)

# --- 5. Residual distribution (histogram) ---
fig_hist, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.7)
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution (Fixed Effects Model)')
ax.axvline(0, color='red', linestyle='--', label='Zero')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'residual_distribution.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'residual_distribution.png'}")
plt.close(fig_hist)

# --- 6. Q-Q plot of residuals ---
fig_qq, ax = plt.subplots(figsize=(8, 5))
sm.qqplot(np.array(residuals), line='45', ax=ax, alpha=0.5)
ax.set_title('Q-Q Plot of Residuals (Fixed Effects Model)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'qq_plot.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'qq_plot.png'}")
plt.close(fig_qq)

# --- 7. Firm fixed effects: True vs Estimated ---
fig_fe, ax = plt.subplots(figsize=(10, 6))
firm_means_y = df.groupby('firm_id')['y'].mean()
ax.scatter(firm_fe, firm_means_y.values, alpha=0.6)
ax.set_xlabel('True Firm Fixed Effect')
ax.set_ylabel('Estimated Firm Mean (Y)')
ax.set_title('Firm Fixed Effects: True vs Estimated\n(FE model absorbs unobserved heterogeneity)')
z = np.polyfit(firm_fe, firm_means_y.values, 1)
p = np.poly1d(z)
ax.plot(sorted(firm_fe), p(sorted(firm_fe)), "r--", alpha=0.8, label='Trend line')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'firm_fixed_effects.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'firm_fixed_effects.png'}")
plt.close(fig_fe)

# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
KEY TAKEAWAYS:

1. PANEL DATA STRUCTURE:
   - {n_firms} firms observed over {n_years} years
   - Total {n_obs} observations

2. VARIATION DECOMPOSITION:
   - X1 (size): {100*var_between_x1/var_total_x1:.1f}% is between-firm, {100*var_within_x1/var_total_x1:.1f}% is within-firm
   - The between-group variation confounds the OLS estimate

3. POOLED OLS BIAS:
   - X1 coefficient is biased because X1 is correlated with firm FE
   - True effect: 1.0, Pooled OLS: {pooled_model.params['x1_size']:.3f}
   
4. FIXED EFFECTS SOLUTION:
   - FE estimator uses within-firm variation only
   - Correctly estimates: {fe_model.params['x1_size']:.3f} (close to true 1.0)
   - Time-invariant heterogeneity is absorbed by firm FE

5. WHAT YOU CAN'T ESTIMATE WITH FE:
   - Any time-invariant variable (e.g., firm location, industry)
   - These are "absorbed" by the entity fixed effects
""")

print("\nScript completed successfully!")