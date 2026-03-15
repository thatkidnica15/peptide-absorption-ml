"""
Evaluation & Visualisation
============================
Regression metrics and pharmaceutical-domain visualisations.

Author: Veronica Pilagov
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)


def regression_report(y_true, y_pred, model_name="Model"):
    """Print regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n{'='*40}")
    print(f"{model_name}")
    print(f"{'='*40}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f} log units")
    print(f"  RMSE: {rmse:.4f} log units")
    
    return {'model': model_name, 'r2': r2, 'mae': mae, 'rmse': rmse}


def plot_predictions_comparison(results: dict, save=True):
    """Predicted vs actual scatter plots for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]
    
    colors = {'Gradient Boosted Trees': '#2196F3', 'Random Forest': '#4CAF50',
              'Neural Network (NumPy)': '#FF5722'}
    
    for ax, (name, data) in zip(axes, results.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        r2 = r2_score(y_true, y_pred)
        color = colors.get(name, '#666')
        
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color=color)
        lims = [min(y_true.min(), y_pred.min()) - 0.2,
                max(y_true.max(), y_pred.max()) + 0.2]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('Actual log(Papp)', fontsize=11)
        ax.set_ylabel('Predicted log(Papp)', fontsize=11)
        ax.set_title(f'{name}\nR² = {r2:.3f}', fontsize=12)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'predictions_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(importance_df, title="Feature Importance", top_n=15, save=True):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance_df.head(top_n).sort_values('importance')
    ax.barh(top['feature'], top['importance'], color='#2196F3', edgecolor='white')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=13)
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_permeability_landscape(df, save=True):
    """
    2D landscape: MW vs LogP coloured by permeability.
    Shows the "druglikeness" space with Lipinski boundaries.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df['molecular_weight'], df['logp_mean'],
        c=df['log_papp'], cmap='RdYlGn', s=15, alpha=0.6,
        vmin=-7, vmax=-4,
    )
    
    # Lipinski boundaries
    ax.axvline(500, color='red', linestyle='--', alpha=0.5, label='Lipinski MW limit (500)')
    ax.axhline(5, color='orange', linestyle='--', alpha=0.5, label='Lipinski LogP limit (5)')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log(Papp) cm/s', fontsize=11)
    ax.set_xlabel('Molecular Weight (Da)', fontsize=12)
    ax.set_ylabel('Mean LogP', fontsize=12)
    ax.set_title('Peptide Permeability Landscape', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'permeability_landscape.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_modification_impact(df, save=True):
    """
    Box plot showing how cyclisation and N-methylation
    affect permeability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Cyclisation
    groups = [df[df['is_cyclic']==0]['log_papp'], df[df['is_cyclic']==1]['log_papp']]
    bp1 = ax1.boxplot(groups, labels=['Linear', 'Cyclic'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#E3F2FD')
    bp1['boxes'][1].set_facecolor('#2196F3')
    ax1.set_ylabel('log(Papp) cm/s', fontsize=11)
    ax1.set_title('Effect of Cyclisation', fontsize=12)
    
    # N-methylation
    n_methyl_groups = []
    labels = []
    for nm in sorted(df['n_methylations'].unique()):
        n_methyl_groups.append(df[df['n_methylations']==nm]['log_papp'])
        labels.append(f'{int(nm)}')
    bp2 = ax2.boxplot(n_methyl_groups, labels=labels, patch_artist=True)
    blues = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0']
    for i, box in enumerate(bp2['boxes']):
        box.set_facecolor(blues[min(i, len(blues)-1)])
    ax2.set_xlabel('Number of N-methylations', fontsize=11)
    ax2.set_ylabel('log(Papp) cm/s', fontsize=11)
    ax2.set_title('Effect of N-Methylation', fontsize=12)
    
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'modification_impact.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_analogue_ranking(analogues_df, parent_papp, save=True):
    """Bar chart of top analogue improvements."""
    if analogues_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    top = analogues_df.head(15)
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in top['delta_log_papp']]
    ax.barh(top['substitution'], top['delta_log_papp'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Δ log(Papp) vs parent', fontsize=11)
    ax.set_title('Top Peptide Modifications by Predicted Absorption Improvement', fontsize=12)
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / 'analogue_ranking.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
