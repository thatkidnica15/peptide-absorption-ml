"""
PeptideAbsorb — Main Pipeline
===============================
End-to-end pipeline for peptide absorption prediction and analogue design.

Usage: python main.py

Author: Veronica Pilagov
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from data.generate_peptide_data import generate_dataset, compute_descriptors
from src.feature_engineering import engineer_features, FEATURE_COLS
from src.models import (
    train_gradient_boosting, train_random_forest,
    NeuralNetRegressor, cross_validate
)
from src.analogue_generator import generate_single_substitutions, suggest_modifications
from src.evaluation import (
    regression_report, plot_predictions_comparison,
    plot_feature_importance, plot_permeability_landscape,
    plot_modification_impact, plot_analogue_ranking
)
from sklearn.ensemble import GradientBoostingRegressor


def main():
    print("=" * 60)
    print("PEPTIDEABSORB")
    print("Peptide Oral Absorption Prediction & Analogue Design")
    print("=" * 60)
    
    # ─── 1. DATA GENERATION ───────────────────────────────────
    print("\n[1/7] Generating peptide dataset...")
    df = generate_dataset(n_samples=1500, seed=42)
    print(f"  Peptides: {len(df)}")
    print(f"  Permeable fraction: {df['permeable'].mean():.1%}")
    print(f"  Length range: {df['length'].min()}-{df['length'].max()} residues")
    
    # ─── 2. FEATURE ENGINEERING ────────────────────────────────
    print("\n[2/7] Engineering pharmaceutical features...")
    df = engineer_features(df)
    print(f"  Features: {len(FEATURE_COLS)} ({len(FEATURE_COLS) - 14} engineered)")
    
    # ─── 3. TRAIN/TEST SPLIT ──────────────────────────────────
    print("\n[3/7] Splitting data...")
    X = df[FEATURE_COLS].values
    y = df['log_papp'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    
    results = {}
    
    # ─── 4. GRADIENT BOOSTED TREES ─────────────────────────────
    print("\n[4/7] Training Gradient Boosted Trees...")
    gbt_model, gbt_importance = train_gradient_boosting(X_train, y_train, FEATURE_COLS)
    gbt_pred = gbt_model.predict(X_test)
    results['Gradient Boosted Trees'] = {'y_true': y_test, 'y_pred': gbt_pred}
    
    gbt_cv = cross_validate(
        GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        X_train, y_train
    )
    print(f"  CV R²: {gbt_cv['r2_mean']:.4f} (+/- {gbt_cv['r2_std']:.4f})")
    
    # ─── 5. RANDOM FOREST ──────────────────────────────────────
    print("\n[5/7] Training Random Forest...")
    rf_model, rf_importance = train_random_forest(X_train, y_train, FEATURE_COLS)
    rf_pred = rf_model.predict(X_test)
    results['Random Forest'] = {'y_true': y_test, 'y_pred': rf_pred}
    
    # ─── 6. NEURAL NETWORK ─────────────────────────────────────
    print("\n[6/7] Training Neural Network (NumPy)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.15, random_state=42
    )
    
    nn = NeuralNetRegressor(n_features=X_tr.shape[1], lr=0.001)
    nn.fit(X_tr, y_tr, X_val, y_val, epochs=300, batch_size=32, verbose=True)
    nn_pred = nn.predict(X_test_scaled)
    results['Neural Network (NumPy)'] = {'y_true': y_test, 'y_pred': nn_pred}
    
    # ─── EVALUATION ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    summary = []
    for name, data in results.items():
        metrics = regression_report(data['y_true'], data['y_pred'], name)
        summary.append(metrics)
    
    print("\n\nSummary:")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # ─── VISUALISATIONS ────────────────────────────────────────
    print("\n\nGenerating visualisations...")
    plot_predictions_comparison(results)
    print("  [x] Predictions comparison")
    plot_feature_importance(gbt_importance, "GBT Feature Importance")
    print("  [x] Feature importance")
    plot_permeability_landscape(df)
    print("  [x] Permeability landscape")
    plot_modification_impact(df)
    print("  [x] Modification impact")
    
    # ─── 7. ANALOGUE GENERATION ─────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALOGUE GENERATION")
    print("=" * 60)
    
    # Pick an example peptide
    example = df.iloc[0]
    print(f"\nParent peptide: {example['sequence']}")
    print(f"  Length: {example['length']} residues")
    print(f"  MW: {example['molecular_weight']:.0f} Da")
    print(f"  log(Papp): {example['log_papp']:.3f}")
    
    # Suggest high-level strategies
    desc = compute_descriptors(example['sequence'])
    desc['is_cyclic'] = example['is_cyclic']
    suggestions = suggest_modifications(example['sequence'], desc)
    
    if suggestions:
        print("\n  Suggested modification strategies:")
        for s in suggestions:
            print(f"    [{s['priority'].upper()}] {s['strategy']}: {s['rationale']}")
    
    # Generate single-residue substitutions
    print("\n  Generating single-residue substitutions...")
    analogues = generate_single_substitutions(
        sequence=example['sequence'],
        model_predict=gbt_model.predict,
        feature_engineer=engineer_features,
        feature_cols=FEATURE_COLS,
        parent_papp=example['log_papp'],
        top_k=15,
    )
    
    if not analogues.empty:
        print(f"\n  Top 5 modifications:")
        for _, row in analogues.head(5).iterrows():
            delta = row['delta_log_papp']
            direction = "+" if delta > 0 else ""
            print(f"    {row['substitution']:8s} | Δlog(Papp) = {direction}{delta:.3f} | "
                  f"{row['papp_fold_change']:.1f}x fold change")
        
        plot_analogue_ranking(analogues, example['log_papp'])
        print("\n  [x] Analogue ranking plot")
    
    print(f"\nAll figures saved to: figures/")
    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
