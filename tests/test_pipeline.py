"""Tests for PeptideAbsorb pipeline."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_peptide_data import generate_dataset, compute_descriptors, generate_random_peptide
from src.feature_engineering import engineer_features, FEATURE_COLS
from src.models import NeuralNetRegressor
from src.analogue_generator import suggest_modifications


def test_dataset_generation():
    df = generate_dataset(n_samples=100)
    assert len(df) == 100
    assert 'log_papp' in df.columns
    assert df['log_papp'].between(-8, -3).all()
    print("  PASS: test_dataset_generation")

def test_descriptors():
    desc = compute_descriptors("AVILF")
    assert desc['length'] == 5
    assert desc['molecular_weight'] > 0
    assert 0 <= desc['hydrophobic_frac'] <= 1
    print("  PASS: test_descriptors")

def test_feature_engineering():
    df = generate_dataset(n_samples=50)
    df = engineer_features(df)
    assert 'lipinski_violations' in df.columns
    assert 'veber_compliant' in df.columns
    assert all(col in df.columns for col in FEATURE_COLS)
    print("  PASS: test_feature_engineering")

def test_neural_net_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 10))
    y = X[:, 0] * 2 + X[:, 1] - 1 + rng.normal(0, 0.1, 100)
    
    nn = NeuralNetRegressor(n_features=10, lr=0.01)
    history = nn.fit(X, y, epochs=100, batch_size=32, verbose=False)
    assert history['train_loss'][-1] < history['train_loss'][0]
    print("  PASS: test_neural_net_regression")

def test_suggestions():
    desc = {'molecular_weight': 900, 'hbd_total': 8, 'psa_total': 1000,
            'logp_mean': -0.5, 'charged_frac': 0.4, 'is_cyclic': False}
    suggestions = suggest_modifications("RKDENSQT", desc)
    assert len(suggestions) > 0
    assert any(s['strategy'] == 'N-methylation' for s in suggestions)
    print("  PASS: test_suggestions")

def test_permeability_range():
    df = generate_dataset(n_samples=500)
    assert df['log_papp'].min() >= -7.5
    assert df['log_papp'].max() <= -3.5
    print("  PASS: test_permeability_range")


if __name__ == '__main__':
    print("Running PeptideAbsorb tests...\n")
    tests = [test_dataset_generation, test_descriptors, test_feature_engineering,
             test_neural_net_regression, test_suggestions, test_permeability_range]
    
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__} - {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
