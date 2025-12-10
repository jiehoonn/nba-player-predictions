"""
Tests for Model Training Module.

Tests high-level functionality of model training pipeline.
Others can extend these tests when adding new models.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_models import train_ridge_models, calculate_baseline_performance


@pytest.fixture
def sample_train_val_data():
    """Create sample train/val data for testing."""
    np.random.seed(42)

    n_samples = 1000
    n_features = 10

    # Create synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create synthetic targets (correlated with features)
    y_pts = X['feature_0'] * 5 + X['feature_1'] * 3 + np.random.randn(n_samples) * 2 + 20
    y_reb = X['feature_2'] * 2 + X['feature_3'] * 1.5 + np.random.randn(n_samples) * 1 + 8
    y_ast = X['feature_4'] * 1.8 + X['feature_5'] * 1.2 + np.random.randn(n_samples) * 0.8 + 5

    # Create rolling features for baseline
    X['PTS_LAST_5'] = y_pts + np.random.randn(n_samples) * 1
    X['REB_LAST_5'] = y_reb + np.random.randn(n_samples) * 0.5
    X['AST_LAST_5'] = y_ast + np.random.randn(n_samples) * 0.4

    # Split into train/val
    split_idx = 800

    train = X.iloc[:split_idx].copy()
    train['PTS'] = y_pts[:split_idx]
    train['REB'] = y_reb[:split_idx]
    train['AST'] = y_ast[:split_idx]

    val = X.iloc[split_idx:].copy()
    val['PTS'] = y_pts[split_idx:]
    val['REB'] = y_reb[split_idx:]
    val['AST'] = y_ast[split_idx:]

    feature_names = [f'feature_{i}' for i in range(n_features)]

    return train, val, feature_names


def test_baseline_calculation(sample_train_val_data):
    """Test baseline (rolling average) performance calculation."""
    _, val, _ = sample_train_val_data

    baselines = calculate_baseline_performance(val)

    # Check structure
    assert 'PTS' in baselines
    assert 'REB' in baselines
    assert 'AST' in baselines

    for target in ['PTS', 'REB', 'AST']:
        assert 'mae' in baselines[target]
        assert 'r2' in baselines[target]

        # Check values are reasonable
        assert baselines[target]['mae'] > 0, f"{target} MAE should be positive"
        assert -1 <= baselines[target]['r2'] <= 1, f"{target} R² should be in [-1, 1]"


def test_ridge_models_trained(sample_train_val_data):
    """Test that Ridge models train successfully."""
    train, val, feature_names = sample_train_val_data

    X_train = train[feature_names]
    X_val = val[feature_names]

    y_train = {
        'PTS': train['PTS'],
        'REB': train['REB'],
        'AST': train['AST']
    }

    y_val = {
        'PTS': val['PTS'],
        'REB': val['REB'],
        'AST': val['AST']
    }

    models, results = train_ridge_models(X_train, y_train, X_val, y_val)

    # Check all three models trained
    assert 'PTS' in models
    assert 'REB' in models
    assert 'AST' in models

    # Check models have predict method
    for target, model in models.items():
        assert hasattr(model, 'predict'), f"{target} model should have predict method"

    # Check results structure
    for target in ['PTS', 'REB', 'AST']:
        assert 'val_mae' in results[target]
        assert 'val_r2' in results[target]
        assert results[target]['val_mae'] > 0
        assert -1 <= results[target]['val_r2'] <= 1


def test_ridge_models_predict_correctly(sample_train_val_data):
    """Test that Ridge models make predictions in correct shape and range."""
    train, val, feature_names = sample_train_val_data

    X_train = train[feature_names]
    X_val = val[feature_names]

    y_train = {
        'PTS': train['PTS'],
        'REB': train['REB'],
        'AST': train['AST']
    }

    y_val = {
        'PTS': val['PTS'],
        'REB': val['REB'],
        'AST': val['AST']
    }

    models, _ = train_ridge_models(X_train, y_train, X_val, y_val)

    # Test PTS model
    pts_pred = models['PTS'].predict(X_val)

    # Check shape
    assert len(pts_pred) == len(val), "Predictions should match validation set size"

    # Check range (PTS should be 0-100 for NBA)
    assert pts_pred.min() >= -10, "PTS predictions unreasonably low"
    assert pts_pred.max() <= 100, "PTS predictions unreasonably high"

    # Check predictions are not all the same
    assert pts_pred.std() > 0.1, "Predictions should have some variance"


def test_ridge_models_beat_constant_prediction(sample_train_val_data):
    """Test that Ridge models beat naive mean prediction."""
    train, val, feature_names = sample_train_val_data

    X_train = train[feature_names]
    X_val = val[feature_names]

    y_train = {
        'PTS': train['PTS'],
        'REB': train['REB'],
        'AST': train['AST']
    }

    y_val = {
        'PTS': val['PTS'],
        'REB': val['REB'],
        'AST': val['AST']
    }

    models, results = train_ridge_models(X_train, y_train, X_val, y_val)

    for target in ['PTS', 'REB', 'AST']:
        # Naive prediction: always predict mean
        mean_pred = np.full(len(val), train[target].mean())
        naive_mae = np.abs(val[target] - mean_pred).mean()

        ridge_mae = results[target]['val_mae']

        assert ridge_mae < naive_mae, \
            f"Ridge {target} MAE ({ridge_mae:.3f}) should beat naive mean ({naive_mae:.3f})"


def test_ridge_alpha_parameter(sample_train_val_data):
    """Test that different alpha values produce different models."""
    train, val, feature_names = sample_train_val_data

    X_train = train[feature_names]
    y_train_pts = train['PTS']
    X_val = val[feature_names]
    y_val_pts = val['PTS']

    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error

    # Train with different alphas
    model_alpha_1 = Ridge(alpha=1.0, random_state=42)
    model_alpha_100 = Ridge(alpha=100.0, random_state=42)

    model_alpha_1.fit(X_train, y_train_pts)
    model_alpha_100.fit(X_train, y_train_pts)

    # Predictions should be different
    pred_alpha_1 = model_alpha_1.predict(X_val)
    pred_alpha_100 = model_alpha_100.predict(X_val)

    # Check predictions are different (not identical)
    assert not np.allclose(pred_alpha_1, pred_alpha_100), \
        "Different alpha values should produce different predictions"


def test_model_serialization(sample_train_val_data, tmp_path):
    """Test that trained models can be saved and loaded."""
    train, val, feature_names = sample_train_val_data

    X_train = train[feature_names]
    X_val = val[feature_names]

    y_train = {
        'PTS': train['PTS'],
        'REB': train['REB'],
        'AST': train['AST']
    }

    y_val = {
        'PTS': val['PTS'],
        'REB': val['REB'],
        'AST': val['AST']
    }

    models, _ = train_ridge_models(X_train, y_train, X_val, y_val)

    # Save model
    model_path = tmp_path / 'test_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(models['PTS'], f)

    # Load model
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    # Check loaded model makes same predictions
    original_pred = models['PTS'].predict(X_val)
    loaded_pred = loaded_model.predict(X_val)

    assert np.allclose(original_pred, loaded_pred), \
        "Loaded model should make identical predictions"


def test_ridge_coefficients_reasonable(sample_train_val_data):
    """Test that Ridge coefficients are in reasonable range."""
    train, val, feature_names = sample_train_val_data

    X_train = train[feature_names]
    X_val = val[feature_names]
    y_train = {
        'PTS': train['PTS'],
        'REB': train['REB'],
        'AST': train['AST']
    }
    y_val = {
        'PTS': val['PTS'],
        'REB': val['REB'],
        'AST': val['AST']
    }

    models, _ = train_ridge_models(X_train, y_train, X_val, y_val)

    # Check PTS model coefficients
    coefs = models['PTS'].coef_

    # Coefficients should be finite (not NaN or inf)
    assert np.all(np.isfinite(coefs)), "Coefficients should be finite"

    # Coefficients should not be all zero
    assert np.any(coefs != 0), "Not all coefficients should be zero"

    # Coefficients should be in reasonable range
    assert np.abs(coefs).max() < 1000, "Coefficients unreasonably large"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
