"""
Pipeline integration tests.

Tests the end-to-end pipeline: data → features → models → predictions.
"""

import pytest
from pathlib import Path
import pandas as pd
import json


def test_data_exists():
    """Test that processed data files exist."""
    data_dir = Path('data/processed')

    required_files = [
        'train.parquet',
        'val.parquet',
        'test.parquet',
        'feature_metadata_v2.json'
    ]

    for file in required_files:
        filepath = data_dir / file
        assert filepath.exists(), f"Missing required file: {file}"

    print("✓ All required data files exist")


def test_feature_metadata_valid():
    """Test that feature metadata is valid JSON."""
    metadata_path = Path('data/processed/feature_metadata_v2.json')

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Check required keys
    assert 'feature_names' in metadata
    assert 'target_columns' in metadata
    assert 'feature_breakdown' in metadata

    # Check feature count
    assert len(metadata['feature_names']) == 38, "Should have 38 features"

    # Check targets
    assert set(metadata['target_columns']) == {'PTS', 'REB', 'AST'}

    print(f"✓ Feature metadata valid: {len(metadata['feature_names'])} features")


def test_train_val_test_splits():
    """Test that train/val/test splits are proper temporal splits."""
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')

    # Convert to datetime
    train['GAME_DATE'] = pd.to_datetime(train['GAME_DATE'])
    val['GAME_DATE'] = pd.to_datetime(val['GAME_DATE'])
    test['GAME_DATE'] = pd.to_datetime(test['GAME_DATE'])

    # Check temporal ordering
    assert train['GAME_DATE'].max() < val['GAME_DATE'].min(), \
        "Train set leaks into validation set"

    assert val['GAME_DATE'].max() < test['GAME_DATE'].min(), \
        "Validation set leaks into test set"

    # Check no overlap
    train_dates = set(train['GAME_DATE'].dt.date)
    val_dates = set(val['GAME_DATE'].dt.date)
    test_dates = set(test['GAME_DATE'].dt.date)

    assert len(train_dates & val_dates) == 0, "Train/val overlap"
    assert len(val_dates & test_dates) == 0, "Val/test overlap"
    assert len(train_dates & test_dates) == 0, "Train/test overlap"

    print(f"✓ Temporal splits valid:")
    print(f"  Train: {train['GAME_DATE'].min().date()} to {train['GAME_DATE'].max().date()}")
    print(f"  Val:   {val['GAME_DATE'].min().date()} to {val['GAME_DATE'].max().date()}")
    print(f"  Test:  {test['GAME_DATE'].min().date()} to {test['GAME_DATE'].max().date()}")


def test_no_missing_values():
    """Test that feature matrices have no missing values."""
    train = pd.read_parquet('data/processed/train.parquet')

    with open('data/processed/feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']
    X_train = train[feature_names]

    missing = X_train.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values in training features"

    print(f"✓ No missing values in {len(feature_names)} features")


def test_models_exist():
    """Test that trained models are saved."""
    models_dir = Path('results/models')

    required_models = [
        'best_ridge_pts.pkl',
        'best_ridge_reb.pkl',
        'best_ridge_ast.pkl'
    ]

    for model_file in required_models:
        filepath = models_dir / model_file
        assert filepath.exists(), f"Missing model: {model_file}"

    print("✓ All Ridge models exist")


def test_results_files_exist():
    """Test that result JSON files exist."""
    results_dir = Path('results')

    required_results = [
        'baseline_models_results.json',
        'advanced_models_results.json',
        'final_test_results.json'
    ]

    for result_file in required_results:
        filepath = results_dir / result_file
        assert filepath.exists(), f"Missing result file: {result_file}"

    print("✓ All result files exist")


def test_final_results_performance():
    """Test that final test results meet minimum performance thresholds."""
    with open('results/final_test_results.json', 'r') as f:
        results = json.load(f)

    # Test PTS performance
    pts_mae = results['results']['PTS']['test_mae']
    pts_r2 = results['results']['PTS']['test_r2']

    assert pts_mae < 6.0, f"PTS MAE too high: {pts_mae:.3f} (should be < 6.0)"
    assert pts_r2 > 0.45, f"PTS R² too low: {pts_r2:.3f} (should be > 0.45)"

    # Test REB performance
    reb_mae = results['results']['REB']['test_mae']
    reb_r2 = results['results']['REB']['test_r2']

    assert reb_mae < 2.5, f"REB MAE too high: {reb_mae:.3f} (should be < 2.5)"
    assert reb_r2 > 0.40, f"REB R² too low: {reb_r2:.3f} (should be > 0.40)"

    # Test AST performance
    ast_mae = results['results']['AST']['test_mae']
    ast_r2 = results['results']['AST']['test_r2']

    assert ast_mae < 2.0, f"AST MAE too high: {ast_mae:.3f} (should be < 2.0)"
    assert ast_r2 > 0.45, f"AST R² too low: {ast_r2:.3f} (should be > 0.45)"

    print("✓ Model performance meets thresholds:")
    print(f"  PTS: MAE={pts_mae:.3f}, R²={pts_r2:.3f}")
    print(f"  REB: MAE={reb_mae:.3f}, R²={reb_r2:.3f}")
    print(f"  AST: MAE={ast_mae:.3f}, R²={ast_r2:.3f}")


def test_generalization():
    """Test that models generalize well (val vs test performance)."""
    with open('results/final_test_results.json', 'r') as f:
        results = json.load(f)

    for target in ['PTS', 'REB', 'AST']:
        degradation = results['results'][target]['mae_degradation_pct']

        # Warning if degradation > 5%
        if abs(degradation) > 5:
            print(f"⚠️  {target}: High degradation ({degradation:+.1f}%)")

        # Fail if degradation > 10%
        assert abs(degradation) < 10, \
            f"{target}: Excessive degradation ({degradation:+.1f}%) - possible overfitting"

    print("✓ Models generalize well (degradation < 10%)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
